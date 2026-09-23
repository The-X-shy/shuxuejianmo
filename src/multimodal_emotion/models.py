"""PyTorch models for aligned, missing-aware multimodal emotion features.

The module deliberately contains model definitions only.  In particular, B3/B4
and M0/M1 are architectural pairs: their different missing-data training
protocols belong to the trainer, not to the forward pass.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


MODALITIES: Tuple[str, str, str] = ("text", "audio", "vision")
DEFAULT_INPUT_DIMS: Dict[str, int] = {"text": 768, "audio": 74, "vision": 35}


def _validate_class_prior(class_prior: Optional[Sequence[float] | Tensor]) -> Tensor:
    if class_prior is None:
        prior = torch.full((3,), 1.0 / 3.0, dtype=torch.float32)
    else:
        prior = torch.as_tensor(class_prior, dtype=torch.float32).detach().clone()
        if prior.shape != (3,):
            raise ValueError("class_prior must contain three class probabilities")
        if not torch.isfinite(prior).all() or (prior < 0).any() or prior.sum() <= 0:
            raise ValueError("class_prior must be finite, non-negative, and have positive sum")
        prior = prior / prior.sum()
    return prior


class _EmotionModel(nn.Module):
    """Shared input, prior, and output handling for all model variants."""

    def __init__(
        self,
        class_prior: Optional[Sequence[float] | Tensor] = None,
        regression_prior: float = 0.0,
    ) -> None:
        super().__init__()
        regression_prior_tensor = torch.as_tensor(regression_prior, dtype=torch.float32).detach().clone()
        if regression_prior_tensor.numel() != 1 or not torch.isfinite(regression_prior_tensor).all():
            raise ValueError("regression_prior must be a finite scalar")
        self.register_buffer("train_class_prior", _validate_class_prior(class_prior))
        self.register_buffer(
            "train_regression_mean",
            regression_prior_tensor.reshape(()).clamp(-3.0, 3.0),
        )

    def set_prior(
        self,
        class_prior: Sequence[float] | Tensor,
        regression_mean: float,
    ) -> None:
        """Set invalid-sample outputs from train-split statistics."""
        prior = _validate_class_prior(class_prior).to(self.train_class_prior.device)
        mean = torch.as_tensor(
            regression_mean,
            dtype=self.train_regression_mean.dtype,
            device=self.train_regression_mean.device,
        )
        if mean.numel() != 1 or not torch.isfinite(mean).all():
            raise ValueError("regression_mean must be a finite scalar")
        self.train_class_prior.copy_(prior)
        self.train_regression_mean.copy_(mean.reshape(()).clamp(-3.0, 3.0))

    def _canonicalize_features(
        self, text: Tensor, audio: Tensor, vision: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        values = (text, audio, vision)
        normalized = []
        for name, value in zip(MODALITIES, values):
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} input must be a torch.Tensor")
            if value.ndim == 1:  # [D] -> [1, 1, D]
                value = value.unsqueeze(0).unsqueeze(0)
            elif value.ndim == 2:  # [B, D] -> [B, 1, D]
                value = value.unsqueeze(1)
            elif value.ndim != 3:
                raise ValueError(f"{name} input must have shape [D], [B,D], or [B,T,D]")
            if hasattr(self, "input_dims"):
                expected = self.input_dims[name]
                if value.shape[-1] != expected:
                    raise ValueError(f"{name} feature dimension is {value.shape[-1]}, expected {expected}")
            normalized.append(value)
        batch_time = {(x.shape[0], x.shape[1]) for x in normalized}
        if len(batch_time) != 1:
            raise ValueError("text, audio, and vision must have matching batch and time dimensions")
        if len({x.device for x in normalized}) != 1:
            raise ValueError("all modality tensors must be on the same device")
        return normalized[0], normalized[1], normalized[2]

    @staticmethod
    def _canonicalize_valid_mask(
        valid_mask: Optional[Tensor], batch: int, steps: int, device: torch.device
    ) -> Tensor:
        if valid_mask is None:
            return torch.ones((batch, steps), dtype=torch.bool, device=device)
        mask = torch.as_tensor(valid_mask, device=device, dtype=torch.bool)
        if mask.ndim == 1:
            if batch == 1 and mask.shape[0] == steps:
                mask = mask.unsqueeze(0)
            elif steps == 1 and mask.shape[0] == batch:
                mask = mask.unsqueeze(1)
        if mask.shape != (batch, steps):
            raise ValueError(f"valid_mask must have shape [{batch},{steps}]")
        return mask

    @staticmethod
    def _canonicalize_observed_mask(
        observed_mask: Optional[Tensor], batch: int, steps: int, device: torch.device
    ) -> Tensor:
        if observed_mask is None:
            return torch.ones((batch, 3, steps), dtype=torch.bool, device=device)
        mask = torch.as_tensor(observed_mask, device=device, dtype=torch.bool)
        if mask.ndim == 2:
            if batch == 1 and mask.shape == (3, steps):
                mask = mask.unsqueeze(0)
            elif steps == 1 and mask.shape == (batch, 3):
                mask = mask.unsqueeze(-1)
        if mask.shape != (batch, 3, steps):
            raise ValueError(f"observed_mask must have shape [{batch},3,{steps}]")
        return mask

    @staticmethod
    def _normalize_weights(weights: Tensor, observed: Tensor, *, name: str) -> Tensor:
        """Apply the observation mask and normalize over available modalities."""
        if weights.shape != observed.shape:
            raise ValueError(f"{name} must have shape [B,3,T]")
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError(f"{name} must contain finite, non-negative values")
        masked = torch.where(observed, weights, torch.zeros_like(weights))
        totals = masked.sum(dim=1, keepdim=True)
        has_observation = observed.any(dim=1, keepdim=True)
        if (has_observation & (totals <= 0)).any():
            raise ValueError(f"{name} must assign positive total weight to every observed time step")
        return masked / totals.clamp_min(torch.finfo(masked.dtype).tiny)

    def _output(
        self,
        logits: Tensor,
        regression: Tensor,
        valid_mask: Tensor,
        observed_mask: Tensor,
        weights: Tensor,
        gate_weights: Optional[Tensor],
    ) -> Dict[str, Optional[Tensor]]:
        invalid_sample = ~valid_mask.any(dim=1)
        no_observed_sample = ~observed_mask.any(dim=(1, 2))
        if invalid_sample.any():
            prior_logits = self.train_class_prior.to(dtype=logits.dtype).clamp_min(
                torch.finfo(logits.dtype).tiny
            ).log()
            logits = torch.where(invalid_sample[:, None], prior_logits[None, :], logits)
            prior_reg = self.train_regression_mean.to(dtype=regression.dtype)
            regression = torch.where(invalid_sample, prior_reg.expand_as(regression), regression)
        return {
            "logits": logits,
            "regression": regression,
            # `weights` is the sample-level, normalized audit summary. For B0-B4
            # it is a visible-ratio summary; for temporal models it aggregates gates.
            "weights": weights,
            "modality_weights": weights,
            # Per-position fusion weights are available for B5/M0/M1.
            "gate_weights": gate_weights,
            "invalid_sample": invalid_sample,
            "low_information": no_observed_sample,
            "valid_mask": valid_mask,
            "observed_mask": observed_mask,
        }


class _PredictionHeads(nn.Module):
    def __init__(self, hidden_dim: int = 128, heads_hidden_dim: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, heads_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(heads_hidden_dim, 3),
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, heads_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(heads_hidden_dim, 1),
        )

    def forward(self, representation: Tensor) -> Tuple[Tensor, Tensor]:
        logits = self.classifier(representation)
        regression = 3.0 * torch.tanh(self.regressor(representation).squeeze(-1))
        return logits, regression


class BaselineMLP(_EmotionModel):
    """B0-B4 masked-mean baselines; B3 and B4 differ in trainer protocol only."""

    def __init__(
        self,
        model_id: str,
        input_dims: Optional[Dict[str, int]] = None,
        dropout: float = 0.2,
        class_prior: Optional[Sequence[float] | Tensor] = None,
        regression_prior: float = 0.0,
    ) -> None:
        if model_id not in {"B0", "B1", "B2", "B3", "B4"}:
            raise ValueError("model_id must be one of B0, B1, B2, B3, B4")
        super().__init__(class_prior, regression_prior)
        self.model_id = model_id
        self.input_dims = dict(DEFAULT_INPUT_DIMS if input_dims is None else input_dims)
        if set(self.input_dims) != set(MODALITIES) or any(v <= 0 for v in self.input_dims.values()):
            raise ValueError("input_dims must provide positive dimensions for text, audio, and vision")
        self.modalities = (MODALITIES[0],) if model_id == "B0" else (
            (MODALITIES[1],) if model_id == "B1" else (
                (MODALITIES[2],) if model_id == "B2" else MODALITIES
            )
        )
        input_dim = sum(self.input_dims[m] for m in self.modalities) + len(self.modalities)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = _PredictionHeads(128, 64, dropout)

    def forward(
        self,
        text: Tensor,
        audio: Tensor,
        vision: Tensor,
        valid_mask: Optional[Tensor] = None,
        observed_mask: Optional[Tensor] = None,
        *,
        observed_mask_override: Optional[Tensor] = None,
        gate_weights_override: Optional[Tensor] = None,
    ) -> Dict[str, Optional[Tensor]]:
        if gate_weights_override is not None:
            raise ValueError("gate_weights_override is only supported by B5, M0, and M1")
        text, audio, vision = self._canonicalize_features(text, audio, vision)
        batch, steps = text.shape[:2]
        valid = self._canonicalize_valid_mask(valid_mask, batch, steps, text.device)
        observed = self._canonicalize_observed_mask(observed_mask, batch, steps, text.device)
        if observed_mask_override is not None:
            observed = self._canonicalize_observed_mask(
                observed_mask_override, batch, steps, text.device
            )
        observed = observed & valid[:, None, :]

        values = {"text": text, "audio": audio, "vision": vision}
        summaries = []
        ratios = []
        for name in MODALITIES:
            mask = observed[:, MODALITIES.index(name), :]
            value = values[name]
            safe_value = torch.where(mask.unsqueeze(-1), value, torch.zeros_like(value))
            count = mask.sum(dim=1, keepdim=True)
            mean = safe_value.sum(dim=1) / count.clamp_min(1).to(value.dtype)
            valid_count = valid.sum(dim=1, keepdim=True)
            ratio = count.to(value.dtype) / valid_count.clamp_min(1).to(value.dtype)
            mean = torch.where(count > 0, mean, torch.zeros_like(mean))
            ratio = torch.where(valid_count > 0, ratio, torch.zeros_like(ratio))
            summaries.append(mean)
            ratios.append(ratio)
        by_name = dict(zip(MODALITIES, zip(summaries, ratios)))
        encoded_input = torch.cat(
            [item for name in self.modalities for item in by_name[name]], dim=-1
        )
        representation = self.encoder(encoded_input)
        logits, regression = self.heads(representation)

        selected_ratios = torch.cat([by_name[name][1] for name in self.modalities], dim=-1)
        selected_share = selected_ratios / selected_ratios.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        weights = text.new_zeros((batch, 3))
        for index, name in enumerate(self.modalities):
            weights[:, MODALITIES.index(name)] = selected_share[:, index]
        return self._output(logits, regression, valid, observed, weights, None)


class B0(BaselineMLP):
    def __init__(self, input_dims=None, dropout=0.2, class_prior=None, regression_prior=0.0):
        super().__init__("B0", input_dims, dropout, class_prior, regression_prior)


class B1(BaselineMLP):
    def __init__(self, input_dims=None, dropout=0.2, class_prior=None, regression_prior=0.0):
        super().__init__("B1", input_dims, dropout, class_prior, regression_prior)


class B2(BaselineMLP):
    def __init__(self, input_dims=None, dropout=0.2, class_prior=None, regression_prior=0.0):
        super().__init__("B2", input_dims, dropout, class_prior, regression_prior)


class B3(BaselineMLP):
    def __init__(self, input_dims=None, dropout=0.2, class_prior=None, regression_prior=0.0):
        super().__init__("B3", input_dims, dropout, class_prior, regression_prior)


class B4(BaselineMLP):
    def __init__(self, input_dims=None, dropout=0.2, class_prior=None, regression_prior=0.0):
        super().__init__("B4", input_dims, dropout, class_prior, regression_prior)


class TemporalFusion(_EmotionModel):
    """Shared temporal body for fixed equal-weight B5 and dynamic M0/M1."""

    def __init__(
        self,
        fusion: str = "dynamic",
        input_dims: Optional[Dict[str, int]] = None,
        hidden_dim: int = 128,
        sequence_length: int = 50,
        dropout: float = 0.2,
        class_prior: Optional[Sequence[float] | Tensor] = None,
        regression_prior: float = 0.0,
    ) -> None:
        if fusion not in {"dynamic", "equal"}:
            raise ValueError("fusion must be 'dynamic' or 'equal'")
        super().__init__(class_prior, regression_prior)
        self.fusion = fusion
        self.input_dims = dict(DEFAULT_INPUT_DIMS if input_dims is None else input_dims)
        if set(self.input_dims) != set(MODALITIES) or any(v <= 0 for v in self.input_dims.values()):
            raise ValueError("input_dims must provide positive dimensions for text, audio, and vision")
        if hidden_dim <= 0 or hidden_dim % 4 != 0 or sequence_length <= 0:
            raise ValueError("hidden_dim must be positive and divisible by four; sequence_length must be positive")
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.projections = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(self.input_dims[name], hidden_dim),
                    nn.LayerNorm(hidden_dim, eps=1e-5),
                    nn.GELU(),
                )
                for name in MODALITIES
            }
        )
        if fusion == "dynamic":
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim + 4, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )
        else:
            self.gate = None
        self.state_projection = nn.Linear(3, hidden_dim, bias=False)
        self.all_missing = nn.Parameter(torch.zeros(hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=256,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            layer_norm_eps=1e-5,
        )
        self.temporal = nn.TransformerEncoder(
            layer,
            num_layers=2,
            norm=nn.LayerNorm(hidden_dim, eps=1e-5),
        )
        self.heads = _PredictionHeads(hidden_dim, 64, dropout)
        self.register_buffer(
            "position_encoding", self._make_position_encoding(sequence_length, hidden_dim).unsqueeze(0)
        )

    @staticmethod
    def _make_position_encoding(length: int, hidden_dim: int) -> Tensor:
        positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / hidden_dim)
        )
        encoding = torch.zeros(length, hidden_dim, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * scale)
        encoding[:, 1::2] = torch.cos(positions * scale[: encoding[:, 1::2].shape[1]])
        return encoding

    def _position_encoding(self, steps: int, reference: Tensor) -> Tensor:
        if steps <= self.position_encoding.shape[1]:
            encoding = self.position_encoding[:, :steps]
        else:
            encoding = self._make_position_encoding(steps, self.hidden_dim).unsqueeze(0)
        return encoding.to(device=reference.device, dtype=reference.dtype)

    def _sample_ratios(self, observed: Tensor, valid: Tensor, dtype: torch.dtype) -> Tensor:
        visible = observed.sum(dim=2).to(dtype)
        valid_count = valid.sum(dim=1, keepdim=True).to(dtype)
        return visible / valid_count.clamp_min(1.0)

    def _gate_weights(
        self,
        projected: Tensor,
        observed: Tensor,
        valid: Tensor,
        gate_weights_override: Optional[Tensor],
    ) -> Tensor:
        batch, _, steps, _ = projected.shape
        if gate_weights_override is not None:
            supplied = torch.as_tensor(
                gate_weights_override, device=projected.device, dtype=projected.dtype
            )
            return self._normalize_weights(supplied, observed, name="gate_weights_override")
        if self.fusion == "equal":
            equal = observed.to(projected.dtype)
            return equal / equal.sum(dim=1, keepdim=True).clamp_min(1.0)

        ratios = self._sample_ratios(observed, valid, projected.dtype)
        state = observed.permute(0, 2, 1).unsqueeze(1).expand(batch, 3, steps, 3)
        ratio_features = ratios[:, :, None, None].expand(batch, 3, steps, 1)
        gate_input = torch.cat((projected, ratio_features, state), dim=-1)
        scores = self.gate(gate_input).squeeze(-1)
        has_observation = observed.any(dim=1, keepdim=True)
        masked_scores = scores.masked_fill(~observed, torch.finfo(scores.dtype).min)
        # Avoid softmax([-inf, -inf, -inf]) at all-missing time steps.
        masked_scores = torch.where(has_observation, masked_scores, torch.zeros_like(masked_scores))
        alpha = torch.softmax(masked_scores, dim=1) * observed.to(scores.dtype)
        return alpha / alpha.sum(dim=1, keepdim=True).clamp_min(torch.finfo(alpha.dtype).tiny)

    def forward(
        self,
        text: Tensor,
        audio: Tensor,
        vision: Tensor,
        valid_mask: Optional[Tensor] = None,
        observed_mask: Optional[Tensor] = None,
        *,
        observed_mask_override: Optional[Tensor] = None,
        gate_weights_override: Optional[Tensor] = None,
    ) -> Dict[str, Optional[Tensor]]:
        text, audio, vision = self._canonicalize_features(text, audio, vision)
        batch, steps = text.shape[:2]
        valid = self._canonicalize_valid_mask(valid_mask, batch, steps, text.device)
        observed = self._canonicalize_observed_mask(observed_mask, batch, steps, text.device)
        if observed_mask_override is not None:
            observed = self._canonicalize_observed_mask(
                observed_mask_override, batch, steps, text.device
            )
        observed = observed & valid[:, None, :]

        values = (text, audio, vision)
        projected = []
        for index, (name, value) in enumerate(zip(MODALITIES, values)):
            mask = observed[:, index, :]
            # Zero hidden/missing inputs before projection, then suppress projection
            # biases at absent positions. Changing a hidden value cannot leak in.
            safe_value = torch.where(mask.unsqueeze(-1), value, torch.zeros_like(value))
            representation = self.projections[name](safe_value)
            representation = torch.where(
                mask.unsqueeze(-1), representation, torch.zeros_like(representation)
            )
            projected.append(representation)
        projected_tensor = torch.stack(projected, dim=1)  # [B,3,T,H]
        alpha = self._gate_weights(projected_tensor, observed, valid, gate_weights_override)
        fused = (projected_tensor * alpha.unsqueeze(-1)).sum(dim=1)
        position_missing = ~observed.any(dim=1)
        fused = torch.where(position_missing.unsqueeze(-1), self.all_missing[None, None, :], fused)
        state = observed.permute(0, 2, 1).to(fused.dtype)
        fused = fused + self.state_projection(state) + self._position_encoding(steps, fused)
        fused = torch.where(valid.unsqueeze(-1), fused, torch.zeros_like(fused))

        # PyTorch attention returns NaNs if every key is masked. Unmask one zero
        # dummy position for those samples; their final prediction is the prior.
        invalid_sample = ~valid.any(dim=1)
        safe_valid = valid.clone()
        if invalid_sample.any():
            safe_valid[invalid_sample, 0] = True
        sequence = self.temporal(fused, src_key_padding_mask=~safe_valid)
        pooled = (sequence * valid.unsqueeze(-1).to(sequence.dtype)).sum(dim=1)
        pooled = pooled / valid.sum(dim=1, keepdim=True).clamp_min(1).to(sequence.dtype)
        logits, regression = self.heads(pooled)

        accumulated = alpha.sum(dim=2)
        weights = accumulated / accumulated.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(accumulated.dtype).tiny
        )
        return self._output(logits, regression, valid, observed, weights, alpha)


class B5(TemporalFusion):
    def __init__(
        self, input_dims=None, hidden_dim=128, sequence_length=50, dropout=0.2,
        class_prior=None, regression_prior=0.0,
    ):
        super().__init__("equal", input_dims, hidden_dim, sequence_length, dropout, class_prior, regression_prior)
        self.model_id = "B5"


class M0(TemporalFusion):
    """Dynamic-gate model trained on complete observations (trainer-defined)."""

    def __init__(
        self, input_dims=None, hidden_dim=128, sequence_length=50, dropout=0.2,
        class_prior=None, regression_prior=0.0,
    ):
        super().__init__("dynamic", input_dims, hidden_dim, sequence_length, dropout, class_prior, regression_prior)
        self.model_id = "M0"


class M1(TemporalFusion):
    """Dynamic-gate model trained with the agreed missing-data protocol."""

    def __init__(
        self, input_dims=None, hidden_dim=128, sequence_length=50, dropout=0.2,
        class_prior=None, regression_prior=0.0,
    ):
        super().__init__("dynamic", input_dims, hidden_dim, sequence_length, dropout, class_prior, regression_prior)
        self.model_id = "M1"


def build_model(model_id: str, **kwargs) -> _EmotionModel:
    """Construct one of the required experiment models by its Spec ID."""
    constructors = {"B0": B0, "B1": B1, "B2": B2, "B3": B3, "B4": B4,
                    "B5": B5, "M0": M0, "M1": M1}
    try:
        return constructors[model_id](**kwargs)
    except KeyError as exc:
        raise ValueError(f"unknown model_id {model_id!r}; expected {', '.join(constructors)}") from exc


# The work-name used in the Spec for the dynamic, missing-aware main model.
MissingAwareFusion = TemporalFusion


__all__ = [
    "MODALITIES", "DEFAULT_INPUT_DIMS", "BaselineMLP", "TemporalFusion", "MissingAwareFusion",
    "B0", "B1", "B2", "B3", "B4", "B5", "M0", "M1", "build_model",
]
