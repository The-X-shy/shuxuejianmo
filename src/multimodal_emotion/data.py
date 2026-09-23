"""Data contracts and preprocessing for the official aligned50 features.

The public boundary is :class:`AlignedSplit.from_arrays`: it accepts the three
official feature arrays in their existing order, a verified position-validity
mask ``P``, and optional labels. It never splits, sorts, or shuffles samples.
The modality order is always ``T, A, V`` with feature sizes ``768, 74, 35``.

Three masks have separate meanings throughout this module:

* ``P`` / ``valid_mask``: a sequence position exists;
* ``O0`` / ``original_observed_mask``: a modality had an original observation;
* ``C`` / ``corruption_mask``: a later artificial deletion, constrained to
  ``P & O0``. ``O`` / ``observed_mask`` is ``O0 & ~C``.

An exactly all-zero feature row inside P is treated as naturally missing. Any
row containing NaN or infinity is logged as an invalid row, replaced with
zeros, and marked naturally unobserved; the sample itself is retained. Fit
normalization only with :func:`fit_normalizer` on the official train split.
Each feature dimension uses train rows satisfying ``P & O0``, population
standard deviation (``ddof=0``), and scale 1 for a zero-variance dimension.
Transforming any split safely fills every unobserved row with zero after
normalization.

NumPy is the only runtime dependency. ``valid_mask`` must come from a verified
source; this code intentionally does not guess padding from all-modality zeros.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


MODALITIES: Tuple[str, ...] = ("T", "A", "V")
FEATURE_DIMS: Mapping[str, int] = {"T": 768, "A": 74, "V": 35}
_INPUT_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "T": ("T", "text", "x_text"),
    "A": ("A", "audio", "x_audio"),
    "V": ("V", "vision", "visual", "x_vision"),
}


class DataContractError(ValueError):
    """Raised when an input does not satisfy the aligned50 data contract."""


@dataclass(frozen=True)
class InvalidValueEvent:
    """Audit record for a feature row containing at least one NaN/Inf value."""

    sample_id: str
    modality: str
    timestep: int
    nonfinite_dimensions: Tuple[int, ...]


def _normalise_feature_mapping(
    features: Optional[Mapping[str, np.ndarray]],
    text: Optional[np.ndarray],
    audio: Optional[np.ndarray],
    vision: Optional[np.ndarray],
) -> Dict[str, np.ndarray]:
    supplied: Dict[str, np.ndarray] = {}
    if features is not None:
        for key, value in features.items():
            canonical = None
            for modality, aliases in _INPUT_ALIASES.items():
                if key in aliases:
                    canonical = modality
                    break
            if canonical is None:
                raise DataContractError("Unknown feature key {!r}; expected T/A/V.".format(key))
            if canonical in supplied:
                raise DataContractError("Feature {!r} was supplied more than once.".format(canonical))
            supplied[canonical] = value
    for modality, value in (("T", text), ("A", audio), ("V", vision)):
        if value is not None:
            if modality in supplied:
                raise DataContractError("Feature {!r} was supplied both directly and in features.".format(modality))
            supplied[modality] = value
    missing = [name for name in MODALITIES if name not in supplied]
    if missing:
        raise DataContractError("Missing aligned feature arrays: {}.".format(", ".join(missing)))
    return supplied


@dataclass(frozen=True)
class AlignedSplit:
    """Sanitized raw aligned50 arrays plus labels and the P/O0 masks.

    ``features`` maps T/A/V to float32 arrays shaped ``[N, 50, D]``.
    ``valid_mask`` is P shaped ``[N, 50]`` and
    ``original_observed_mask`` is O0 shaped ``[N, 3, 50]``. Input order and
    sample IDs are preserved exactly as supplied.
    """

    split: str
    sample_ids: Tuple[str, ...]
    features: Mapping[str, np.ndarray]
    valid_mask: np.ndarray
    original_observed_mask: np.ndarray
    y_class: Optional[np.ndarray] = None
    y_reg: Optional[np.ndarray] = None
    invalid_value_events: Tuple[InvalidValueEvent, ...] = ()

    @classmethod
    def from_arrays(
        cls,
        *,
        split: str,
        sample_ids: Sequence[str],
        valid_mask: np.ndarray,
        features: Optional[Mapping[str, np.ndarray]] = None,
        text: Optional[np.ndarray] = None,
        audio: Optional[np.ndarray] = None,
        vision: Optional[np.ndarray] = None,
        y_class: Optional[Sequence[int]] = None,
        y_reg: Optional[Sequence[float]] = None,
    ) -> "AlignedSplit":
        """Validate arrays and derive P/O0 without changing official order.

        ``valid_mask`` is required because padding cannot safely be inferred
        from the feature values. Supply arrays either through ``features``
        (keys T/A/V, text/audio/vision, or x_text/x_audio/x_vision) or through
        the three named arguments.
        """
        if not isinstance(split, str) or not split.strip():
            raise DataContractError("split must be a non-empty name such as train/valid/test.")
        ids = tuple(str(value) for value in sample_ids)
        if not ids:
            raise DataContractError("A split must contain at least one sample.")
        if len(set(ids)) != len(ids):
            raise DataContractError("sample_ids must be unique within a split.")

        raw_features = _normalise_feature_mapping(features, text, audio, vision)
        valid = np.asarray(valid_mask, dtype=np.bool_)
        if valid.shape != (len(ids), 50):
            raise DataContractError("valid_mask must have shape [N, 50]; got {}.".format(valid.shape))

        clean: Dict[str, np.ndarray] = {}
        observed_parts = []
        events = []
        for modality in MODALITIES:
            expected = (len(ids), 50, FEATURE_DIMS[modality])
            raw = np.asarray(raw_features[modality])
            if raw.shape != expected:
                raise DataContractError(
                    "{} feature array must have shape {}; got {}.".format(modality, expected, raw.shape)
                )
            if not np.issubdtype(raw.dtype, np.number):
                raise DataContractError("{} features must be numeric.".format(modality))

            # Work in float64 for safe finite checks before the canonical float32 cast.
            values = np.asarray(raw, dtype=np.float64)
            # A finite float64 can still overflow the contract's float32 model
            # inputs, so check the canonical cast before accepting a row.
            with np.errstate(over="ignore", invalid="ignore"):
                float32_values = values.astype(np.float32)
            row_finite = np.isfinite(float32_values).all(axis=-1)
            bad_rows = np.argwhere(~row_finite)
            for sample_index, timestep in bad_rows:
                bad_dims = tuple(np.flatnonzero(~np.isfinite(float32_values[sample_index, timestep])).tolist())
                events.append(
                    InvalidValueEvent(
                        sample_id=ids[int(sample_index)],
                        modality=modality,
                        timestep=int(timestep),
                        nonfinite_dimensions=bad_dims,
                    )
                )

            modality_observed = valid & row_finite & np.any(values != 0.0, axis=-1)
            safe_values = np.zeros(expected, dtype=np.float32)
            # Assign only safe, eligible rows. This also makes padding, natural
            # zero rows, and invalid rows harmless to downstream arithmetic.
            eligible = modality_observed
            if np.any(eligible):
                safe_values[eligible] = float32_values[eligible]
            clean[modality] = safe_values
            observed_parts.append(modality_observed)

        original_observed = np.stack(observed_parts, axis=1)
        class_array = _validate_class_labels(y_class, len(ids))
        reg_array = _validate_regression_labels(y_reg, len(ids))
        return cls(
            split=split,
            sample_ids=ids,
            features=clean,
            valid_mask=valid.copy(),
            original_observed_mask=original_observed,
            y_class=class_array,
            y_reg=reg_array,
            invalid_value_events=tuple(events),
        )

    @property
    def P(self) -> np.ndarray:
        """Alias for the sequence-validity mask."""
        return self.valid_mask

    @property
    def O0(self) -> np.ndarray:
        """Alias for the original-observation mask."""
        return self.original_observed_mask


def _validate_class_labels(values: Optional[Sequence[int]], n: int) -> Optional[np.ndarray]:
    if values is None:
        return None
    labels = np.asarray(values)
    if labels.shape != (n,):
        raise DataContractError("y_class must have shape [N].")
    if not np.issubdtype(labels.dtype, np.integer):
        if not np.all(np.isfinite(labels)) or not np.all(labels == np.floor(labels)):
            raise DataContractError("y_class must contain integer class IDs 0/1/2.")
    labels = labels.astype(np.int64, copy=True)
    if np.any((labels < 0) | (labels > 2)):
        raise DataContractError("y_class IDs must use the fixed order 0=Negative, 1=Neutral, 2=Positive.")
    return labels


def _validate_regression_labels(values: Optional[Sequence[float]], n: int) -> Optional[np.ndarray]:
    if values is None:
        return None
    labels = np.asarray(values, dtype=np.float64)
    if labels.shape != (n,):
        raise DataContractError("y_reg must have shape [N].")
    if not np.all(np.isfinite(labels)):
        raise DataContractError("y_reg labels must be finite; resolve source label errors before training.")
    return labels.astype(np.float32)


@dataclass(frozen=True)
class SampleBatch:
    """Standardized aligned50 model inputs with separate P/O0/C/O masks.

    ``text/audio/vision`` are float32 arrays shaped ``[N,50,D]``. P has
    shape ``[N,50]`` and O0/C/O each have shape ``[N,3,50]``. ``sample_ids``
    retain source order. ``with_corruption`` safely zero-fills hidden rows;
    ``as_torch`` lazily converts numeric fields while leaving IDs as strings.
    """

    split: str
    sample_ids: Tuple[str, ...]
    text: np.ndarray
    audio: np.ndarray
    vision: np.ndarray
    valid_mask: np.ndarray
    original_observed_mask: np.ndarray
    corruption_mask: np.ndarray
    observed_mask: np.ndarray
    y_class: Optional[np.ndarray] = None
    y_reg: Optional[np.ndarray] = None
    invalid_value_events: Tuple[InvalidValueEvent, ...] = ()

    @property
    def features(self) -> Mapping[str, np.ndarray]:
        return {"T": self.text, "A": self.audio, "V": self.vision}

    @property
    def P(self) -> np.ndarray:
        return self.valid_mask

    @property
    def O0(self) -> np.ndarray:
        return self.original_observed_mask

    @property
    def C(self) -> np.ndarray:
        return self.corruption_mask

    @property
    def O(self) -> np.ndarray:
        return self.observed_mask

    @property
    def x_text(self) -> np.ndarray:
        return self.text

    @property
    def x_audio(self) -> np.ndarray:
        return self.audio

    @property
    def x_vision(self) -> np.ndarray:
        return self.vision

    def with_corruption(self, corruption_mask: np.ndarray) -> "SampleBatch":
        """Return a copy with C applied; reject any deletion outside P & O0."""
        corruption = np.asarray(corruption_mask, dtype=np.bool_)
        if corruption.shape != self.original_observed_mask.shape:
            raise DataContractError(
                "corruption_mask must have shape {}; got {}.".format(
                    self.original_observed_mask.shape, corruption.shape
                )
            )
        allowed = self.valid_mask[:, None, :] & self.original_observed_mask
        if np.any(corruption & ~allowed):
            raise DataContractError("Artificial corruption C must be a subset of P & O0.")
        observed = self.original_observed_mask & ~corruption
        output = {}
        for modality_index, modality in enumerate(MODALITIES):
            values = np.array(self.features[modality], dtype=np.float32, copy=True)
            values[~observed[:, modality_index, :]] = 0.0
            output[modality] = values
        return SampleBatch(
            split=self.split,
            sample_ids=self.sample_ids,
            text=output["T"],
            audio=output["A"],
            vision=output["V"],
            valid_mask=self.valid_mask.copy(),
            original_observed_mask=self.original_observed_mask.copy(),
            corruption_mask=corruption.copy(),
            observed_mask=observed,
            y_class=None if self.y_class is None else self.y_class.copy(),
            y_reg=None if self.y_reg is None else self.y_reg.copy(),
            invalid_value_events=self.invalid_value_events,
        )

    def as_torch(self, device: Optional[object] = None) -> Dict[str, object]:
        """Convert numeric fields to torch tensors, retaining IDs as strings."""
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("SampleBatch.as_torch requires PyTorch to be installed.") from exc
        result: Dict[str, object] = {
            "text": torch.as_tensor(self.text, dtype=torch.float32, device=device),
            "audio": torch.as_tensor(self.audio, dtype=torch.float32, device=device),
            "vision": torch.as_tensor(self.vision, dtype=torch.float32, device=device),
            "valid_mask": torch.as_tensor(self.valid_mask, dtype=torch.bool, device=device),
            "original_observed_mask": torch.as_tensor(
                self.original_observed_mask, dtype=torch.bool, device=device
            ),
            "observed_mask": torch.as_tensor(self.observed_mask, dtype=torch.bool, device=device),
            "corruption_mask": torch.as_tensor(self.corruption_mask, dtype=torch.bool, device=device),
            "sample_ids": self.sample_ids,
            "split": self.split,
        }
        if self.y_class is not None:
            result["y_class"] = torch.as_tensor(self.y_class, dtype=torch.long, device=device)
        if self.y_reg is not None:
            result["y_reg"] = torch.as_tensor(self.y_reg, dtype=torch.float32, device=device)
        return result


ModelBatch = SampleBatch


@dataclass(frozen=True)
class Normalizer:
    """Train-fitted per-feature statistics for all aligned50 splits."""

    mean: Mapping[str, np.ndarray]
    scale: Mapping[str, np.ndarray]
    count: Mapping[str, int]
    fitted_split: str = "train"
    ddof: int = 0

    def transform(self, split: AlignedSplit) -> SampleBatch:
        """Apply train statistics and zero-fill P/O0-missing rows safely."""
        transformed: Dict[str, np.ndarray] = {}
        for modality_index, modality in enumerate(MODALITIES):
            x = split.features[modality].astype(np.float64, copy=False)
            obs = split.original_observed_mask[:, modality_index, :]
            normalized = np.zeros(x.shape, dtype=np.float32)
            if np.any(obs):
                # Avoid arithmetic on unobserved rows altogether.
                values = (x[obs] - self.mean[modality]) / self.scale[modality]
                max_float32 = np.finfo(np.float32).max
                if not np.all(np.isfinite(values) & (np.abs(values) <= max_float32)):
                    raise DataContractError(
                        "Standardized {} values exceed finite float32 range in split {!r}.".format(
                            modality, split.split
                        )
                    )
                normalized[obs] = values.astype(np.float32)
            transformed[modality] = normalized
        no_corruption = np.zeros_like(split.original_observed_mask, dtype=np.bool_)
        return SampleBatch(
            split=split.split,
            sample_ids=split.sample_ids,
            text=transformed["T"],
            audio=transformed["A"],
            vision=transformed["V"],
            valid_mask=split.valid_mask.copy(),
            original_observed_mask=split.original_observed_mask.copy(),
            corruption_mask=no_corruption,
            observed_mask=split.original_observed_mask.copy(),
            y_class=None if split.y_class is None else split.y_class.copy(),
            y_reg=None if split.y_reg is None else split.y_reg.copy(),
            invalid_value_events=split.invalid_value_events,
        )


def fit_normalizer(train_split: AlignedSplit) -> Normalizer:
    """Fit ddof=0 feature statistics on train positions where P & O0 are true.

    This function rejects non-train input and any feature dimension with no
    observed train values. Validation, test, and special splits cannot
    contribute statistics.
    """
    if train_split.split != "train":
        raise DataContractError("Normalization statistics may only be fitted on split='train'.")
    means: Dict[str, np.ndarray] = {}
    scales: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {}
    for modality_index, modality in enumerate(MODALITIES):
        selected_mask = train_split.valid_mask & train_split.original_observed_mask[:, modality_index, :]
        selected = train_split.features[modality][selected_mask].astype(np.float64, copy=False)
        if selected.shape[0] == 0:
            raise DataContractError("No observable train positions for modality {}.".format(modality))
        mean = selected.mean(axis=0, dtype=np.float64)
        std = selected.std(axis=0, ddof=0, dtype=np.float64)
        # A dimension with zero variance keeps its centered value at zero.
        scale = np.where(std == 0.0, 1.0, std)
        if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(scale))):
            raise DataContractError("Non-finite normalization statistics in modality {}.".format(modality))
        means[modality] = mean
        scales[modality] = scale
        counts[modality] = int(selected.shape[0])
    return Normalizer(mean=means, scale=scales, count=counts, fitted_split="train", ddof=0)
