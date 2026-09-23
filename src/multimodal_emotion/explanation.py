"""Faithful occlusion explanations with separate modality and local effects."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .data import SampleBatch


MODALITIES = ("T", "A", "V")
ATTRIBUTE_NAMES = ("text", "audio", "vision")


def _sample_inputs(batch: SampleBatch, row: int, device: torch.device) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    inputs = [torch.as_tensor(getattr(batch, name)[row : row + 1], dtype=torch.float32, device=device) for name in ATTRIBUTE_NAMES]
    valid = torch.as_tensor(batch.valid_mask[row : row + 1], dtype=torch.bool, device=device)
    observed = torch.as_tensor(batch.original_observed_mask[row : row + 1], dtype=torch.bool, device=device)
    return inputs, valid, observed


def _forward(model: torch.nn.Module, inputs: list[torch.Tensor], valid: torch.Tensor, observed: torch.Tensor) -> tuple[np.ndarray, float]:
    output = model(*inputs, valid, observed)
    probability = torch.softmax(output["logits"], dim=-1)[0].detach().cpu().numpy()
    regression = float(output["regression"][0].detach().cpu())
    if not np.isfinite(probability).all() or not np.isfinite(regression):
        raise FloatingPointError("non-finite prediction during explanation")
    return probability, regression


def _contiguous_valid_runs(valid: np.ndarray) -> list[tuple[int, int]]:
    positions = np.flatnonzero(valid)
    if not positions.size:
        return []
    split_at = np.flatnonzero(np.diff(positions) != 1) + 1
    groups = np.split(positions, split_at)
    return [(int(group[0]), int(group[-1]) + 1) for group in groups]


def _window_candidates(valid: np.ndarray, observed: np.ndarray, window_length: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for start, stop in _contiguous_valid_runs(valid):
        for left in range(start, stop - window_length + 1):
            right = left + window_length
            if observed[left:right].any():
                candidates.append((left, right))
    return candidates


def _stable_rng(*parts: object) -> np.random.Generator:
    key = "|".join(str(part) for part in parts)
    seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big", signed=False)
    return np.random.Generator(np.random.PCG64(seed))


def _mapping_for_window(mapping: Sequence[Mapping[str, Any]] | None, start: int, stop: int) -> dict[str, Any]:
    if mapping is None:
        return {"time_start": None, "time_end": None, "mapping_source": None, "text": None}
    rows = list(mapping[start:stop])
    times = [(r.get("start"), r.get("end")) for r in rows if r.get("start") is not None and r.get("end") is not None]
    return {
        "time_start": min((x[0] for x in times), default=None),
        "time_end": max((x[1] for x in times), default=None),
        "mapping_source": next((r.get("mapping_source") for r in rows if r.get("mapping_source")), None),
        "text": " ".join(str(r.get("text")) for r in rows if r.get("text")),
    }


def explain_sample(
    model: torch.nn.Module,
    batch: SampleBatch,
    row: int,
    *,
    config: Mapping[str, Any] | None = None,
    position_mapping: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    device: str | torch.device = "cpu",
    run_random_controls: bool = True,
) -> dict[str, Any]:
    """Explain one clean sample without changing the original prediction.

    The result reports occlusion sensitivity, not causal contribution. If no
    verified position-to-time map is passed, local evidence indices remain
    useful but times are null and `mapping_status` is unresolved.
    """
    if not 0 <= row < len(batch.sample_ids):
        raise IndexError("row index outside sample batch")
    if np.any(batch.corruption_mask[row]):
        raise ValueError("explanations must start from a clean, uncorrupted sample")
    cfg = {} if config is None else config
    exp_cfg = cfg.get("explanation", {})
    max_windows = int(exp_cfg.get("top_nonoverlap_windows_per_modality", 3))
    requested_window = int(exp_cfg.get("window_length", 3))
    stride = int(exp_cfg.get("stride", 1))
    random_matches = int(exp_cfg.get("random_matches_per_window", 10)) if run_random_controls else 0
    eps = float(exp_cfg.get("zero_sum_epsilon", 1e-8))
    tie_tolerance = float(exp_cfg.get("dominant_modality_tie_tolerance", 1e-6))
    valid_np = batch.valid_mask[row].astype(bool)
    observed_np = batch.original_observed_mask[row].astype(bool)
    valid_length = int(valid_np.sum())
    window_length = min(requested_window, valid_length) if valid_length else 0
    sample_id = batch.sample_ids[row]
    maps = {} if position_mapping is None else position_mapping
    tensors, valid, observed = _sample_inputs(batch, row, torch.device(device))

    previous_mode = model.training
    model.eval()
    try:
        with torch.no_grad():
            base_probs, base_reg = _forward(model, tensors, valid, observed)
            target_class = int(np.argmax(base_probs))
            class_deltas: dict[str, float] = {}
            reg_deltas: dict[str, float] = {}
            for m, modality in enumerate(MODALITIES):
                ablated_observed = observed.clone()
                ablated_observed[:, m, :] = False
                probs, reg = _forward(model, tensors, valid, ablated_observed)
                class_deltas[modality] = float(base_probs[target_class] - probs[target_class])
                reg_deltas[modality] = float(base_reg - reg)

            class_abs = np.asarray([abs(class_deltas[m]) for m in MODALITIES], dtype=np.float64)
            reg_abs = np.asarray([abs(reg_deltas[m]) for m in MODALITIES], dtype=np.float64)
            class_weights = np.zeros(3, dtype=np.float64) if class_abs.sum() <= eps else class_abs / class_abs.sum()
            reg_weights = np.zeros(3, dtype=np.float64) if reg_abs.sum() <= eps else reg_abs / reg_abs.sum()
            def dominant(values: np.ndarray) -> str | None:
                if values.sum() <= eps:
                    return None
                top = values.max()
                matches = np.flatnonzero(np.abs(values - top) <= tie_tolerance)
                return MODALITIES[int(matches[0])] if len(matches) == 1 else None

            point_curves: dict[str, list[dict[str, Any]]] = {}
            evidence: dict[str, list[dict[str, Any]]] = {}
            for m, modality in enumerate(MODALITIES):
                curve: list[dict[str, Any]] = []
                for t in np.flatnonzero(valid_np & observed_np[m]):
                    perturbed = observed.clone()
                    perturbed[:, m, int(t)] = False
                    probs, reg = _forward(model, tensors, valid, perturbed)
                    signed_class = float(base_probs[target_class] - probs[target_class])
                    signed_reg = float(base_reg - reg)
                    curve.append({
                        "index": int(t), "signed_class": signed_class,
                        "absolute_class": abs(signed_class), "signed_reg": signed_reg,
                        "absolute_reg": abs(signed_reg),
                    })
                point_curves[modality] = curve

                if not window_length:
                    evidence[modality] = []
                    continue
                curve_by_index = {item["index"]: item for item in curve}
                valid_modality = valid_np & observed_np[m]
                candidates = _window_candidates(valid_np, valid_modality, window_length)
                # Keep starts aligned to the declared stride.
                candidates = [(a, b) for a, b in candidates if a % stride == 0]
                ranked = []
                for start, stop in candidates:
                    score = sum(curve_by_index.get(t, {}).get("absolute_class", 0.0) for t in range(start, stop))
                    ranked.append((float(score), start, stop))
                ranked.sort(key=lambda item: (-item[0], item[1]))
                selected: list[tuple[float, int, int]] = []
                for candidate in ranked:
                    _, start, stop = candidate
                    if any(not (stop <= a or start >= b) for _, a, b in selected):
                        continue
                    selected.append(candidate)
                    if len(selected) >= max_windows:
                        break

                evidence[modality] = []
                for rank, (score, start, stop) in enumerate(selected, start=1):
                    joint_observed = observed.clone()
                    joint_observed[:, m, start:stop] = False
                    joint_probs, joint_reg = _forward(model, tensors, valid, joint_observed)
                    mapping = _mapping_for_window(maps.get(modality), start, stop)
                    item: dict[str, Any] = {
                        "rank": rank, "index_start": start, "index_end": stop,
                        "single_point_rank_score": score,
                        "signed_class": float(base_probs[target_class] - joint_probs[target_class]),
                        "absolute_class": abs(float(base_probs[target_class] - joint_probs[target_class])),
                        "signed_reg": float(base_reg - joint_reg),
                        "absolute_reg": abs(float(base_reg - joint_reg)),
                        **mapping,
                        "random_controls": [],
                    }
                    if random_matches:
                        selected_observed_count = int(valid_modality[start:stop].sum())
                        random_candidates = [
                            (a, b) for a, b in _window_candidates(valid_np, valid_modality, stop - start)
                            if int(valid_np[a:b].sum()) == int(valid_np[start:stop].sum())
                            and int(valid_modality[a:b].sum()) == selected_observed_count
                        ]
                        rng = _stable_rng("explanation-v1", sample_id, modality, start)
                        chosen = rng.choice(len(random_candidates), size=random_matches, replace=len(random_candidates) < random_matches) if random_candidates else []
                        for match_no, candidate_idx in enumerate(chosen):
                            a, b = random_candidates[int(candidate_idx)]
                            random_observed = observed.clone()
                            random_observed[:, m, a:b] = False
                            random_probs, random_reg = _forward(model, tensors, valid, random_observed)
                            item["random_controls"].append({
                                "index_start": a, "index_end": b,
                                "signed_class": float(base_probs[target_class] - random_probs[target_class]),
                                "absolute_class": abs(float(base_probs[target_class] - random_probs[target_class])),
                                "signed_reg": float(base_reg - random_reg),
                                "absolute_reg": abs(float(base_reg - random_reg)),
                                "match_no": match_no,
                            })
                    evidence[modality].append(item)
    finally:
        if previous_mode:
            model.train()

    evidence_modalities = [m for m in MODALITIES if evidence[m]]
    if not evidence_modalities:
        mapping_status = "not_applicable"
    elif position_mapping is None:
        mapping_status = "unresolved"
    elif all(item.get("mapping_source") for m in evidence_modalities for item in evidence[m]):
        mapping_status = "verified"
    else:
        mapping_status = "approximate"
    return {
        "sample_id": sample_id,
        "predicted_class": target_class,
        "probabilities": base_probs.tolist(),
        "intensity": base_reg,
        "target_class_fixed_for_occlusion": target_class,
        "primary_modality": dominant(class_abs),
        "primary_modality_reg": dominant(reg_abs),
        "modality_class_delta": class_deltas,
        "modality_reg_delta": reg_deltas,
        "weight_class": {m: float(class_weights[i]) for i, m in enumerate(MODALITIES)},
        "weight_reg": {m: float(reg_weights[i]) for i, m in enumerate(MODALITIES)},
        "point_curves": point_curves,
        "evidence": evidence,
        "mapping_status": mapping_status,
        "explanation_status": "complete" if any(evidence.values()) else "no_observable_evidence",
        "quality_flag": "invalid_sample" if not valid_np.any() else ("low_information" if not (observed_np & valid_np[None, :]).any() else "ok"),
    }
