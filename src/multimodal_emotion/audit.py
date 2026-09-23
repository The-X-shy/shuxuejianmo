"""Structural and label audits that report conflicts without rewriting data."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping

import numpy as np

from .data import AlignedSplit


def audit_aligned_splits(splits: Mapping[str, AlignedSplit]) -> dict[str, Any]:
    required = {"train", "valid", "test"}
    if set(splits) != required:
        raise ValueError(f"expected exactly train/valid/test splits, got {sorted(splits)}")
    exact_ids: dict[str, list[str]] = defaultdict(list)
    video_splits: dict[str, set[str]] = defaultdict(set)
    split_summary: dict[str, Any] = {}
    issues: list[dict[str, Any]] = []
    label_conflicts: list[dict[str, Any]] = []

    for name in ("train", "valid", "test"):
        split = splits[name]
        class_counts = None
        y_range = None
        if split.y_class is not None:
            class_counts = {str(i): int(np.count_nonzero(split.y_class == i)) for i in range(3)}
        if split.y_reg is not None:
            y_range = [float(np.min(split.y_reg)), float(np.max(split.y_reg))]
        for sample_id in split.sample_ids:
            exact_ids[sample_id].append(name)
            video_id = sample_id.split("$_$", 1)[0]
            video_splits[video_id].add(name)
        split_summary[name] = {
            "sample_count": len(split.sample_ids),
            "feature_shapes": {mod: list(value.shape) for mod, value in split.features.items()},
            "valid_position_count_min": int(split.valid_mask.sum(axis=1).min()),
            "valid_position_count_max": int(split.valid_mask.sum(axis=1).max()),
            "class_counts": class_counts,
            "regression_range": y_range,
            "invalid_value_event_count": len(split.invalid_value_events),
        }
        if split.y_class is not None and split.y_reg is not None:
            expected_class = np.where(split.y_reg < 0, 0, np.where(split.y_reg == 0, 1, 2))
            mismatch = np.flatnonzero(expected_class != split.y_class)
            for idx in mismatch:
                label_conflicts.append({
                    "split": name,
                    "sample_id": split.sample_ids[int(idx)],
                    "classification_label": int(split.y_class[idx]),
                    "regression_label": float(split.y_reg[idx]),
                    "class_from_regression_sign": int(expected_class[idx]),
                })

    for sample_id, locations in exact_ids.items():
        if len(locations) > 1:
            issues.append({"kind": "duplicate_sample_id_across_splits", "sample_id": sample_id, "splits": locations})
    for video_id, locations in video_splits.items():
        if len(locations) > 1:
            issues.append({"kind": "video_id_crosses_splits", "video_id": video_id, "splits": sorted(locations), "action": "report_and_review; do_not_resplit"})
    if label_conflicts:
        issues.append({"kind": "classification_regression_label_conflict", "count": len(label_conflicts)})
    return {
        "status": "PASS" if not issues else "REVIEW_REQUIRED",
        "split_summary": split_summary,
        "issues": issues,
        "label_conflicts": label_conflicts,
        "official_split_policy_preserved": True,
    }
