"""Submission CSV writers and output-contract validation."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


CLASS_NAMES = ("Negative", "Neutral", "Positive")
PREDICTION_COLUMNS = (
    "sample_id", "polarity", "intensity", "p_negative", "p_neutral", "p_positive",
    "quality_flag", "feature_version", "model_version",
)
EXPLANATION_COLUMNS = PREDICTION_COLUMNS + (
    "primary_modality", "primary_modality_reg", "weight_text", "weight_audio", "weight_vision",
    "signed_class_text", "signed_class_audio", "signed_class_vision",
    "signed_reg_text", "signed_reg_audio", "signed_reg_vision",
    "evidence_text_json", "evidence_audio_json", "evidence_vision_json",
    "mapping_status", "explanation_status",
)


def validate_prediction_rows(rows: Sequence[Mapping[str, Any]], expected_ids: Sequence[str] | None = None) -> None:
    ids = [str(row["sample_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("sample_id values must be unique")
    if expected_ids is not None and ids != [str(v) for v in expected_ids]:
        raise ValueError("prediction IDs must cover inputs exactly once and preserve input order")
    for row in rows:
        if row.get("polarity") not in CLASS_NAMES:
            raise ValueError(f"illegal polarity for {row['sample_id']}")
        intensity = float(row["intensity"])
        probs = np.asarray([row[k] for k in ("p_negative", "p_neutral", "p_positive")], dtype=np.float64)
        if not np.isfinite(intensity) or not -3 <= intensity <= 3:
            raise ValueError(f"intensity outside [-3,3] for {row['sample_id']}")
        if not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)) or not np.isclose(probs.sum(), 1.0, atol=1e-6):
            raise ValueError(f"invalid class probabilities for {row['sample_id']}")


def write_prediction_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], *, expected_ids: Sequence[str] | None = None) -> None:
    validate_prediction_rows(rows, expected_ids)
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in PREDICTION_COLUMNS})


def write_explanation_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], *, expected_ids: Sequence[str] | None = None) -> None:
    validate_prediction_rows(rows, expected_ids)
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXPLANATION_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            converted = dict(row)
            for key in ("evidence_text_json", "evidence_audio_json", "evidence_vision_json"):
                value = converted.get(key, [])
                if not isinstance(value, str):
                    converted[key] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            writer.writerow({k: converted.get(k, "") for k in EXPLANATION_COLUMNS})


def prediction_rows_from_outputs(outputs: Mapping[str, Any], *, feature_version: str, model_version: str) -> list[dict[str, Any]]:
    rows = []
    for index, sample_id in enumerate(outputs["sample_ids"]):
        probs = np.asarray(outputs["probabilities"][index], dtype=np.float64)
        predicted = int(outputs["predicted_class"][index])
        invalid = bool(outputs["invalid_sample"][index])
        low_information = bool(outputs["low_information"][index])
        rows.append({
            "sample_id": str(sample_id),
            "polarity": CLASS_NAMES[predicted],
            "intensity": float(outputs["intensity"][index]),
            "p_negative": float(probs[0]), "p_neutral": float(probs[1]), "p_positive": float(probs[2]),
            "quality_flag": "invalid_sample" if invalid else ("low_information" if low_information else "ok"),
            "feature_version": feature_version,
            "model_version": model_version,
        })
    return rows


def explanation_row_from_record(record: Mapping[str, Any], *, feature_version: str, model_version: str) -> dict[str, Any]:
    probs = record["probabilities"]
    weights = record["weight_class"]
    class_delta = record["modality_class_delta"]
    reg_delta = record["modality_reg_delta"]
    evidence = record["evidence"]
    return {
        "sample_id": record["sample_id"],
        "polarity": CLASS_NAMES[int(record["predicted_class"])],
        "intensity": float(record["intensity"]),
        "p_negative": float(probs[0]), "p_neutral": float(probs[1]), "p_positive": float(probs[2]),
        "quality_flag": record.get("quality_flag", "ok"),
        "feature_version": feature_version,
        "model_version": model_version,
        "primary_modality": record["primary_modality"],
        "primary_modality_reg": record["primary_modality_reg"],
        "weight_text": float(weights["T"]), "weight_audio": float(weights["A"]), "weight_vision": float(weights["V"]),
        "signed_class_text": float(class_delta["T"]), "signed_class_audio": float(class_delta["A"]), "signed_class_vision": float(class_delta["V"]),
        "signed_reg_text": float(reg_delta["T"]), "signed_reg_audio": float(reg_delta["A"]), "signed_reg_vision": float(reg_delta["V"]),
        "evidence_text_json": evidence["T"], "evidence_audio_json": evidence["A"], "evidence_vision_json": evidence["V"],
        "mapping_status": record["mapping_status"],
        "explanation_status": record["explanation_status"],
    }
