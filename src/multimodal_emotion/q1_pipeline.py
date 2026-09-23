"""Question 1 orchestration around version-locked feature extractors.

The external decoder, BERT/CTC aligner, openSMILE, and OpenFace commands are
provided as injected callbacks. This keeps tool versions and clock conversion
explicit instead of pretending unavailable external binaries were validated.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .q1_alignment import aggregate_q1_sample


@dataclass(frozen=True)
class Q1Input:
    sample_id: str
    video_path: str
    text: str
    label: float | None = None
    annotation: str | None = None


@dataclass(frozen=True)
class Q1Extractors:
    """Locked tool callbacks; timestamps must already use video-clock seconds."""

    probe: Callable[[str], Mapping[str, Any]]
    align_and_embed_text: Callable[[str, str, Mapping[str, Any]], Sequence[Mapping[str, Any]]]
    extract_audio_lld: Callable[[str, Mapping[str, Any]], Mapping[str, Any]]
    extract_vision: Callable[[str, Mapping[str, Any]], Mapping[str, Any]]
    versions: Mapping[str, str]


def process_q1_input(item: Q1Input, extractors: Q1Extractors, *, vision_confidence_min: float = 0.8) -> dict[str, Any]:
    """Process one item, retaining a failure record rather than fake zero success."""
    base: dict[str, Any] = {
        "sample_id": item.sample_id,
        "video_path": item.video_path,
        "text": item.text,
        "label": item.label,
        "annotation": item.annotation,
        "status": "failed",
    }
    try:
        metadata = dict(extractors.probe(item.video_path))
        duration = float(metadata["duration_seconds"])
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("probe duration_seconds must be finite and positive")
        words = extractors.align_and_embed_text(item.video_path, item.text, metadata)
        audio = dict(extractors.extract_audio_lld(item.video_path, metadata))
        vision = dict(extractors.extract_vision(item.video_path, metadata))
        for name, output in (("audio", audio), ("vision", vision)):
            if output.get("clock") != "video_seconds":
                raise ValueError(f"{name} extractor timestamps must be converted to video-clock seconds")
        audio_features = np.asarray(audio["features"])
        audio_times = np.asarray(audio["timestamps"], dtype=np.float64).reshape(-1)
        vision_features = np.asarray(vision["features"])
        vision_times = np.asarray(vision["timestamps"], dtype=np.float64).reshape(-1)
        vision_success = np.asarray(vision["success"]).reshape(-1)
        vision_confidence = np.asarray(vision["confidence"], dtype=np.float64).reshape(-1)
        if vision_times.shape != vision_success.shape or vision_times.shape != vision_confidence.shape:
            raise ValueError("vision timestamps, success, and confidence arrays must have matching lengths")
        aggregate = aggregate_q1_sample(
            duration,
            words,
            audio_features,
            audio_times,
            vision_features,
            vision_times,
            vision_success,
            vision_confidence,
            vision_confidence_min=vision_confidence_min,
            bins=50,
        )
        vision_usable = (vision_success == 1) & np.isfinite(vision_confidence) & (vision_confidence >= vision_confidence_min)
        for alignment_row in aggregate["alignment"]:
            start, end = float(alignment_row["start"]), float(alignment_row["end"])
            last = int(alignment_row["bin_index"]) == 49
            audio_keep = np.isfinite(audio_times) & (audio_times >= start) & ((audio_times <= end) if last else (audio_times < end))
            vision_keep = np.isfinite(vision_times) & (vision_times >= start) & ((vision_times <= end) if last else (vision_times < end)) & vision_usable
            alignment_row["audio_frame_indices"] = np.flatnonzero(audio_keep).astype(int).tolist()
            alignment_row["audio_frame_times"] = audio_times[audio_keep].astype(float).tolist()
            alignment_row["vision_frame_indices"] = np.flatnonzero(vision_keep).astype(int).tolist()
            alignment_row["vision_frame_times"] = vision_times[vision_keep].astype(float).tolist()
        base.update({
            "status": "success",
            "duration_seconds": duration,
            "features": {k: aggregate[k] for k in ("text", "audio", "vision")},
            "alignment": aggregate["alignment"],
            "audio_counts": aggregate["audio_counts"],
            "vision_counts": aggregate["vision_counts"],
            "metadata": metadata,
            "audio_metadata": {k: v for k, v in audio.items() if k not in {"features", "timestamps"}},
            "vision_metadata": {k: v for k, v in vision.items() if k not in {"features", "timestamps", "success", "confidence"}},
            "quality_flag": {
                "no_text_windows": int(sum(row["text_status"] == "alignment_unavailable" for row in aggregate["alignment"])),
                "no_audio_windows": int(np.sum(aggregate["audio_counts"] == 0)),
                "no_vision_windows": int(np.sum(aggregate["vision_counts"] == 0)),
                "speaker_ambiguous": bool(vision.get("speaker_ambiguous", False)),
            },
        })
    except Exception as exc:
        base["failure_reason"] = f"{type(exc).__name__}: {exc}"
    return base


def process_q1_inputs(
    inputs: Sequence[Q1Input],
    extractors: Q1Extractors,
    *,
    expected_samples: int = 100,
    vision_confidence_min: float = 0.8,
) -> list[dict[str, Any]]:
    ids = [item.sample_id for item in inputs]
    if len(ids) != len(set(ids)):
        raise ValueError("Q1 inputs contain duplicate sample IDs")
    if len(inputs) != expected_samples:
        raise ValueError(f"expected {expected_samples} Q1 inputs, got {len(inputs)}")
    return [process_q1_input(item, extractors, vision_confidence_min=vision_confidence_min) for item in inputs]


def save_q1_outputs(results: Sequence[Mapping[str, Any]], output_dir: str | Path, *, tool_versions: Mapping[str, str], vision_confidence_min: float = 0.8) -> None:
    """Write inventory and, only after full success, the Q1 feature bundle."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ids = [str(row["sample_id"]) for row in results]
    if len(ids) != 100 or len(set(ids)) != 100:
        raise ValueError("Q1 final output requires exactly 100 unique input IDs")

    inventory_fields = ["sample_id", "status", "duration_seconds", "label", "annotation", "quality_flag", "failure_reason"]
    with (output / "q1_inventory.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=inventory_fields, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow({**row, "quality_flag": json.dumps(row.get("quality_flag", {}), ensure_ascii=False)})

    modality_fields = ("sample_id", "modality", "feature_version", "dimension", "time_bins", "observed_bins", "observed_ratio", "status")
    with (output / "q1_inventory_by_modality.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=modality_fields)
        writer.writeheader()
        for row in results:
            for modality, dimension in (("T", 768), ("A", 25), ("V", 29)):
                if row.get("status") != "success":
                    observed = None
                elif modality == "T":
                    observed = sum(item["text_status"] == "aligned_text" for item in row["alignment"])
                elif modality == "A":
                    observed = int(np.count_nonzero(row["audio_counts"]))
                else:
                    observed = int(np.count_nonzero(row["vision_counts"]))
                writer.writerow({
                    "sample_id": row["sample_id"], "modality": modality,
                    "feature_version": "q1_equal_time50_bert_egemaps_openface_v1",
                    "dimension": dimension, "time_bins": 50,
                    "observed_bins": observed,
                    "observed_ratio": None if observed is None else observed / 50.0,
                    "status": row.get("status", "failed"),
                })

    failed = [row for row in results if row.get("status") != "success"]
    if failed:
        with (output / "q1_issues.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=("sample_id", "failure_reason"))
            writer.writeheader()
            writer.writerows({"sample_id": row["sample_id"], "failure_reason": row.get("failure_reason", "unknown")} for row in failed)
        raise RuntimeError(f"Q1 bundle not written: {len(failed)} of 100 samples did not complete")

    for row in results:
        features = row["features"]
        if features["text"].shape != (50, 768) or features["audio"].shape != (50, 25) or features["vision"].shape != (50, 29):
            raise ValueError(f"wrong Q1 feature shape for {row['sample_id']}")
        if not all(np.isfinite(features[name]).all() for name in ("text", "audio", "vision")):
            raise ValueError(f"non-finite Q1 feature values for {row['sample_id']}")

    np.savez_compressed(
        output / "q1_features.npz",
        sample_ids=np.asarray(ids),
        text=np.stack([row["features"]["text"] for row in results]).astype(np.float32),
        audio=np.stack([row["features"]["audio"] for row in results]).astype(np.float32),
        vision=np.stack([row["features"]["vision"] for row in results]).astype(np.float32),
    )
    schema = {
        "feature_version": "q1_equal_time50_bert_egemaps_openface_v1",
        "sample_count": 100,
        "sequence_length": 50,
        "dimensions": {"T": 768, "A": 25, "V": 29},
        "window_rule": "50 equal-duration bins over presented video duration; half-open except final endpoint",
        "vision_confidence_min": float(vision_confidence_min),
        "tool_versions": dict(tool_versions),
        "features_compatible_with_official_aligned50": False,
    }
    (output / "q1_feature_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "alignment_q1.jsonl").open("w", encoding="utf-8") as f:
        for row in results:
            for item in row["alignment"]:
                record = {
                    "sample_id": row["sample_id"],
                    **item,
                    "audio_valid_frame_count": int(row["audio_counts"][item["bin_index"]]),
                    "vision_valid_frame_count": int(row["vision_counts"][item["bin_index"]]),
                    "quality_flag": row["quality_flag"],
                }
                f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
