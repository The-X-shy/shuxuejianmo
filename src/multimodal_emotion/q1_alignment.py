"""Time-bin aggregation for already extracted Q1 features.

This module does not invoke FFmpeg, OpenFace, openSMILE, BERT, or CTC. It
converts timestamped outputs from those tools into the documented 50 bins.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def make_bin_edges(duration: float, bins: int = 50) -> np.ndarray:
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    if bins <= 0:
        raise ValueError("bins must be positive")
    return np.linspace(0.0, float(duration), int(bins) + 1, dtype=np.float64)


def aggregate_word_embeddings(words: Sequence[Mapping[str, Any]], duration: float, bins: int = 50, dimension: int = 768) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Overlap-weighted word embeddings; words without verified times are skipped."""
    edges = make_bin_edges(duration, bins)
    sums = np.zeros((bins, dimension), dtype=np.float64)
    weights = np.zeros(bins, dtype=np.float64)
    used: list[dict[str, Any]] = []
    for i, word in enumerate(words):
        start, end = word.get("start"), word.get("end")
        vector = np.asarray(word.get("embedding", []), dtype=np.float64)
        if start is None or end is None or not np.isfinite([start, end]).all() or end <= start:
            continue
        if vector.shape != (dimension,) or not np.isfinite(vector).all():
            raise ValueError(f"word {i} has an invalid embedding")
        start = max(0.0, float(start))
        end = min(float(duration), float(end))
        if end <= start:
            continue
        first = max(0, int(np.searchsorted(edges, start, side="right") - 1))
        last = min(bins - 1, int(np.searchsorted(edges, end, side="left")))
        for b in range(first, last + 1):
            overlap = max(0.0, min(end, edges[b + 1]) - max(start, edges[b]))
            if overlap:
                sums[b] += overlap * vector
                weights[b] += overlap
        used.append({"word_index": i, "start": start, "end": end, "text": word.get("text"), "mapping_source": word.get("mapping_source")})
    out = np.zeros_like(sums, dtype=np.float32)
    present = weights > 0
    out[present] = (sums[present] / weights[present, None]).astype(np.float32)
    statuses = ["aligned_text" if present[b] else "alignment_unavailable" for b in range(bins)]
    return out, [{"bin_index": b, "start": float(edges[b]), "end": float(edges[b + 1]), "text_status": statuses[b], "words": [w for w in used if w["start"] < edges[b + 1] and w["end"] > edges[b]]} for b in range(bins)]


def aggregate_frame_features(features: np.ndarray, timestamps: np.ndarray, duration: float, *, bins: int = 50, valid: np.ndarray | None = None, expected_dim: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Mean timestamped frame features into bins; return values and counts."""
    x = np.asarray(features, dtype=np.float64)
    t = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[0] != t.size:
        raise ValueError("features must have shape [frames, dimension] matching timestamps")
    if expected_dim is not None and x.shape[1] != expected_dim:
        raise ValueError(f"expected {expected_dim} feature columns, got {x.shape[1]}")
    ok = np.isfinite(t) & np.isfinite(x).all(axis=1) & (t >= 0) & (t <= duration)
    if valid is not None:
        v = np.asarray(valid, dtype=bool).reshape(-1)
        if v.shape != t.shape:
            raise ValueError("valid mask shape does not match timestamps")
        ok &= v
    edges = make_bin_edges(duration, bins)
    out = np.zeros((bins, x.shape[1]), dtype=np.float32)
    counts = np.zeros(bins, dtype=np.int64)
    idx = np.searchsorted(edges, t[ok], side="right") - 1
    idx = np.clip(idx, 0, bins - 1)
    for b in range(bins):
        selected = x[ok][idx == b]
        if selected.size:
            out[b] = selected.mean(axis=0).astype(np.float32)
            counts[b] = selected.shape[0]
    return out, counts


def aggregate_q1_sample(duration: float, words: Sequence[Mapping[str, Any]], audio_features: np.ndarray, audio_timestamps: np.ndarray, vision_features: np.ndarray, vision_timestamps: np.ndarray, vision_success: np.ndarray, vision_confidence: np.ndarray, *, vision_confidence_min: float = 0.8, bins: int = 50) -> dict[str, Any]:
    text, alignment = aggregate_word_embeddings(words, duration, bins=bins, dimension=768)
    audio, audio_counts = aggregate_frame_features(audio_features, audio_timestamps, duration, bins=bins, expected_dim=25)
    vision_valid = (np.asarray(vision_success).reshape(-1) == 1) & (np.asarray(vision_confidence).reshape(-1) >= vision_confidence_min)
    vision, vision_counts = aggregate_frame_features(vision_features, vision_timestamps, duration, bins=bins, valid=vision_valid, expected_dim=29)
    return {
        "text": text,
        "audio": audio,
        "vision": vision,
        "audio_counts": audio_counts,
        "vision_counts": vision_counts,
        "alignment": alignment,
        "bin_edges": make_bin_edges(duration, bins),
    }
