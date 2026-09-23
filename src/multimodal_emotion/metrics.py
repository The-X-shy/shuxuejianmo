"""Metrics with explicit handling of absent classes and undefined Pearson r."""

from __future__ import annotations

from typing import Any

import numpy as np


CLASS_NAMES = ("Negative", "Neutral", "Positive")


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if true.shape != pred.shape or true.size == 0:
        raise ValueError("classification labels must be non-empty and have matching shapes")
    if np.any((true < 0) | (true > 2)) or np.any((pred < 0) | (pred > 2)):
        raise ValueError("class ids must be 0=Negative, 1=Neutral, 2=Positive")

    cm = np.zeros((3, 3), dtype=np.int64)
    np.add.at(cm, (true, pred), 1)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    tp = np.diag(cm).astype(np.float64)
    precision = np.divide(tp, predicted, out=np.zeros(3), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(3), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(3), where=(precision + recall) != 0)
    total = int(support.sum())
    return {
        "accuracy": float(tp.sum() / total),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.dot(f1, support) / total),
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(3)
        },
        "confusion_matrix": cm.tolist(),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if true.shape != pred.shape or true.size == 0:
        raise ValueError("regression values must be non-empty and have matching shapes")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("regression metrics require finite values")
    pearson: float | None
    reason: str | None = None
    if true.size < 2:
        pearson, reason = None, "fewer than two observations"
    elif np.std(true) == 0:
        pearson, reason = None, "constant target"
    elif np.std(pred) == 0:
        pearson, reason = None, "constant prediction"
    else:
        pearson = float(np.corrcoef(true, pred)[0, 1])
        if not np.isfinite(pearson):
            pearson, reason = None, "non-finite correlation"
    return {
        "mae": float(np.mean(np.abs(true - pred))),
        "pearson": pearson,
        "pearson_undefined_reason": reason,
    }


def calculate_metrics(y_class: np.ndarray, pred_class: np.ndarray, y_reg: np.ndarray, pred_reg: np.ndarray) -> dict[str, Any]:
    return {
        **classification_metrics(y_class, pred_class),
        **regression_metrics(y_reg, pred_reg),
    }
