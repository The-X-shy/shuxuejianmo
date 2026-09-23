"""Deterministic early stopping on clean plus six fixed validation conditions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def early_stopping_score(clean_mae: float, middle_30_mae: dict[str, float]) -> float:
    required = {"T", "A", "V", "TA", "TV", "AV"}
    if set(middle_30_mae) != required:
        raise ValueError(f"early-stop grid must contain exactly {sorted(required)}")
    values = [float(clean_mae), *(float(middle_30_mae[k]) for k in sorted(required))]
    if not all(torch.isfinite(torch.tensor(v)).item() for v in values):
        raise ValueError("early-stopping metrics must be finite")
    return 0.5 * values[0] + 0.5 * sum(values[1:]) / 6.0


@dataclass
class EarlyStopping:
    patience: int = 8
    min_delta: float = 1e-4
    best_score: float = float("inf")
    best_epoch: int | None = None
    bad_epochs: int = 0
    best_state: dict[str, Any] | None = None

    def update(self, epoch: int, score: float, model: torch.nn.Module) -> bool:
        """Record an improvement and return whether training should stop."""
        if not torch.isfinite(torch.tensor(float(score))).item():
            raise ValueError("early-stopping score is non-finite")
        if self.best_epoch is None or score < self.best_score - self.min_delta:
            self.best_score = float(score)
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def restore_best(self, model: torch.nn.Module) -> None:
        if self.best_state is None:
            raise RuntimeError("no finite checkpoint was recorded")
        model.load_state_dict(self.best_state)
