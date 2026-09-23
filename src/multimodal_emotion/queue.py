"""Run-queue checks that do not open data or start jobs."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any


def validate_run_queue(path: str | Path, config: dict[str, Any]) -> list[str]:
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    errors: list[str] = []
    expected_models = config["training"]["model_execution_order"]
    expected_seeds = config["training"]["seed_execution_order"]
    if len(rows) != len(expected_models) * len(expected_seeds):
        errors.append(f"expected {len(expected_models) * len(expected_seeds)} rows, got {len(rows)}")
    if len({r.get("run_id") for r in rows}) != len(rows):
        errors.append("run_id values are not unique")
    if len({r.get("output_directory") for r in rows}) != len(rows):
        errors.append("output_directory values are not unique")

    for seed in expected_seeds:
        seed_rows = [r for r in rows if int(r["seed"]) == int(seed)]
        actual = [r["experiment_id"] for r in seed_rows]
        if actual != expected_models:
            errors.append(f"seed {seed} model order mismatch: {actual}")
    expected = Counter((model, int(seed)) for model in expected_models for seed in expected_seeds)
    actual = Counter((r["experiment_id"], int(r["seed"])) for r in rows)
    if actual != expected:
        errors.append("model/seed pairs do not match the configured core matrix")
    for row in rows:
        if row.get("status") != "PLANNED":
            errors.append(f"unexpected non-planned status for {row.get('run_id')}")
        if int(row["max_epochs"]) != int(config["training"]["max_epochs"]):
            errors.append(f"max_epochs mismatch for {row.get('run_id')}")
        if int(row["batch_size"]) != int(config["training"]["batch_size"]):
            errors.append(f"batch_size mismatch for {row.get('run_id')}")
    if errors:
        raise ValueError("; ".join(errors))
    return rows
