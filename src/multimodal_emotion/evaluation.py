"""Fixed validation-grid evaluation and prediction helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from .corruption import ScenarioSpec, generate_validation_scenarios, scenario_mask, validation_earlystop_scenarios
from .data import SampleBatch
from .metrics import calculate_metrics
from .selection import select_candidate


def predict_split(model: torch.nn.Module, batch: SampleBatch, *, device: str | torch.device = "cpu", observed_mask_override: np.ndarray | None = None) -> dict[str, Any]:
    """Run one clean pass and retain per-ID outputs in input order."""
    tensors = batch.as_torch(device)
    override = None if observed_mask_override is None else torch.as_tensor(observed_mask_override, dtype=torch.bool, device=device)
    previous_mode = model.training
    model.eval()
    try:
        with torch.no_grad():
            output = model(
                tensors["text"], tensors["audio"], tensors["vision"],
                tensors["valid_mask"], tensors["observed_mask"],
                observed_mask_override=override,
            )
            probabilities = torch.softmax(output["logits"], dim=-1)
            predicted_class = probabilities.argmax(dim=-1)
    finally:
        if previous_mode:
            model.train()
    result: dict[str, Any] = {
        "sample_ids": batch.sample_ids,
        "probabilities": probabilities.cpu().numpy(),
        "predicted_class": predicted_class.cpu().numpy(),
        "intensity": output["regression"].cpu().numpy(),
        "invalid_sample": output["invalid_sample"].cpu().numpy(),
        "low_information": output["low_information"].cpu().numpy(),
        "weights": output["weights"].cpu().numpy(),
        "gate_weights": None if output["gate_weights"] is None else output["gate_weights"].cpu().numpy(),
    }
    if batch.y_class is not None:
        result["y_class"] = batch.y_class.copy()
    if batch.y_reg is not None:
        result["y_reg"] = batch.y_reg.copy()
    return result


def _metric_summary(prediction: Mapping[str, Any]) -> dict[str, Any]:
    if "y_class" not in prediction or "y_reg" not in prediction:
        raise ValueError("evaluation requires both class and regression labels")
    return calculate_metrics(
        np.asarray(prediction["y_class"]),
        np.asarray(prediction["predicted_class"]),
        np.asarray(prediction["y_reg"]),
        np.asarray(prediction["intensity"]),
    )


def evaluate_early_stopping(model: torch.nn.Module, valid_batch: SampleBatch, *, device: str | torch.device = "cpu") -> tuple[float, dict[str, float]]:
    """Evaluate clean MAE plus the six fixed 30%-middle cases."""
    clean = predict_split(model, valid_batch, device=device)
    clean_mae = float(_metric_summary(clean)["mae"])
    missing: dict[str, float] = {}
    for scenario in validation_earlystop_scenarios():
        corruption = np.stack([
            scenario_mask(valid_batch.valid_mask[i], valid_batch.original_observed_mask[i], valid_batch.sample_ids[i], scenario).corruption_mask
            for i in range(len(valid_batch.sample_ids))
        ])
        observed = valid_batch.original_observed_mask & ~corruption
        prediction = predict_split(model, valid_batch, device=device, observed_mask_override=observed)
        missing["".join(scenario.modalities)] = float(_metric_summary(prediction)["mae"])
    return clean_mae, missing


def evaluate_scenario(model: torch.nn.Module, batch: SampleBatch, scenario: ScenarioSpec | None = None, *, device: str | torch.device = "cpu") -> dict[str, Any]:
    corruption = np.zeros_like(batch.original_observed_mask)
    if scenario is None:
        observed = batch.original_observed_mask
    else:
        corruption = np.stack([
            scenario_mask(batch.valid_mask[i], batch.original_observed_mask[i], batch.sample_ids[i], scenario).corruption_mask
            for i in range(len(batch.sample_ids))
        ])
        observed = batch.original_observed_mask & ~corruption
    prediction = predict_split(model, batch, device=device, observed_mask_override=observed)
    metrics = _metric_summary(prediction)
    return {
        "scenario_id": "clean" if scenario is None else scenario.scenario_id,
        "group": "clean" if scenario is None else scenario.group,
        "modalities": [] if scenario is None else list(scenario.modalities),
        "ratio": None if scenario is None else scenario.ratio,
        "position": "clean" if scenario is None else scenario.position,
        "metrics": metrics,
        "predictions": prediction,
        "corruption_mask": corruption,
    }


def evaluate_fixed_grid(model: torch.nn.Module, valid_batch: SampleBatch, *, device: str | torch.device = "cpu") -> list[dict[str, Any]]:
    """Evaluate clean + 90 main + 15 TAV + 9 async conditions with one checkpoint."""
    results = [evaluate_scenario(model, valid_batch, None, device=device)]
    results.extend(
        evaluate_scenario(model, valid_batch, scenario, device=device)
        for scenario in generate_validation_scenarios()
    )
    if len(results) != 115:
        raise RuntimeError(f"fixed evaluation grid has {len(results)} views, expected 115")
    return results


def aggregate_core_evaluations(
    run_results: Mapping[tuple[str, int], list[dict[str, Any]]],
    config: Mapping[str, Any],
    parameter_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Aggregate three-seed valid evaluations and apply the frozen selector.

    `run_results[(model_id, seed)]` must be the clean + 114 grid returned by
    :func:`evaluate_fixed_grid`. No test or special-test metrics belong here.
    """
    model_ids = list(config["training"]["model_execution_order"])
    seeds = [int(s) for s in config["training"]["seed_execution_order"]]
    expected_pairs = {(model, seed) for model in model_ids for seed in seeds}
    if set(run_results) != expected_pairs:
        raise ValueError("run result keys must contain every configured core model/seed pair exactly once")
    if set(parameter_counts) != set(model_ids):
        raise ValueError("parameter_counts must contain every core model exactly once")
    for key, rows in run_results.items():
        if len(rows) != int(config["evaluation"]["total_views_per_run"]):
            raise ValueError(f"{key} has {len(rows)} evaluation views; expected 115")
        if rows[0]["scenario_id"] != "clean":
            raise ValueError(f"{key} must place clean evaluation first")

    condition_rows: list[dict[str, Any]] = []
    candidates: dict[str, dict[str, float | int]] = {}
    for model_id in model_ids:
        seed_rows = {seed: run_results[(model_id, seed)] for seed in seeds}
        condition_ids = [row["scenario_id"] for row in seed_rows[seeds[0]]]
        if len(condition_ids) != 115 or any([r["scenario_id"] for r in seed_rows[s]] != condition_ids for s in seeds):
            raise ValueError(f"evaluation condition mismatch across seeds for {model_id}")
        for view_index, condition_id in enumerate(condition_ids):
            metrics_by_seed = [seed_rows[seed][view_index]["metrics"] for seed in seeds]
            metric_names = ("accuracy", "macro_f1", "weighted_f1", "mae", "pearson")
            summary: dict[str, Any] = {"model_id": model_id, "scenario_id": condition_id}
            for metric in metric_names:
                values = np.asarray([m[metric] for m in metrics_by_seed], dtype=np.float64)
                finite = values[np.isfinite(values)]
                summary[metric + "_mean"] = float(finite.mean()) if finite.size else None
                summary[metric + "_std"] = float(finite.std(ddof=1)) if finite.size > 1 else (0.0 if finite.size == 1 else None)
            condition_rows.append(summary)

        main_by_seed = []
        for seed in seeds:
            main = [row["metrics"]["mae"] for row in seed_rows[seed][1:] if row["group"] == "main"]
            if len(main) != 90:
                raise ValueError(f"{model_id} seed={seed} main grid must contain 90 conditions")
            main_by_seed.append(float(np.mean(main)))
        clean_metrics = [seed_rows[seed][0]["metrics"] for seed in seeds]
        candidates[model_id] = {
            "R": float(np.mean(main_by_seed)),
            "R_std_across_seeds": float(np.std(main_by_seed, ddof=1)),
            "clean_mae": float(np.mean([m["mae"] for m in clean_metrics])),
            "clean_macro_f1": float(np.mean([m["macro_f1"] for m in clean_metrics])),
            "main_macro_f1": float(np.mean([
                row["metrics"]["macro_f1"]
                for seed in seeds for row in seed_rows[seed][1:] if row["group"] == "main"
            ])),
            "parameter_count": int(parameter_counts[model_id]),
        }
    selection = select_candidate({name: candidates[name] for name in config["selection"]["candidate_ids"]}, config)
    return {"candidates": candidates, "condition_summaries": condition_rows, "selection": selection, "source": "valid_only"}
