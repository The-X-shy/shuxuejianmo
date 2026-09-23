"""Training loop for one preregistered core run.

This module is not invoked by preflight. Real training happens only when an
explicit caller supplies a labeled train split and validation split.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch import nn

from .config import config_hash
from .corruption import make_training_corruption, stable_seed
from .data import SampleBatch
from .early_stopping import EarlyStopping, early_stopping_score
from .evaluation import evaluate_early_stopping, predict_split
from .models import build_model


CORRUPTION_MODELS = {"B4", "B5", "M1"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:  # pragma: no cover - older supported PyTorch
        torch.use_deterministic_algorithms(True)


def initialize_model(model_id: str, train_batch: SampleBatch, config: Mapping[str, Any], *, seed: int) -> nn.Module:
    if train_batch.y_class is None or train_batch.y_reg is None:
        raise ValueError("training split must include both class and regression labels")
    seed_everything(seed)
    counts = np.bincount(train_batch.y_class.astype(np.int64), minlength=3).astype(np.float64)
    if counts.sum() == 0:
        raise ValueError("training split has no labels")
    dims = {"text": int(train_batch.text.shape[-1]), "audio": int(train_batch.audio.shape[-1]), "vision": int(train_batch.vision.shape[-1])}
    model = build_model(
        model_id,
        input_dims=dims,
        hidden_dim=int(config["model"]["hidden_dim"]),
        sequence_length=int(config["data"]["sequence_length"]),
        dropout=float(config["model"]["dropout"]),
        class_prior=counts / counts.sum(),
        regression_prior=float(np.mean(train_batch.y_reg)),
    )
    return model


def copy_common_initialization(source: nn.Module, destination: nn.Module) -> list[str]:
    """Copy same-shaped parameters for paired ablations, leaving unique layers fresh."""
    src, dst = source.state_dict(), destination.state_dict()
    shared = [key for key in src.keys() & dst.keys() if src[key].shape == dst[key].shape]
    if not shared:
        raise ValueError("models have no same-shaped state to share")
    for key in shared:
        dst[key] = src[key].detach().clone()
    destination.load_state_dict(dst)
    return sorted(shared)


def _array_fingerprint(batch: SampleBatch) -> str:
    digest = hashlib.sha256()
    digest.update("\n".join(batch.sample_ids).encode("utf-8"))
    for array in (batch.text, batch.audio, batch.vision, batch.valid_mask, batch.original_observed_mask):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _mini_batch_tensors(batch: SampleBatch, indices: np.ndarray, corruption: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    observed = batch.original_observed_mask[indices] & ~corruption
    tensors: dict[str, torch.Tensor] = {}
    for index, name in enumerate(("text", "audio", "vision")):
        values = np.array(getattr(batch, name)[indices], dtype=np.float32, copy=True)
        values[~observed[:, index, :]] = 0.0
        tensors[name] = torch.as_tensor(values, dtype=torch.float32, device=device)
    tensors["valid_mask"] = torch.as_tensor(batch.valid_mask[indices], dtype=torch.bool, device=device)
    tensors["observed_mask"] = torch.as_tensor(observed, dtype=torch.bool, device=device)
    tensors["y_class"] = torch.as_tensor(batch.y_class[indices], dtype=torch.long, device=device)
    tensors["y_reg"] = torch.as_tensor(batch.y_reg[indices], dtype=torch.float32, device=device)
    return tensors


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fit_one_run(
    model: nn.Module,
    model_id: str,
    train_batch: SampleBatch,
    valid_batch: SampleBatch,
    config: Mapping[str, Any],
    *,
    seed: int,
    output_dir: str | Path | None = None,
    device: str | torch.device = "cpu",
    validation_evaluator: Callable[[nn.Module, SampleBatch], tuple[float, dict[str, float]]] | None = None,
) -> dict[str, Any]:
    """Train one model/seed and restore the earliest best validation epoch.

    This function intentionally has no implicit data loading. Call it only
    after auditing the official inputs, deriving the train-only normalizer,
    and locking the config/data/code hashes.
    """
    if model_id not in {"B0", "B1", "B2", "B3", "B4", "B5", "M0", "M1"}:
        raise ValueError(f"unsupported core model {model_id}")
    if train_batch.split != "train" or valid_batch.split != "valid":
        raise ValueError("fit_one_run requires the official train and valid splits")
    if train_batch.y_class is None or train_batch.y_reg is None or valid_batch.y_class is None or valid_batch.y_reg is None:
        raise ValueError("train and valid splits must include both labels")
    if not np.isfinite(train_batch.y_reg).all() or not np.isfinite(valid_batch.y_reg).all():
        raise ValueError("labels must be finite")
    if not np.isfinite(train_batch.text).all() or not np.isfinite(train_batch.audio).all() or not np.isfinite(train_batch.vision).all():
        raise ValueError("training features contain NaN/Inf; fix the input contract before training")

    config_copy = json.loads(json.dumps(config))
    run_id = f"{model_id}_seed{seed}"
    run_dir = None if output_dir is None else Path(output_dir)
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(run_dir / "config.json", config_copy)
        _write_json(run_dir / "manifest.json", {
            "run_id": run_id,
            "experiment_id": model_id,
            "seed": int(seed),
            "config_hash": config_hash(config_copy),
            "train_data_hash": _array_fingerprint(train_batch),
            "valid_data_hash": _array_fingerprint(valid_batch),
            "train_sample_count": len(train_batch.sample_ids),
            "valid_sample_count": len(valid_batch.sample_ids),
            "device": str(device),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        })

    seed_everything(seed)
    device_obj = torch.device(device)
    model.to(device_obj)
    counts = np.bincount(train_batch.y_class.astype(np.int64), minlength=3).astype(np.float64)
    model.set_prior(counts / counts.sum(), float(np.mean(train_batch.y_reg)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    batch_size = int(config["training"]["batch_size"])
    max_epochs = int(config["training"]["max_epochs"])
    stop_cfg = config["training"]["early_stop"]
    early_stopping = EarlyStopping(
        patience=int(stop_cfg["patience"]),
        min_delta=float(stop_cfg["min_delta"]),
    )
    evaluate = validation_evaluator or (lambda m, b: evaluate_early_stopping(m, b, device=device_obj))
    epoch_rows: list[dict[str, Any]] = []
    start_time = time.monotonic()

    try:
        for epoch in range(1, max_epochs + 1):
            model.train()
            shuffle_rng = np.random.Generator(np.random.PCG64(stable_seed(f"shuffle-v1|{seed}|{epoch}")))
            order = shuffle_rng.permutation(len(train_batch.sample_ids))
            loss_sum = 0.0
            seen_count = 0
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                corruption = np.zeros((len(indices), 3, 50), dtype=np.bool_)
                if model_id in CORRUPTION_MODELS:
                    for local_index, sample_index in enumerate(indices):
                        draw = make_training_corruption(
                            train_batch.valid_mask[sample_index],
                            train_batch.original_observed_mask[sample_index],
                            int(seed), epoch, train_batch.sample_ids[sample_index],
                        )
                        corruption[local_index] = draw.corruption_mask
                x = _mini_batch_tensors(train_batch, indices, corruption, device_obj)
                optimizer.zero_grad(set_to_none=True)
                output = model(x["text"], x["audio"], x["vision"], x["valid_mask"], x["observed_mask"])
                if not torch.isfinite(output["logits"]).all() or not torch.isfinite(output["regression"]).all():
                    bad_ids = [train_batch.sample_ids[int(i)] for i in indices]
                    raise FloatingPointError(f"non-finite model output in {run_id}, epoch={epoch}, sample_ids={bad_ids}")
                classification_loss = nn.functional.cross_entropy(output["logits"], x["y_class"])
                regression_loss = nn.functional.smooth_l1_loss(output["regression"], x["y_reg"], beta=1.0)
                loss = classification_loss + regression_loss
                if not torch.isfinite(loss):
                    bad_ids = [train_batch.sample_ids[int(i)] for i in indices]
                    raise FloatingPointError(f"non-finite loss in {run_id}, epoch={epoch}, sample_ids={bad_ids}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip_norm"]))
                optimizer.step()
                count = len(indices)
                loss_sum += float(loss.detach().cpu()) * count
                seen_count += count

            clean_mae, missing_mae = evaluate(model, valid_batch)
            score = early_stopping_score(clean_mae, missing_mae)
            should_stop = early_stopping.update(epoch, score, model)
            row = {
                "epoch": epoch,
                "train_loss": loss_sum / max(seen_count, 1),
                "clean_valid_mae": float(clean_mae),
                "early_stop_score": float(score),
                "best_score": float(early_stopping.best_score),
                "bad_epochs": int(early_stopping.bad_epochs),
            }
            epoch_rows.append(row)
            if should_stop:
                break

        early_stopping.restore_best(model)
        elapsed = time.monotonic() - start_time
        result = {
            "run_id": run_id,
            "status": "completed",
            "seed": int(seed),
            "best_epoch": int(early_stopping.best_epoch),
            "best_early_stop_score": float(early_stopping.best_score),
            "epochs_completed": len(epoch_rows),
            "duration_seconds": elapsed,
        }
        if run_dir is not None:
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, run_dir / "best_weights.pt")
            _write_json(run_dir / "best_epoch.json", {k: result[k] for k in ("best_epoch", "best_early_stop_score", "epochs_completed")})
            with (run_dir / "epoch_metrics.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(epoch_rows[0]))
                writer.writeheader()
                writer.writerows(epoch_rows)
            clean = predict_split(model, valid_batch, device=device_obj)
            rows = []
            for i, sample_id in enumerate(valid_batch.sample_ids):
                pred = int(clean["predicted_class"][i])
                probs = clean["probabilities"][i]
                quality = "invalid_sample" if clean["invalid_sample"][i] else ("low_information" if clean["low_information"][i] else "ok")
                rows.append({
                    "sample_id": sample_id,
                    "polarity": ("Negative", "Neutral", "Positive")[pred],
                    "intensity": float(clean["intensity"][i]),
                    "p_negative": float(probs[0]), "p_neutral": float(probs[1]), "p_positive": float(probs[2]),
                    "quality_flag": quality, "feature_version": config["data"]["feature_version"], "model_version": model_id,
                })
            from .export import write_prediction_csv
            write_prediction_csv(run_dir / "valid_clean_predictions.csv", rows, expected_ids=valid_batch.sample_ids)
            _write_json(run_dir / "run_status.json", result)
        return result
    except Exception as exc:
        if run_dir is not None:
            _write_json(run_dir / "run_status.json", {
                "run_id": run_id, "status": "failed", "seed": int(seed),
                "error_type": type(exc).__name__, "failure_reason": str(exc),
            })
        raise
