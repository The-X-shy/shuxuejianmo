"""Configuration loading and consistency checks for the experiment plan."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.json"


class ConfigError(ValueError):
    """Raised when an experiment configuration is internally inconsistent."""


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        config = json.load(f)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any], *, require_paths: bool = False) -> None:
    """Validate key cross-field invariants without accessing any data files.

    `require_paths=True` is intended for a real run. It deliberately fails on
    the planning template's null attachment paths.
    """
    errors: list[str] = []
    required_sections = ("paths_to_fill_at_E00", "data", "model", "training", "corruption", "evaluation", "selection", "budget")
    for section in required_sections:
        if section not in config or not isinstance(config[section], dict):
            errors.append(f"missing configuration section: {section}")
    if errors:
        raise ConfigError("; ".join(errors))

    paths = config["paths_to_fill_at_E00"]
    if require_paths:
        for key in ("attachment1", "attachment2", "attachment3", "attachment4", "project_root"):
            if not paths.get(key):
                errors.append(f"path is not configured: {key}")

    data = config["data"]
    if data.get("feature_version") != "official_aligned50":
        errors.append("core model must use official_aligned50 features")
    if int(data.get("sequence_length", -1)) != 50:
        errors.append("aligned sequence_length must be 50")
    if data.get("dimensions") != {"T": 768, "A": 74, "V": 35}:
        errors.append("aligned feature dimensions must be T=768, A=74, V=35")
    if data.get("normalization_fit_split") != "train" or data.get("normalization_ddof") != 0:
        errors.append("normalization must use train only and ddof=0")
    if data.get("raw_text_and_text_bert_as_predictor_input") is not False:
        errors.append("raw_text/text_bert may not enter the predictor")

    training = config["training"]
    model_order = training.get("model_execution_order", [])
    seeds = training.get("seed_execution_order", [])
    if len(model_order) != 8 or len(set(model_order)) != 8:
        errors.append("core run queue requires eight unique models")
    if seeds != [42, 17, 2026]:
        errors.append("seed execution order must remain [42, 17, 2026]")
    if config["budget"].get("default_core_runs") != len(model_order) * len(seeds):
        errors.append("default_core_runs must equal model_count × seed_count")
    if training.get("drop_last") is not False:
        errors.append("drop_last must be false to retain all training samples")

    corruption = config["corruption"]
    probs = corruption.get("type_probabilities", {})
    if abs(sum(float(v) for v in probs.values()) - 1.0) > 1e-12:
        errors.append("training corruption probabilities must sum to one")
    if not corruption.get("only_originally_observed_positions") or not corruption.get("valid_mask_never_changed"):
        errors.append("corruption must preserve valid positions and only hide observed inputs")

    evaluation = config["evaluation"]
    expected_views = sum(int(evaluation.get(k, -10_000)) for k in ("clean_count", "main_count", "tav_count", "async_count"))
    if expected_views != evaluation.get("total_views_per_run"):
        errors.append("evaluation view count does not match its components")
    if (evaluation.get("main_count"), evaluation.get("tav_count"), evaluation.get("async_count")) != (90, 15, 9):
        errors.append("fixed evaluation grid must contain 90 + 15 + 9 scenarios")
    if not config["selection"].get("no_special_test_selection"):
        errors.append("special test sets may not be used for model selection")

    selection = config["selection"]
    if not selection.get("baseline_rule"):
        errors.append("selection.baseline_rule must explicitly define the B4/B3 fallback branch")

    if errors:
        raise ConfigError("; ".join(errors))


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
