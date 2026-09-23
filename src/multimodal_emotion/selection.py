"""Validation-only model selection helpers; specialized test sets are excluded."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def eligible_by_clean_guards(candidate: Mapping[str, Any], b3: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    rule = config["selection"]
    mae_ok = float(candidate["clean_mae"]) <= float(b3["clean_mae"]) * float(rule["clean_mae_max_relative_to_B3"])
    f1_ok = float(candidate["clean_macro_f1"]) >= float(b3["clean_macro_f1"]) - float(rule["clean_macro_f1_max_absolute_drop_from_B3"])
    return mae_ok and f1_ok


def select_candidate(results: Mapping[str, Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    """Apply predeclared validation rules to three-seed aggregate results.

    Each result must contain `R`, `clean_mae`, `clean_macro_f1`,
    `main_macro_f1`, and `parameter_count`. `results` must not contain test or
    special-test metrics.
    """
    selection = config["selection"]
    required = set(selection["candidate_ids"])
    if set(results) != required:
        raise ValueError(f"candidate set mismatch; expected {sorted(required)}, got {sorted(results)}")
    b3 = results["B3"]
    eligible = {name: row for name, row in results.items() if eligible_by_clean_guards(row, b3, config)}
    if not eligible:
        raise ValueError("B3 must pass its own clean-performance guards")

    augmented = set(eligible) - {"B3"}
    tolerance = float(selection["primary_near_tie_tolerance"])

    def beats(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        difference = float(left["R"]) - float(right["R"])
        if difference < -tolerance:
            return True
        if difference > tolerance:
            return False
        f1_difference = float(left["main_macro_f1"]) - float(right["main_macro_f1"])
        if abs(f1_difference) > 1e-12:
            return f1_difference > 0
        return int(left["parameter_count"]) < int(right["parameter_count"])

    if not augmented:
        chosen = "B3"
        reason = "no augmented candidate meets the clean-performance guards"
    else:
        complex_ids = {"B5", "M0", "M1"}
        b4_ok = "B4" in eligible
        complex_beats_b4 = any(name in eligible and beats(eligible[name], eligible["B4"]) for name in complex_ids) if b4_ok else True
        if b4_ok and not complex_beats_b4:
            chosen = "B4"
            reason = "B4 passes clean guards and no complex candidate improves its robust score"
        else:
            best_r = min(float(eligible[name]["R"]) for name in augmented)
            tied = [name for name in augmented if float(eligible[name]["R"]) <= best_r + tolerance]
            chosen = min(tied, key=lambda name: (-float(eligible[name]["main_macro_f1"]), int(eligible[name]["parameter_count"]), name))
            reason = "lowest robust score among clean-eligible augmented candidates"
    return {
        "chosen_model": chosen,
        "reason": reason,
        "eligible_candidates": sorted(eligible),
        "selection_source": "valid_only",
        "special_test_used": False,
    }
