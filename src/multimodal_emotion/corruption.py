"""Deterministic natural/artificial missingness masks for aligned50 inputs.

All generated corruption masks C have shape ``[3,50]`` and obey
``C <= P & O0``. Applying C creates current observation mask ``O=O0 & ~C``
and safely zero-fills hidden feature rows through ``SampleBatch.with_corruption``.

``generate_validation_scenarios`` returns the fixed 90-condition main grid,
15 shared TAV stress conditions, and 9 asynchronous stress conditions. Their
starts are generated per sample at application time, so the stored scenario
list is independent of feature values and can be shared across model seeds.
Training masks use one SHA256-derived NumPy PCG64 stream per
``mask-v1|train|seed|epoch|sample_id`` key; they never depend on shuffle or
DataLoader order.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .data import DataContractError, MODALITIES, SampleBatch


PROTOCOL_VERSION = "mask-v1"
VALIDATION_SEED = 3407
RATIOS: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5)
MAIN_COMBINATIONS: Tuple[Tuple[str, ...], ...] = (
    ("T",), ("A",), ("V",), ("T", "A"), ("T", "V"), ("A", "V")
)
LOCAL_COMBINATIONS: Tuple[Tuple[str, ...], ...] = MAIN_COMBINATIONS + (("T", "A", "V"),)
POSITIONS: Tuple[str, ...] = ("front", "middle", "back")


@dataclass(frozen=True)
class ScenarioSpec:
    """One fixed evaluation condition; asynchronous starts are per sample."""

    scenario_id: str
    group: str
    modalities: Tuple[str, ...]
    ratio: float
    position: str
    replicate: int = 0
    shared_start: bool = True
    seed: int = VALIDATION_SEED


@dataclass(frozen=True)
class CorruptionRecord:
    """Compact audit details for one sample/modality corruption interval."""

    sample_id: str
    scenario_id: str
    group: str
    modality: str
    interval_start: Optional[int]
    interval_stop: Optional[int]
    requested_length: Optional[int]
    actual_added_points: int
    actual_ratio: Optional[float]
    requested_ratio: Optional[float]
    degenerate: bool = False


@dataclass(frozen=True)
class ScenarioMask:
    """A sample's C mask and compact per-modality audit records."""

    corruption_mask: np.ndarray
    records: Tuple[CorruptionRecord, ...]


@dataclass(frozen=True)
class TrainingMask:
    """Per-sample training corruption and the sampled protocol choices."""

    corruption_mask: np.ndarray
    corruption_type: str
    modalities: Tuple[str, ...]
    requested_ratio: Optional[float]
    shared_start: Optional[bool]
    records: Tuple[CorruptionRecord, ...]


def generate_validation_scenarios() -> Tuple[ScenarioSpec, ...]:
    """Return the fixed 90 + 15 + 9 validation scenarios in protocol order."""
    scenarios: List[ScenarioSpec] = []
    for modalities in MAIN_COMBINATIONS:
        label = "".join(modalities)
        for ratio in RATIOS:
            percent = int(round(ratio * 100))
            for position in POSITIONS:
                scenarios.append(
                    ScenarioSpec(
                        scenario_id="main_{}_{}_{}".format(label, percent, position),
                        group="main",
                        modalities=modalities,
                        ratio=ratio,
                        position=position,
                    )
                )
    for ratio in RATIOS:
        percent = int(round(ratio * 100))
        for position in POSITIONS:
            scenarios.append(
                ScenarioSpec(
                    scenario_id="tav_TAV_{}_{}".format(percent, position),
                    group="tav",
                    modalities=("T", "A", "V"),
                    ratio=ratio,
                    position=position,
                    shared_start=True,
                )
            )
    for modalities in (("T", "A"), ("T", "V"), ("A", "V")):
        label = "".join(modalities)
        for replicate in (1, 2, 3):
            scenarios.append(
                ScenarioSpec(
                    scenario_id="async_{}_30_r{}".format(label, replicate),
                    group="async",
                    modalities=modalities,
                    ratio=0.3,
                    position="random",
                    replicate=replicate,
                    shared_start=False,
                    seed=VALIDATION_SEED,
                )
            )
    return tuple(scenarios)


def validation_earlystop_scenarios() -> Tuple[ScenarioSpec, ...]:
    """Return the six exact 30%-middle cases already present in the main grid."""
    lookup = {scenario.scenario_id: scenario for scenario in generate_validation_scenarios()}
    return tuple(
        lookup["main_{}_30_middle".format("".join(modalities))]
        for modalities in MAIN_COMBINATIONS
    )


def stable_seed(key: str) -> int:
    """Map a string key to an unsigned big-endian SHA256-derived 64-bit seed."""
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _generator(key: str) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(stable_seed(key)))


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return half-open contiguous true runs from a one-dimensional mask."""
    indexes = np.flatnonzero(mask)
    if indexes.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(indexes) > 1) + 1
    groups = np.split(indexes, cuts)
    return [(int(group[0]), int(group[-1]) + 1) for group in groups]


def _valid_length(valid_mask: np.ndarray) -> int:
    return int(np.count_nonzero(valid_mask))


def _requested_length(valid_mask: np.ndarray, ratio: float) -> Optional[int]:
    length = _valid_length(valid_mask)
    if length == 0:
        return None
    # Python round is ties-to-even, matching the fixed protocol.
    return max(1, int(round(float(ratio) * length)))


def _candidate_intervals(valid_mask: np.ndarray, requested: int) -> List[Tuple[int, int, int]]:
    """Legal intervals as (start, stop, effective-valid ordinal start).

    A block never crosses an invalid position. If no contiguous valid run is
    long enough, use the longest run and report the shorter realized span.
    """
    runs = _runs(valid_mask)
    if not runs:
        return []
    actual = min(requested, max(stop - start for start, stop in runs))
    candidates: List[Tuple[int, int, int]] = []
    valid_before = 0
    run_ordinal_starts: Dict[int, int] = {}
    for start, stop in runs:
        run_ordinal_starts[start] = valid_before
        valid_before += stop - start
    for start, stop in runs:
        run_length = stop - start
        if run_length < actual:
            continue
        ordinal_base = run_ordinal_starts[start]
        for block_start in range(start, stop - actual + 1):
            ordinal_start = ordinal_base + block_start - start
            candidates.append((block_start, block_start + actual, ordinal_start))
    return candidates


def _choose_interval(
    valid_mask: np.ndarray,
    requested: Optional[int],
    position: str,
    rng: Optional[np.random.Generator] = None,
    avoid_start: Optional[int] = None,
) -> Optional[Tuple[int, int, int, bool]]:
    """Select an interval; output (start, stop, requested length, degenerate)."""
    if requested is None:
        return None
    candidates = _candidate_intervals(valid_mask, requested)
    if not candidates:
        return None
    actual_length = candidates[0][1] - candidates[0][0]
    length = _valid_length(valid_mask)
    target = {
        "front": 0,
        "middle": (length - actual_length) // 2,
        "back": length - actual_length,
    }.get(position)
    if position == "random":
        selectable = candidates
        if avoid_start is not None and len(candidates) > 1:
            different = [candidate for candidate in candidates if candidate[0] != avoid_start]
            if different:
                selectable = different
        if rng is None:
            raise ValueError("A NumPy RNG is required for a random interval.")
        chosen = selectable[int(rng.integers(0, len(selectable)))]
    elif target is not None:
        # Pick the legal interval nearest the ideal position in the compacted
        # valid-position order. Ties resolve toward the earlier interval.
        chosen = min(candidates, key=lambda item: (abs(item[2] - target), item[0]))
    else:
        raise ValueError("Unknown position {!r}.".format(position))
    shortened = actual_length < requested
    single_start = position == "random" and len(candidates) == 1
    return chosen[0], chosen[1], requested, shortened or single_start


def scenario_mask(
    valid_mask: np.ndarray,
    original_observed_mask: np.ndarray,
    sample_id: str,
    scenario: ScenarioSpec,
) -> ScenarioMask:
    """Build one fixed scenario's C mask for a sample and return audit rows."""
    p = np.asarray(valid_mask, dtype=np.bool_)
    o0 = np.asarray(original_observed_mask, dtype=np.bool_)
    if p.shape != (50,):
        raise DataContractError("A sample valid_mask must have shape [50].")
    if o0.shape != (3, 50):
        raise DataContractError("A sample original_observed_mask must have shape [3,50].")
    if not scenario.modalities or any(name not in MODALITIES for name in scenario.modalities):
        raise DataContractError("Scenario modalities must be a non-empty subset of T/A/V.")
    if not (0.0 < scenario.ratio <= 1.0):
        raise DataContractError("Scenario ratio must be in (0,1].")
    if scenario.shared_start and scenario.position == "random":
        raise DataContractError("A random scenario must specify per-modality starts (shared_start=False).")

    corruption = np.zeros((3, 50), dtype=np.bool_)
    requested = _requested_length(p, scenario.ratio)
    records: List[CorruptionRecord] = []
    if requested is None:
        for modality in scenario.modalities:
            records.append(
                CorruptionRecord(
                    sample_id=str(sample_id),
                    scenario_id=scenario.scenario_id,
                    group=scenario.group,
                    modality=modality,
                    interval_start=None,
                    interval_stop=None,
                    requested_length=None,
                    actual_added_points=0,
                    actual_ratio=None,
                    requested_ratio=scenario.ratio,
                )
            )
        return ScenarioMask(corruption, tuple(records))

    seed_key = "{}|{}|{}|{}|{}".format(
        PROTOCOL_VERSION, scenario.seed, sample_id, scenario.scenario_id, scenario.replicate
    )
    rng = _generator(seed_key)
    chosen_intervals: Dict[str, Optional[Tuple[int, int, int, bool]]] = {}
    if scenario.shared_start:
        shared = _choose_interval(p, requested, scenario.position)
        for modality in scenario.modalities:
            chosen_intervals[modality] = shared
    else:
        prior_start = None
        for modality in scenario.modalities:
            chosen = _choose_interval(p, requested, scenario.position, rng=rng, avoid_start=prior_start)
            chosen_intervals[modality] = chosen
            if chosen is not None:
                prior_start = chosen[0]

    valid_count = _valid_length(p)
    for modality in scenario.modalities:
        idx = MODALITIES.index(modality)
        chosen = chosen_intervals[modality]
        if chosen is None:
            start = stop = req = None
            degenerate = False
        else:
            start, stop, req, degenerate = chosen
            corruption[idx, start:stop] = o0[idx, start:stop] & p[start:stop]
        added = int(corruption[idx].sum())
        records.append(
            CorruptionRecord(
                sample_id=str(sample_id),
                scenario_id=scenario.scenario_id,
                group=scenario.group,
                modality=modality,
                interval_start=start,
                interval_stop=stop,
                requested_length=req,
                actual_added_points=added,
                actual_ratio=(float(added) / valid_count) if valid_count else None,
                requested_ratio=scenario.ratio,
                degenerate=degenerate,
            )
        )
    return ScenarioMask(corruption_mask=corruption, records=tuple(records))


def apply_scenario(
    batch: SampleBatch,
    row: Optional[int] = None,
    scenario: Optional[ScenarioSpec] = None,
) -> SampleBatch:
    """Apply a scenario to one row or the whole batch and return a fresh batch.

    ``row=None`` corrupts all samples, which is the normal evaluation path.
    Passing an integer changes only that batch row and leaves other rows clean.
    For saved interval metadata, call :func:`apply_scenario_with_records`.
    """
    # Convenience form: apply_scenario(batch, scenario) means the whole batch.
    if scenario is None and isinstance(row, ScenarioSpec):
        scenario = row
        row = None
    if scenario is None:
        raise TypeError("scenario is required")
    corrupted, _ = apply_scenario_with_records(batch, scenario, row=row)
    return corrupted


def apply_scenario_with_records(
    batch: SampleBatch,
    scenario: ScenarioSpec,
    row: Optional[int] = None,
) -> Tuple[SampleBatch, Tuple[CorruptionRecord, ...]]:
    """Apply one scenario and return (new batch, interval audit records)."""
    n = len(batch.sample_ids)
    if row is not None and not (0 <= row < n):
        raise IndexError("row index {} is outside a batch of {} samples.".format(row, n))
    rows = range(n) if row is None else (row,)
    corruption = np.zeros_like(batch.original_observed_mask, dtype=np.bool_)
    records: List[CorruptionRecord] = []
    for index in rows:
        outcome = scenario_mask(
            batch.valid_mask[index],
            batch.original_observed_mask[index],
            batch.sample_ids[index],
            scenario,
        )
        corruption[index] = outcome.corruption_mask
        records.extend(outcome.records)
    return batch.with_corruption(corruption), tuple(records)


def make_training_corruption(
    valid_mask: np.ndarray,
    original_observed_mask: np.ndarray,
    seed: int,
    epoch: int,
    sample_id: str,
) -> TrainingMask:
    """Draw one deterministic B4/B5/M1-style training mask for one sample.

    The random stream key is exactly
    ``mask-v1|train|seed|epoch|sample_id``. Its draws occur in the fixed order
    type -> modality combination -> ratio -> sharing mode -> start. A sample's
    output therefore does not change when the epoch shuffle or loader workers
    change.
    """
    p = np.asarray(valid_mask, dtype=np.bool_)
    o0 = np.asarray(original_observed_mask, dtype=np.bool_)
    if p.shape != (50,) or o0.shape != (3, 50):
        raise DataContractError("Expected sample masks P=[50] and O0=[3,50].")
    if epoch < 1:
        raise ValueError("Formal training epochs are one-based and must be >= 1.")
    rng = _generator("{}|train|{}|{}|{}".format(PROTOCOL_VERSION, int(seed), int(epoch), sample_id))
    draw = float(rng.random())
    corruption = np.zeros((3, 50), dtype=np.bool_)
    if draw < 0.30:
        return TrainingMask(corruption, "none", (), None, None, ())

    if draw < 0.90:
        modalities = LOCAL_COMBINATIONS[int(rng.integers(0, len(LOCAL_COMBINATIONS)))]
        ratio = RATIOS[int(rng.integers(0, len(RATIOS)))]
        if len(modalities) == 3:
            shared = True
        elif len(modalities) == 2:
            shared = bool(float(rng.random()) < 0.5)
        else:
            shared = True
        requested = _requested_length(p, ratio)
        if requested is None:
            empty_records = tuple(
                CorruptionRecord(
                    sample_id=str(sample_id),
                    scenario_id="train_{}_{}_{}".format(int(round(ratio * 100)), "".join(modalities), epoch),
                    group="train_local",
                    modality=modality,
                    interval_start=None,
                    interval_stop=None,
                    requested_length=None,
                    actual_added_points=0,
                    actual_ratio=None,
                    requested_ratio=ratio,
                )
                for modality in modalities
            )
            return TrainingMask(corruption, "local_span", modalities, ratio, shared, empty_records)

        intervals: Dict[str, Optional[Tuple[int, int, int, bool]]] = {}
        if shared:
            interval = _choose_interval(p, requested, "random", rng=rng)
            for modality in modalities:
                intervals[modality] = interval
        else:
            for modality in modalities:
                intervals[modality] = _choose_interval(p, requested, "random", rng=rng)
        group = "".join(modalities)
        records = []
        valid_count = _valid_length(p)
        for modality in modalities:
            modality_index = MODALITIES.index(modality)
            chosen = intervals[modality]
            if chosen is None:
                start = stop = req = None
                degenerate = False
            else:
                start, stop, req, degenerate = chosen
                corruption[modality_index, start:stop] = p[start:stop] & o0[modality_index, start:stop]
            added = int(corruption[modality_index].sum())
            records.append(
                CorruptionRecord(
                    sample_id=str(sample_id),
                    scenario_id="train_{}_{}_{}".format(int(round(ratio * 100)), group, epoch),
                    group="train_local",
                    modality=modality,
                    interval_start=start,
                    interval_stop=stop,
                    requested_length=req,
                    actual_added_points=added,
                    actual_ratio=(float(added) / valid_count) if valid_count else None,
                    requested_ratio=ratio,
                    degenerate=degenerate,
                )
            )
        return TrainingMask(corruption, "local_span", modalities, ratio, shared, tuple(records))

    # Whole-modality corruption is sampled only from the six non-empty proper
    # subsets, as specified. Every available observed position is deleted.
    modalities = MAIN_COMBINATIONS[int(rng.integers(0, len(MAIN_COMBINATIONS)))]
    records = []
    valid_count = _valid_length(p)
    for modality in modalities:
        modality_index = MODALITIES.index(modality)
        corruption[modality_index] = p & o0[modality_index]
        added = int(corruption[modality_index].sum())
        records.append(
            CorruptionRecord(
                sample_id=str(sample_id),
                scenario_id="train_whole_{}_{}".format("".join(modalities), epoch),
                group="train_whole_modality",
                modality=modality,
                interval_start=None,
                interval_stop=None,
                requested_length=None,
                actual_added_points=added,
                actual_ratio=(float(added) / valid_count) if valid_count else None,
                requested_ratio=None,
            )
        )
    return TrainingMask(corruption, "whole_modality", modalities, None, None, tuple(records))
