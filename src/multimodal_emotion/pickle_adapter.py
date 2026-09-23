"""Strict adapter for trusted official aligned50 pickle files.

Pickle deserialization can execute arbitrary code. This adapter refuses to
unpickle unless the caller explicitly passes ``trusted_source=True`` and
documents that the file is from a trusted competition source. It requires the
official ``train``, ``valid``, and ``test`` keys, preserves each split's sample
order, and delegates aligned feature validation to :class:`AlignedSplit`.

Padding is never inferred from zero-valued features. Callers must supply either
verified ``valid_masks`` with shape ``[N,50]`` for every split or verified
``valid_lengths`` with one prefix length per sample for every split. The
returned audit object includes the source file SHA256, file size, split shapes,
ID-order hashes, selected field names, and the validity-mask evidence source.

Default feature/id/label key names match common aligned50 exports. A
``PickleSchema`` can specify different field names (including dotted paths
such as ``labels.M``) when the supplied artifact's documented schema differs.
Integer class annotations always require an explicit source-to-internal class
mapping. Exact official strings ``Negative``, ``Neutral``, and ``Positive``
map explicitly to 0, 1, and 2; a supplied mapping is checked against that
fixed string mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .data import AlignedSplit, DataContractError


OFFICIAL_STRING_CLASS_MAP: Mapping[str, int] = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2,
}
REQUIRED_SPLITS: Tuple[str, ...] = ("train", "valid", "test")


class PickleAdapterError(ValueError):
    """Raised for unsafe opt-in, schema, annotation, or mask-evidence errors."""


@dataclass(frozen=True)
class PickleSchema:
    """Field paths in each split object of the pickle.

    Paths may be nested mapping paths separated by dots (for example,
    ``labels.M``). ``classification_label_key`` may be ``None`` to request
    discovery among the standard class-label aliases.
    """

    text_key: str = "text"
    audio_key: str = "audio"
    vision_key: str = "vision"
    sample_id_key: str = "id"
    classification_label_key: Optional[str] = "classification_labels"
    regression_label_key: Optional[str] = "regression_labels"


@dataclass(frozen=True)
class SplitAudit:
    """Small structural summary for a single official split."""

    split: str
    sample_count: int
    sample_ids_sha256: str
    feature_shapes: Mapping[str, Tuple[int, ...]]
    feature_dtypes: Mapping[str, str]
    class_annotation_dtype: str
    regression_label_shape: Tuple[int, ...]
    valid_mask_source: str


@dataclass(frozen=True)
class PickleAudit:
    """Lightweight source and schema provenance, safe to serialize as JSON."""

    source_path: str
    file_size_bytes: int
    file_sha256: str
    splits: Mapping[str, SplitAudit]
    class_mapping: Tuple[Tuple[str, int], ...]
    selected_fields: Mapping[str, str]

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-friendly metadata without feature values or labels."""
        return {
            "source_path": self.source_path,
            "file_size_bytes": self.file_size_bytes,
            "file_sha256": self.file_sha256,
            "splits": {
                name: {
                    "sample_count": item.sample_count,
                    "sample_ids_sha256": item.sample_ids_sha256,
                    "feature_shapes": {key: list(shape) for key, shape in item.feature_shapes.items()},
                    "feature_dtypes": dict(item.feature_dtypes),
                    "class_annotation_dtype": item.class_annotation_dtype,
                    "regression_label_shape": list(item.regression_label_shape),
                    "valid_mask_source": item.valid_mask_source,
                }
                for name, item in self.splits.items()
            },
            "class_mapping": [
                {"source_label": key, "internal_class": value} for key, value in self.class_mapping
            ],
            "selected_fields": dict(self.selected_fields),
        }


@dataclass(frozen=True)
class AlignedPickle:
    """Three converted official splits and their structural audit metadata."""

    splits: Mapping[str, AlignedSplit]
    audit: PickleAudit

    def __getitem__(self, split: str) -> AlignedSplit:
        return self.splits[split]

    @property
    def train(self) -> AlignedSplit:
        return self.splits["train"]

    @property
    def valid(self) -> AlignedSplit:
        return self.splits["valid"]

    @property
    def test(self) -> AlignedSplit:
        return self.splits["test"]


def _read_trusted_pickle(path: Path) -> Tuple[Any, int, str]:
    """Hash and load through the same open file; caller enforces trust opt-in."""
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            size += len(block)
            hasher.update(block)
        stream.seek(0)
        value = pickle.load(stream)
    return value, size, hasher.hexdigest()


def _lookup_path(container: Any, key_path: str) -> Any:
    current = container
    for component in key_path.split("."):
        if isinstance(current, Mapping) and component in current:
            current = current[component]
        else:
            raise KeyError(key_path)
    return current


def _resolve_field(
    record: Mapping[str, Any],
    preferred: Optional[str],
    aliases: Sequence[str] = (),
    required: bool = True,
) -> Tuple[Optional[Any], Optional[str]]:
    if preferred is not None:
        try:
            return _lookup_path(record, preferred), preferred
        except KeyError:
            pass
    found = []
    for alias in aliases:
        try:
            found.append((alias, _lookup_path(record, alias)))
        except KeyError:
            continue
    if len(found) > 1:
        raise PickleAdapterError(
            "Multiple candidate fields are present ({}); select one explicitly with PickleSchema.".format(
                ", ".join(name for name, _ in found)
            )
        )
    if found:
        name, value = found[0]
        return value, name
    if required:
        raise PickleAdapterError(
            "Required field {!r} is missing; configure its documented path with PickleSchema.".format(preferred)
        )
    return None, None


def _as_vector(values: Any, name: str, count: int) -> np.ndarray:
    result = np.asarray(values)
    if result.shape == (count, 1):
        result = result[:, 0]
    if result.shape != (count,):
        raise PickleAdapterError("{} must have shape [N] or [N,1]; got {}.".format(name, result.shape))
    return result


def _decode_ids(values: Any, count: int) -> Tuple[str, ...]:
    vector = _as_vector(values, "sample_ids", count)
    decoded = []
    for value in vector:
        item = value.item() if isinstance(value, np.generic) else value
        if isinstance(item, bytes):
            try:
                item = item.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise PickleAdapterError("Sample ID bytes are not valid UTF-8.") from exc
        if not isinstance(item, (str, int, float)):
            raise PickleAdapterError("Sample IDs must be scalar strings or numbers; got {}.".format(type(item)))
        if isinstance(item, float) and not np.isfinite(item):
            raise PickleAdapterError("Sample IDs cannot be NaN or infinity.")
        decoded.append(str(item))
    if len(set(decoded)) != len(decoded):
        raise PickleAdapterError("Sample IDs must be unique within each official split.")
    return tuple(decoded)


def _mask_for_split(
    split: str,
    sample_ids: Sequence[str],
    valid_masks: Optional[Mapping[str, Any]],
    valid_lengths: Optional[Mapping[str, Any]],
) -> Tuple[np.ndarray, str]:
    if (valid_masks is None) == (valid_lengths is None):
        raise PickleAdapterError("Provide exactly one of verified valid_masks or verified valid_lengths.")
    n = len(sample_ids)
    if valid_masks is not None:
        if set(valid_masks) != set(REQUIRED_SPLITS):
            raise PickleAdapterError("valid_masks must provide exactly train, valid, and test entries.")
        raw = np.asarray(valid_masks[split])
        if raw.shape != (n, 50):
            raise PickleAdapterError("valid_masks[{}] must have shape [N,50]; got {}.".format(split, raw.shape))
        if raw.dtype != np.bool_:
            if not np.issubdtype(raw.dtype, np.number) or not np.all(np.isfinite(raw)):
                raise PickleAdapterError("valid_masks[{}] must contain only boolean/0/1 values.".format(split))
            if not np.all((raw == 0) | (raw == 1)):
                raise PickleAdapterError("valid_masks[{}] contains values other than 0/1.".format(split))
        return np.asarray(raw, dtype=np.bool_).copy(), "verified_valid_masks"

    assert valid_lengths is not None
    if set(valid_lengths) != set(REQUIRED_SPLITS):
        raise PickleAdapterError("valid_lengths must provide exactly train, valid, and test entries.")
    lengths = np.asarray(valid_lengths[split])
    if lengths.shape != (n,):
        raise PickleAdapterError("valid_lengths[{}] must have shape [N]; got {}.".format(split, lengths.shape))
    if not np.issubdtype(lengths.dtype, np.number) or not np.all(np.isfinite(lengths)):
        raise PickleAdapterError("valid_lengths[{}] must contain finite integers.".format(split))
    if not np.all(lengths == np.floor(lengths)):
        raise PickleAdapterError("valid_lengths[{}] contains non-integer values.".format(split))
    int_lengths = lengths.astype(np.int64)
    if np.any((int_lengths < 0) | (int_lengths > 50)):
        raise PickleAdapterError("valid_lengths[{}] values must be between 0 and 50.".format(split))
    mask = np.arange(50, dtype=np.int64)[None, :] < int_lengths[:, None]
    return mask, "verified_prefix_lengths"


def _label_key(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _class_mapping_for_all(
    raw_by_split: Mapping[str, np.ndarray],
    provided: Optional[Mapping[Any, int]],
) -> Tuple[Dict[Any, int], Tuple[Tuple[str, int], ...]]:
    all_values = []
    for values in raw_by_split.values():
        for value in values:
            item = _label_key(value)
            if isinstance(item, bytes):
                try:
                    item = item.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise PickleAdapterError("Class annotation bytes must be valid UTF-8.") from exc
            if isinstance(item, bool) or not isinstance(item, (str, int, float)):
                raise PickleAdapterError(
                    "Classification annotations must be scalar strings or numbers; got {}.".format(type(item))
                )
            if isinstance(item, float) and not np.isfinite(item):
                raise PickleAdapterError("Classification annotations cannot be NaN or infinity.")
            all_values.append(item)
    unique = set(all_values)
    if not unique:
        raise PickleAdapterError("Classification annotations must not be empty.")

    all_strings = all(isinstance(value, str) for value in unique)
    if all_strings:
        normalized_unique = {value.strip() for value in unique}
        if not normalized_unique.issubset(set(OFFICIAL_STRING_CLASS_MAP)):
            raise PickleAdapterError(
                "String annotations must be exactly Negative/Neutral/Positive; got {}.".format(
                    sorted(normalized_unique)
                )
            )
        inferred = dict(OFFICIAL_STRING_CLASS_MAP)
        if provided is not None:
            normalized_provided = {}
            for key, class_id in provided.items():
                source_key = _label_key(key)
                if isinstance(source_key, str):
                    source_key = source_key.strip()
                normalized_provided[source_key] = int(class_id)
            for value in normalized_unique:
                if normalized_provided.get(value) != inferred[value]:
                    raise PickleAdapterError(
                        "Provided class mapping disagrees with official string mapping for {!r}.".format(value)
                    )
            inferred = normalized_provided
        mapping = inferred
    else:
        if provided is None:
            raise PickleAdapterError(
                "Integer/non-official classification annotations require explicit class_mapping."
            )
        mapping = {}
        for key, class_id in provided.items():
            source_key = _label_key(key)
            if isinstance(source_key, str):
                source_key = source_key.strip()
            mapped = int(class_id)
            if mapped not in (0, 1, 2):
                raise PickleAdapterError("class_mapping values must be 0/1/2 in Negative/Neutral/Positive order.")
            mapping[source_key] = mapped
    if not set(mapping.values()).issubset({0, 1, 2}):
        raise PickleAdapterError("class_mapping values must use internal classes 0/1/2.")
    missing = unique - set(mapping)
    # Recheck string bytes/whitespace against normalized mapping keys.
    if missing:
        normalized_all = {
            value.strip() if isinstance(value, str) else value
            for value in unique
        }
        missing = normalized_all - set(mapping)
    if missing:
        raise PickleAdapterError("class_mapping does not cover annotations: {}.".format(sorted(map(repr, missing))))
    used_labels = {
        value.strip() if isinstance(value, str) else value
        for value in unique
    }
    used_class_ids = [mapping[value] for value in used_labels]
    if len(set(used_class_ids)) != len(used_class_ids):
        raise PickleAdapterError("Distinct source annotations must map to distinct internal class IDs.")
    mapping_audit = tuple(sorted(((repr(key), value) for key, value in mapping.items()), key=lambda row: row[0]))
    return mapping, mapping_audit


def _map_class_vector(values: np.ndarray, mapping: Mapping[Any, int], split: str) -> np.ndarray:
    mapped = []
    for value in values:
        item = _label_key(value)
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        if isinstance(item, str):
            item = item.strip()
        if item not in mapping:
            raise PickleAdapterError("Unmapped classification label in {}: {!r}.".format(split, item))
        mapped.append(mapping[item])
    result = np.asarray(mapped, dtype=np.int64)
    if np.any((result < 0) | (result > 2)):
        raise PickleAdapterError("Mapped y_class must use internal class IDs 0/1/2.")
    return result


def load_aligned_pickle(
    path: os.PathLike[str] | str,
    *,
    trusted_source: bool = False,
    valid_masks: Optional[Mapping[str, Any]] = None,
    valid_lengths: Optional[Mapping[str, Any]] = None,
    class_mapping: Optional[Mapping[Any, int]] = None,
    schema: PickleSchema = PickleSchema(),
    include_test_labels: bool = False,
) -> AlignedPickle:
    """Load official aligned50 splits after explicit trust and mask evidence.

    Args:
        path: Source pickle file.
        trusted_source: Must be explicitly ``True``. Only use for a known,
            trusted competition attachment; untrusted pickle files can execute
            arbitrary code during loading.
        valid_masks: Verified boolean/0-1 P arrays for each official split.
        valid_lengths: Alternatively, verified contiguous prefix lengths for
            each official split. Exactly one mask source is required.
        class_mapping: Explicit original-label to internal class-ID mapping for
            integer or other noncanonical annotations. Exact canonical English
            class-name strings use the fixed Negative=0, Neutral=1, Positive=2
            mapping; any supplied mapping is checked against it.
        schema: Documented field paths within each split record.
        include_test_labels: False by default to keep test targets out of the
            development process. Set True only for the one-time E13 evaluation
            after the model and all selection rules are frozen.

    Returns:
        An :class:`AlignedPickle` with train/valid/test :class:`AlignedSplit`
        instances in their original order and lightweight source audit data.
    """
    if trusted_source is not True:
        raise PickleAdapterError(
            "Refusing to unpickle without trusted_source=True; only load a trusted competition attachment."
        )
    if (valid_masks is None) == (valid_lengths is None):
        raise PickleAdapterError("Provide exactly one of valid_masks or valid_lengths.")
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(str(source))

    try:
        payload, file_size, file_hash = _read_trusted_pickle(source)
    except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, IndexError, TypeError, ValueError) as exc:
        raise PickleAdapterError("Could not read the trusted pickle: {}".format(exc)) from exc
    if not isinstance(payload, Mapping):
        raise PickleAdapterError("Pickle root must be a mapping with train/valid/test entries.")
    missing_splits = [name for name in REQUIRED_SPLITS if name not in payload]
    if missing_splits:
        raise PickleAdapterError("Pickle is missing required official splits: {}.".format(missing_splits))

    raw_class_vectors: Dict[str, np.ndarray] = {}
    split_masks: Dict[str, np.ndarray] = {}
    split_mask_sources: Dict[str, str] = {}
    ids_by_split: Dict[str, Tuple[str, ...]] = {}
    selected_fields: Dict[str, str] = {}
    raw_regression: Dict[str, np.ndarray] = {}
    raw_features: Dict[str, Dict[str, np.ndarray]] = {}

    for split in REQUIRED_SPLITS:
        record = payload[split]
        if not isinstance(record, Mapping):
            raise PickleAdapterError("Top-level {!r} entry must be a mapping.".format(split))
        text, text_field = _resolve_field(record, schema.text_key)
        audio, audio_field = _resolve_field(record, schema.audio_key)
        vision, vision_field = _resolve_field(record, schema.vision_key)
        ids_value, id_field = _resolve_field(
            record, schema.sample_id_key, aliases=("sample_ids", "ids", "sample_id")
        )
        text_array = np.asarray(text)
        audio_array = np.asarray(audio)
        vision_array = np.asarray(vision)
        if text_array.ndim != 3:
            raise PickleAdapterError("{}.{} must have shape [N,50,768].".format(split, text_field))
        n = int(text_array.shape[0])
        ids = _decode_ids(ids_value, n)
        mask, mask_source = _mask_for_split(split, ids, valid_masks, valid_lengths)
        labels_locked = split == "test" and not include_test_labels
        if not labels_locked:
            class_value, class_field = _resolve_field(
                record,
                schema.classification_label_key,
                aliases=("class_labels", "y_class", "annotations"),
            )
            reg_value, reg_field = _resolve_field(
                record,
                schema.regression_label_key,
                aliases=("y_reg",),
            )
            class_vector = _as_vector(class_value, "{}.{}".format(split, class_field), n)
            reg_vector = _as_vector(reg_value, "{}.{}".format(split, reg_field), n)
            if not np.issubdtype(np.asarray(reg_vector).dtype, np.number):
                raise PickleAdapterError("{}.{} regression labels must be numeric.".format(split, reg_field))
            if not np.all(np.isfinite(reg_vector.astype(np.float64))):
                raise PickleAdapterError("{}.{} regression labels must be finite.".format(split, reg_field))
            raw_class_vectors[split] = class_vector
            raw_regression[split] = reg_vector.astype(np.float32)
        else:
            class_field = reg_field = "LOCKED_UNREAD"
        raw_features[split] = {"T": text_array, "A": audio_array, "V": vision_array}
        ids_by_split[split] = ids
        split_masks[split] = mask
        split_mask_sources[split] = mask_source
        selected_fields[split + ".text"] = str(text_field)
        selected_fields[split + ".audio"] = str(audio_field)
        selected_fields[split + ".vision"] = str(vision_field)
        selected_fields[split + ".sample_ids"] = str(id_field)
        selected_fields[split + ".y_class"] = str(class_field)
        selected_fields[split + ".y_reg"] = str(reg_field)

    mapped_classes, mapping_audit = _class_mapping_for_all(raw_class_vectors, class_mapping)
    aligned_splits: Dict[str, AlignedSplit] = {}
    split_audit: Dict[str, SplitAudit] = {}
    for split in REQUIRED_SPLITS:
        y_class = _map_class_vector(raw_class_vectors[split], mapped_classes, split) if split in raw_class_vectors else None
        try:
            aligned = AlignedSplit.from_arrays(
                split=split,
                sample_ids=ids_by_split[split],
                valid_mask=split_masks[split],
                features=raw_features[split],
                y_class=y_class,
                y_reg=raw_regression.get(split),
            )
        except DataContractError as exc:
            raise PickleAdapterError("{} split violates aligned50 contract: {}".format(split, exc)) from exc
        aligned_splits[split] = aligned
        id_digest = hashlib.sha256("\n".join(ids_by_split[split]).encode("utf-8")).hexdigest()
        split_audit[split] = SplitAudit(
            split=split,
            sample_count=len(ids_by_split[split]),
            sample_ids_sha256=id_digest,
            feature_shapes={name: tuple(raw_features[split][name].shape) for name in ("T", "A", "V")},
            feature_dtypes={name: str(raw_features[split][name].dtype) for name in ("T", "A", "V")},
            class_annotation_dtype=str(raw_class_vectors[split].dtype) if split in raw_class_vectors else "locked_unread",
            regression_label_shape=tuple(raw_regression[split].shape) if split in raw_regression else (),
            valid_mask_source=split_mask_sources[split],
        )

    audit = PickleAudit(
        source_path=str(source),
        file_size_bytes=file_size,
        file_sha256=file_hash,
        splits=split_audit,
        class_mapping=mapping_audit,
        selected_fields=selected_fields,
    )
    return AlignedPickle(splits=aligned_splits, audit=audit)
