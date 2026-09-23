import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from multimodal_emotion.pickle_adapter import PickleAdapterError, load_aligned_pickle


def tiny_payload(class_values=("Negative", "Positive")):
    payload = {}
    for split in ("train", "valid", "test"):
        n = 2
        payload[split] = {
            "id": np.array([f"{split}$_$a", f"{split}$_$b"]),
            "text": np.ones((n, 50, 768), dtype=np.float32),
            "audio": np.ones((n, 50, 74), dtype=np.float32),
            "vision": np.ones((n, 50, 35), dtype=np.float32),
            "classification_labels": np.asarray(class_values),
            "regression_labels": np.array([-1.0, 1.0], dtype=np.float32),
        }
    return payload


class PickleAdapterTests(unittest.TestCase):
    def test_explicit_trust_and_verified_masks_are_required(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "features.pkl"
            with path.open("wb") as f:
                pickle.dump(tiny_payload(), f)
            with self.assertRaises(PickleAdapterError):
                load_aligned_pickle(path)
            with self.assertRaises(PickleAdapterError):
                load_aligned_pickle(path, trusted_source=True)

    def test_test_labels_are_locked_by_default_and_ids_keep_order(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "features.pkl"
            with path.open("wb") as f:
                pickle.dump(tiny_payload(), f)
            lengths = {split: np.array([50, 37]) for split in ("train", "valid", "test")}
            loaded = load_aligned_pickle(path, trusted_source=True, valid_lengths=lengths)
            self.assertEqual(loaded.train.sample_ids, ("train$_$a", "train$_$b"))
            self.assertIsNotNone(loaded.valid.y_class)
            self.assertIsNone(loaded.test.y_class)
            self.assertIsNone(loaded.test.y_reg)
            self.assertEqual(loaded.audit.splits["test"].class_annotation_dtype, "locked_unread")
            self.assertEqual(loaded.audit.splits["train"].sample_count, 2)

    def test_integer_classes_need_explicit_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "features.pkl"
            with path.open("wb") as f:
                pickle.dump(tiny_payload((0, 2)), f)
            masks = {split: np.ones((2, 50), dtype=bool) for split in ("train", "valid", "test")}
            with self.assertRaises(PickleAdapterError):
                load_aligned_pickle(path, trusted_source=True, valid_masks=masks)
            loaded = load_aligned_pickle(
                path, trusted_source=True, valid_masks=masks,
                class_mapping={0: 0, 1: 1, 2: 2},
            )
            np.testing.assert_array_equal(loaded.train.y_class, [0, 2])


if __name__ == "__main__":
    unittest.main()
