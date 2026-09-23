import unittest

import numpy as np

from multimodal_emotion.corruption import (
    apply_scenario,
    generate_validation_scenarios,
    make_training_corruption,
    validation_earlystop_scenarios,
)
from multimodal_emotion.data import AlignedSplit, DataContractError, fit_normalizer


def make_split(split="train", n=3, feature_value=1.0):
    rng = np.random.default_rng(13)
    t = np.full((n, 50, 768), feature_value, dtype=np.float32)
    a = rng.normal(size=(n, 50, 74)).astype(np.float32)
    v = rng.normal(size=(n, 50, 35)).astype(np.float32)
    p = np.ones((n, 50), dtype=bool)
    p[:, 40:] = False
    t[:, 40:] = 0
    a[:, 40:] = 0
    v[:, 40:] = 0
    t[0, 3] = 0  # natural missing row
    a[0, 4, 0] = np.nan  # invalid row must be logged and masked
    return AlignedSplit.from_arrays(
        split=split,
        sample_ids=[f"s{i}" for i in range(n)],
        valid_mask=p,
        text=t,
        audio=a,
        vision=v,
        y_class=np.arange(n, dtype=np.int64) % 3,
        y_reg=np.linspace(-1, 1, n, dtype=np.float32),
    )


class DataContractTests(unittest.TestCase):
    def test_p_o0_nonfinite_and_zero_rows_are_distinct(self):
        split = make_split()
        self.assertEqual(split.P.shape, (3, 50))
        self.assertFalse(split.O0[0, 0, 3])
        self.assertFalse(split.O0[0, 1, 4])
        self.assertFalse(split.P[0, 40])
        self.assertEqual(len(split.invalid_value_events), 1)
        self.assertTrue(np.isfinite(split.features["A"]).all())

    def test_train_only_normalization_and_safe_corruption(self):
        train = make_split("train", feature_value=2.0)
        normalizer = fit_normalizer(train)
        valid = make_split("valid", feature_value=200.0)
        normalized_train = normalizer.transform(train)
        normalized_valid = normalizer.transform(valid)
        np.testing.assert_allclose(normalizer.mean["T"], 2.0)
        self.assertFalse(np.allclose(normalized_train.text[0, 0], normalized_valid.text[0, 0]))
        corruption = np.zeros((3, 3, 50), dtype=bool)
        corruption[0, 0, 1:3] = True
        damaged = normalized_train.with_corruption(corruption)
        self.assertTrue(np.all(damaged.text[0, 1:3] == 0))
        self.assertTrue(np.array_equal(damaged.valid_mask, normalized_train.valid_mask))
        self.assertTrue(np.array_equal(damaged.original_observed_mask, normalized_train.original_observed_mask))

    def test_bad_corruption_and_nontrain_normalizer_are_rejected(self):
        valid = make_split("valid")
        with self.assertRaises(DataContractError):
            fit_normalizer(valid)
        train = make_split()
        normalized = fit_normalizer(train).transform(train)
        illegal = np.zeros_like(normalized.corruption_mask)
        illegal[0, 0, 40] = True  # padding is never corruptible
        with self.assertRaises(DataContractError):
            normalized.with_corruption(illegal)

    def test_mask_scenarios_are_fixed_and_obey_p_and_o0(self):
        train = make_split()
        batch = fit_normalizer(train).transform(train)
        scenarios = generate_validation_scenarios()
        self.assertEqual(len(scenarios), 114)
        self.assertEqual(len(validation_earlystop_scenarios()), 6)
        for scenario in scenarios:
            damaged = apply_scenario(batch, scenario=scenario)
            self.assertTrue(np.all(damaged.corruption_mask <= batch.valid_mask[:, None, :]))
            self.assertTrue(np.all(damaged.corruption_mask <= batch.original_observed_mask))
            self.assertTrue(np.all(damaged.valid_mask == batch.valid_mask))

    def test_training_mask_is_keyed_by_seed_epoch_and_sample(self):
        train = make_split()
        p, o0, sid = train.valid_mask[0], train.original_observed_mask[0], train.sample_ids[0]
        first = make_training_corruption(p, o0, 42, 3, sid)
        repeat = make_training_corruption(p, o0, 42, 3, sid)
        np.testing.assert_array_equal(first.corruption_mask, repeat.corruption_mask)
        self.assertTrue(np.all(first.corruption_mask <= p[None, :]))
        self.assertTrue(np.all(first.corruption_mask <= o0))


if __name__ == "__main__":
    unittest.main()
