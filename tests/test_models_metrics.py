import unittest

import numpy as np
import torch

from multimodal_emotion.metrics import classification_metrics, regression_metrics
from multimodal_emotion.models import build_model


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.dims = {"text": 8, "audio": 4, "vision": 3}
        self.inputs = [torch.randn(2, 5, d) for d in (8, 4, 3)]
        self.valid = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]], dtype=torch.bool)
        self.observed = torch.ones((2, 3, 5), dtype=torch.bool)
        self.observed[0, :, 2] = False
        self.observed[1, :, 4] = False

    def test_all_models_forward_and_parameter_budget(self):
        for model_id in ("B0", "B1", "B2", "B3", "B4", "B5", "M0", "M1"):
            model = build_model(model_id, input_dims=self.dims)
            model.eval()
            with torch.no_grad():
                output = model(*self.inputs, self.valid, self.observed)
            self.assertTrue(torch.isfinite(output["logits"]).all())
            self.assertTrue(torch.isfinite(output["regression"]).all())
            self.assertEqual(tuple(output["logits"].shape), (2, 3))
            self.assertEqual(tuple(output["regression"].shape), (2,))
            self.assertLess(sum(p.numel() for p in model.parameters()), 1_200_000)

    def test_unobserved_values_cannot_change_prediction(self):
        for model_id in ("B3", "B5", "M1"):
            model = build_model(model_id, input_dims=self.dims).eval()
            altered = [x.clone() for x in self.inputs]
            hidden = ~self.observed
            for modality in range(3):
                altered[modality][hidden[:, modality]] = 1e6
            with torch.no_grad():
                first = model(*self.inputs, self.valid, self.observed)
                second = model(*altered, self.valid, self.observed)
            torch.testing.assert_close(first["logits"], second["logits"], rtol=0, atol=1e-6)
            torch.testing.assert_close(first["regression"], second["regression"], rtol=0, atol=1e-6)

    def test_all_missing_and_all_invalid_are_finite_and_distinct(self):
        model = build_model("M1", input_dims=self.dims)
        model.set_prior([0.2, 0.3, 0.5], -1.25)
        all_missing = torch.zeros_like(self.observed)
        with torch.no_grad():
            output = model(*self.inputs, self.valid, all_missing)
            invalid = model(*self.inputs, torch.zeros_like(self.valid), all_missing)
        self.assertTrue(output["low_information"].all())
        self.assertFalse(output["invalid_sample"].any())
        self.assertTrue(torch.isfinite(output["logits"]).all())
        self.assertTrue(invalid["invalid_sample"].all())
        torch.testing.assert_close(torch.softmax(invalid["logits"], dim=-1), torch.tensor([[0.2, 0.3, 0.5]]).expand(2, -1), atol=1e-6, rtol=0)
        torch.testing.assert_close(invalid["regression"], torch.full((2,), -1.25), atol=1e-6, rtol=0)

    def test_temporal_gate_weights_obey_observation_mask(self):
        for model_id in ("B5", "M0"):
            model = build_model(model_id, input_dims=self.dims).eval()
            with torch.no_grad():
                out = model(*self.inputs, self.valid, self.observed)
            gates = out["gate_weights"]
            self.assertTrue(torch.isfinite(gates).all())
            self.assertTrue(torch.all(gates[~self.observed] == 0))
            sums = gates.sum(dim=1)
            has = self.observed.any(dim=1)
            torch.testing.assert_close(sums[has], torch.ones_like(sums[has]), atol=1e-6, rtol=0)
            self.assertTrue(torch.all(sums[~has] == 0))


class MetricTests(unittest.TestCase):
    def test_three_class_metrics_include_absent_class_with_zero_division(self):
        metrics = classification_metrics(np.array([0, 1, 1]), np.array([0, 0, 1]))
        self.assertEqual(metrics["confusion_matrix"], [[1, 0, 0], [1, 1, 0], [0, 0, 0]])
        self.assertEqual(metrics["per_class"]["Positive"]["f1"], 0.0)
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)

    def test_undefined_pearson_is_null_with_reason(self):
        metrics = regression_metrics(np.ones(3), np.array([0.0, 1.0, 2.0]))
        self.assertIsNone(metrics["pearson"])
        self.assertEqual(metrics["pearson_undefined_reason"], "constant target")


if __name__ == "__main__":
    unittest.main()
