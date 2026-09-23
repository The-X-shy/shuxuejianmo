import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from multimodal_emotion.config import DEFAULT_CONFIG, load_config
from multimodal_emotion.export import validate_prediction_rows, write_prediction_csv
from multimodal_emotion.data import SampleBatch
from multimodal_emotion.early_stopping import EarlyStopping, early_stopping_score
from multimodal_emotion.corruption import generate_validation_scenarios
from multimodal_emotion.evaluation import aggregate_core_evaluations, evaluate_early_stopping, evaluate_fixed_grid
from multimodal_emotion.explanation import explain_sample
from multimodal_emotion.models import build_model
from multimodal_emotion.q1_alignment import aggregate_frame_features, aggregate_word_embeddings
from multimodal_emotion.queue import validate_run_queue
from multimodal_emotion.selection import select_candidate


ROOT = Path(__file__).resolve().parents[1]


class ContractTests(unittest.TestCase):
    def test_default_config_and_queue(self):
        config = load_config(DEFAULT_CONFIG)
        rows = validate_run_queue(ROOT / "configs" / "core_experiment_queue.csv", config)
        self.assertEqual(len(rows), 24)

    def test_q1_word_overlap_and_frame_bins(self):
        words = [
            {"text": "hello", "start": 0.0, "end": 1.0, "embedding": np.ones(4), "mapping_source": "synthetic"},
            {"text": "unknown", "start": None, "end": None, "embedding": np.ones(4)},
        ]
        text, alignment = aggregate_word_embeddings(words, 2.0, bins=2, dimension=4)
        np.testing.assert_allclose(text[0], 1.0)
        np.testing.assert_allclose(text[1], 0.0)
        self.assertEqual(alignment[0]["text_status"], "aligned_text")
        self.assertEqual(alignment[1]["text_status"], "alignment_unavailable")
        frame_features, counts = aggregate_frame_features(np.array([[1.0], [3.0]]), np.array([0.2, 1.2]), 2.0, bins=2)
        np.testing.assert_allclose(frame_features[:, 0], [1.0, 3.0])
        np.testing.assert_array_equal(counts, [1, 1])

    def test_export_preserves_ids_and_rejects_invalid_rows(self):
        rows = [{
            "sample_id": "a", "polarity": "Neutral", "intensity": 0.0,
            "p_negative": 0.2, "p_neutral": 0.6, "p_positive": 0.2,
            "quality_flag": "ok", "feature_version": "official_aligned50", "model_version": "M1",
        }]
        validate_prediction_rows(rows, ["a"])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pred.csv"
            write_prediction_csv(path, rows, expected_ids=["a"])
            self.assertTrue(path.read_text(encoding="utf-8").startswith("sample_id,polarity,intensity"))
        bad = [dict(rows[0], p_negative=0.5)]
        with self.assertRaises(ValueError):
            validate_prediction_rows(bad)

    def test_selection_never_claims_special_test_as_source(self):
        config = load_config(DEFAULT_CONFIG)
        results = {
            "B3": {"R": 0.8, "clean_mae": 0.8, "clean_macro_f1": 0.5, "main_macro_f1": 0.45, "parameter_count": 146000},
            "B4": {"R": 0.7, "clean_mae": 0.82, "clean_macro_f1": 0.49, "main_macro_f1": 0.48, "parameter_count": 146000},
            "B5": {"R": 0.72, "clean_mae": 0.81, "clean_macro_f1": 0.49, "main_macro_f1": 0.47, "parameter_count": 396000},
            "M0": {"R": 0.75, "clean_mae": 0.82, "clean_macro_f1": 0.49, "main_macro_f1": 0.46, "parameter_count": 400000},
            "M1": {"R": 0.71, "clean_mae": 0.83, "clean_macro_f1": 0.49, "main_macro_f1": 0.47, "parameter_count": 400000},
        }
        choice = select_candidate(results, config)
        self.assertEqual(choice["chosen_model"], "B4")
        self.assertFalse(choice["special_test_used"])

    def test_three_seed_aggregation_uses_only_the_fixed_validation_grid(self):
        config = load_config(DEFAULT_CONFIG)
        scenarios = generate_validation_scenarios()
        views = [{"scenario_id": "clean", "group": "clean", "metrics": {
            "accuracy": 0.6, "macro_f1": 0.5, "weighted_f1": 0.6, "mae": 0.5, "pearson": 0.2,
        }}]
        views.extend({"scenario_id": item.scenario_id, "group": item.group, "metrics": {
            "accuracy": 0.6, "macro_f1": 0.5, "weighted_f1": 0.6, "mae": 0.5, "pearson": 0.2,
        }} for item in scenarios)
        run_results = {
            (model_id, seed): views
            for model_id in config["training"]["model_execution_order"]
            for seed in config["training"]["seed_execution_order"]
        }
        parameter_counts = {model_id: 1000 + index for index, model_id in enumerate(config["training"]["model_execution_order"])}
        summary = aggregate_core_evaluations(run_results, config, parameter_counts)
        self.assertEqual(summary["source"], "valid_only")
        self.assertEqual(len(summary["condition_summaries"]), 8 * 115)
        self.assertFalse(summary["selection"]["special_test_used"])

    def test_explanation_keeps_prediction_fixed_and_marks_unknown_time(self):
        rng = np.random.default_rng(7)
        p = np.zeros((1, 50), dtype=bool)
        p[0, :4] = True
        o0 = np.zeros((1, 3, 50), dtype=bool)
        o0[0, :, :4] = True
        batch = SampleBatch(
            split="valid", sample_ids=("toy",),
            text=rng.normal(size=(1, 50, 8)).astype(np.float32),
            audio=rng.normal(size=(1, 50, 4)).astype(np.float32),
            vision=rng.normal(size=(1, 50, 3)).astype(np.float32),
            valid_mask=p, original_observed_mask=o0,
            corruption_mask=np.zeros_like(o0), observed_mask=o0,
        )
        model = build_model("B3", input_dims={"text": 8, "audio": 4, "vision": 3}).eval()
        before = model(*[
            torch.as_tensor(batch.text), torch.as_tensor(batch.audio), torch.as_tensor(batch.vision),
            torch.as_tensor(batch.valid_mask), torch.as_tensor(batch.observed_mask),
        ])
        result = explain_sample(model, batch, 0, run_random_controls=False)
        after = model(*[
            torch.as_tensor(batch.text), torch.as_tensor(batch.audio), torch.as_tensor(batch.vision),
            torch.as_tensor(batch.valid_mask), torch.as_tensor(batch.observed_mask),
        ])
        torch.testing.assert_close(before["logits"], after["logits"], atol=0, rtol=0)
        self.assertEqual(result["mapping_status"], "unresolved")
        self.assertEqual(result["sample_id"], "toy")
        self.assertEqual(len(result["point_curves"]["T"]), 4)

    def test_early_stopping_grid_formula_and_first_finite_checkpoint(self):
        middle = {name: 1.0 for name in ("T", "A", "V", "TA", "TV", "AV")}
        self.assertAlmostEqual(early_stopping_score(2.0, middle), 1.5)
        model = build_model("B0", input_dims={"text": 8, "audio": 4, "vision": 3})
        stopper = EarlyStopping(patience=2, min_delta=0.01)
        self.assertFalse(stopper.update(1, 1.0, model))
        self.assertFalse(stopper.update(2, 0.995, model))
        self.assertEqual(stopper.best_epoch, 1)  # improvement below min_delta is ignored

    def test_early_stop_and_full_grid_are_115_prediction_views(self):
        rng = np.random.default_rng(19)
        p = np.zeros((1, 50), dtype=bool)
        p[0, :5] = True
        o0 = np.zeros((1, 3, 50), dtype=bool)
        o0[0, :, :5] = True
        batch = SampleBatch(
            split="valid", sample_ids=("eval-toy",),
            text=rng.normal(size=(1, 50, 8)).astype(np.float32),
            audio=rng.normal(size=(1, 50, 4)).astype(np.float32),
            vision=rng.normal(size=(1, 50, 3)).astype(np.float32),
            valid_mask=p, original_observed_mask=o0,
            corruption_mask=np.zeros_like(o0), observed_mask=o0,
            y_class=np.array([1]), y_reg=np.array([0.0], dtype=np.float32),
        )
        model = build_model("B3", input_dims={"text": 8, "audio": 4, "vision": 3}).eval()
        clean_mae, early = evaluate_early_stopping(model, batch)
        self.assertTrue(np.isfinite(clean_mae))
        self.assertEqual(set(early), {"T", "A", "V", "TA", "TV", "AV"})
        grid = evaluate_fixed_grid(model, batch)
        self.assertEqual(len(grid), 115)
        self.assertEqual(grid[0]["scenario_id"], "clean")
        self.assertEqual(sum(row["group"] == "main" for row in grid), 90)
        self.assertEqual(sum(row["group"] == "tav" for row in grid), 15)
        self.assertEqual(sum(row["group"] == "async" for row in grid), 9)


if __name__ == "__main__":
    unittest.main()
