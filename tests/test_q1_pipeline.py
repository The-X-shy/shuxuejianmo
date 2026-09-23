import tempfile
import unittest
from pathlib import Path

import numpy as np

from multimodal_emotion.q1_pipeline import Q1Extractors, Q1Input, process_q1_inputs, save_q1_outputs


class Q1PipelineTests(unittest.TestCase):
    def test_synthetic_callbacks_keep_all_samples_and_emit_expected_contract(self):
        def probe(_path):
            return {"duration_seconds": 2.0, "time_origin": "first_presented_frame", "source_sha256": "synthetic"}

        def align(_path, _text, _metadata):
            return [{"text": "hello", "start": 0.1, "end": 0.3, "embedding": np.ones(768), "mapping_source": "synthetic_alignment"}]

        def audio(_path, _metadata):
            return {"clock": "video_seconds", "features": np.ones((1, 25)), "timestamps": np.array([0.2]), "sample_rate": 16000}

        def vision(_path, _metadata):
            return {"clock": "video_seconds", "features": np.ones((1, 29)), "timestamps": np.array([0.2]), "success": np.array([1]), "confidence": np.array([0.95]), "speaker_ambiguous": False}

        extractors = Q1Extractors(
            probe=probe, align_and_embed_text=align,
            extract_audio_lld=audio, extract_vision=vision,
            versions={"aligner": "synthetic", "audio": "synthetic", "vision": "synthetic"},
        )
        inputs = [Q1Input(f"video{i}$_$clip", f"/video/{i}.mp4", "hello") for i in range(100)]
        results = process_q1_inputs(inputs, extractors)
        self.assertEqual(len(results), 100)
        self.assertTrue(all(row["status"] == "success" for row in results))
        self.assertEqual(results[0]["features"]["text"].shape, (50, 768))
        self.assertEqual(results[0]["alignment"][5]["audio_frame_indices"], [0])
        with tempfile.TemporaryDirectory() as temp:
            save_q1_outputs(results, temp, tool_versions=extractors.versions)
            bundle = np.load(Path(temp) / "q1_features.npz")
            self.assertEqual(bundle["text"].shape, (100, 50, 768))
            self.assertEqual(bundle["audio"].shape, (100, 50, 25))
            self.assertEqual(bundle["vision"].shape, (100, 50, 29))
            self.assertEqual(len((Path(temp) / "alignment_q1.jsonl").read_text().splitlines()), 5000)

    def test_extraction_failure_is_not_reported_as_success(self):
        def fail(*_args):
            raise RuntimeError("tool missing")

        extractors = Q1Extractors(fail, fail, fail, fail, {})
        row = Q1Input("x$_$y", "/missing/video.mp4", "text")
        result = process_q1_inputs([row], extractors, expected_samples=1)[0]
        self.assertEqual(result["status"], "failed")
        self.assertIn("tool missing", result["failure_reason"])
        self.assertNotIn("features", result)


if __name__ == "__main__":
    unittest.main()
