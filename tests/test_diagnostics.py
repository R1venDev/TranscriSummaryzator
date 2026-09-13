import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("diagnostics", ROOT / "scripts" / "diagnostics.py")
diagnostics = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(diagnostics)


class DiagnosticsTests(unittest.TestCase):
    def test_decision_records_metrics_thresholds_and_redacts_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostics.jsonl"
            diagnostics.configure(path, component="test", run_id="r1", job_id=7)
            diagnostics.decision(
                "speaker_match", "accepted", candidates=["A", "B"],
                metrics={"best": .72, "margin": .15, "api_token": "hidden"},
                thresholds={"score": .62, "margin": .12}, reasons=["both_thresholds_passed"],
                refs={"word_ids": ["W1"]},
            )
            item = json.loads(path.read_text())
            self.assertEqual(item["metrics"]["best"], .72)
            self.assertEqual(item["thresholds"]["score"], .62)
            self.assertEqual(item["metrics"]["api_token"], "[REDACTED]")

    def test_publish_builds_machine_readable_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            diagnostics.configure(source, component="test", run_id="r2")
            diagnostics.event("stage", category="stage", outcome="completed")
            summary = diagnostics.publish(source, root / "output")
            self.assertEqual(summary["events"], 1)
            self.assertTrue((root / "output" / "diagnostics.jsonl").is_file())
            self.assertTrue((root / "output" / "diagnostics_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
