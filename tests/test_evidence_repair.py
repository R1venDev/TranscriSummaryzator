import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("evidence_repair", ROOT / "scripts" / "evidence_repair.py")
repair = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(repair)


class EvidenceRepairTests(unittest.TestCase):
    def fact(self):
        return {"fact_id": "F1", "risk_level": "CRITICAL", "semantic_risks": ["quantity"], "uncertainty": {}, "evidence": [{"id": "U1", "start": 10, "end": 12, "text": "порог 0.5", "speaker": "A"}]}

    def test_planner_adds_asymmetric_audio_context(self):
        request = repair.repair_requests([self.fact()])[0]
        self.assertEqual((request["start"], request["end"]), (8.0, 16.0))

    def test_number_disagreement_fails_closed(self):
        facts, report = repair.reconcile_repairs([self.fact()], [{"evidence_id": "U1", "original_text": "порог 0.5", "text": "порог 0.7", "audio_clip_sha256": "x", "model": "gigaam"}])
        self.assertEqual(report["disagreed"], 1)
        self.assertTrue(facts[0]["uncertainty"]["needs_review"])
        self.assertIn("repair_disagreement", facts[0]["semantic_risks"])

    def test_long_audio_window_is_split_below_gigaam_limit(self):
        ranges = repair.audio_chunk_ranges(90 * 16000, 16000)
        self.assertEqual(ranges[0][0], 0)
        self.assertEqual(ranges[-1][1], 90 * 16000)
        self.assertTrue(all(right - left <= 20 * 16000 for left, right in ranges))
        self.assertTrue(all(ranges[index][1] == ranges[index + 1][0] for index in range(len(ranges) - 1)))


if __name__ == "__main__":
    unittest.main()
