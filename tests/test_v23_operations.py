import json
import pathlib
import tempfile
import unittest

import pipeline
from scripts import diagnostics
from scripts.summary_worker import ensure_closing_schedule_question


class GlobalStageCacheTests(unittest.TestCase):
    def test_same_stage_key_restores_artifact_across_jobs(self):
        with tempfile.TemporaryDirectory() as root:
            root = pathlib.Path(root)
            old = pipeline.GLOBAL_STAGE_CACHE
            pipeline.GLOBAL_STAGE_CACHE = root / "global"
            try:
                first, second = root / "job-a", root / "job-b"
                first.mkdir(); second.mkdir()
                (first / "audio.wav").write_bytes(b"immutable-audio")
                pipeline.mark_stage_cached(first, "audio", "abcdef", ["audio.wav"])
                self.assertTrue(pipeline.stage_cache_valid(second, "audio", "abcdef", ["audio.wav"]))
                self.assertEqual((second / "audio.wav").read_bytes(), b"immutable-audio")
                meta = json.loads((second / "stage_cache.json").read_text())
                self.assertEqual(meta["audio"]["source"], "global_content_addressed")
            finally:
                pipeline.GLOBAL_STAGE_CACHE = old


class DiagnosticAggregationTests(unittest.TestCase):
    def test_per_item_decisions_go_to_trace_and_summary_has_operations(self):
        with tempfile.TemporaryDirectory() as root:
            root = pathlib.Path(root)
            main, trace = root / "diagnostics.jsonl", root / "diagnostics.trace.jsonl"
            diagnostics.configure(main, trace_path=trace, component="test", run_id="r")
            diagnostics.decision("claim_lifecycle", "active", refs={"claim_id": "C1"})
            diagnostics.event("llm_request", category="stage", outcome="completed", metrics={"prompt_tokens": 10, "output_tokens": 2}, duration_ms=120)
            summary = diagnostics.summarize(main)
            self.assertEqual(summary["events"], 1)
            self.assertEqual(summary["calls"], 1)
            self.assertEqual(summary["tokens"]["prompt_tokens"], 10)
            self.assertTrue(trace.is_file())


class ClosingScheduleRecoveryTests(unittest.TestCase):
    def test_conflicting_end_of_meeting_times_leave_exact_time_open(self):
        utterances = [
            {"id": "U1", "start": 100, "end": 101, "speaker": "@A", "text": "Когда вам будет удобнее провести созвончик?", "source_word_ids": ["W1"]},
            {"id": "U2", "start": 102, "end": 103, "speaker": "@B", "text": "Давайте во вторник", "source_word_ids": ["W2"]},
            {"id": "U3", "start": 104, "end": 105, "speaker": "@A", "text": "В 19:00 или 20:00?", "source_word_ids": ["W3"]},
        ]
        records = ensure_closing_schedule_question([], utterances)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["question_status"], "partially_answered")
        self.assertEqual(records[0]["requested_slots"], ["day", "exact_time"])
        self.assertIn("W3", records[0]["source_word_ids"])


if __name__ == "__main__":
    unittest.main()
