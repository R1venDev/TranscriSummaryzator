"""Real summary entry with fake Batch transport for the segmented route."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from summary.gemini_v1.batch import MODEL as GEMINI_MODEL, Reply as GeminiReply
from summary.luna_v1.engine import SEGMENT_REVIEW_QUALITY_POLICY_VERSION, poll_once, submit
from summary.luna_v1.ledger import Ledger
from tests.test_gemini_engine import TwoRoleStore
from tests.test_luna_engine import FakeClient, arm_poll


class SegmentGemini:
    submissions = {}
    submit_calls = 0

    def __init__(self, token):
        assert token == "synthetic-openrouter-judge"

    def submit(self, custom_id, request_body):
        self.__class__.submit_calls += 1
        identifier = f"batch_synthetic_segment_{self.__class__.submit_calls}"
        self.__class__.submissions[identifier] = (custom_id, request_body)
        return GeminiReply(202, {"id": identifier, "status": "validating",
                                 "endpoint": "/v1/chat/completions", "model": GEMINI_MODEL})

    def get(self, identifier):
        custom_id, request = self.__class__.submissions[identifier]
        payload = json.loads(request["messages"][1]["content"])
        if payload["MODE"] == "segment_review":
            segment = payload["SOURCE_SEGMENT"]
            first = segment["primary_utterances"][0]
            second = segment["segment_id"] == "S02"
            schema_id = request["response_format"]["json_schema"]["name"]
            item = {"kind": "action", "claim": first["text"],
                    "source_ids": [first["id"]], "actor": None,
                    "recipient": None, "modality": "proposed", "condition": None,
                    "alternatives": [], "correction_of": []}
            if schema_id == "gemini_segment_review_v1":
                item.update({"speaker": first["speaker"], "action": first["text"],
                             "uncertainty": None})
            findings, patches = [], []
            if second:
                task = {"title": "Записать результат проверки",
                        "description": "После проверки записать результат в журнал.",
                        "discussion_status": "proposed", "assignee": None,
                        "due": None, "priority": None, "recipient": None,
                        "source_ids": [first["id"]],
                        "field_sources": {"action": [first["id"]], "assignee": [],
                                          "due": [], "priority": [], "recipient": [],
                                          "discussion_status": [first["id"]]}}
                findings = [{"severity": "major", "kind": "omission",
                             "description": "Пропущена запись результата.",
                             "source_ids": [first["id"]], "affected": [],
                             "status": "repaired", "patch_indices": [0]}]
                patches = [{"section": "tasks", "operation": "insert", "index": 1,
                            "item_json": json.dumps(task, ensure_ascii=False)}]
            if schema_id == "gemini_segment_review_v3":
                item = {"k": item["kind"], "c": item["claim"], "s": item["source_ids"],
                        "m": item["modality"], "a": item["actor"], "r": item["recipient"],
                        "if": item["condition"], "or": item["alternatives"],
                        "fix": item["correction_of"],
                        "v": "missing" if second else "represented",
                        "t": [] if second else ["tasks:0"], "f": [0] if second else []}
            report = {"schema_version": schema_id,
                      "segment_id": segment["segment_id"],
                      "coverage": [{"window_id": row["window_id"],
                                    "start_id": row["start_id"], "end_id": row["end_id"],
                                    "assessment": "material_items", "item_indices": [0]}
                                   for row in segment["coverage_windows"]],
                      "items": [item],
                      "draft_assessments": [] if schema_id in {"gemini_segment_review_v2",
                                                          "gemini_segment_review_v3"} else
                          [{"unit_id": unit["unit_id"],
                                              "status": "supported",
                                              "source_ids": [first["id"]],
                                              "item_indices": [], "finding_indices": []}
                                             for unit in payload["DRAFT_UNITS_TO_CHECK"]],
                      "findings": findings, "patches": patches}
            if schema_id != "gemini_segment_review_v3":
                report["item_assessments"] = [{"item_index": 0,
                    "status": "missing" if second else "represented",
                    "draft_targets": [] if second else [{"section": "tasks", "index": 0}],
                    "finding_indices": [0] if second else []}]
        else:
            assert payload["MODE"] == "verify"
            inventory = payload["INDEPENDENT_SOURCE_INVENTORY"]
            report = {"schema_version": "gemini_inventory_reconcile_v1",
                      "source_window_assessments": [],
                      "inventory_assessments": [
                          {"item_id": item["item_id"], "status": "represented",
                           "draft_targets": [{"section": "tasks", "index":
                                              1 if item["source_ids"] == ["U00002"] else 0}],
                           "finding_indices": []} for item in inventory["items"]],
                      "draft_assessments": [
                          {"unit_id": unit["unit_id"], "status": "supported",
                           "source_ids": ["U00002" if unit["unit_id"].startswith("tasks:1")
                                          else "U00001"],
                           "inventory_ids": [inventory["items"][0]["item_id"]],
                           "finding_indices": []} for unit in payload["DRAFT_UNITS"]],
                      "findings": [], "patches": []}
        body = {"model": GEMINI_MODEL, "choices": [{"message": {
            "role": "assistant", "content": json.dumps(report, ensure_ascii=False)},
            "finish_reason": "stop"}], "usage": {"prompt_tokens": 400,
            "completion_tokens": 350, "total_tokens": 750}}
        return GeminiReply(200, {"id": identifier, "status": "completed",
                                "endpoint": "/v1/chat/completions", "model": GEMINI_MODEL,
                                "request_counts": {"total": 1, "completed": 1, "failed": 0},
                                "usage": {"cost": 0.000807, "is_byok": False,
                                          "prompt_tokens": 400, "completion_tokens": 350,
                                          "total_tokens": 750},
                                "results": [{"custom_id": custom_id, "response": {
                                    "status_code": 200, "body": body}, "error": None}]})

    def delete(self, identifier):
        return GeminiReply(204, {})


class SegmentEngineTests(unittest.TestCase):
    def setUp(self):
        # Historical Gemini flow runs only after restoring its logging-OFF policy.
        logging_policy = patch("summary.luna_v1.engine.OPUS_WORKSPACE_IO_LOGGING_ENABLED", False)
        logging_policy.start()
        self.addCleanup(logging_policy.stop)

    def test_serial_segments_patch_verify_and_zero_call_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "meeting"
            output.mkdir()
            source = output / "transcript.json"
            source.write_text(json.dumps({"source": "Синтетика.mkv",
                "duration_seconds": 12, "speakers": {"p1": "А"}, "utterances": [
                    {"start": 0.2, "end": 5.0, "speaker": "p1",
                     "text": "Предлагаю проверить X или Y, не оба."},
                    {"start": 5.2, "end": 11.0, "speaker": "p1",
                     "text": "Ещё предлагаю записать результат в журнал."},
                ]}, ensure_ascii=False))
            private = root / "private"
            writer_route = SimpleNamespace(reserve_microusd=lambda payload, **kwargs: 20_000,
                workspace_id="writer-workspace", prompt_usd_per_token="0.00000005",
                completion_usd_per_token="0.00000025",
                cache_write_usd_per_token="0.0000000625", request_usd="0")
            judge_route = SimpleNamespace(reserve_microusd=lambda: 15_000,
                input_tokens=1500, context_bound_tokens=2100,
                workspace_id="judge-workspace")
            FakeClient.submit_calls = 0
            FakeClient.submissions = {}
            FakeClient.report_factory = None
            SegmentGemini.submit_calls = 0
            SegmentGemini.submissions = {}
            with patch("summary.luna_v1.engine._credential_store", return_value=TwoRoleStore()), \
                 patch("summary.luna_v1.engine.verify_batch_route", return_value=writer_route), \
                 patch("summary.luna_v1.engine.verify_gemini_batch_route", return_value=judge_route):
                started = submit(transcript_path=source, output_dir=output,
                                 private_root=private, client_factory=FakeClient,
                                 quality_policy_version=SEGMENT_REVIEW_QUALITY_POLICY_VERSION)
                self.assertEqual(started["status"], "submitted")
                arm_poll(private)
                poll_once(private_root=private, client_factory=FakeClient,
                          gemini_client_factory=SegmentGemini)
                self.assertEqual(SegmentGemini.submit_calls, 1)
                for expected in (2, 3, 3):
                    arm_poll(private)
                    poll_once(private_root=private, client_factory=FakeClient,
                              gemini_client_factory=SegmentGemini)
                    if SegmentGemini.submit_calls != expected:
                        diagnostic = Ledger(private)
                        stages = [(row[0], row[1], row[2]) for row in diagnostic.db.execute(
                            "SELECT kind, status, error_code FROM jobs ORDER BY created_at")]
                        diagnostic.close()
                        self.fail(f"expected {expected} stage submissions, got {SegmentGemini.submit_calls}: {stages}")
                again = submit(transcript_path=source, output_dir=output,
                               private_root=private, client_factory=FakeClient,
                               quality_policy_version=SEGMENT_REVIEW_QUALITY_POLICY_VERSION)
            self.assertEqual(again["status"], "accepted_cache_hit")
            self.assertEqual(FakeClient.submit_calls, 1)
            ledger = Ledger(private)
            parent = ledger.get(started["job_id"])
            self.assertEqual(parent["status"], "accepted")
            self.assertEqual({row[0] for row in ledger.db.execute(
                "SELECT kind FROM jobs WHERE root_job_id=?", (parent["id"],))},
                {"summary", "segment_1", "segment_2", "verify"})
            ledger.close()
            pointer = json.loads((output / "summary_current.json").read_text())
            generation = output / "summary_generations" / pointer["generation_id"]
            tasks = json.loads((generation / "tasks.json").read_text())
            self.assertEqual(len(tasks), 2)
            self.assertIsNone(tasks[1]["assignee"])
            review = json.loads((generation / "run_manifest.json").read_text())["quality_review"]
            self.assertEqual(review["status"], "model_segment_reviewed_checked")
            self.assertEqual(len(review["segment_job_ids"]), 2)
            for _, request in list(SegmentGemini.submissions.values())[:2]:
                payload = json.loads(request["messages"][1]["content"])
                self.assertIn("DRAFT_DOCUMENT", payload)
                self.assertEqual(len(payload["SOURCE_SEGMENT"]["primary_utterances"]), 1)


if __name__ == "__main__":
    unittest.main()
