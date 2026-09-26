"""Production summary entry with two fake providers and an artificial source."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from summary.gemini_v1.batch import Reply as GeminiReply
from summary.luna_v1.engine import _cost_micros, poll_once, submit
from summary.luna_v1.ledger import Ledger
from tests.test_luna_engine import FakeClient, arm_poll


class TwoRoleStore:
    def dispatch_candidates(self, role="writer"):
        if role == "judge":
            return [{"id": "judge-key", "version": 1, "role": "judge",
                     "workspace_id": "judge-workspace"}]
        return [{"id": "writer-key", "version": 1, "role": "writer",
                 "workspace_id": "writer-workspace"}]

    def reveal_for_dispatch(self, identifier, version, role="writer"):
        assert version == 1 and identifier == ("judge-key" if role == "judge" else "writer-key")
        return "synthetic-secret-never-sent" if role == "writer" else "synthetic-openrouter-judge"

    def reveal_for_existing_job(self, identifier, version):
        assert version == 1
        return "synthetic-openrouter-judge" if identifier == "judge-key" else "synthetic-secret-never-sent"


class FakeGemini:
    submissions = {}
    submit_calls = 0
    mode = "patch"

    def __init__(self, token):
        assert token == "synthetic-openrouter-judge"

    def submit(self, custom_id, request_body):
        self.__class__.submit_calls += 1
        name = f"batch_synthetic{self.__class__.submit_calls}"
        self.__class__.submissions[name] = (custom_id, request_body)
        return GeminiReply(202, {"id": name, "status": "validating",
                                 "endpoint": "/v1/chat/completions",
                                 "model": "google/gemini-3.7-flash:batch"})

    def get(self, name):
        custom_id, request = self.__class__.submissions[name]
        payload = json.loads(request["messages"][1]["content"])
        windows = payload["SOURCE_WINDOWS"]
        report = {
            "schema_version": "luna_summary_audit_v1",
            "coverage": [{"window_id": window["window_id"],
                          "start_id": window["start_id"], "end_id": window["end_id"],
                          "salient": "Обсуждены проверка и запись результата.",
                          "draft_coverage": "covered", "finding_indices": []}
                         for window in windows],
            "findings": [], "patches": [],
        }
        if payload["MODE"] == "audit" and self.__class__.mode == "patch":
            report["coverage"][1]["draft_coverage"] = "missing"
            report["coverage"][1]["finding_indices"] = [0]
            task = {
                "title": "Записать результат проверки", "description": "После проверки записать результат в журнал.",
                "discussion_status": "proposed", "assignee": None, "due": None,
                "priority": None, "recipient": None, "source_ids": ["U00002"],
                "field_sources": {"action": ["U00002"], "assignee": [], "due": [],
                                  "priority": [], "recipient": [], "discussion_status": ["U00002"]},
            }
            report["findings"] = [{"severity": "major", "kind": "omission",
                                  "description": "Пропущено предложение записать результат.",
                                  "source_ids": ["U00002"], "affected": [],
                                  "status": "repaired", "patch_indices": [0]}]
            report["patches"] = [{"section": "tasks", "operation": "insert",
                                  "index": 1, "item_json": json.dumps(task, ensure_ascii=False)}]
        response = {"model": "google/gemini-3.7-flash:batch", "choices": [{
            "message": {"role": "assistant", "content": json.dumps(report, ensure_ascii=False)},
            "finish_reason": "stop"}], "usage": {"prompt_tokens": 400,
                "completion_tokens": 350, "total_tokens": 750}}
        return GeminiReply(200, {"id": name, "status": "completed",
            "endpoint": "/v1/chat/completions",
            "model": "google/gemini-3.7-flash:batch",
            "request_counts": {"total": 1, "completed": 1, "failed": 0},
            "usage": {"cost": 0.000807, "prompt_tokens": 400,
                      "completion_tokens": 350, "total_tokens": 750},
            "results": [{"custom_id": custom_id, "response": {
                "status_code": 200, "body": response}, "error": None}]})

    def delete(self, name):
        return GeminiReply(204, {})


class GeminiEngineTests(unittest.TestCase):
    def test_byok_gateway_fee_never_releases_full_reserve(self):
        self.assertIsNone(_cost_micros({"cost": 0.000001, "is_byok": True}))
        self.assertEqual(_cost_micros({"cost": 0.000807, "is_byok": False}), 807)

    def test_writer_then_openrouter_gemini_audit_patch_verify_and_zero_call_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "meeting"
            output.mkdir()
            source = output / "transcript.json"
            source.write_text(json.dumps({"source": "Синтетика.mkv", "duration_seconds": 12,
                "speakers": {"p1": "А"}, "utterances": [
                    {"start": 0.2, "end": 5.0, "speaker": "p1", "text": "Предлагаю проверить X или Y, не оба."},
                    {"start": 5.2, "end": 11.0, "speaker": "p1", "text": "Ещё предлагаю записать результат в журнал."},
                ]}, ensure_ascii=False))
            private = root / "private"
            writer_route = SimpleNamespace(
                reserve_microusd=lambda payload, **kwargs: 20_000,
                workspace_id="writer-workspace", prompt_usd_per_token="0.00000005",
                completion_usd_per_token="0.00000025",
                cache_write_usd_per_token="0.0000000625", request_usd="0")
            judge_route = SimpleNamespace(reserve_microusd=lambda: 25_000,
                                          input_tokens=1500, workspace_id="judge-workspace")
            FakeClient.submit_calls = 0
            FakeClient.submissions = {}
            FakeClient.report_factory = None
            FakeGemini.submit_calls = 0
            FakeGemini.submissions = {}
            FakeGemini.mode = "patch"
            with patch("summary.luna_v1.engine._credential_store", return_value=TwoRoleStore()), \
                 patch("summary.luna_v1.engine.verify_batch_route", return_value=writer_route), \
                 patch("summary.luna_v1.engine.verify_gemini_batch_route", return_value=judge_route):
                started = submit(transcript_path=source, output_dir=output,
                                 private_root=private, client_factory=FakeClient)
                self.assertEqual(started["status"], "submitted")
                outcomes = []
                for _ in range(3):
                    arm_poll(private)
                    outcomes.append(poll_once(private_root=private, client_factory=FakeClient,
                                              gemini_client_factory=FakeGemini))
                again = submit(transcript_path=source, output_dir=output,
                               private_root=private, client_factory=FakeClient)
            self.assertEqual(FakeClient.submit_calls, 1)
            self.assertEqual(FakeGemini.submit_calls, 2, outcomes)
            self.assertEqual(again["status"], "accepted_cache_hit")
            ledger = Ledger(private)
            root_job = ledger.get(started["job_id"])
            audit = ledger.stage(root_job["id"], "audit")
            verify = ledger.stage(root_job["id"], "verify")
            self.assertEqual(root_job["status"], "accepted")
            self.assertEqual((audit["workspace_id"], verify["workspace_id"]),
                             ("judge-workspace", "judge-workspace"))
            self.assertEqual((audit["credential_id"], verify["credential_id"]),
                             ("judge-key", "judge-key"))
            self.assertEqual(audit["billed_microusd"], 807)
            ledger.close()
            pointer = json.loads((output / "summary_current.json").read_text())
            generation = output / "summary_generations" / pointer["generation_id"]
            tasks = json.loads((generation / "tasks.json").read_text())
            self.assertEqual(len(tasks), 2)
            self.assertEqual(tasks[1]["assignee"], None)
            review = json.loads((generation / "run_manifest.json").read_text())["quality_review"]
            self.assertEqual(review["status"], "checked")
            self.assertEqual(review["audit_job_id"], audit["id"])
            request = FakeGemini.submissions[next(iter(FakeGemini.submissions))][1]
            self.assertEqual(request["response_format"]["type"], "json_schema")
            self.assertEqual(request["plugins"], [])


if __name__ == "__main__":
    unittest.main()
