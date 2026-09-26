"""The actual summary entry with fake Batch transport and invented dialogue."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from summary.gemini_v1.batch import MODEL as GEMINI_MODEL, Reply as GeminiReply
from summary.luna_v1.engine import (INVENTORY_RECONCILE_QUALITY_POLICY_VERSION,
                                     _pin_stage_files, poll_once, submit)
from summary.luna_v1.ledger import Ledger
from tests.test_gemini_engine import TwoRoleStore
from tests.test_luna_engine import FakeClient, arm_poll


class InventoryGemini:
    submissions = {}
    submit_calls = 0

    def __init__(self, token):
        assert token == "synthetic-openrouter-judge"

    def submit(self, custom_id, request_body):
        self.__class__.submit_calls += 1
        identifier = f"batch_synthetic_inventory_{self.__class__.submit_calls}"
        self.__class__.submissions[identifier] = (custom_id, request_body)
        return GeminiReply(202, {"id": identifier, "status": "validating",
                                 "endpoint": "/v1/chat/completions", "model": GEMINI_MODEL})

    def get(self, identifier):
        custom_id, request = self.__class__.submissions[identifier]
        payload = json.loads(request["messages"][1]["content"])
        if "SOURCE_SEGMENT" in payload:
            segment = payload["SOURCE_SEGMENT"]
            first = segment["primary_utterances"][0]
            report = {
                "schema_version": "gemini_source_inventory_v1",
                "segment_id": segment["segment_id"],
                "coverage": [{"window_id": row["window_id"],
                              "start_id": row["start_id"], "end_id": row["end_id"],
                              "assessment": "material_items", "item_indices": [0]}
                             for row in segment["coverage_windows"]],
                "items": [{"kind": "action", "claim": first["text"],
                           "source_ids": [first["id"]], "speaker": first["speaker"],
                           "actor": None, "recipient": None, "action": first["text"],
                           "modality": "proposed", "condition": None,
                           "alternatives": [], "correction_of": [], "uncertainty": None}],
            }
        else:
            inventory = payload["INDEPENDENT_SOURCE_INVENTORY"]
            verifying = payload["MODE"] == "verify"
            items = inventory["items"]
            report = {
                "schema_version": "gemini_inventory_reconcile_v1",
                "source_window_assessments": [],
                "inventory_assessments": [{
                    "item_id": item["item_id"],
                    "status": "represented" if verifying or item["source_ids"] == ["U00001"] else "missing",
                    "draft_targets": [{"section": "tasks", "index": 1 if verifying and item["source_ids"] == ["U00002"] else 0}]
                    if verifying or item["source_ids"] == ["U00001"] else [],
                    "finding_indices": [] if verifying or item["source_ids"] == ["U00001"] else [0],
                } for item in items],
                "draft_assessments": [{
                    "unit_id": unit["unit_id"], "status": "supported",
                    "source_ids": ["U00002" if unit["unit_id"].startswith("tasks:1") else "U00001"],
                    "inventory_ids": [items[0]["item_id"]] if items else [],
                    "finding_indices": [],
                } for unit in payload["DRAFT_UNITS"]],
                "findings": [], "patches": [],
            }
            if not verifying:
                task = {"title": "Записать результат проверки",
                        "description": "После проверки записать результат в журнал.",
                        "discussion_status": "proposed", "assignee": None, "due": None,
                        "priority": None, "recipient": None,
                        "source_ids": ["U00002"],
                        "field_sources": {"action": ["U00002"], "assignee": [],
                                          "due": [], "priority": [], "recipient": [],
                                          "discussion_status": ["U00002"]}}
                report["findings"] = [{"severity": "major", "kind": "omission",
                                       "description": "Пропущено предложение записать результат.",
                                       "source_ids": ["U00002"], "affected": [],
                                       "status": "repaired", "patch_indices": [0]}]
                report["patches"] = [{"section": "tasks", "operation": "insert",
                                      "index": 1, "item_json": json.dumps(task, ensure_ascii=False)}]
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


class InventoryEngineTests(unittest.TestCase):
    def test_reserved_stage_rebuilds_missing_sidecars_before_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory)
            manifest = {"request_sha256": "pinned", "stage": "inventory_1"}
            request = {"messages": [{"role": "user", "content": "synthetic"}]}
            target = {"segment_id": "S01"}
            # Simulate the old interrupted write order: manifest exists, but
            # neither replay input does. The next reserved attempt repairs it.
            (artifacts / "manifest.json").write_text(json.dumps(manifest))
            _pin_stage_files(artifacts, manifest, request, target)
            self.assertEqual(json.loads((artifacts / "request.json").read_text()), request)
            self.assertEqual(json.loads((artifacts / "target.json").read_text()), target)
            (artifacts / "target.json").write_text(json.dumps({"segment_id": "other"}))
            with self.assertRaisesRegex(ValueError, "saved_quality_sidecar_changed"):
                _pin_stage_files(artifacts, manifest, request, target)

    def test_two_source_only_segments_reconcile_patch_verify_and_reuse(self):
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
            InventoryGemini.submit_calls = 0
            InventoryGemini.submissions = {}
            with patch("summary.luna_v1.engine._credential_store", return_value=TwoRoleStore()), \
                 patch("summary.luna_v1.engine.verify_batch_route", return_value=writer_route), \
                 patch("summary.luna_v1.engine.verify_gemini_batch_route", return_value=judge_route), \
                 patch("summary.luna_v1.engine.QUALITY_POLICY_VERSION",
                       INVENTORY_RECONCILE_QUALITY_POLICY_VERSION):
                started = submit(transcript_path=source, output_dir=output,
                                 private_root=private, client_factory=FakeClient)
                self.assertEqual(started["status"], "submitted")
                for _ in range(5):
                    arm_poll(private)
                    poll_once(private_root=private, client_factory=FakeClient,
                              gemini_client_factory=InventoryGemini)
                again = submit(transcript_path=source, output_dir=output,
                               private_root=private, client_factory=FakeClient)
            self.assertEqual(again["status"], "accepted_cache_hit")
            self.assertEqual(FakeClient.submit_calls, 1)
            self.assertEqual(InventoryGemini.submit_calls, 4)
            ledger = Ledger(private)
            parent = ledger.get(started["job_id"])
            self.assertEqual(parent["status"], "accepted")
            kinds = [row[0] for row in ledger.db.execute(
                "SELECT kind FROM jobs WHERE root_job_id=? ORDER BY created_at,id",
                (parent["id"],))]
            self.assertEqual(set(kinds), {"summary", "inventory_1", "inventory_2",
                                          "reconcile", "verify"})
            ledger.close()
            pointer = json.loads((output / "summary_current.json").read_text())
            generation = output / "summary_generations" / pointer["generation_id"]
            tasks = json.loads((generation / "tasks.json").read_text())
            self.assertEqual(len(tasks), 2)
            self.assertIsNone(tasks[1]["assignee"])
            review = json.loads((generation / "run_manifest.json").read_text())["quality_review"]
            self.assertEqual(review["status"], "model_reconciled_checked")
            for _, request in InventoryGemini.submissions.values():
                payload = json.loads(request["messages"][1]["content"])
                if "SOURCE_SEGMENT" in payload:
                    self.assertNotIn("DRAFT_DOCUMENT", payload)


if __name__ == "__main__":
    unittest.main()
