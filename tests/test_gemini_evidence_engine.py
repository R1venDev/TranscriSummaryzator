"""Production-entry evidence policy with a long invented source and fake transport."""

from __future__ import annotations

import json
import socket
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from summary.gemini_v1.batch import MODEL as GEMINI_MODEL, Reply as GeminiReply
from summary.luna_v1.engine import (
    EVIDENCE_RECONCILE_QUALITY_POLICY_VERSION, poll_once, submit,
)
from summary.luna_v1.ledger import Ledger
from summary.luna_v1.source import load_source
from tests.test_gemini_engine import TwoRoleStore
from tests.test_luna_engine import FakeClient, arm_poll


class EvidenceGemini:
    """One terminal fake Batch item per serial quality stage."""

    submissions = {}
    submit_calls = 0
    active = set()
    stage_order = []
    bad_quote_segment = None

    def __init__(self, token):
        assert token == "synthetic-openrouter-judge"

    def submit(self, custom_id, request_body):
        cls = self.__class__
        if cls.active:
            raise AssertionError("concurrent Gemini quality dispatch")
        cls.submit_calls += 1
        name = f"batch_synthetic_evidence_{cls.submit_calls}"
        cls.submissions[name] = (custom_id, request_body)
        cls.active.add(name)
        payload = json.loads(request_body["messages"][1]["content"])
        cls.stage_order.append(
            payload["SOURCE_SEGMENT"]["segment_id"] if "SOURCE_SEGMENT" in payload
            else f"reconcile_{payload['RECONCILE_SCOPE']['part']}"
        )
        return GeminiReply(202, {"id": name, "status": "validating",
                                 "endpoint": "/v1/chat/completions", "model": GEMINI_MODEL})

    def get(self, name):
        cls = self.__class__
        custom_id, request = cls.submissions[name]
        payload = json.loads(request["messages"][1]["content"])
        schema_id = request["response_format"]["json_schema"]["name"]
        if "SOURCE_SEGMENT" in payload:
            segment = payload["SOURCE_SEGMENT"]
            items = []
            for row in segment["primary_utterances"]:
                number = int(row["id"][1:])
                claim = {
                    1: "Предложено проверить X или Y, только один вариант.",
                    2: "Предложено записать результат проверки в журнал.",
                    3: "Обсудили проверку данных.",
                    4: "Повторили контекст проверки данных.",
                }[number]
                quote = row["text"].split(". ", 1)[0] + "."
                if segment["segment_id"] == cls.bad_quote_segment:
                    quote = "Это дословной цитатой источника не является."
                kind = "technical" if number >= 3 else "action"
                items.append({"kind": kind, "claim": claim, "source_ids": [row["id"]],
                              "speaker": row["speaker"], "actor": None, "recipient": None,
                              "action": claim if kind == "action" else None,
                              "modality": "proposed" if kind == "action" else "explanation",
                              "condition": None, "alternatives": ["X", "Y"] if number == 1 else [],
                              "correction_of": [], "uncertainty": None,
                              "source_quote": quote})
            report = {
                "schema_version": schema_id, "segment_id": segment["segment_id"],
                "coverage": [{"window_id": window["window_id"],
                              "start_id": window["start_id"], "end_id": window["end_id"],
                              "assessment": "material_items", "item_indices": [next(
                                  i for i, item in enumerate(items)
                                  if window["start_id"] in item["source_ids"])]}
                             for window in segment["coverage_windows"]],
                "items": items,
            }
        else:
            inventory = payload["INDEPENDENT_SOURCE_INVENTORY"]
            self._assert_v2_reconcile_input(payload, schema_id)
            visible_ids = {row["id"] for row in payload["SOURCE_EXCERPTS"]}
            assessments = []
            for item in inventory["items"]:
                source_id = item["source_ids"][0]
                if source_id == "U00002" and len(payload["DRAFT_DOCUMENT"]["tasks"]) == 1:
                    assessments.append({"item_id": item["item_id"], "status": "missing",
                                        "draft_targets": [], "draft_evidence": [],
                                        "finding_indices": [0]})
                else:
                    section, index, unit_id, quote = (
                        ("tasks", 1, "tasks:1:action", "После проверки записать результат")
                        if source_id == "U00002" else
                        ("tasks", 0, "tasks:0:action", "Проверить один выбранный вариант X или Y")
                        if source_id == "U00001" else
                        ("meeting", 0, "meeting:0", "Проверка данных")
                    )
                    assessments.append({
                        "item_id": item["item_id"], "status": "represented",
                        "draft_targets": [{"section": section, "index": index}],
                        "draft_evidence": [{"unit_id": unit_id, "quote": quote}],
                        "finding_indices": [],
                    })
            report = {
                "schema_version": schema_id,
                "source_window_assessments": [{
                    "window_id": window["window_id"],
                    "status": "material_in_inventory",
                    "source_ids": [window["start_id"]],
                    "finding_indices": [],
                } for window in payload["SOURCE_WINDOWS_TO_RECHECK"]],
                "inventory_assessments": assessments,
                "draft_assessments": [{
                    "unit_id": unit["unit_id"], "status": "supported",
                    "source_ids": ["U00002" if unit["unit_id"].startswith("tasks:1")
                                   and "U00002" in visible_ids else
                                   "U00001" if "U00001" in visible_ids else
                                   sorted(visible_ids)[0]],
                    "inventory_ids": [item["item_id"] for item in inventory["items"]
                                      if item["source_ids"] ==
                                      (["U00002"] if unit["unit_id"].startswith("tasks:1")
                                       else ["U00001"])],
                    "finding_indices": [],
                } for unit in payload["DRAFT_UNITS"]],
                "findings": [], "patches": [],
            }
            if any(row["status"] == "missing" for row in assessments):
                task = {"title": "Записать результат проверки",
                        "description": "После проверки записать результат в журнал.",
                        "discussion_status": "proposed", "assignee": None,
                        "due": None, "priority": None, "recipient": None,
                        "source_ids": ["U00002"],
                        "field_sources": {"action": ["U00002"],
                                          "assignee": [], "due": [], "priority": [],
                                          "recipient": [], "discussion_status": ["U00002"]}}
                report["findings"] = [{"severity": "major", "kind": "omission",
                                       "description": "Пропущено предложение записать результат.",
                                       "source_ids": ["U00002"], "affected": [],
                                       "status": "repaired", "patch_indices": [0]}]
                report["patches"] = [{"section": "tasks", "operation": "insert",
                                      "index": 1, "item_json": json.dumps(task, ensure_ascii=False)}]
        body = {"model": GEMINI_MODEL, "choices": [{"message": {
            "role": "assistant", "content": json.dumps(report, ensure_ascii=False)},
            "finish_reason": "stop"}], "usage": {
                "prompt_tokens": 400, "completion_tokens": 350, "total_tokens": 750}}
        cls.active.discard(name)
        return GeminiReply(200, {"id": name, "status": "completed",
                                "endpoint": "/v1/chat/completions", "model": GEMINI_MODEL,
                                "request_counts": {"total": 1, "completed": 1, "failed": 0},
                                "usage": {"cost": 0.000807, "is_byok": False,
                                          "prompt_tokens": 400, "completion_tokens": 350,
                                          "total_tokens": 750},
                                "results": [{"custom_id": custom_id,
                                             "response": {"status_code": 200, "body": body},
                                             "error": None}]})

    @staticmethod
    def _assert_v2_reconcile_input(payload, schema_id):
        assert schema_id == "gemini_inventory_reconcile_v2"
        assert payload["INDEPENDENT_SOURCE_INVENTORY"]["schema_version"] == "gemini_source_inventory_merged_v2"
        assert "SOURCE_RISK_WARNINGS" in payload
        assert all("source_quote" in item for item in
                   payload["INDEPENDENT_SOURCE_INVENTORY"]["items"])
        assert payload["RECONCILE_SCOPE"]["part"] in (1, 2)

    def delete(self, name):
        return GeminiReply(204, {})


class EvidenceEngineTests(unittest.TestCase):
    def test_long_source_uses_three_inventories_then_two_bounded_reconciliations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "meeting"
            output.mkdir()
            source = output / "transcript.json"
            filler = "Служебное пояснение без новой работы. " * 650
            source.write_text(json.dumps({
                "source": "01.01.2030 — Синтетика.mkv", "duration_seconds": 24,
                "speakers": {"p1": "А"}, "utterances": [
                    {"start": 0.0, "end": 5.0, "speaker": "p1",
                     "text": "Предлагаю проверить X или Y, не оба. " + filler},
                    {"start": 6.0, "end": 11.0, "speaker": "p1",
                     "text": "Ещё предлагаю записать результат в журнал. " + filler},
                    {"start": 12.0, "end": 17.0, "speaker": "p1",
                     "text": "Обсудили проверку данных. " + filler},
                    {"start": 18.0, "end": 23.0, "speaker": "p1",
                     "text": "Повторили контекст проверки данных. " + filler},
                ]}, ensure_ascii=False), encoding="utf-8")
            source_text, source_index, _ = load_source(source)
            self.assertGreater(len(source_text), 60_000)
            self.assertEqual(len(source_index["by_id"]), 4)
            private = root / "private"
            writer_route = SimpleNamespace(
                reserve_microusd=lambda payload, **kwargs: 20_000,
                workspace_id="writer-workspace", prompt_usd_per_token="0.00000005",
                completion_usd_per_token="0.00000025",
                cache_write_usd_per_token="0.0000000625", request_usd="0")
            judge_route = SimpleNamespace(reserve_microusd=lambda: 15_000,
                input_tokens=1500, context_bound_tokens=2100,
                workspace_id="judge-workspace")
            FakeClient.submit_calls = 0
            FakeClient.submissions = {}
            FakeClient.report_factory = None
            EvidenceGemini.submit_calls = 0
            EvidenceGemini.submissions = {}
            EvidenceGemini.active = set()
            EvidenceGemini.stage_order = []
            EvidenceGemini.bad_quote_segment = None
            with patch("summary.luna_v1.engine._credential_store", return_value=TwoRoleStore()), \
                 patch("summary.luna_v1.engine.verify_batch_route", return_value=writer_route), \
                 patch("summary.luna_v1.engine.verify_gemini_batch_route", return_value=judge_route), \
                 patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")):
                started = submit(
                    transcript_path=source, output_dir=output,
                    private_root=private, client_factory=FakeClient,
                    quality_policy_version=EVIDENCE_RECONCILE_QUALITY_POLICY_VERSION)
                self.assertEqual(started["status"], "submitted")
                for _ in range(10):
                    arm_poll(private)
                    poll_once(private_root=private, client_factory=FakeClient,
                              gemini_client_factory=EvidenceGemini)
                    ledger = Ledger(private)
                    status = ledger.get(started["job_id"])["status"]
                    ledger.close()
                    if status == "accepted":
                        break
            self.assertEqual(status, "accepted")
            self.assertEqual(FakeClient.submit_calls, 1)
            self.assertEqual(EvidenceGemini.submit_calls, 5)
            self.assertEqual(EvidenceGemini.stage_order,
                             ["S01", "S02", "S03", "reconcile_1", "reconcile_2"])
            self.assertLessEqual(FakeClient.submit_calls + EvidenceGemini.submit_calls, 6)
            for label, (_, request) in zip(EvidenceGemini.stage_order,
                                           EvidenceGemini.submissions.values()):
                payload = json.loads(request["messages"][1]["content"])
                if label.startswith("S"):
                    self.assertEqual(request["response_format"]["json_schema"]["name"],
                                     "gemini_source_inventory_v2")
                    self.assertNotIn("DRAFT_DOCUMENT", payload)
                    self.assertNotIn("INDEPENDENT_SOURCE_INVENTORY", payload)
                else:
                    self.assertIn("DRAFT_DOCUMENT", payload)
                    self.assertIn("SOURCE_RISK_WARNINGS", payload)
            ledger = Ledger(private)
            parent = ledger.get(started["job_id"])
            self.assertEqual(parent["dispatches"], 1)
            stage_rows = [ledger.stage(parent["id"], kind) for kind in
                          ("inventory_1", "inventory_2", "inventory_3",
                           "reconcile_1", "reconcile_2")]
            self.assertTrue(all(stage is not None and stage["status"] == "stage_complete"
                                for stage in stage_rows))
            self.assertIsNone(ledger.stage(parent["id"], "verify"))
            targets = [json.loads((Path(stage["artifact_dir"]) / "target.json").read_text())
                       for stage in stage_rows[-2:]]
            self.assertEqual(len(targets[0]["draft"]["tasks"]), 1)
            self.assertEqual(len(targets[1]["draft"]["tasks"]), 2)
            primary = [source_id for target in targets
                       for source_id in target["scope"]["primary_source_ids"]]
            self.assertEqual(primary, list(source_index["by_id"]))
            inventory_ids = {item["item_id"] for target in targets
                             for item in target["inventory"]["items"]}
            merged = json.loads((Path(parent["artifact_dir"]) / "merged_inventory.json").read_text())
            self.assertEqual(inventory_ids, {item["item_id"] for item in merged["items"]})
            for stage in stage_rows:
                artifacts = Path(stage["artifact_dir"])
                self.assertTrue((artifacts / "request.json").is_file())
                self.assertTrue((artifacts / "native_response.json").is_file())
                self.assertTrue((artifacts / "batch_terminal.json").is_file())
                self.assertGreater(stage["billed_microusd"], 0)
            ledger.close()
            pointer = json.loads((output / "summary_current.json").read_text())
            generation = output / "summary_generations" / pointer["generation_id"]
            review = json.loads((generation / "run_manifest.json").read_text())["quality_review"]
            self.assertEqual(review["status"], "model_reconciled_unverified")
            self.assertEqual(review["reason"],
                             "verification_not_run_after_two_part_reconciliation")
            self.assertFalse(review["verification_performed"])
            self.assertEqual(review["verification_reason"],
                             "two_part_reconciliation_without_verification")
            self.assertEqual(review["reconcile_job_ids"], [stage["id"] for stage in stage_rows[-2:]])
            tasks = json.loads((generation / "tasks.json").read_text())
            self.assertEqual(len(tasks), 2)
            self.assertIsNone(tasks[1]["assignee"])

    def test_two_inventory_parts_do_not_claim_six_dispatches_used(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "meeting"
            output.mkdir()
            source = output / "transcript.json"
            source.write_text(json.dumps({
                "source": "01.01.2030 — Синтетика.mkv", "duration_seconds": 24,
                "speakers": {"p1": "А"}, "utterances": [
                    {"start": 0, "end": 5, "speaker": "p1",
                     "text": "Предлагаю проверить X или Y, не оба."},
                    {"start": 6, "end": 11, "speaker": "p1",
                     "text": "Ещё предлагаю записать результат в журнал."},
                    {"start": 12, "end": 17, "speaker": "p1",
                     "text": "Обсудили проверку данных."},
                    {"start": 18, "end": 23, "speaker": "p1",
                     "text": "Повторили контекст проверки данных."},
                ]}, ensure_ascii=False), encoding="utf-8")
            private = root / "private"
            writer_route = SimpleNamespace(
                reserve_microusd=lambda payload, **kwargs: 20_000,
                workspace_id="writer-workspace", prompt_usd_per_token="0.00000005",
                completion_usd_per_token="0.00000025",
                cache_write_usd_per_token="0.0000000625", request_usd="0")
            judge_route = SimpleNamespace(reserve_microusd=lambda: 15_000,
                input_tokens=1500, context_bound_tokens=2100,
                workspace_id="judge-workspace")
            FakeClient.submit_calls = 0
            FakeClient.submissions = {}
            FakeClient.report_factory = None
            EvidenceGemini.submit_calls = 0
            EvidenceGemini.submissions = {}
            EvidenceGemini.active = set()
            EvidenceGemini.stage_order = []
            EvidenceGemini.bad_quote_segment = None
            with patch("summary.luna_v1.engine._credential_store", return_value=TwoRoleStore()), \
                 patch("summary.luna_v1.engine.verify_batch_route", return_value=writer_route), \
                 patch("summary.luna_v1.engine.verify_gemini_batch_route", return_value=judge_route), \
                 patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")):
                started = submit(
                    transcript_path=source, output_dir=output, private_root=private,
                    client_factory=FakeClient,
                    quality_policy_version=EVIDENCE_RECONCILE_QUALITY_POLICY_VERSION)
                for _ in range(10):
                    arm_poll(private)
                    poll_once(private_root=private, client_factory=FakeClient,
                              gemini_client_factory=EvidenceGemini)
                    ledger = Ledger(private)
                    status = ledger.get(started["job_id"])["status"]
                    ledger.close()
                    if status == "accepted":
                        break
            self.assertEqual(status, "accepted")
            self.assertEqual(EvidenceGemini.stage_order,
                             ["S01", "S02", "reconcile_1", "reconcile_2"])
            self.assertEqual(FakeClient.submit_calls + EvidenceGemini.submit_calls, 5)
            pointer = json.loads((output / "summary_current.json").read_text())
            generation = output / "summary_generations" / pointer["generation_id"]
            review = json.loads((generation / "run_manifest.json").read_text())["quality_review"]
            self.assertNotEqual(review["reason"], "verification_not_run_six_dispatch_cap")
            self.assertEqual(review["verification_reason"],
                             "two_part_reconciliation_without_verification")

    def test_quote_mismatch_keeps_raw_and_marks_published_quality_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "meeting"
            output.mkdir()
            source = output / "transcript.json"
            source.write_text(json.dumps({
                "source": "01.01.2030 — Синтетика.mkv", "duration_seconds": 12,
                "speakers": {"p1": "А"}, "utterances": [
                    {"start": 0.0, "end": 5.0, "speaker": "p1",
                     "text": "Предлагаю проверить X или Y, не оба."},
                    {"start": 6.0, "end": 11.0, "speaker": "p1",
                     "text": "Ещё предлагаю записать результат в журнал."},
                ]}, ensure_ascii=False), encoding="utf-8")
            private = root / "private"
            writer_route = SimpleNamespace(
                reserve_microusd=lambda payload, **kwargs: 20_000,
                workspace_id="writer-workspace", prompt_usd_per_token="0.00000005",
                completion_usd_per_token="0.00000025",
                cache_write_usd_per_token="0.0000000625", request_usd="0")
            judge_route = SimpleNamespace(reserve_microusd=lambda: 15_000,
                input_tokens=1500, context_bound_tokens=2100,
                workspace_id="judge-workspace")
            FakeClient.submit_calls = 0
            FakeClient.submissions = {}
            FakeClient.report_factory = None
            EvidenceGemini.submit_calls = 0
            EvidenceGemini.submissions = {}
            EvidenceGemini.active = set()
            EvidenceGemini.stage_order = []
            EvidenceGemini.bad_quote_segment = "S02"
            with patch("summary.luna_v1.engine._credential_store", return_value=TwoRoleStore()), \
                 patch("summary.luna_v1.engine.verify_batch_route", return_value=writer_route), \
                 patch("summary.luna_v1.engine.verify_gemini_batch_route", return_value=judge_route), \
                 patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")):
                started = submit(transcript_path=source, output_dir=output,
                    private_root=private, client_factory=FakeClient,
                    quality_policy_version=EVIDENCE_RECONCILE_QUALITY_POLICY_VERSION)
                self.assertEqual(started["status"], "submitted")
                for _ in range(9):
                    arm_poll(private)
                    poll_once(private_root=private, client_factory=FakeClient,
                              gemini_client_factory=EvidenceGemini)
                    ledger = Ledger(private)
                    status = ledger.get(started["job_id"])["status"]
                    ledger.close()
                    if status == "accepted":
                        break
            self.assertEqual(status, "accepted")
            ledger = Ledger(private)
            second = ledger.stage(started["job_id"], "inventory_2")
            self.assertEqual(second["status"], "stage_complete")
            artifacts = Path(second["artifact_dir"])
            self.assertTrue((artifacts / "native_response.json").is_file())
            self.assertTrue((artifacts / "batch_terminal.json").is_file())
            warning = json.loads((artifacts / "coverage_warnings.json").read_text())["warnings"]
            self.assertIn("source_quote_mismatch", {row["code"] for row in warning})
            merged = json.loads((Path(ledger.get(started["job_id"])["artifact_dir"]) /
                                 "merged_inventory.json").read_text())
            self.assertIn("source_quote_mismatch",
                          {row["code"] for row in merged["risk_warnings"]})
            ledger.close()
            pointer = json.loads((output / "summary_current.json").read_text())
            generation = output / "summary_generations" / pointer["generation_id"]
            review = json.loads((generation / "run_manifest.json").read_text())["quality_review"]
            self.assertEqual(review["status"], "coverage_incomplete")
            self.assertGreater(review["coverage_warning_count"], 0)
            self.assertFalse(review["verification_performed"])


if __name__ == "__main__":
    unittest.main()
