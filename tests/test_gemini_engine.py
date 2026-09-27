"""Production summary entry with two fake providers and an artificial source."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from summary.gemini_v1.batch import MODEL as GEMINI_MODEL, RESOLVED_MODEL, Reply as GeminiReply
from summary.gemini_v1.audit_v2 import GEMINI_AUDIT_SCHEMA_ID_V2
from summary.luna_v1.engine import (EVIDENCE_RECONCILE_QUALITY_POLICY_VERSION,
                                    INVENTORY_V2_QUALITY_POLICY_VERSION,
                                    _cost_micros, _finish_raw, _quality_contract,
                                    _quality_route, poll_once, submit)
from summary.luna_v1.audit import AUDIT_SCHEMA, AUDIT_SCHEMA_ID
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
    remote_model = GEMINI_MODEL

    def __init__(self, token):
        assert token == "synthetic-openrouter-judge"

    def submit(self, custom_id, request_body):
        self.__class__.submit_calls += 1
        name = f"batch_synthetic{self.__class__.submit_calls}"
        self.__class__.submissions[name] = (custom_id, request_body)
        return GeminiReply(202, {"id": name, "status": "validating",
                                 "endpoint": "/v1/chat/completions",
                                 "model": self.__class__.remote_model})

    def get(self, name):
        custom_id, request = self.__class__.submissions[name]
        payload = json.loads(request["messages"][1]["content"])
        windows = payload["SOURCE_WINDOWS"]
        report = {
            "schema_version": request["response_format"]["json_schema"]["name"],
            "coverage": [{"window_id": window["window_id"],
                          "start_id": window["start_id"], "end_id": window["end_id"],
                          "salient": "Обсуждены проверка и запись результата.",
                          "draft_coverage": "covered", "finding_indices": [],
                          **({"material_items": [{
                              "kind": "action", "claim": "Предложено конкретное действие.",
                              "source_ids": [window["start_id"]],
                              "draft_targets": (
                                  [{"section": "tasks", "index": number}]
                                  if number == 0 or payload["MODE"] == "verify" else []),
                              "finding_indices": (
                                  [0] if number == 1 and payload["MODE"] == "audit"
                                  and self.__class__.mode == "patch" else []),
                          }]} if request["response_format"]["json_schema"]["name"]
                              == GEMINI_AUDIT_SCHEMA_ID_V2 else {})}
                         for number, window in enumerate(windows)],
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
        if self.__class__.mode == "empty_inventory":
            for row in report["coverage"]:
                row["material_items"] = []
        response = {"model": GEMINI_MODEL, "choices": [{
            "message": {"role": "assistant", "content": json.dumps(report, ensure_ascii=False)},
            "finish_reason": "stop"}], "usage": {"prompt_tokens": 400,
                "completion_tokens": 350, "total_tokens": 750}}
        return GeminiReply(200, {"id": name, "status": "completed",
            "endpoint": "/v1/chat/completions",
            "model": self.__class__.remote_model,
            "request_counts": {"total": 1, "completed": 1, "failed": 0},
            "usage": {"cost": 0.000807, "is_byok": False, "prompt_tokens": 400,
                      "completion_tokens": 350, "total_tokens": 750},
            "results": [{"custom_id": custom_id, "response": {
                "status_code": 200, "body": response}, "error": None}]})

    def delete(self, name):
        return GeminiReply(204, {})


class GeminiEngineTests(unittest.TestCase):
    def _saved_gemini_writer(self, output, private):
        output.mkdir()
        source = output / "transcript.json"
        source.write_text(json.dumps({"source": "Синтетика.mkv", "duration_seconds": 12,
            "speakers": {"p1": "А"}, "utterances": [
                {"start": 0.2, "end": 5.0, "speaker": "p1",
                 "text": "Предлагаю проверить X или Y, не оба."},
                {"start": 5.2, "end": 11.0, "speaker": "p1",
                 "text": "Ещё предлагаю записать результат в журнал."},
            ]}, ensure_ascii=False), encoding="utf-8")
        writer_route = SimpleNamespace(
            reserve_microusd=lambda payload, **kwargs: 20_000,
            workspace_id="writer-workspace", prompt_usd_per_token="0.00000005",
            completion_usd_per_token="0.00000025",
            cache_write_usd_per_token="0.0000000625", request_usd="0")
        FakeClient.submit_calls = 0
        FakeClient.submissions = {}
        FakeClient.report_factory = None
        FakeGemini.submit_calls = 0
        FakeGemini.submissions = {}
        FakeGemini.mode = "empty_inventory"
        FakeGemini.remote_model = GEMINI_MODEL
        with patch("summary.luna_v1.engine._credential_store", return_value=TwoRoleStore()), \
             patch("summary.luna_v1.engine.QUALITY_POLICY_VERSION", INVENTORY_V2_QUALITY_POLICY_VERSION), \
             patch("summary.luna_v1.engine.verify_batch_route", return_value=writer_route):
            started = submit(transcript_path=source, output_dir=output,
                             private_root=private, client_factory=FakeClient)
        self.assertEqual(started["status"], "submitted")
        return started

    def test_pending_saved_gemini_root_does_not_dispatch_with_opus_logging_on(self):
        with tempfile.TemporaryDirectory() as directory:
            output, private = Path(directory) / "meeting", Path(directory) / "private"
            started = self._saved_gemini_writer(output, private)
            arm_poll(private)
            with patch("summary.luna_v1.engine._credential_store", return_value=TwoRoleStore()):
                outcomes = poll_once(private_root=private, client_factory=FakeClient,
                                     gemini_client_factory=FakeGemini)
            self.assertIn("accepted", {row["status"] for row in outcomes})
            self.assertEqual(FakeGemini.submit_calls, 0)
            ledger = Ledger(private)
            self.assertIsNone(ledger.stage(started["job_id"], "audit"))
            ledger.close()
            pointer = json.loads((output / "summary_current.json").read_text())
            generation = output / "summary_generations" / pointer["generation_id"]
            review = json.loads((generation / "run_manifest.json").read_text())["quality_review"]
            self.assertEqual(review["status"], "audit_unavailable")
            self.assertEqual(review["reason"], "gemini_workspace_logging_policy_mismatch")

    def test_fresh_legacy_gemini_writer_is_blocked_before_credential_or_post(self):
        with tempfile.TemporaryDirectory() as directory:
            output, private = Path(directory) / "meeting", Path(directory) / "private"
            output.mkdir()
            source = output / "transcript.json"
            source.write_text(json.dumps({
                "source": "Синтетика.mkv", "duration_seconds": 1,
                "speakers": {"p1": "А"},
                "utterances": [{"start": 0, "end": 1, "speaker": "p1", "text": "Проверить."}],
            }, ensure_ascii=False), encoding="utf-8")
            with patch("summary.luna_v1.engine._credential_store", return_value=TwoRoleStore()):
                outcome = submit(transcript_path=source, output_dir=output,
                                 private_root=private,
                                 client_factory=lambda _token: self.fail("legacy writer client created"),
                                 quality_policy_version=EVIDENCE_RECONCILE_QUALITY_POLICY_VERSION)
            self.assertEqual(outcome, {"status": "privacy_policy_blocked",
                                       "reason": "gemini_workspace_logging_policy_mismatch"})
            ledger = Ledger(private)
            self.assertEqual(ledger.db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 0)
            ledger.close()

    def test_reserved_saved_gemini_stage_is_cancelled_before_post(self):
        with tempfile.TemporaryDirectory() as directory:
            output, private = Path(directory) / "meeting", Path(directory) / "private"
            started = self._saved_gemini_writer(output, private)
            ledger = Ledger(private)
            writer = ledger.get(started["job_id"])
            reserved = ledger.reserve(
                semantic_key="a" * 64, source_sha256=writer["source_sha256"],
                output_dir=output, credential_id="judge-key", credential_version=1,
                workspace_id="judge-workspace", max_cost_microusd=25_000,
                kind="audit", root_job_id=writer["id"])
            self.assertEqual(reserved.kind, "new")
            ledger.close()
            arm_poll(private)
            with patch("summary.luna_v1.engine._credential_store", return_value=TwoRoleStore()):
                poll_once(private_root=private, client_factory=FakeClient,
                          gemini_client_factory=FakeGemini)
            ledger = Ledger(private)
            self.assertEqual(ledger.get(reserved.job_id)["status"], "cancelled_before_submit")
            ledger.close()
            self.assertEqual(FakeGemini.submit_calls, 0)

    def test_submitted_saved_gemini_stage_finishes_with_opus_logging_on(self):
        with tempfile.TemporaryDirectory() as directory:
            output, private = Path(directory) / "meeting", Path(directory) / "private"
            started = self._saved_gemini_writer(output, private)
            judge_route = SimpleNamespace(reserve_microusd=lambda: 25_000,
                                          input_tokens=1500, context_bound_tokens=2100,
                                          workspace_id="judge-workspace")
            arm_poll(private)
            with patch("summary.luna_v1.engine._credential_store", return_value=TwoRoleStore()), \
                 patch("summary.luna_v1.engine.QUALITY_POLICY_VERSION", INVENTORY_V2_QUALITY_POLICY_VERSION), \
                 patch("summary.luna_v1.engine.verify_gemini_batch_route", return_value=judge_route):
                poll_once(private_root=private, client_factory=FakeClient,
                          gemini_client_factory=FakeGemini)
            self.assertEqual(FakeGemini.submit_calls, 1)
            ledger = Ledger(private)
            self.assertEqual(ledger.stage(started["job_id"], "audit")["status"], "submitted")
            ledger.close()
            arm_poll(private)
            with patch("summary.luna_v1.engine._credential_store", return_value=TwoRoleStore()):
                outcomes = poll_once(private_root=private, client_factory=FakeClient,
                                     gemini_client_factory=FakeGemini)
            self.assertIn("accepted", {row["status"] for row in outcomes})
            self.assertEqual(FakeGemini.submit_calls, 1)

    def test_previous_gemini_policy_keeps_its_pinned_contract(self):
        old = "gemini_openrouter_judge_repair_v2"
        prompt, schema, schema_id = _quality_contract("gemini", old)
        self.assertEqual(prompt.name, "prompt_audit_v1.md")
        self.assertIs(schema, AUDIT_SCHEMA)
        self.assertEqual(schema_id, AUDIT_SCHEMA_ID)
        self.assertEqual(_quality_route({
            "quality_provider": "openrouter_gemini",
            "quality_policy_version": old,
        }), "gemini")

    def test_byok_gateway_fee_never_releases_full_reserve(self):
        self.assertIsNone(_cost_micros({"cost": 0.000001, "is_byok": True}))
        self.assertEqual(_cost_micros({"cost": 0.000807, "is_byok": False}), 807)

    def _assert_writer_then_openrouter_gemini_audit(self, remote_model, *, mode="patch"):
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
                                          input_tokens=1500, context_bound_tokens=2100,
                                          workspace_id="judge-workspace")
            FakeClient.submit_calls = 0
            FakeClient.submissions = {}
            FakeClient.report_factory = None
            FakeGemini.submit_calls = 0
            FakeGemini.submissions = {}
            FakeGemini.mode = mode
            FakeGemini.remote_model = remote_model
            with patch("summary.luna_v1.engine._credential_store", return_value=TwoRoleStore()), \
                 patch("summary.luna_v1.engine.QUALITY_POLICY_VERSION", INVENTORY_V2_QUALITY_POLICY_VERSION), \
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
            self.assertEqual(FakeGemini.submit_calls, 2 if mode == "patch" else 1, outcomes)
            self.assertEqual(again["status"], "accepted_cache_hit")
            ledger = Ledger(private)
            root_job = ledger.get(started["job_id"])
            audit = ledger.stage(root_job["id"], "audit")
            verify = ledger.stage(root_job["id"], "verify")
            self.assertEqual(root_job["status"], "accepted")
            self.assertEqual(audit["workspace_id"], "judge-workspace")
            self.assertEqual(audit["credential_id"], "judge-key")
            if mode == "patch":
                self.assertEqual(verify["workspace_id"], "judge-workspace")
                self.assertEqual(verify["credential_id"], "judge-key")
            else:
                self.assertIsNone(verify)
            self.assertEqual(audit["billed_microusd"], 807)
            for quality_job in (audit, verify) if verify else (audit,):
                artifacts = Path(quality_job["artifact_dir"])
                submission = json.loads((artifacts / "submit_response.json").read_text())
                terminal = json.loads((artifacts / "batch_terminal.json").read_text())
                self.assertEqual((submission["model"], terminal["model"]),
                                 (remote_model, remote_model))
                self.assertEqual(submission["endpoint"], "/v1/chat/completions")
                self.assertEqual(terminal["endpoint"], "/v1/chat/completions")
                self.assertEqual(terminal["results"][0]["response"]["body"]["model"], GEMINI_MODEL)
                self.assertIs(terminal["usage"]["is_byok"], False)
            ledger.close()
            pointer = json.loads((output / "summary_current.json").read_text())
            generation = output / "summary_generations" / pointer["generation_id"]
            tasks = json.loads((generation / "tasks.json").read_text())
            self.assertEqual(len(tasks), 2 if mode == "patch" else 1)
            if mode == "patch":
                self.assertEqual(tasks[1]["assignee"], None)
            review = json.loads((generation / "run_manifest.json").read_text())["quality_review"]
            self.assertEqual(review["status"],
                             "model_audit_unverified" if mode == "patch" else "coverage_incomplete")
            if mode == "empty_inventory":
                warnings = json.loads((Path(audit["artifact_dir"]) / "coverage_warnings.json").read_text())
                self.assertIn("material_inventory_empty",
                              {item["code"] for item in warnings["warnings"]})
            self.assertEqual(review["audit_job_id"], audit["id"])
            request = FakeGemini.submissions[next(iter(FakeGemini.submissions))][1]
            self.assertEqual(request["response_format"]["type"], "json_schema")
            self.assertEqual(request["plugins"], [])

    def test_writer_then_existing_gemini_batch_alias(self):
        self._assert_writer_then_openrouter_gemini_audit(GEMINI_MODEL)

    def test_writer_then_resolved_gemini_revision_audit_patch_verify_and_reuse(self):
        self._assert_writer_then_openrouter_gemini_audit(RESOLVED_MODEL)

    def test_empty_inventory_is_published_with_visible_incomplete_status(self):
        self._assert_writer_then_openrouter_gemini_audit(GEMINI_MODEL, mode="empty_inventory")

    def test_unrelated_or_mixed_gemini_batch_model_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory)
            (artifacts / "manifest.json").write_text(json.dumps({"provider": "openrouter_gemini"}))
            job = {"artifact_dir": str(artifacts), "remote_id": "batch_abc123",
                   "custom_id": "audit-001"}
            for submission_model, terminal_model in (
                ("google/gemini-3.7-flash-20260814", "google/gemini-3.7-flash-20260814"),
                (GEMINI_MODEL, RESOLVED_MODEL),
            ):
                with self.subTest(submission_model=submission_model, terminal_model=terminal_model):
                    (artifacts / "submit_response.json").write_text(json.dumps({
                        "id": job["remote_id"], "model": submission_model,
                        "endpoint": "/v1/chat/completions"}))
                    (artifacts / "batch_terminal.json").write_text(json.dumps({
                        "id": job["remote_id"], "model": terminal_model,
                        "endpoint": "/v1/chat/completions", "status": "completed"}))
                    with self.assertRaisesRegex(ValueError, "gemini_batch_submission_identity_mismatch"):
                        _finish_raw(None, job)


if __name__ == "__main__":
    unittest.main()
