"""Offline engine and ledger integration for the saved Opus Batch judge route."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlsplit

from summary.gemini_v1.batch import MODEL as GEMINI_MODEL
from summary.luna_v1.engine import (
    OPUS_QUALITY_POLICY_VERSION,
    OPUS_QUALITY_PROVIDER,
    OPUS_PRIVACY_MODE,
    OPUS_WRITER_PRIVACY_MODE,
    PREVIOUS_GEMINI_QUALITY_POLICY_VERSION,
    _quality_contract,
    _quality_route,
    _quality_stage_key,
    _semantic_identity,
    poll_once,
    submit,
)
from summary.luna_v1.ledger import (JOB_CAP_MICROUSD, OPUS_GROUP_CAP_MICROUSD,
                                    OPUS_STAGE_CAP_MICROUSD, Ledger)
from summary.opus_v1 import BatchClient as OpusBatchClient
from summary.opus_v1 import MODEL as OPUS_MODEL
from summary.opus_v1 import OPUS_AUDIT_SCHEMA_ID
from tests.test_luna_engine import FakeClient as WriterClient


class _SavedKeys:
    """The existing OpenRouter judge role is reused for Opus."""

    writer = {"id": "writer-key", "version": 1, "role": "writer",
              "workspace_id": "writer-workspace"}
    judge = {"id": "former-gemini-judge-key", "version": 1, "role": "judge",
             "workspace_id": "judge-workspace"}

    def dispatch_candidates(self, role="writer"):
        return [dict(self.judge if role == "judge" else self.writer)]

    def list(self):
        return {"keys": [dict(self.writer), dict(self.judge)]}

    def reveal_for_dispatch(self, identifier, version, role="writer"):
        expected = self.judge if role == "judge" else self.writer
        assert identifier == expected["id"] and version == 1
        return "synthetic-openrouter-judge" if role == "judge" else "synthetic-secret-never-sent"

    def reveal_for_existing_job(self, identifier, version):
        assert version == 1
        if identifier == self.judge["id"]:
            return "synthetic-openrouter-judge"
        assert identifier == self.writer["id"]
        return "synthetic-secret-never-sent"


class _HttpResponse:
    def __init__(self, status, document):
        self.status = status
        self.body = json.dumps(document, ensure_ascii=False).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, limit):
        return self.body[:limit]


class _OpusHttp:
    """Stub HTTP only; BatchClient still formats and validates every request."""

    def __init__(self, *, mode="clean", pending_rounds=0):
        self.mode = mode
        self.pending_rounds = pending_rounds
        self.calls = []
        self.submissions = {}
        self.get_counts = {}

    @property
    def posts(self):
        return [payload for method, path, payload in self.calls
                if method == "POST" and path == "/api/v1/batches"]

    @staticmethod
    def _route(path):
        price = {"prompt": "0.0000002", "completion": "0.000002", "request": "0"}
        if path == "/api/v1/key":
            return {"data": {"workspace_id": "judge-workspace", "limit_remaining": "1"}}
        if path == "/api/v1/models/user":
            return {"data": [{"id": OPUS_MODEL}]}
        if path == "/api/v1/model/anthropic/claude-opus-5.5:batch":
            return {"data": {"id": OPUS_MODEL, "context_length": 1_000_000,
                             "top_provider": {"max_completion_tokens": 128_000},
                             "supported_parameters": ["response_format", "reasoning", "max_tokens"],
                             "pricing": price}}
        if path == "/api/v1/models/anthropic/claude-opus-5.5%3Abatch/endpoints":
            return {"data": {"id": OPUS_MODEL, "endpoints": [{
                "tag": "anthropic/us", "provider_name": "Anthropic",
                "context_length": 1_000_000, "max_completion_tokens": 128_000,
                "supported_parameters": ["response_format", "reasoning"], "pricing": price,
            }]}}
        raise AssertionError(f"unexpected Opus route: {path}")

    def _report(self, request):
        payload = json.loads(request["messages"][1]["content"])
        source = payload["TRANSCRIPT_SOURCE"]
        utterances = {item["id"]: item for item in source["utterances"]}
        coverage = []
        for window in payload["SOURCE_WINDOWS"]:
            source_id = window["start_id"]
            coverage.append({
                "window_id": window["window_id"], "start_id": source_id,
                "end_id": window["end_id"], "salient": "Обсуждение вопроса и проверки.",
                "source_quote": utterances[source_id]["text"][:100],
                "draft_coverage": "covered", "finding_indices": [],
            })
        report = {"schema_version": OPUS_AUDIT_SCHEMA_ID,
                  "coverage": coverage, "findings": [], "patches": []}
        if payload["MODE"] == "audit" and self.mode in {"patch", "out_of_scope_verify"}:
            window = next(row for row in coverage if row["start_id"] == "U00002")
            window["draft_coverage"] = "missing"
            window["finding_indices"] = [0]
            question = {"text": "Когда получим исходные данные?", "source_ids": ["U00002"]}
            report["findings"] = [{
                "severity": "major", "kind": "omission",
                "description": "Пропущен вопрос о сроке получения данных.",
                "evidence_quote": "Когда получим исходные данные?",
                "source_ids": ["U00002"], "affected": [],
                "status": "repaired", "patch_indices": [0],
            }]
            report["patches"] = [{"section": "questions", "operation": "insert",
                                  "index": 0, "item_json": json.dumps(question, ensure_ascii=False)}]
        if payload["MODE"] == "verify" and self.mode == "out_of_scope_verify":
            # U00005 exists in the full transcript but was not supplied to
            # this focused verify request (which contains U00001..U00003).
            report["findings"] = [{
                "severity": "major", "kind": "omission",
                "description": "Недопустимая ссылка на непереданную реплику.",
                "evidence_quote": "На этом обсуждение завершено.",
                "source_ids": ["U00005"], "affected": [],
                "status": "unresolved", "patch_indices": [],
            }]
        return report

    def open(self, request, timeout):
        assert timeout == 45
        assert request.get_header("Authorization") == "Bearer synthetic-openrouter-judge"
        method, path = request.get_method(), urlsplit(request.full_url).path
        payload = json.loads(request.data) if request.data else None
        self.calls.append((method, path, payload))
        if method == "GET" and not path.startswith("/api/v1/batches/"):
            return _HttpResponse(200, self._route(path))
        if method == "POST" and path == "/api/v1/batches":
            batch_id = f"batch_opus_{len(self.submissions) + 1:03d}"
            self.submissions[batch_id] = payload["requests"][0]
            if self.mode == "unknown_post":
                # The provider may have accepted this POST, but the client did
                # not receive a receipt. A restart must keep it single-flight.
                raise OSError("synthetic connection drop after POST")
            return _HttpResponse(202, {"id": batch_id, "status": "validating",
                                       "endpoint": "/v1/chat/completions", "model": OPUS_MODEL})
        if path.startswith("/api/v1/batches/"):
            batch_id = path.rsplit("/", 1)[1]
            assert batch_id in self.submissions
            if method == "DELETE":
                return _HttpResponse(204, {})
            assert method == "GET"
            self.get_counts[batch_id] = self.get_counts.get(batch_id, 0) + 1
            item = self.submissions[batch_id]
            body = item["body"]
            mode = json.loads(body["messages"][1]["content"])["MODE"]
            if self.get_counts[batch_id] <= self.pending_rounds:
                return _HttpResponse(200, {"id": batch_id, "status": "in_progress",
                                           "endpoint": "/v1/chat/completions", "model": OPUS_MODEL})
            if mode == "audit" and self.mode == "failed":
                return _HttpResponse(200, {"id": batch_id, "status": "failed",
                                           "endpoint": "/v1/chat/completions", "model": OPUS_MODEL,
                                           "usage": {"cost": 0.001}})
            response = {"model": OPUS_MODEL, "choices": [{
                "finish_reason": "length" if mode == "audit" and self.mode == "length" else "stop",
                "message": {"role": "assistant", "content": json.dumps(
                    self._report(body), ensure_ascii=False)},
            }]}
            return _HttpResponse(200, {
                "id": batch_id, "status": "completed", "model": OPUS_MODEL,
                "endpoint": "/v1/chat/completions", "usage": {"cost": 0.001},
                "request_counts": {"total": 1, "completed": 1, "failed": 0},
                "results": [{"custom_id": item["custom_id"], "error": None,
                             "response": {"status_code": 200, "body": response}}],
            })
        raise AssertionError(f"unexpected Opus HTTP call: {method} {path}")


def _arm(private):
    ledger = Ledger(private)
    ledger.db.execute("UPDATE jobs SET next_poll_at=0")
    ledger.close()


def _source(output):
    output.mkdir()
    path = output / "transcript.json"
    path.write_text(json.dumps({
        "source": "Синтетическая встреча.mkv", "duration_seconds": 20,
        "speakers": {"p1": "А"}, "utterances": [
            {"start": 0.2, "end": 3.0, "speaker": "p1",
             "text": "Предлагаю проверить X или Y, не оба."},
            {"start": 3.1, "end": 6.0, "speaker": "p1",
             "text": "Когда получим исходные данные?"},
            {"start": 6.1, "end": 9.0, "speaker": "p1",
             "text": "Проверку начнём после получения данных."},
            {"start": 9.1, "end": 12.0, "speaker": "p1",
             "text": "Материалы встречи лежат в папке."},
            {"start": 12.1, "end": 15.0, "speaker": "p1",
             "text": "На этом обсуждение завершено."},
        ],
    }, ensure_ascii=False), encoding="utf-8")
    return path


class OpusEngineTests(unittest.TestCase):
    def setUp(self):
        WriterClient.submit_calls = 0
        WriterClient.submissions = {}
        WriterClient.report_factory = None

    @staticmethod
    def _writer_route():
        return SimpleNamespace(
            reserve_microusd=lambda payload, **kwargs: 20_000,
            workspace_id="writer-workspace",
            prompt_usd_per_token="0.00000005",
            completion_usd_per_token="0.00000025",
            cache_write_usd_per_token="0.0000000625", request_usd="0",
        )

    def _start(self, output, private, opener):
        WriterClient.submit_calls = 0
        WriterClient.submissions = {}
        WriterClient.report_factory = None
        with patch("summary.luna_v1.engine._credential_store", return_value=_SavedKeys()), \
             patch("summary.luna_v1.engine.verify_batch_route", return_value=self._writer_route()):
            started = submit(transcript_path=output / "transcript.json", output_dir=output,
                             private_root=private, client_factory=WriterClient)
        self.assertEqual(started["status"], "submitted")
        self.assertEqual(WriterClient.submit_calls, 1)
        return started

    def _poll(self, private, opener):
        _arm(private)
        with patch("summary.luna_v1.engine._credential_store", return_value=_SavedKeys()):
            return poll_once(private_root=private, client_factory=WriterClient,
                             opus_client_factory=lambda token: OpusBatchClient(token, opener=opener))

    @staticmethod
    def _published(output):
        pointer = json.loads((output / "summary_current.json").read_text(encoding="utf-8"))
        generation = output / "summary_generations" / pointer["generation_id"]
        return generation, json.loads((generation / "run_manifest.json").read_text(encoding="utf-8"))

    def test_single_full_source_audit_reuses_saved_judge_key_and_never_reposts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, private = root / "meeting", root / "private"
            _source(output)
            opener = _OpusHttp(pending_rounds=1)
            started = self._start(output, private, opener)
            first = self._poll(private, opener)
            self.assertIn("opus_audit_submitted", {row["status"] for row in first})
            ledger = Ledger(private)
            audit = ledger.stage(started["job_id"], "audit")
            self.assertEqual((audit["credential_id"], audit["workspace_id"]),
                             (_SavedKeys.judge["id"], _SavedKeys.judge["workspace_id"]))
            writer = ledger.get(started["job_id"])
            writer_manifest = json.loads((Path(writer["artifact_dir"]) / "manifest.json").read_text())
            audit_manifest = json.loads((Path(audit["artifact_dir"]) / "manifest.json").read_text())
            self.assertEqual(writer_manifest["privacy_mode"], OPUS_WRITER_PRIVACY_MODE)
            self.assertIs(writer_manifest["workspace_io_logging_enabled"], True)
            self.assertEqual(writer_manifest["audit_privacy_mode"], OPUS_PRIVACY_MODE)
            self.assertIs(writer_manifest["audit_workspace_io_logging_enabled"], True)
            self.assertEqual(audit_manifest["privacy_mode"], OPUS_PRIVACY_MODE)
            self.assertIs(audit_manifest["workspace_io_logging_enabled"], True)
            self.assertIn("logging_on_min_3mo_or_longer", OPUS_PRIVACY_MODE)
            self.assertIsNone(ledger.stage(started["job_id"], "verify"))
            ledger.close()

            second = self._poll(private, opener)
            self.assertIn("opus_audit_pending", {row["status"] for row in second})
            self.assertEqual(len(opener.posts), 1)
            third = self._poll(private, opener)
            self.assertIn("accepted", {row["status"] for row in third})
            generation, run = self._published(output)
            self.assertEqual(run["quality_review"]["status"], "opus_audited_unverified")
            self.assertIsNone(run["quality_review"]["verify_job_id"])
            self.assertIn("Claude Opus 5.5 сопоставил всю доступную стенограмму",
                          (generation / "summary.md").read_text(encoding="utf-8"))
            request = opener.posts[0]
            self.assertEqual(request["model"], OPUS_MODEL)
            self.assertEqual(request["provider"], {"only": ["anthropic"]})
            self.assertEqual(len(request["requests"]), 1)
            content = json.loads(request["requests"][0]["body"]["messages"][1]["content"])
            self.assertEqual(content["MODE"], "audit")
            self.assertEqual(len(content["TRANSCRIPT_SOURCE"]["utterances"]), 5)
            self.assertEqual(len(content["SOURCE_WINDOWS"]), 5)

            # A new scheduler/ledger invocation and a duplicate submit reuse the
            # durable result, including the accepted publication.
            self._poll(private, opener)
            with patch("summary.luna_v1.engine._credential_store", return_value=_SavedKeys()):
                again = submit(transcript_path=output / "transcript.json", output_dir=output,
                               private_root=private, client_factory=WriterClient)
            self.assertEqual(again["status"], "accepted_cache_hit")
            self.assertEqual((WriterClient.submit_calls, len(opener.posts)), (1, 1))

    def test_patch_gets_one_focused_verify_using_same_saved_judge_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, private = root / "meeting", root / "private"
            _source(output)
            opener = _OpusHttp(mode="patch")
            started = self._start(output, private, opener)
            self.assertIn("opus_audit_submitted", {row["status"] for row in self._poll(private, opener)})
            second = self._poll(private, opener)
            self.assertIn("opus_verify_submitted", {row["status"] for row in second})
            third = self._poll(private, opener)
            self.assertIn("accepted", {row["status"] for row in third})
            self.assertEqual(len(opener.posts), 2)
            audit_request, verify_request = [post["requests"][0]["body"] for post in opener.posts]
            audit_payload = json.loads(audit_request["messages"][1]["content"])
            verify_payload = json.loads(verify_request["messages"][1]["content"])
            self.assertEqual([item["id"] for item in audit_payload["TRANSCRIPT_SOURCE"]["utterances"]],
                             [f"U{number:05d}" for number in range(1, 6)])
            self.assertEqual(verify_payload["MODE"], "verify")
            self.assertEqual([item["id"] for item in verify_payload["TRANSCRIPT_SOURCE"]["utterances"]],
                             ["U00001", "U00002", "U00003"])
            self.assertEqual(len(verify_payload["PRIOR_FINDINGS"]), 1)
            self.assertIn("DRAFT_DOCUMENT", verify_payload)
            self.assertNotIn("DRAFT_TARGETS", verify_payload)
            self.assertEqual(verify_payload["DRAFT_DOCUMENT"]["questions"], [
                {"text": "Когда получим исходные данные?", "source_ids": ["U00002"]}])
            self.assertEqual(audit_request["response_format"]["json_schema"]["name"],
                             verify_request["response_format"]["json_schema"]["name"])
            ledger = Ledger(private)
            audit = ledger.stage(started["job_id"], "audit")
            verify = ledger.stage(started["job_id"], "verify")
            self.assertEqual((audit["credential_id"], verify["credential_id"]),
                             (_SavedKeys.judge["id"], _SavedKeys.judge["id"]))
            self.assertEqual((audit["workspace_id"], verify["workspace_id"]),
                             (_SavedKeys.judge["workspace_id"],) * 2)
            writer = ledger.get(started["job_id"])
            candidate = json.loads((Path(writer["artifact_dir"]) / "candidate_document.json").read_text())
            self.assertEqual(candidate["questions"], [{"text": "Когда получим исходные данные?",
                                                      "source_ids": ["U00002"]}])
            ledger.close()
            generation, run = self._published(output)
            self.assertEqual(run["quality_review"]["status"], "opus_self_verified")
            self.assertIn("повторно проверил внесённые правки",
                          (generation / "summary.md").read_text(encoding="utf-8"))
            self._poll(private, opener)
            self.assertEqual(len(opener.posts), 2)

    def test_verify_rejects_source_id_outside_sent_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, private = root / "meeting", root / "private"
            _source(output)
            opener = _OpusHttp(mode="out_of_scope_verify")
            started = self._start(output, private, opener)
            self._poll(private, opener)
            audit_result = self._poll(private, opener)
            self.assertIn("opus_verify_submitted", {row["status"] for row in audit_result})
            verify_result = self._poll(private, opener)
            self.assertIn("failed_validation", {row["status"] for row in verify_result})
            self.assertIn("accepted", {row["status"] for row in verify_result})
            ledger = Ledger(private)
            verify = ledger.stage(started["job_id"], "verify")
            self.assertEqual(verify["status"], "failed_validation")
            self.assertIn("source_outside_verify_scope", verify["error_code"])
            writer = ledger.get(started["job_id"])
            candidate = json.loads((Path(writer["artifact_dir"]) / "candidate_document.json").read_text())
            self.assertEqual(candidate["questions"], [
                {"text": "Когда получим исходные данные?", "source_ids": ["U00002"]}])
            ledger.close()
            generation, run = self._published(output)
            self.assertEqual(run["quality_review"]["status"], "opus_verify_unavailable")
            self.assertIn("адресная повторная проверка не завершилась",
                          (generation / "summary.md").read_text(encoding="utf-8"))
            self.assertEqual(len(opener.posts), 2)

    def test_unknown_post_is_never_repeated_after_scheduler_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, private = root / "meeting", root / "private"
            _source(output)
            opener = _OpusHttp(mode="unknown_post")
            started = self._start(output, private, opener)
            first = self._poll(private, opener)
            self.assertIn("opus_audit_submission_unknown", {row["status"] for row in first})
            ledger = Ledger(private)
            audit = ledger.stage(started["job_id"], "audit")
            self.assertEqual(audit["status"], "submission_unknown")
            self.assertEqual(audit["dispatches"], 1)
            ledger.close()
            second = self._poll(private, opener)
            self.assertIn("accepted", {row["status"] for row in second})
            self._poll(private, opener)
            generation, run = self._published(output)
            self.assertEqual(run["quality_review"]["status"], "opus_audit_unavailable")
            self.assertIn("Проверка черновика Claude Opus 5.5 не завершилась",
                          (generation / "summary.md").read_text(encoding="utf-8"))
            self.assertEqual(len(opener.posts), 1)

    def test_failed_or_length_limited_audit_preserves_draft_and_visible_status(self):
        for mode in ("failed", "length"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output, private = root / "meeting", root / "private"
                _source(output)
                opener = _OpusHttp(mode=mode)
                started = self._start(output, private, opener)
                self._poll(private, opener)
                result = self._poll(private, opener)
                self.assertIn("accepted", {row["status"] for row in result})
                ledger = Ledger(private)
                writer = ledger.get(started["job_id"])
                audit = ledger.stage(started["job_id"], "audit")
                draft = json.loads((Path(writer["artifact_dir"]) / "draft_document.json").read_text())
                candidate = json.loads((Path(writer["artifact_dir"]) / "candidate_document.json").read_text())
                self.assertEqual(candidate, draft)
                self.assertIsNone(ledger.stage(started["job_id"], "verify"))
                self.assertIn(audit["status"], {"remote_failed", "failed_validation"})
                ledger.close()
                generation, run = self._published(output)
                self.assertEqual(run["quality_review"]["status"], "opus_audit_unavailable")
                self.assertIn("Проверка черновика Claude Opus 5.5 не завершилась",
                              (generation / "summary.md").read_text(encoding="utf-8"))
                self.assertEqual(len(opener.posts), 1)

    def test_saved_gemini_policy_still_routes_to_its_original_contract(self):
        manifest = {"quality_policy_version": PREVIOUS_GEMINI_QUALITY_POLICY_VERSION,
                    "quality_provider": "openrouter_gemini", "audit_model": GEMINI_MODEL}
        self.assertEqual(_quality_route(manifest), "gemini")
        prompt, schema, schema_id = _quality_contract(
            "gemini", PREVIOUS_GEMINI_QUALITY_POLICY_VERSION)
        self.assertEqual(prompt.name, "prompt_audit_v1.md")
        self.assertIsInstance(schema, dict)
        self.assertNotEqual(schema_id, OPUS_AUDIT_SCHEMA_ID)
        self.assertEqual(_quality_route({"quality_policy_version": OPUS_QUALITY_POLICY_VERSION,
                                         "quality_provider": OPUS_QUALITY_PROVIDER,
                                         "audit_model": OPUS_MODEL}), "opus")

    def test_logging_choice_changes_only_opus_semantic_keys(self):
        args = ("a" * 64, "source", "writer-workspace", "b" * 64,
                "c" * 64, None, "judge-workspace")
        opus_root = _semantic_identity(*args, OPUS_QUALITY_POLICY_VERSION)
        gemini_root = _semantic_identity(*args, PREVIOUS_GEMINI_QUALITY_POLICY_VERSION)
        root = {"semantic_key": "a" * 64}
        request = {"messages": []}
        opus_stage = _quality_stage_key(root, "audit", request, "opus", OPUS_QUALITY_POLICY_VERSION)
        gemini_stage = _quality_stage_key(root, "audit", request, "gemini",
                                          PREVIOUS_GEMINI_QUALITY_POLICY_VERSION)
        with patch("summary.luna_v1.engine.OPUS_WORKSPACE_IO_LOGGING_ENABLED", False):
            self.assertNotEqual(_semantic_identity(*args, OPUS_QUALITY_POLICY_VERSION), opus_root)
            self.assertNotEqual(_quality_stage_key(root, "audit", request, "opus",
                                                   OPUS_QUALITY_POLICY_VERSION), opus_stage)
            self.assertEqual(_semantic_identity(*args, PREVIOUS_GEMINI_QUALITY_POLICY_VERSION),
                             gemini_root)
            self.assertEqual(_quality_stage_key(root, "audit", request, "gemini",
                                                PREVIOUS_GEMINI_QUALITY_POLICY_VERSION), gemini_stage)

    def test_old_opus_root_without_logging_disclosure_cannot_start_judge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, private = root / "meeting", root / "private"
            _source(output)
            opener = _OpusHttp()
            started = self._start(output, private, opener)
            ledger = Ledger(private)
            writer = ledger.get(started["job_id"])
            manifest_path = Path(writer["artifact_dir"]) / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["privacy_mode"] = "batch_gateway_retention_up_to_30d_provider_zdr_off_user_authorized"
            manifest["audit_privacy_mode"] = "openrouter_batch_30d_anthropic_user_authorized"
            del manifest["workspace_io_logging_enabled"]
            del manifest["audit_workspace_io_logging_enabled"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            ledger.close()
            result = self._poll(private, opener)
            self.assertIn("accepted", {row["status"] for row in result})
            self.assertEqual(opener.posts, [])
            _generation, published = self._published(output)
            self.assertEqual(published["quality_review"]["status"], "opus_audit_unavailable")

    def test_ledger_raises_only_pinned_opus_stage_and_group_caps(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Ledger(Path(directory) / "private")
            self.assertEqual((JOB_CAP_MICROUSD, OPUS_STAGE_CAP_MICROUSD,
                              OPUS_GROUP_CAP_MICROUSD), (100_000, 450_000, 600_000))

            def writer(key, source_sha, policy, provider):
                row = ledger.reserve(
                    semantic_key=key * 64, source_sha256=source_sha * 64,
                    output_dir=Path(directory) / f"meeting-{key}",
                    credential_id="writer-key", credential_version=1,
                    workspace_id="writer-workspace", max_cost_microusd=50_000,
                )
                self.assertEqual(row.kind, "new")
                artifacts = Path(ledger.get(row.job_id)["artifact_dir"])
                (artifacts / "manifest.json").write_text(json.dumps({
                    "quality_policy_version": policy, "quality_provider": provider,
                    "judge_workspace_id": "judge-workspace",
                }), encoding="utf-8")
                return row.job_id

            def stage(root_id, key, kind, reserve, source_sha):
                return ledger.reserve(
                    semantic_key=key * 64, source_sha256=source_sha * 64,
                    output_dir=Path(ledger.get(root_id)["output_dir"]),
                    credential_id="former-gemini-judge-key", credential_version=1,
                    workspace_id="judge-workspace", max_cost_microusd=reserve,
                    kind=kind, root_job_id=root_id,
                )

            opus_root = writer("a", "1", OPUS_QUALITY_POLICY_VERSION,
                               OPUS_QUALITY_PROVIDER)
            self.assertEqual(stage(opus_root, "b", "audit", 450_001, "1").reason,
                             "job_budget_exceeded")
            self.assertEqual(stage(opus_root, "b", "audit", 450_000, "1").kind,
                             "new")
            self.assertEqual(stage(opus_root, "c", "verify", 100_000, "1").kind,
                             "new")
            self.assertEqual(stage(opus_root, "d", "verify", 1, "1").reason,
                             "logical_job_budget_exceeded")

            gemini_root = writer("e", "2", PREVIOUS_GEMINI_QUALITY_POLICY_VERSION,
                                 "openrouter_gemini")
            self.assertEqual(stage(gemini_root, "f", "audit", 100_001, "2").reason,
                             "job_budget_exceeded")
            ledger.close()


if __name__ == "__main__":
    unittest.main()
