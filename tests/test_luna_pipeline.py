"""Summary-only integration through the real pipeline entry points.

All credentials, transcript content and Batch replies here are synthetic.
No provider request or speech worker is started.
"""

import hashlib
import base64
from contextlib import contextmanager
from email.message import Message
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pipeline
from summary_credentials import make_admin_password_hash
from summary.luna_v1.ledger import Ledger
from summary.luna_v1 import load_source
from summary.luna_v1.publication import publish_document
from summary.luna_v1.tasks import TaskStore
from tests.test_luna_task_api import fake_document


def _job(db, output, work, *, summary_status="queued", attempt_id=None):
    cursor = db.execute(
        """INSERT INTO jobs (fingerprint,source_path,original_name,status,stage,job_dir,
              output_dir,summary_status,summary_attempt_id,summary_started_at,created_at,updated_at)
              VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (hashlib.sha256(str(output).encode()).hexdigest(), "/unused/source.wav", "sample.wav", "done", "done", str(work),
         str(output), summary_status, attempt_id, pipeline.now(), pipeline.now(), pipeline.now()),
    )
    db.commit()
    return cursor.lastrowid


class LunaPipelineIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.state = self.root / "state"
        self.output = self.root / "output"
        self.work = self.root / "work"
        self.state.mkdir(); self.output.mkdir(); self.work.mkdir()
        (self.output / "transcript.json").write_text('{"synthetic":true}\n', encoding="utf-8")
        self.patches = [
            patch.object(pipeline, "STATE", self.state),
            patch.object(pipeline, "DB_PATH", self.state / "queue.sqlite3"),
            patch.object(pipeline, "config", return_value={"summary_backend": "luna_batch", "summary_enabled": True,
                                                           "poll_seconds": 1}),
            patch.object(pipeline, "configure_diagnostics"),
            patch.object(pipeline, "diagnostic_event"),
            patch.object(pipeline, "process_job", side_effect=AssertionError("speech pipeline called")),
            patch.object(pipeline, "extract_audio", side_effect=AssertionError("audio preparation called")),
            patch.object(pipeline, "summary_python", return_value=sys.executable),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def _seal_luna_generation(self, *, job_id, source_sha, semantic_key, remote_batch_id):
        generation = "20260925-000000-" + "e" * 12
        package = self.output / "summary_generations" / generation
        package.mkdir(parents=True)
        digests = {}
        for name in pipeline.LUNA_GENERATION_FILES:
            payload = (json.dumps({
                "contract_version": "luna_summary_v1", "job_id": job_id,
                "source_sha256": source_sha, "semantic_key": semantic_key,
                "remote_batch_id": remote_batch_id,
            }) + "\n").encode() if name == "run_manifest.json" else name.encode()
            (package / name).write_bytes(payload)
            digests[name] = hashlib.sha256(payload).hexdigest()
        pipeline.write_json(package / "generation_manifest.json", {
            "contract_version": "luna_summary_v1", "generation_id": generation,
            "source_sha256": source_sha, "artifact_sha256": digests,
            "verified_artifact_sha256": digests["summary.md"],
        })
        pipeline.write_json(self.output / "summary_current.json", {
            "generation_id": generation, "verified_artifact_sha256": digests["summary.md"],
        })
        return generation, package

    def test_protected_stage_keys_keep_legacy_identity_only_for_reviewed_source(self):
        source = (pipeline.ROOT / "pipeline.py").read_bytes()
        legacy = "f310dd064f3515cfb24a29b80a85037203b3602d954110360878a3cf4e1f0115"
        self.assertEqual(pipeline._stage_pipeline_sha256(source), legacy)
        for stage in ("audio", "asr"):
            inputs = {"synthetic_input": "unchanged"}
            code = {}
            for relative in pipeline.STAGE_DEPENDENCIES[stage]:
                path = pipeline.ROOT / relative
                if path.is_file():
                    code[relative] = legacy if relative == "pipeline.py" else hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(pipeline.stage_cache_key(stage, inputs),
                             pipeline._json_hash({"stage": stage, "inputs": inputs, "code": code}))
        altered = source.replace(b"def extract_audio(", b"def extract_audio_changed(", 1)
        self.assertNotEqual(altered, source)
        self.assertEqual(pipeline._stage_pipeline_sha256(altered), hashlib.sha256(altered).hexdigest())
        self.assertNotEqual(pipeline._stage_pipeline_sha256(altered), legacy)

    def test_key_order_and_disable_hold_dispatch_guard_through_rpc(self):
        handler = object.__new__(pipeline.DashboardHandler)
        held = []
        calls = []

        @contextmanager
        def guard(path):
            self.assertEqual(path, pipeline.SUMMARY_CREDENTIAL_DB)
            held.append(True)
            try:
                yield
            finally:
                held.pop()

        def rpc(action, body):
            self.assertTrue(held)
            calls.append((action, body))
            return {"synthetic": True}

        with patch.object(pipeline, "credential_dispatch_guard", guard), \
             patch.object(pipeline.DashboardHandler, "require_summary_admin", return_value=True), \
             patch.object(pipeline.DashboardHandler, "summary_admin_request_json",
                          side_effect=[{"ids": ["synthetic"]}, {"id": "synthetic", "enabled": False}]), \
             patch.object(pipeline.DashboardHandler, "summary_credential_rpc", side_effect=rpc), \
             patch.object(pipeline.DashboardHandler, "send_summary_admin_json") as response:
            handler.handle_summary_credential_write("/api/summary/credentials/order")
            handler.handle_summary_credential_write("/api/summary/credentials/enabled")
        self.assertEqual([action for action, _ in calls], ["order", "enabled"])
        self.assertFalse(held)
        self.assertEqual(response.call_count, 2)

    def test_luna_dispatch_uses_isolated_worker_and_never_calls_speech(self):
        db = pipeline.connect()
        job_id = _job(db, self.output, self.work)
        db.close()
        seen = []

        def fake_run(command, log, **kwargs):
            seen.append((command, kwargs))
            pipeline.write_json(self.output / "summary_luna_attempt.json", {
                "status": "submitted", "job_id": "b" * 32, "remote_id": "batch_synthetic",
            })

        with patch.object(pipeline, "run_command", side_effect=fake_run):
            self.assertTrue(pipeline.process_summary(job_id))
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][0][:3], [sys.executable, str(pipeline.ROOT / "scripts/luna_summary_worker.py"), "submit"])
        self.assertIn("--private-root", seen[0][0])
        self.assertNotIn("--force-nonce", seen[0][0])
        row = pipeline.connect().execute("SELECT status,summary_status FROM jobs WHERE id=?", (job_id,)).fetchone()
        self.assertEqual((row["status"], row["summary_status"]), ("done", "pending_batch"))
        self.assertEqual(pipeline.load_json(self.output / "summary_attempt.json")["remote_batch_id"], "batch_synthetic")

    def test_worker_preflight_refusal_stays_blocked_not_submission_unknown(self):
        db = pipeline.connect(); job_id = _job(db, self.output, self.work); db.close()

        def fake_run(*_args, **_kwargs):
            pipeline.write_json(self.output / "summary_luna_attempt.json", {
                "status": "preflight_blocked", "reason": "model_not_allowed",
            })
            raise RuntimeError("worker exited 2")

        with patch.object(pipeline, "run_command", side_effect=fake_run):
            self.assertFalse(pipeline.process_summary(job_id))
        row = pipeline.connect().execute("SELECT summary_status,summary_error FROM jobs WHERE id=?", (job_id,)).fetchone()
        self.assertEqual(row["summary_status"], "blocked")
        self.assertEqual(row["summary_error"], "model_not_allowed")

    def test_recovery_uses_ledger_and_never_submits_again(self):
        db = pipeline.connect()
        job_id = _job(db, self.output, self.work, summary_status="running", attempt_id="attempt-synthetic")
        db.close()
        ledger = Ledger(self.state / "summary_private")
        decision = ledger.reserve(semantic_key="c" * 64, source_sha256="d" * 64,
                                  output_dir=self.output, credential_id="key-synthetic",
                                  credential_version=1, workspace_id="workspace-synthetic",
                                  max_cost_microusd=1000)
        self.assertEqual(decision.kind, "new")
        ledger.mark_submitting(decision.job_id)
        ledger.submission_result(decision.job_id, remote_id="batch_synthetic")
        ledger.close()
        pipeline.write_json(self.output / "summary_luna_attempt.json", {"status": "submitted", "job_id": decision.job_id})
        pipeline.write_json(self.output / "summary_attempt.json", {"attempt_id": "attempt-synthetic"})
        with patch.object(pipeline, "run_next_summary", side_effect=AssertionError("duplicate dispatch")):
            self.assertEqual(pipeline.reconcile_luna_summary_queue(), 1)
        row = pipeline.connect().execute("SELECT summary_status FROM jobs WHERE id=?", (job_id,)).fetchone()
        self.assertEqual(row["summary_status"], "pending_batch")

    def test_queued_retry_waits_for_unresolved_batch_in_same_output(self):
        db = pipeline.connect()
        job_id = _job(db, self.output, self.work, summary_status="queued_force")
        db.close()
        ledger = Ledger(self.state / "summary_private")
        decision = ledger.reserve(semantic_key="c" * 64, source_sha256="d" * 64,
                                  output_dir=self.output, credential_id="key-synthetic",
                                  credential_version=1, workspace_id="workspace-synthetic",
                                  max_cost_microusd=1000)
        ledger.mark_submitting(decision.job_id)
        ledger.submission_result(decision.job_id, remote_id="batch_synthetic")
        second_output = self.root / "second-output"
        second_output.mkdir()
        (second_output / "transcript.json").write_bytes((self.output / "transcript.json").read_bytes())
        second_work = self.root / "second-work"
        second_work.mkdir()
        db = pipeline.connect()
        second_job_id = _job(db, second_output, second_work, summary_status="queued_force")
        db.close()
        ledger.attach_consumer("c" * 64, "d" * 64, second_output)
        ledger.close()
        with patch.object(pipeline, "process_summary", side_effect=AssertionError("duplicate paid dispatch")):
            self.assertFalse(pipeline.run_next_summary())
        row = pipeline.connect().execute("SELECT summary_status FROM jobs WHERE id=?", (job_id,)).fetchone()
        self.assertEqual(row["summary_status"], "queued_force")
        second = pipeline.connect().execute("SELECT summary_status FROM jobs WHERE id=?", (second_job_id,)).fetchone()
        self.assertEqual(second["summary_status"], "queued_force")

    def test_quality_pending_blocks_new_force_after_source_change_and_updates_display(self):
        db = pipeline.connect()
        job_id = _job(db, self.output, self.work, summary_status="queued_force", attempt_id="attempt-quality")
        db.close()
        original_sha = hashlib.sha256((self.output / "transcript.json").read_bytes()).hexdigest()
        ledger = Ledger(self.state / "summary_private")
        decision = ledger.reserve(semantic_key="q" * 64, source_sha256=original_sha,
            output_dir=self.output, credential_id="key-synthetic", credential_version=1,
            workspace_id="workspace-synthetic", max_cost_microusd=1000)
        self.assertEqual(decision.kind, "new")
        ledger.mark_submitting(decision.job_id)
        ledger.submission_result(decision.job_id, remote_id="batch_quality_synthetic")
        ledger.poll_result(decision.job_id, "completed", usage_cost_microusd=200)
        ledger.mark_quality_pending(decision.job_id)
        ledger.close()
        (self.output / "transcript.json").write_text('{"synthetic":true,"revision":2}\n', encoding="utf-8")

        with patch.object(pipeline, "process_summary", side_effect=AssertionError("duplicate paid dispatch")):
            self.assertFalse(pipeline.run_next_summary())
        db = pipeline.connect()
        row = db.execute("SELECT summary_status FROM jobs WHERE id=?", (job_id,)).fetchone()
        self.assertEqual(row["summary_status"], "queued_force")
        db.execute("UPDATE jobs SET summary_status='pending_batch',summary_stage='summary_pending_batch' WHERE id=?", (job_id,))
        db.commit()
        db.close()
        pipeline.write_json(self.output / "summary_luna_attempt.json", {"job_id": decision.job_id})
        pipeline.write_json(self.output / "summary_attempt.json", {"attempt_id": "attempt-quality"})
        with patch.object(pipeline, "run_command", side_effect=AssertionError("external call")):
            self.assertEqual(pipeline.reconcile_luna_summary_queue(), 1)
            self.assertEqual(pipeline.reconcile_luna_summary_queue(), 0)
        row = pipeline.connect().execute(
            "SELECT summary_status,summary_stage,summary_progress,summary_detail FROM jobs WHERE id=?",
            (job_id,)).fetchone()
        self.assertEqual((row["summary_status"], row["summary_stage"]),
                         ("pending_batch", "summary_quality_pending"))
        self.assertEqual(row["summary_progress"], 80)
        self.assertIn("проверяю", row["summary_detail"])

    def test_second_consumer_recovers_from_ledger_without_attempt_file(self):
        source_sha = hashlib.sha256((self.output / "transcript.json").read_bytes()).hexdigest()
        ledger = Ledger(self.state / "summary_private")
        first = ledger.reserve(semantic_key="c" * 64, source_sha256=source_sha,
            output_dir=self.root / "first-output", credential_id="key-synthetic",
            credential_version=1, workspace_id="workspace-synthetic", max_cost_microusd=1000)
        db = pipeline.connect()
        job_id = _job(db, self.output, self.work, summary_status="pending_batch", attempt_id="second-attempt")
        db.close()
        attached = ledger.attach_consumer("c" * 64, source_sha, self.output)
        self.assertEqual(attached["id"], first.job_id)
        ledger.mark_submitting(first.job_id)
        ledger.submission_result(first.job_id, remote_id="batch_synthetic")
        ledger.db.execute("UPDATE jobs SET status='accepted' WHERE id=?", (first.job_id,))
        ledger.close()
        self._seal_luna_generation(job_id=first.job_id, source_sha=source_sha,
                                   semantic_key="c" * 64, remote_batch_id="batch_synthetic")
        self.assertEqual(pipeline.reconcile_luna_summary_queue(), 1)
        row = pipeline.connect().execute("SELECT summary_status FROM jobs WHERE id=?", (job_id,)).fetchone()
        self.assertEqual(row["summary_status"], "done")

    def test_accepted_ledger_reconciles_stale_attempt_display_without_dispatch(self):
        db = pipeline.connect()
        job_id = _job(db, self.output, self.work, summary_status="submission_unknown", attempt_id="attempt-synthetic")
        db.close()
        source_sha = hashlib.sha256((self.output / "transcript.json").read_bytes()).hexdigest()
        ledger = Ledger(self.state / "summary_private")
        decision = ledger.reserve(semantic_key="c" * 64, source_sha256=source_sha,
                                  output_dir=self.output, credential_id="key-synthetic",
                                  credential_version=1, workspace_id="workspace-synthetic",
                                  max_cost_microusd=1000)
        ledger.mark_submitting(decision.job_id)
        ledger.submission_result(decision.job_id, remote_id="batch_synthetic")
        ledger.db.execute("UPDATE jobs SET status='accepted' WHERE id=?", (decision.job_id,))
        ledger.close()
        raw_attempt = {"status": "submission_unknown", "job_id": decision.job_id}
        pipeline.write_json(self.output / "summary_luna_attempt.json", raw_attempt)
        pipeline.write_json(self.output / "summary_attempt.json", {
            "schema_version": 2, "job_id": job_id, "attempt_id": "attempt-synthetic",
            "attempt_status": "submission_unknown", "luna_job_id": decision.job_id,
            "remote_batch_id": None, "displayed_generation_id": None,
        })
        generation, _ = self._seal_luna_generation(job_id=decision.job_id, source_sha=source_sha,
            semantic_key="c" * 64, remote_batch_id="batch_synthetic")

        with patch.object(pipeline, "run_command", side_effect=AssertionError("new external call")):
            self.assertEqual(pipeline.reconcile_luna_summary_queue(), 1)
            self.assertEqual(pipeline.reconcile_luna_summary_queue(), 0)
        row = pipeline.connect().execute("SELECT summary_status FROM jobs WHERE id=?", (job_id,)).fetchone()
        self.assertEqual(row["summary_status"], "done")
        display = pipeline.load_json(self.output / "summary_attempt.json")
        self.assertEqual(display["attempt_status"], "accepted")
        self.assertEqual(display["luna_job_id"], decision.job_id)
        self.assertEqual(display["remote_batch_id"], "batch_synthetic")
        self.assertEqual(display["displayed_generation_id"], generation)
        self.assertEqual(pipeline.load_json(self.output / "summary_luna_attempt.json"), raw_attempt)

        # A finished queue row can repair its mutable display directly; this
        # never reopens the queue or calls the provider.
        display["attempt_status"] = "submission_unknown"
        pipeline.write_json(self.output / "summary_attempt.json", display)
        ledger = Ledger(self.state / "summary_private")
        accepted = ledger.get(decision.job_id)
        ledger.close()
        with patch.object(pipeline, "run_command", side_effect=AssertionError("new external call")):
            self.assertTrue(pipeline.reconcile_luna_accepted_attempt(
                self.output, job_id, "attempt-synthetic", accepted))
        self.assertEqual(pipeline.load_json(self.output / "summary_attempt.json")["attempt_status"], "accepted")
        self.assertEqual(pipeline.connect().execute(
            "SELECT summary_status FROM jobs WHERE id=?", (job_id,)).fetchone()[0], "done")

    def test_accepted_identity_mismatch_stays_recoverable_without_dispatch(self):
        db = pipeline.connect()
        job_id = _job(db, self.output, self.work, summary_status="submission_unknown", attempt_id="attempt-synthetic")
        db.close()
        source = (self.output / "transcript.json").read_bytes()
        source_sha = hashlib.sha256(source).hexdigest()
        ledger = Ledger(self.state / "summary_private")
        decision = ledger.reserve(semantic_key="c" * 64, source_sha256=source_sha,
                                  output_dir=self.output, credential_id="key-synthetic",
                                  credential_version=1, workspace_id="workspace-synthetic",
                                  max_cost_microusd=1000)
        ledger.mark_submitting(decision.job_id)
        ledger.submission_result(decision.job_id, remote_id="batch_synthetic")
        ledger.db.execute("UPDATE jobs SET status='accepted' WHERE id=?", (decision.job_id,))
        accepted = ledger.get(decision.job_id)
        ledger.close()
        pipeline.write_json(self.output / "summary_luna_attempt.json", {
            "status": "submission_unknown", "job_id": decision.job_id,
        })
        stale = {"attempt_id": "attempt-synthetic", "attempt_status": "submission_unknown"}
        pipeline.write_json(self.output / "summary_attempt.json", stale)
        _, package = self._seal_luna_generation(job_id=decision.job_id, source_sha=source_sha,
            semantic_key="c" * 64, remote_batch_id="batch_synthetic")

        run_path = package / "run_manifest.json"
        valid_run = pipeline.load_json(run_path)
        wrong_run = dict(valid_run, semantic_key="f" * 64)
        pipeline.write_json(run_path, wrong_run)
        sealed_path = package / "generation_manifest.json"
        sealed = pipeline.load_json(sealed_path)
        sealed["artifact_sha256"]["run_manifest.json"] = hashlib.sha256(run_path.read_bytes()).hexdigest()
        pipeline.write_json(sealed_path, sealed)
        self.assertIsNotNone(pipeline.current_summary_output(self.output))
        with patch.object(pipeline, "run_command", side_effect=AssertionError("new external call")):
            self.assertEqual(pipeline.reconcile_luna_summary_queue(), 1)
        self.assertEqual(pipeline.connect().execute(
            "SELECT summary_status FROM jobs WHERE id=?", (job_id,)).fetchone()[0], "submission_unknown")
        self.assertEqual(pipeline.load_json(self.output / "summary_attempt.json"), stale)

        # Even a correct sealed run must not be accepted for changed source bytes.
        pipeline.write_json(run_path, valid_run)
        sealed["artifact_sha256"]["run_manifest.json"] = hashlib.sha256(run_path.read_bytes()).hexdigest()
        pipeline.write_json(sealed_path, sealed)
        (self.output / "transcript.json").write_bytes(source + b" ")
        self.assertIsNotNone(pipeline.current_summary_output(self.output))
        self.assertFalse(pipeline.reconcile_luna_accepted_attempt(
            self.output, job_id, "attempt-synthetic", accepted))
        self.assertEqual(pipeline.load_json(self.output / "summary_attempt.json"), stale)

    def test_watcher_restart_does_not_requeue_luna_summary(self):
        db = pipeline.connect()
        job_id = _job(db, self.output, self.work, summary_status="running", attempt_id="active")
        db.close()
        with patch.object(pipeline, "start_dashboard", return_value=None), \
             patch.object(pipeline, "scan_inbox"), \
             patch.object(pipeline, "run_next", return_value=False), \
             patch.object(pipeline, "run_next_summary", side_effect=AssertionError("duplicate dispatch")), \
             patch.object(pipeline.time, "sleep", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                pipeline.watch()
        row = pipeline.connect().execute("SELECT summary_status FROM jobs WHERE id=?", (job_id,)).fetchone()
        self.assertEqual(row["summary_status"], "running")

    def test_luna_reader_accepts_only_its_complete_versioned_manifest(self):
        generation = "20260925-000000-" + "e" * 12
        package = self.output / "summary_generations" / generation
        package.mkdir(parents=True)
        digests = {}
        for name in pipeline.LUNA_GENERATION_FILES:
            payload = name.encode()
            (package / name).write_bytes(payload)
            digests[name] = hashlib.sha256(payload).hexdigest()
        manifest = {"contract_version": "luna_summary_v1", "generation_id": generation,
                    "artifact_sha256": digests, "verified_artifact_sha256": digests["summary.md"]}
        pipeline.write_json(package / "generation_manifest.json", manifest)
        pipeline.write_json(self.output / "summary_current.json", {"generation_id": generation,
                              "verified_artifact_sha256": digests["summary.md"]})
        self.assertEqual(pipeline.current_summary_output(self.output), package)
        (package / "tasks.json").write_bytes(b"tampered")
        self.assertIsNone(pipeline.current_summary_output(self.output))
        (package / "tasks.json").write_bytes(b"tasks.json")
        manifest["contract_version"] = "unknown_future_schema"
        pipeline.write_json(package / "generation_manifest.json", manifest)
        self.assertIsNone(pipeline.current_summary_output(self.output))

    def test_legacy_reader_remains_valid_without_luna_contract(self):
        generation = "20260925-000000-" + "f" * 12
        package = self.output / "summary_generations" / generation
        package.mkdir(parents=True)
        digests = {}
        for name in pipeline.REQUIRED_GENERATION_FILES:
            payload = name.encode()
            target = package / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            digests[name] = hashlib.sha256(payload).hexdigest()
        pipeline.write_json(package / "generation_manifest.json", {"generation_id": generation,
                            "artifact_sha256": digests, "verified_artifact_sha256": digests["summary.md"]})
        pipeline.write_json(self.output / "summary_current.json", {"generation_id": generation,
                            "verified_artifact_sha256": digests["summary.md"]})
        self.assertEqual(pipeline.current_summary_output(self.output), package)

    def test_summary_scheduler_does_one_poll_and_no_speech(self):
        with patch.object(pipeline, "reconcile_luna_summary_queue", return_value=0) as reconcile, \
             patch.object(pipeline, "run_next_summary", return_value=False) as dispatch, \
             patch.object(pipeline, "run_command") as worker:
            self.assertFalse(pipeline.luna_scheduler_tick())
        self.assertEqual(reconcile.call_count, 2)
        dispatch.assert_called_once_with()
        worker.assert_called_once()
        self.assertEqual(worker.call_args.args[0][2], "poll")

    def test_summary_only_http_route_requires_admin_and_only_queues_summary(self):
        db = pipeline.connect(); job_id = _job(db, self.output, self.work, summary_status="done"); db.close()
        password = "synthetic-admin-password-for-tests"
        verifier = make_admin_password_hash(password)

        def request(authorized):
            handler = object.__new__(pipeline.DashboardHandler)
            handler.command = "POST"
            handler.path = "/api/summary?id={}".format(job_id)
            handler.request_version = "HTTP/1.1"
            handler.requestline = "POST " + handler.path + " HTTP/1.1"
            handler.client_address = ("127.0.0.1", 12345)
            handler.wfile = BytesIO(); handler.rfile = BytesIO()
            handler.headers = Message()
            handler.headers["Host"] = "127.0.0.1:8765"
            handler.headers["Origin"] = "http://127.0.0.1:8765"
            handler.headers["X-Requested-With"] = "TranscriSummaryzator-Admin"
            if authorized:
                handler.headers["Authorization"] = "Basic " + base64.b64encode(("admin:" + password).encode()).decode()
            handler.do_POST()
            return int(handler.wfile.getvalue().split(b"\r\n", 1)[0].split()[1])

        with patch.dict(os.environ, {"TRANSCRI_SUMMARY_ADMIN_PASSWORD_HASH": verifier,
                                  "TRANSCRI_SUMMARY_ADMIN_ORIGIN": "http://127.0.0.1:8765"}):
            self.assertEqual(request(False), 401)
            self.assertEqual(pipeline.connect().execute("SELECT summary_status FROM jobs WHERE id=?", (job_id,)).fetchone()[0], "done")
            self.assertEqual(request(True), 200)
        self.assertEqual(pipeline.connect().execute("SELECT summary_status FROM jobs WHERE id=?", (job_id,)).fetchone()[0], "queued_force")

    def test_summary_page_and_download_show_same_edited_card_and_real_anchor(self):
        self.output.joinpath("transcript.json").write_text(json.dumps({
            "source": "01.09.2026 — Test.mkv", "duration_seconds": 8,
            "speakers": {"p1": "А"},
            "utterances": [{"start": 0.4, "end": 7.5, "speaker": "p1",
                            "text": "Предлагаю проверить X или Y, не оба."}],
        }, ensure_ascii=False), encoding="utf-8")
        document = fake_document()
        _, source_index, source_sha = load_source(self.output / "transcript.json")
        store = TaskStore(self.state / "summary_private" / "tasks.sqlite3")
        plan = store.preview_reconcile(source_sha, document["tasks"])
        publish_document(document=document, source_index=source_index,
            transcript_path=self.output / "transcript.json", output_dir=self.output,
            semantic_key="synthetic", job_id="synthetic", remote_batch_id="batch_synthetic",
            credential_id="key-synthetic", prompt_sha256="a" * 64, schema_sha256="b" * 64,
            effective_tasks=plan.effective_tasks, before_pointer=lambda: store.commit_reconcile(plan))
        action_id = plan.effective_tasks[0]["action_id"]
        store.update(action_id, 0, {"assignee": "А", "description": "Проверить выбранный X либо Y."}, "admin")
        db = pipeline.connect(); job_id = _job(db, self.output, self.work, summary_status="done"); db.close()

        def get(path):
            handler = object.__new__(pipeline.DashboardHandler)
            handler.command = "GET"; handler.path = path
            handler.request_version = "HTTP/1.1"
            handler.requestline = "GET " + path + " HTTP/1.1"
            handler.client_address = ("127.0.0.1", 12345)
            handler.wfile = BytesIO(); handler.rfile = BytesIO()
            handler.headers = Message(); handler.headers["Host"] = "127.0.0.1:8765"
            handler.do_GET()
            response_headers, body = handler.wfile.getvalue().split(b"\r\n\r\n", 1)
            return int(response_headers.split(b" ")[1]), body.decode("utf-8")

        status, page = get("/summary?id={}".format(job_id))
        self.assertEqual(status, 200)
        self.assertIn('href="/result?id={}#t-400"'.format(job_id), page)
        self.assertIn("Проверить выбранный X либо Y.", page)
        self.assertIn("/summary-tasks?id={}".format(job_id), page)
        status, markdown = get("/download?id={}&file=summary.md".format(job_id))
        self.assertEqual(status, 200)
        self.assertIn("**Исполнитель:** А", markdown)
        self.assertIn("Проверить выбранный X либо Y.", markdown)
        status, html_summary = get("/download?id={}&file=summary.html".format(job_id))
        self.assertEqual(status, 200)
        self.assertIn("Проверить выбранный X либо Y.", html_summary)
        self.assertIn("<details>", html_summary)
        status, tasks_text = get("/download?id={}&file=tasks.json".format(job_id))
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(tasks_text)[0]["description"], "Проверить выбранный X либо Y.")
        status, transcript_html = get("/download?id={}&file=transcript.html".format(job_id))
        self.assertEqual(status, 200)
        self.assertIn('id="t-400"', transcript_html)


if __name__ == "__main__":
    unittest.main()
