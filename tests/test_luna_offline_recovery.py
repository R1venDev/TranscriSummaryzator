"""Saved-audit recovery uses only synthetic data and no HTTP client."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from summary.luna_v1 import PROMPT_PATH, SCHEMA, load_source
from summary.luna_v1.audit import AUDIT_PROMPT_PATH, AUDIT_SCHEMA, AUDIT_SCHEMA_ID
from summary.luna_v1.engine import _semantic_identity, submit
from summary.luna_v1.ledger import Ledger, write_private_json
from summary.luna_v1.recovery import RecoveryExpected, recover_saved_v3, recovery_code_sha256
from tests.test_luna_task_api import fake_document


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha(value: object) -> str:
    return sha(json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")).encode("utf-8"))


class SavedAuditRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.output = root / "output"
        self.output.mkdir()
        self.source = self.output / "transcript.json"
        self.source.write_text(json.dumps({
            "source": "01.09.2026 — Test.mkv", "duration_seconds": 8,
            "speakers": {"p1": "А"},
            "utterances": [{"start": 0.4, "end": 7.5, "speaker": "p1",
                            "text": "Предлагаю проверить X или Y, не оба."}],
        }, ensure_ascii=False), encoding="utf-8")
        self.private = root / "private"
        source_text, _, source_sha = load_source(self.source)
        self.source_sha = source_sha
        self.draft = fake_document()
        # The old validator rejected evidence for an intentionally null
        # optional field.  The corrected contract leaves this evidence intact.
        self.draft["tasks"][0]["field_sources"]["due"] = ["U00001"]
        replacement = {"text": "Обсудили выбор одного из X или Y; оба не нужны.",
                       "source_ids": ["U00001"]}
        self.report = {
            "schema_version": AUDIT_SCHEMA_ID,
            "coverage": [{"window_id": "W01", "start_id": "U00001", "end_id": "U00001",
                          "salient": "Предложение проверить один вариант.",
                          "draft_coverage": "partial", "finding_indices": [0, 1, 99]}],
            "findings": [
                {"severity": "major", "kind": "omission", "description": "Не указана альтернатива Y.",
                 "source_ids": ["U00001"], "affected": [], "status": "repaired", "patch_indices": [0]},
                {"severity": "minor", "kind": "time", "description": "Срок не определён.",
                 "source_ids": ["U00001"], "affected": [], "status": "unresolved", "patch_indices": []},
            ],
            "patches": [{"section": "main", "operation": "replace", "index": 0,
                         "item_json": json.dumps(replacement, ensure_ascii=False)}],
        }
        prompt_sha = sha(PROMPT_PATH.read_bytes())
        schema_sha = canonical_sha(SCHEMA)
        audit_prompt_sha = sha(AUDIT_PROMPT_PATH.read_bytes())
        audit_schema_sha = canonical_sha(AUDIT_SCHEMA)
        semantic_key = _semantic_identity(source_sha, source_text, "synthetic-workspace",
                                          prompt_sha, schema_sha, None)
        ledger = Ledger(self.private)
        try:
            decision = ledger.reserve(
                semantic_key=semantic_key, source_sha256=source_sha, output_dir=self.output,
                credential_id="fake-credential", credential_version=1,
                workspace_id="synthetic-workspace", max_cost_microusd=20_000,
            )
            self.root_id = decision.job_id
            root_row = ledger.get(self.root_id)
            audit_decision = ledger.reserve(
                semantic_key="a" * 64, source_sha256=source_sha, output_dir=self.output,
                credential_id="fake-credential", credential_version=1,
                workspace_id="synthetic-workspace", max_cost_microusd=20_000,
                kind="audit", root_job_id=self.root_id,
            )
            self.audit_id = audit_decision.job_id
            audit_row = ledger.get(self.audit_id)
            ledger.db.execute(
                "UPDATE jobs SET status='failed_validation',remote_id=?,dispatches=1,"
                "billed_microusd=4000,error_code=? WHERE id=?",
                ("batch-writer-synthetic", "ValueError:tasks[0]: due and its source references disagree",
                 self.root_id),
            )
            ledger.db.execute(
                "UPDATE jobs SET status='stage_complete',remote_id=?,dispatches=1,"
                "billed_microusd=3000,accepted_document_path=? WHERE id=?",
                ("batch-audit-synthetic", str(Path(audit_row["artifact_dir"]) / "audit_report.json"),
                 self.audit_id),
            )
        finally:
            ledger.close()
        self.root_artifacts = self.private / "jobs" / self.root_id
        self.audit_artifacts = self.private / "jobs" / self.audit_id
        write_private_json(self.root_artifacts / "manifest.json", {
            "job_id": self.root_id, "source_sha256": source_sha,
            "quality_policy_version": "luna_auto_audit_v3",
            "prompt_sha256": prompt_sha, "schema_sha256": schema_sha,
            "audit_prompt_sha256": audit_prompt_sha,
            "audit_schema_sha256": audit_schema_sha,
        })
        write_private_json(self.root_artifacts / "draft_document.json", self.draft)
        write_private_json(self.root_artifacts / "candidate_document.json", self.draft)
        write_private_json(self.root_artifacts / "native_response.json", {
            "text": json.dumps(self.draft, ensure_ascii=False), "usage": {},
        })
        write_private_json(self.root_artifacts / "batch_terminal.json", {
            "id": "batch-writer-synthetic", "status": "completed",
        })
        write_private_json(self.root_artifacts / "quality_decision.json", {
            "quality_review": {"status": "audit_unavailable", "reason": "invalid_audit_patch",
                               "unresolved_count": 0, "coverage_warning_count": 1,
                               "audit_job_id": self.audit_id, "verify_job_id": None},
        })
        write_private_json(self.root_artifacts / "audit_apply_error.json", {
            "reason": "tasks[0]: due and its source references disagree",
        })
        write_private_json(self.audit_artifacts / "audit_report.json", self.report)
        write_private_json(self.audit_artifacts / "manifest.json", {
            "root_job_id": self.root_id, "source_sha256": source_sha,
            "target_document_sha256": canonical_sha(self.draft),
            "prompt_sha256": audit_prompt_sha, "schema_sha256": audit_schema_sha,
        })
        write_private_json(self.audit_artifacts / "native_response.json", {
            "text": json.dumps(self.report, ensure_ascii=False), "usage": {},
        })
        write_private_json(self.audit_artifacts / "batch_terminal.json", {
            "id": "batch-audit-synthetic", "status": "completed",
        })
        write_private_json(self.audit_artifacts / "coverage_warnings.json", {
            "warnings": [{"code": "unknown_finding_index", "coverage_index": 0,
                          "finding_index": 99}],
        })
        self.expected = RecoveryExpected(
            source_sha256=source_sha,
            root_manifest_sha256=sha((self.root_artifacts / "manifest.json").read_bytes()),
            draft_sha256=sha((self.root_artifacts / "draft_document.json").read_bytes()),
            audit_sha256=sha((self.audit_artifacts / "audit_report.json").read_bytes()),
            code_sha256=recovery_code_sha256(),
            error_code="ValueError:tasks[0]: due and its source references disagree",
        )

    def _recover(self, *, apply: bool):
        return recover_saved_v3(private_root=self.private, root_job_id=self.root_id,
                                audit_job_id=self.audit_id, expected=self.expected, apply=apply)

    def test_preserves_failed_artifacts_and_reuses_accepted_result_without_http(self):
        original = {str(path): path.read_bytes() for path in (
            self.root_artifacts / "draft_document.json",
            self.root_artifacts / "candidate_document.json",
            self.root_artifacts / "native_response.json",
            self.root_artifacts / "audit_apply_error.json",
            self.audit_artifacts / "audit_report.json",
        )}
        self.assertEqual(self._recover(apply=False)["status"], "ready_offline")
        self.assertFalse((self.output / "summary_current.json").exists())
        result = self._recover(apply=True)
        self.assertEqual(result["status"], "accepted_recovered")
        self.assertEqual(result["new_api_dispatches"], 0)
        for name, data in original.items():
            self.assertEqual(Path(name).read_bytes(), data)
        self.assertEqual(json.loads((self.root_artifacts / "quality_decision_before_recovery.json").read_text())
                         ["quality_review"]["status"], "audit_unavailable")
        pointer = json.loads((self.output / "summary_current.json").read_text())
        generation = self.output / "summary_generations" / pointer["generation_id"]
        sealed = json.loads((generation / "model_document.json").read_text())
        self.assertEqual(sealed["tasks"][0]["field_sources"]["due"], ["U00001"])
        self.assertIsNone(sealed["tasks"][0]["due"])
        self.assertEqual(len(sealed["verification"]), 1)
        review = json.loads((generation / "run_manifest.json").read_text())["quality_review"]
        self.assertEqual(review["status"], "audit_recovered_unverified")
        self.assertEqual((review["unresolved_count"], review["coverage_warning_count"]), (1, 1))
        ledger = Ledger(self.private)
        try:
            root = ledger.get(self.root_id)
            audit = ledger.get(self.audit_id)
            self.assertEqual(root["status"], "accepted")
            self.assertEqual(audit["status"], "stage_complete")
            self.assertEqual((root["dispatches"], audit["dispatches"]), (1, 1))
            self.assertEqual((root["billed_microusd"], audit["billed_microusd"]), (4000, 3000))
            self.assertEqual(ledger.consumer_rows(root["semantic_key"])[0]["status"], "published")
        finally:
            ledger.close()
        self.assertEqual(self._recover(apply=True)["status"], "already_recovered")
        self.assertEqual(len(list((self.output / "summary_generations").iterdir())), 1)

        class FakeStore:
            path = self.private / "credentials.sqlite3"

            def dispatch_candidates(self):
                return [{"id": "fake-credential", "version": 1,
                         "workspace_id": "synthetic-workspace"}]

        def forbidden_client(_token):
            raise AssertionError("offline reuse attempted HTTP")

        with patch("summary.luna_v1.engine._credential_store", return_value=FakeStore()):
            cached = submit(transcript_path=self.source, output_dir=self.output,
                            private_root=self.private, client_factory=forbidden_client)
        self.assertEqual(cached["status"], "accepted_cache_hit")
        self.assertEqual(cached["new_generations"], 0)

    def test_pointer_before_ledger_cas_recovers_idempotently(self):
        with patch.object(Ledger, "accept_recovered_failed", side_effect=RuntimeError("simulated ledger crash")):
            with self.assertRaisesRegex(RuntimeError, "simulated ledger crash"):
                self._recover(apply=True)
        self.assertTrue((self.output / "summary_current.json").exists())
        ledger = Ledger(self.private)
        try:
            self.assertEqual(ledger.get(self.root_id)["status"], "failed_validation")
        finally:
            ledger.close()
        self.assertEqual(self._recover(apply=True)["status"], "accepted_recovered")
        self.assertEqual(len(list((self.output / "summary_generations").iterdir())), 1)

    def test_changed_saved_audit_blocks_publication(self):
        self.expected = RecoveryExpected(**{**self.expected.__dict__, "audit_sha256": "0" * 64})
        with self.assertRaisesRegex(ValueError, "audit SHA-256 changed"):
            self._recover(apply=True)
        self.assertFalse((self.output / "summary_current.json").exists())

    def test_foreign_pointer_arriving_after_preflight_cannot_be_replaced(self):
        from summary.luna_v1 import recovery

        foreign_id = "20260925-000000-" + "f" * 12
        foreign_pointer = {"generation_id": foreign_id,
                           "verified_artifact_sha256": "e" * 64}
        real_publish = recovery.publish_document

        def race_before_publication(**kwargs):
            (self.output / "summary_current.json").write_text(
                json.dumps(foreign_pointer), encoding="utf-8")
            return real_publish(**kwargs)

        with patch.object(recovery, "publish_document", side_effect=race_before_publication):
            with self.assertRaisesRegex(ValueError, "different generation was selected before recovery commit"):
                self._recover(apply=True)
        self.assertEqual(json.loads((self.output / "summary_current.json").read_text()), foreign_pointer)
        with closing(sqlite3.connect(self.private / "tasks.sqlite3")) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM task_identity").fetchone()[0], 0)
        ledger = Ledger(self.private)
        try:
            self.assertEqual(ledger.get(self.root_id)["status"], "failed_validation")
        finally:
            ledger.close()


if __name__ == "__main__":
    unittest.main()
