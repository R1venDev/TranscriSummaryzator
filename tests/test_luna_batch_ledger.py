import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from summary.luna_v1.batch import BatchClient, extract_one_completed, valid_batch_id
from summary.luna_v1.ledger import Ledger


class _Response:
    status = 202

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, _cap):
        return b'{"id":"batch_abc123","status":"validating"}'


class _Opener:
    def __init__(self):
        self.calls = []

    def open(self, request, timeout):
        self.calls.append((request, timeout))
        return _Response()


class LedgerTests(unittest.TestCase):
    @staticmethod
    def _accepted_inventory_root(ledger, root, *, quality_status="inventory_unavailable",
                                 cost=50_000, dynamic=False):
        base = dict(source_sha256="b" * 64, output_dir=root / "meeting",
                    credential_id="key-1", credential_version=1,
                    workspace_id="workspace-1")
        writer = ledger.reserve(semantic_key="a" * 64, max_cost_microusd=90_000,
            dynamic_group_authorization_ref=("user-20260927-inventory-continuation"
                if dynamic else None), **base)
        assert writer.kind == "new"
        assert ledger.mark_submitting(writer.job_id)
        ledger.submission_result(writer.job_id, remote_id="batch_accepted")
        ledger.poll_result(writer.job_id, "completed", usage_cost_microusd=cost)
        artifacts = Path(ledger.get(writer.job_id)["artifact_dir"])
        (artifacts / "draft_document.json").write_text("{}", encoding="utf-8")
        ledger.mark_quality_pending(writer.job_id)
        review = {"status": quality_status, "reason": "inventory_3:failed_validation"}
        (artifacts / "quality_decision.json").write_text(
            json.dumps({"quality_review": review}), encoding="utf-8")
        document = artifacts / "candidate_document.json"
        document.write_text("{}", encoding="utf-8")
        generation_id = "20260927-120000-" + writer.job_id[:12]
        generation = base["output_dir"] / "summary_generations" / generation_id
        generation.mkdir(parents=True)
        run_path = generation / "run_manifest.json"
        run_path.write_text(json.dumps({"job_id": writer.job_id,
            "source_sha256": base["source_sha256"], "quality_review": review}),
            encoding="utf-8")
        (generation / "generation_manifest.json").write_text(json.dumps({
            "generation_id": generation_id, "source_sha256": base["source_sha256"],
            "artifact_sha256": {"run_manifest.json": hashlib.sha256(
                run_path.read_bytes()).hexdigest()},
        }), encoding="utf-8")
        ledger.accepted(writer.job_id, document, generation_id)
        ledger.mark_consumer("a" * 64, base["output_dir"], generation_id=generation_id)
        return writer, base

    @staticmethod
    def _failed_writer(ledger, root, *, key="a", cost=50_000):
        base = dict(source_sha256="b" * 64, output_dir=root / "meeting",
                    credential_id="key-1", credential_version=1,
                    workspace_id="workspace-1")
        writer = ledger.reserve(semantic_key=key * 64,
                                max_cost_microusd=90_000, **base)
        assert writer.kind == "new"
        assert ledger.mark_submitting(writer.job_id)
        ledger.submission_result(writer.job_id, remote_id="batch_" + key * 8)
        ledger.poll_result(writer.job_id, "completed", usage_cost_microusd=cost)
        ledger.failed_validation(writer.job_id, "saved_writer_test_failure")
        (Path(ledger.get(writer.job_id)["artifact_dir"]) / "draft_document.json").write_text(
            "{}", encoding="utf-8")
        return writer, base

    @staticmethod
    def _ready_continuation(ledger, writer, base):
        continuation = ledger.reserve(semantic_key="c" * 64,
            max_cost_microusd=0, continuation_of_job_id=writer.job_id, **base)
        assert continuation.kind == "new"
        artifacts = Path(ledger.get(continuation.job_id)["artifact_dir"])
        (artifacts / "draft_document.json").write_text("{}", encoding="utf-8")
        (artifacts / "manifest.json").write_text("{}", encoding="utf-8")
        assert ledger.mark_continuation_ready(continuation.job_id)
        return continuation

    def test_remote_batch_id_accepts_live_and_documented_forms(self):
        self.assertTrue(valid_batch_id("batch-1790364440-Gq1lAnlrV1xmZGmroHeB"))
        self.assertTrue(valid_batch_id("batch_abc123"))
        self.assertFalse(valid_batch_id("batch-../../other"))
        self.assertFalse(valid_batch_id("batch_"))

    def test_single_flight_restart_and_unknown_charge_hold_weekly_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = dict(semantic_key="a" * 64, source_sha256="b" * 64,
                        credential_id="key-1", credential_version=1,
                        workspace_id="workspace-1", max_cost_microusd=90_000)
            one = Ledger(root)
            first = one.reserve(output_dir=root / "meeting-one", **args)
            self.assertEqual(first.kind, "new")
            second = one.reserve(output_dir=root / "meeting-two", **args)
            self.assertEqual(second.kind, "pending")
            self.assertEqual(second.job_id, first.job_id)
            self.assertEqual(len(one.consumers("a" * 64)), 2)
            self.assertTrue(one.mark_submitting(first.job_id))
            self.assertFalse(one.mark_submitting(first.job_id))
            one.submission_result(first.job_id, remote_id=None, error_code="transport_unknown")
            one.close()
            restarted = Ledger(root)
            self.assertEqual(restarted.get(first.job_id)["status"], "submission_unknown")
            self.assertFalse(restarted.mark_submitting(first.job_id))
            # Eleven such unknown submissions would exceed the one shared USD,
            # even if each uses another key and output directory.
            for index in range(1, 11):
                decision = restarted.reserve(
                    semantic_key=f"{index:064x}", source_sha256="b" * 64,
                    output_dir=root / f"meeting-{index}", credential_id=f"key-{index}",
                    credential_version=1, workspace_id="workspace-1",
                    max_cost_microusd=90_000,
                )
                self.assertEqual(decision.kind, "new")
            blocked = restarted.reserve(
                semantic_key="f" * 64, source_sha256="b" * 64,
                output_dir=root / "over", credential_id="key-12", credential_version=1,
                workspace_id="workspace-1", max_cost_microusd=90_000,
            )
            self.assertEqual(blocked.reason, "weekly_budget_exceeded")

    def test_quality_calls_share_one_logical_job_cap_and_do_not_create_consumers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = Ledger(root)
            base = dict(source_sha256="b" * 64, output_dir=root / "meeting",
                        credential_id="key-1", credential_version=1,
                        workspace_id="workspace-1")
            parent = ledger.reserve(semantic_key="a" * 64,
                                    max_cost_microusd=50_000, **base)
            audit = ledger.reserve(semantic_key="c" * 64,
                                   max_cost_microusd=30_000, kind="audit",
                                   root_job_id=parent.job_id, **base)
            self.assertEqual(audit.kind, "new")
            self.assertEqual(ledger.consumers("c" * 64), [])
            blocked = ledger.reserve(semantic_key="d" * 64,
                                     max_cost_microusd=30_000, kind="verify",
                                     root_job_id=parent.job_id, **base)
            self.assertEqual(blocked.reason, "logical_job_budget_exceeded")
            ledger.db.execute("UPDATE jobs SET billed_microusd=10000 WHERE id=?", (parent.job_id,))
            allowed = ledger.reserve(semantic_key="d" * 64,
                                     max_cost_microusd=30_000, kind="verify",
                                     root_job_id=parent.job_id, **base)
            self.assertEqual(allowed.kind, "new")
            ledger.close()

    def test_six_dispatches_are_shared_by_writer_and_all_quality_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = Ledger(root)
            base = dict(source_sha256="b" * 64, output_dir=root / "meeting",
                        credential_id="key-1", credential_version=1,
                        workspace_id="workspace-1", max_cost_microusd=1_000)
            parent = ledger.reserve(semantic_key="a" * 64, **base)
            self.assertTrue(ledger.mark_submitting(parent.job_id))
            for number, kind in enumerate(("inventory_1", "inventory_2",
                                           "inventory_3", "reconcile", "verify"), 1):
                child = ledger.reserve(semantic_key=f"{number:064x}", kind=kind,
                                       root_job_id=parent.job_id, **base)
                self.assertEqual(child.kind, "new")
                self.assertTrue(ledger.mark_submitting(child.job_id))
            blocked = ledger.reserve(semantic_key="f" * 64, kind="audit",
                                     root_job_id=parent.job_id, **base)
            self.assertEqual(blocked.reason, "logical_job_dispatch_limit")
            self.assertEqual(ledger.db.execute(
                "SELECT SUM(dispatches) FROM jobs WHERE root_job_id=?",
                (parent.job_id,)).fetchone()[0], 6)
            ledger.close()

    def test_saved_writer_continuation_keeps_original_cost_and_stage_namespace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = Ledger(root)
            base = dict(source_sha256="b" * 64, output_dir=root / "meeting",
                        credential_id="key-1", credential_version=1,
                        workspace_id="workspace-1")
            writer = ledger.reserve(semantic_key="a" * 64,
                                    max_cost_microusd=50_000, **base)
            self.assertTrue(ledger.mark_submitting(writer.job_id))
            ledger.submission_result(writer.job_id, remote_id="batch_writer")
            ledger.poll_result(writer.job_id, "completed", usage_cost_microusd=4_000)
            writer_artifacts = Path(ledger.get(writer.job_id)["artifact_dir"])
            (writer_artifacts / "draft_document.json").write_text("{}", encoding="utf-8")
            ledger.mark_quality_pending(writer.job_id)
            old_segment = ledger.reserve(semantic_key="c" * 64,
                                         max_cost_microusd=40_000, kind="segment_1",
                                         root_job_id=writer.job_id, **base)
            self.assertTrue(ledger.mark_submitting(old_segment.job_id))
            ledger.submission_result(old_segment.job_id, remote_id="batch_segment")
            ledger.failed_quality(writer.job_id, "segment_1:pending")
            inflight = ledger.reserve(semantic_key="d" * 64,
                                      max_cost_microusd=0,
                                      continuation_of_job_id=writer.job_id, **base)
            self.assertEqual(inflight.reason, "continuation_group_inflight")
            ledger.poll_result(old_segment.job_id, "completed", usage_cost_microusd=21_000)
            ledger.failed_validation(old_segment.job_id, "MAX_TOKENS")

            next_root = ledger.reserve(semantic_key="d" * 64,
                                       max_cost_microusd=0,
                                       continuation_of_job_id=writer.job_id, **base)
            self.assertEqual(next_root.kind, "new")
            row = ledger.get(next_root.job_id)
            self.assertEqual(row["status"], "preparing")
            self.assertEqual(row["writer_parent_id"], writer.job_id)
            self.assertEqual(row["billing_group_id"], writer.job_id)
            self.assertEqual(row["root_job_id"], next_root.job_id)
            self.assertIsNone(row["remote_id"])
            self.assertEqual(row["dispatches"], 0)
            self.assertFalse(ledger.mark_submitting(next_root.job_id))
            self.assertNotIn(next_root.job_id,
                             [item["id"] for item in ledger.quality_pending()])
            self.assertIsNone(ledger.stage(next_root.job_id, "segment_1"))
            self.assertEqual(ledger.reserve(semantic_key="d" * 64,
                                            max_cost_microusd=0,
                                            continuation_of_job_id=writer.job_id,
                                            **base).job_id, next_root.job_id)
            blocked_unsealed = ledger.reserve(semantic_key="e" * 64,
                                             max_cost_microusd=75_000, kind="segment_1",
                                             root_job_id=next_root.job_id, **base)
            self.assertEqual(blocked_unsealed.reason, "continuation_not_ready")
            with self.assertRaisesRegex(ValueError, "artifacts not sealed"):
                ledger.mark_continuation_ready(next_root.job_id)
            next_artifacts = Path(row["artifact_dir"])
            (next_artifacts / "draft_document.json").write_text("{}", encoding="utf-8")
            (next_artifacts / "manifest.json").write_text("{}", encoding="utf-8")
            self.assertTrue(ledger.mark_continuation_ready(next_root.job_id))
            self.assertFalse(ledger.mark_continuation_ready(next_root.job_id))
            self.assertIn(next_root.job_id,
                          [item["id"] for item in ledger.quality_pending()])

            remaining = ledger.reserve(semantic_key="e" * 64,
                                       max_cost_microusd=75_000, kind="segment_1",
                                       root_job_id=next_root.job_id, **base)
            self.assertEqual(remaining.kind, "new")
            self.assertEqual(ledger.get(remaining.job_id)["billing_group_id"], writer.job_id)
            self.assertTrue(ledger.mark_submitting(remaining.job_id))
            blocked = ledger.reserve(semantic_key="f" * 64,
                                     max_cost_microusd=1, kind="segment_2",
                                     root_job_id=next_root.job_id, **base)
            self.assertEqual(blocked.reason, "logical_job_budget_exceeded")
            self.assertEqual(ledger.db.execute(
                "SELECT SUM(dispatches) FROM jobs WHERE billing_group_id=?",
                (writer.job_id,)).fetchone()[0], 3)
            ledger.close()

    def test_saved_writer_continuation_requires_terminal_root_and_draft(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = Ledger(root)
            base = dict(source_sha256="b" * 64, output_dir=root / "meeting",
                        credential_id="key-1", credential_version=1,
                        workspace_id="workspace-1")
            writer = ledger.reserve(semantic_key="a" * 64,
                                    max_cost_microusd=10_000, **base)
            args = dict(semantic_key="d" * 64, max_cost_microusd=0,
                        continuation_of_job_id=writer.job_id, **base)
            with self.assertRaisesRegex(ValueError, "terminal failed summary"):
                ledger.reserve(**args)
            self.assertEqual(ledger.db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 1)
            ledger.cancel_before_submit(writer.job_id, "preflight")
            ledger.db.execute("UPDATE jobs SET status='failed_validation' WHERE id=?", (writer.job_id,))
            with self.assertRaisesRegex(ValueError, "sealed writer draft"):
                ledger.reserve(**args)
            (Path(ledger.get(writer.job_id)["artifact_dir"]) / "draft_document.json").write_text(
                "{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "credential scope mismatch"):
                ledger.reserve(**{**args, "credential_id": "different-key"})
            self.assertEqual(ledger.reserve(**args).kind, "new")
            ledger.close()

    def test_accepted_inventory_unavailable_continuation_keeps_published_root_and_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = Ledger(root)
            writer, base = self._accepted_inventory_root(ledger, root, dynamic=True)
            old = ledger.get(writer.job_id)
            old_consumer = ledger.consumer_rows("a" * 64)
            args = dict(semantic_key="c" * 64, max_cost_microusd=0,
                continuation_of_job_id=writer.job_id,
                continuation_mode="accepted_inventory_unavailable", **base)
            decision = ledger.reserve(**args)
            self.assertEqual(decision.kind, "new")
            continuation = ledger.get(decision.job_id)
            self.assertEqual(continuation["status"], "preparing")
            self.assertEqual(continuation["writer_parent_id"], writer.job_id)
            self.assertEqual(continuation["billing_group_id"], writer.job_id)
            self.assertEqual(continuation["root_job_id"], decision.job_id)
            self.assertEqual(continuation["dispatches"], 0)
            self.assertIsNone(continuation["remote_id"])
            self.assertFalse(ledger.mark_submitting(decision.job_id))
            self.assertEqual(ledger.reserve(**args).job_id, decision.job_id)
            self.assertEqual(ledger.get(writer.job_id), old)
            self.assertEqual(ledger.consumer_rows("a" * 64), old_consumer)
            self.assertEqual(ledger.logical_group_cap_microusd(writer.job_id), None)
            ledger.close()

    def test_accepted_continuation_rejects_other_decisions_and_mismatched_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = Ledger(root)
            writer, base = self._accepted_inventory_root(
                ledger, root, quality_status="checked")
            args = dict(semantic_key="c" * 64, max_cost_microusd=0,
                continuation_of_job_id=writer.job_id,
                continuation_mode="accepted_inventory_unavailable", **base)
            with self.assertRaisesRegex(ValueError, "sealed inventory_unavailable"):
                ledger.reserve(**args)
            quality = Path(ledger.get(writer.job_id)["artifact_dir"]) / "quality_decision.json"
            quality.write_text(json.dumps({"quality_review": {
                "status": "inventory_unavailable", "reason": "inventory_3:failed_validation"}}),
                encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sealed inventory_unavailable"):
                ledger.reserve(**args)  # published generation still says checked
            with self.assertRaisesRegex(ValueError, "matching summary"):
                ledger.reserve(**{**args, "source_sha256": "d" * 64})
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                ledger.reserve(**{**args, "credential_id": "other-key"})
            with self.assertRaisesRegex(ValueError, "invalid continuation mode"):
                ledger.reserve(**{**args, "continuation_of_job_id": None})
            with self.assertRaisesRegex(ValueError, "terminal failed summary"):
                ledger.reserve(**{**args, "continuation_mode": "failed_writer"})
            self.assertEqual(ledger.db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 1)
            ledger.close()

    def test_accepted_inventory_continuation_inherits_group_spending_and_dispatch_caps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = Ledger(root)
            writer, base = self._accepted_inventory_root(ledger, root, cost=90_000)
            decision = ledger.reserve(semantic_key="c" * 64, max_cost_microusd=0,
                continuation_of_job_id=writer.job_id,
                continuation_mode="accepted_inventory_unavailable", **base)
            self.assertEqual(decision.kind, "new")
            artifacts = Path(ledger.get(decision.job_id)["artifact_dir"])
            (artifacts / "draft_document.json").write_text("{}", encoding="utf-8")
            (artifacts / "manifest.json").write_text("{}", encoding="utf-8")
            self.assertTrue(ledger.mark_continuation_ready(decision.job_id))
            blocked = ledger.reserve(semantic_key="d" * 64, kind="reconcile_1",
                root_job_id=decision.job_id, max_cost_microusd=10_001, **base)
            self.assertEqual(blocked.reason, "logical_job_budget_exceeded")
            allowed = ledger.reserve(semantic_key="d" * 64, kind="reconcile_1",
                root_job_id=decision.job_id, max_cost_microusd=10_000, **base)
            self.assertEqual(allowed.kind, "new")
            self.assertEqual(ledger.get(allowed.job_id)["billing_group_id"], writer.job_id)
            ledger.db.execute("UPDATE jobs SET dispatches=6 WHERE id=?", (writer.job_id,))
            self.assertFalse(ledger.mark_submitting(allowed.job_id))
            ledger.cancel_before_submit(allowed.job_id, "test_release_reservation")
            self.assertEqual(ledger.reserve(semantic_key="e" * 64, kind="reconcile_2",
                root_job_id=decision.job_id, max_cost_microusd=1, **base).reason,
                "logical_job_dispatch_limit")
            ledger.close()

    def test_accepted_inventory_continuation_requires_publication_and_idle_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = Ledger(root)
            writer, base = self._accepted_inventory_root(ledger, root)
            args = dict(semantic_key="c" * 64, max_cost_microusd=0,
                continuation_of_job_id=writer.job_id,
                continuation_mode="accepted_inventory_unavailable", **base)
            stage = ledger.reserve(semantic_key="d" * 64, kind="inventory_3",
                root_job_id=writer.job_id, max_cost_microusd=1_000, **base)
            self.assertEqual(stage.kind, "new")
            self.assertEqual(ledger.reserve(**args).reason, "continuation_group_inflight")
            ledger.cancel_before_submit(stage.job_id, "test_idle_group")
            document = Path(ledger.get(writer.job_id)["accepted_document_path"])
            document.unlink()
            with self.assertRaisesRegex(ValueError, "publication unavailable"):
                ledger.reserve(**args)
            document.write_text("{}", encoding="utf-8")
            generation = base["output_dir"] / "summary_generations" / ledger.get(writer.job_id)["generation_id"]
            (generation / "generation_manifest.json").unlink()
            with self.assertRaisesRegex(ValueError, "publication unavailable"):
                ledger.reserve(**args)
            self.assertEqual(ledger.db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 2)
            ledger.close()

    def test_dynamic_spending_is_explicit_durable_and_scoped_to_one_terminal_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = Ledger(root)
            writer, base = self._failed_writer(ledger, root, cost=50_000)
            self.assertEqual(ledger.logical_group_cap_microusd(writer.job_id), 100_000)
            with self.assertRaisesRegex(ValueError, "authorization reference"):
                ledger.authorize_dynamic_group(writer.job_id, "user approved in chat")
            with self.assertRaisesRegex(ValueError, "terminal failed original writer"):
                ledger.authorize_dynamic_group("nonexistent", "user-message-20260927")
            self.assertTrue(ledger.authorize_dynamic_group(writer.job_id,
                "user-message-20260927-dynamic-full-source"))
            self.assertFalse(ledger.authorize_dynamic_group(writer.job_id,
                "user-message-20260927-dynamic-full-source"))
            with self.assertRaisesRegex(ValueError, "different dynamic spending authorization"):
                ledger.authorize_dynamic_group(writer.job_id, "different-user-decision-1")
            row = ledger.db.execute("SELECT * FROM group_spending_authorizations").fetchone()
            self.assertEqual(row["original_root_id"], writer.job_id)
            self.assertEqual(row["authorization_ref"],
                             "user-message-20260927-dynamic-full-source")
            ledger.close()

            ledger = Ledger(root)  # the specific authorization survives restart
            self.assertIsNone(ledger.logical_group_cap_microusd(writer.job_id))
            continuation = self._ready_continuation(ledger, writer, base)
            self.assertFalse(ledger.authorize_dynamic_group(writer.job_id,
                "user-message-20260927-dynamic-full-source"))
            stage = ledger.reserve(semantic_key="d" * 64, kind="segment_1",
                root_job_id=continuation.job_id, max_cost_microusd=90_000, **base)
            self.assertEqual(stage.kind, "new")  # $0.05 billed + $0.09 reserve
            self.assertEqual(ledger.get(stage.job_id)["billing_group_id"], writer.job_id)
            self.assertEqual(ledger.logical_group_cap_microusd(continuation.job_id), 100_000)
            with self.assertRaisesRegex(ValueError, "invalid quality root or workspace"):
                ledger.reserve(semantic_key="e" * 64, kind="segment_2",
                    root_job_id=continuation.job_id, max_cost_microusd=1_000,
                    **{**base, "workspace_id": "unapproved-workspace"})
            self.assertEqual(ledger.reserve(semantic_key="f" * 64,
                max_cost_microusd=90_000, **{**base, "output_dir": root / "other"}).kind,
                "new")
            separate = ledger.db.execute("SELECT id FROM jobs WHERE semantic_key=?",
                                         ("f" * 64,)).fetchone()[0]
            self.assertEqual(ledger.logical_group_cap_microusd(separate), 100_000)
            blocked = ledger.reserve(semantic_key="1" * 64, kind="audit",
                root_job_id=separate, max_cost_microusd=20_000,
                **{**base, "output_dir": root / "other"})
            self.assertEqual(blocked.reason, "logical_job_budget_exceeded")
            ledger.close()

    def test_new_evidence_trial_has_explicit_dynamic_cap_and_shared_weekly_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = Ledger(root)
            base = dict(source_sha256="b" * 64, output_dir=root / "meeting",
                        credential_id="key-1", credential_version=1,
                        workspace_id="workspace-1")
            with self.assertRaisesRegex(ValueError, "new summary root"):
                ledger.reserve(semantic_key="a" * 64, max_cost_microusd=1_000,
                    kind="audit", root_job_id="missing", **base,
                    dynamic_group_authorization_ref="user-20260927-try-evidence")
            writer = ledger.reserve(semantic_key="a" * 64,
                max_cost_microusd=30_000, **base,
                dynamic_group_authorization_ref="user-20260927-try-evidence")
            self.assertEqual(writer.kind, "new")
            self.assertIsNone(ledger.logical_group_cap_microusd(writer.job_id))
            self.assertEqual(ledger.db.execute(
                "SELECT authorization_ref FROM group_spending_authorizations WHERE billing_group_id=?",
                (writer.job_id,)).fetchone()[0], "user-20260927-try-evidence")
            stage = ledger.reserve(semantic_key="c" * 64,
                max_cost_microusd=90_000, kind="inventory_1",
                root_job_id=writer.job_id, **base)
            self.assertEqual(stage.kind, "new")
            self.assertEqual(ledger.get(stage.job_id)["billing_group_id"], writer.job_id)
            self.assertEqual(ledger.reserve(semantic_key="d" * 64,
                max_cost_microusd=100_001, kind="inventory_2",
                root_job_id=writer.job_id, **base).reason, "job_budget_exceeded")
            ledger.close()

    def test_dynamic_spending_keeps_weekly_per_dispatch_and_six_call_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = Ledger(root)
            writer, base = self._failed_writer(ledger, root, cost=50_000)
            self.assertTrue(ledger.authorize_dynamic_group(writer.job_id,
                "user-message-20260927-dynamic-full-source"))
            continuation = self._ready_continuation(ledger, writer, base)
            per_call = ledger.reserve(semantic_key="2" * 64, kind="segment_1",
                root_job_id=continuation.job_id, max_cost_microusd=100_001, **base)
            self.assertEqual(per_call.reason, "job_budget_exceeded")
            for number in range(8):
                other = ledger.reserve(semantic_key=f"{number + 10:064x}",
                    max_cost_microusd=100_000,
                    **{**base, "output_dir": root / f"other-{number}"})
                self.assertEqual(other.kind, "new")
            first = ledger.reserve(semantic_key="3" * 64, kind="segment_1",
                root_job_id=continuation.job_id, max_cost_microusd=100_000, **base)
            self.assertEqual(first.kind, "new")  # weekly total $0.95
            weekly = ledger.reserve(semantic_key="4" * 64, kind="segment_2",
                root_job_id=continuation.job_id, max_cost_microusd=60_000, **base)
            self.assertEqual(weekly.reason, "weekly_budget_exceeded")
            self.assertEqual(ledger.db.execute(
                "SELECT COUNT(*) FROM jobs WHERE semantic_key=?", ("4" * 64,)).fetchone()[0], 0)
            # The submit transition independently rechecks the shared call
            # count, including all earlier attempts in this billing group.
            ledger.db.execute("UPDATE jobs SET dispatches=5 WHERE id=?", (writer.job_id,))
            self.assertTrue(ledger.mark_submitting(first.job_id))
            self.assertFalse(ledger.mark_submitting(first.job_id))
            limit = ledger.reserve(semantic_key="5" * 64, kind="segment_2",
                root_job_id=continuation.job_id, max_cost_microusd=1_000, **base)
            self.assertEqual(limit.reason, "logical_job_dispatch_limit")
            ledger.close()

    def test_isolated_opus_v3_weekly_authorization_counts_old_costs_and_excludes_other_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = Ledger(root)
            common = dict(source_sha256="b" * 64, credential_id="key-1",
                          credential_version=1, workspace_id="workspace-1")
            # These nine unrelated, already billed jobs retain their full cost
            # in the one shared rolling-week sum.
            for number in range(9):
                old = ledger.reserve(semantic_key=f"{number + 10:064x}",
                    output_dir=root / f"old-{number}", max_cost_microusd=100_000,
                    **common)
                self.assertEqual(old.kind, "new")
                ledger.db.execute("UPDATE jobs SET status='failed_validation', "
                                  "billed_microusd=100000 WHERE id=?", (old.job_id,))

            key = "a" * 64
            authorization = dict(semantic_key=key, source_sha256=common["source_sha256"],
                output_dir=root / "isolated", quality_policy_version=
                "claude_opus_5_5_partitioned_audit_v3",
                authorization_ref="user-20260928-opus-v3-isolated-2usd")
            with self.assertRaisesRegex(ValueError, "isolated weekly authorization"):
                ledger.authorize_weekly_cap_for_run(**{**authorization,
                    "quality_policy_version": "claude_opus_5_5_partitioned_audit_v2"})
            self.assertTrue(ledger.authorize_weekly_cap_for_run(**authorization))
            self.assertFalse(ledger.authorize_weekly_cap_for_run(**authorization))
            with self.assertRaisesRegex(ValueError, "already assigned"):
                ledger.authorize_weekly_cap_for_run(**{**authorization,
                    "semantic_key": "c" * 64})
            with self.assertRaisesRegex(ValueError, "identity changed"):
                ledger.reserve(semantic_key=key, output_dir=root / "wrong-output",
                    max_cost_microusd=50_000,
                    quality_policy_version=authorization["quality_policy_version"],
                    **common)
            with self.assertRaisesRegex(ValueError, "identity changed"):
                ledger.reserve(semantic_key=key, output_dir=root / "isolated",
                    max_cost_microusd=50_000,
                    quality_policy_version=authorization["quality_policy_version"],
                    **{**common, "source_sha256": "d" * 64})
            with self.assertRaisesRegex(ValueError, "identity changed"):
                ledger.reserve(semantic_key=key, output_dir=root / "isolated",
                    max_cost_microusd=50_000,
                    quality_policy_version="claude_opus_5_5_partitioned_audit_v2",
                    **common)
            writer = ledger.reserve(semantic_key=key,
                output_dir=root / "isolated", max_cost_microusd=50_000,
                quality_policy_version=authorization["quality_policy_version"], **common)
            self.assertEqual(writer.kind, "new")
            self.assertEqual(ledger.weekly_cap_microusd(writer.job_id), 2_000_000)
            (Path(ledger.get(writer.job_id)["artifact_dir"]) / "manifest.json").write_text(
                json.dumps({"quality_provider": "openrouter_claude_opus",
                    "quality_policy_version": authorization["quality_policy_version"]}),
                encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "output changed"):
                ledger.reserve(semantic_key="1" * 64,
                    output_dir=root / "wrong-output", max_cost_microusd=200_000,
                    kind="segment_1", root_job_id=writer.job_id, **common)
            with self.assertRaisesRegex(ValueError, "invalid quality root"):
                ledger.reserve(semantic_key="1" * 64,
                    output_dir=root / "isolated", max_cost_microusd=200_000,
                    kind="segment_1", root_job_id=writer.job_id,
                    **{**common, "source_sha256": "d" * 64})
            self.assertEqual(ledger.reserve(semantic_key="e" * 64,
                output_dir=root / "ordinary", max_cost_microusd=60_000,
                quality_policy_version=authorization["quality_policy_version"],
                **common).reason, "weekly_budget_exceeded")
            competing_worker = Ledger(root)
            self.assertEqual(competing_worker.reserve(semantic_key="f" * 64,
                output_dir=root / "ordinary", max_cost_microusd=60_000,
                **common).reason, "weekly_budget_exceeded")
            competing_worker.close()
            stage = ledger.reserve(semantic_key="1" * 64,
                output_dir=root / "isolated", max_cost_microusd=200_000,
                kind="segment_1", root_job_id=writer.job_id, **common)
            self.assertEqual(stage.kind, "new")  # $0.90 old + $0.05 writer + $0.20 stage
            self.assertEqual(ledger.get(stage.job_id)["billing_group_id"], writer.job_id)
            self.assertEqual(ledger.db.execute("""SELECT SUM(COALESCE(billed_microusd,
                reserved_microusd)) FROM jobs""").fetchone()[0], 1_150_000)
            ledger.close()

            restarted = Ledger(root)
            self.assertEqual(restarted.weekly_cap_microusd(writer.job_id), 2_000_000)
            second = restarted.reserve(semantic_key="2" * 64,
                output_dir=root / "isolated", max_cost_microusd=200_000,
                kind="segment_2", root_job_id=writer.job_id, **common)
            self.assertEqual(second.kind, "new")
            self.assertEqual(restarted.reserve(semantic_key="3" * 64,
                output_dir=root / "ordinary", max_cost_microusd=1_000,
                **common).reason, "weekly_budget_exceeded")
            # A later authoritative charge can exceed its reservation. The
            # next reservation still sees it and stops at the $2 ceiling.
            restarted.db.execute("UPDATE jobs SET billed_microusd=850000 WHERE semantic_key=?",
                                 (f"{10:064x}",))
            self.assertEqual(restarted.reserve(semantic_key="4" * 64,
                output_dir=root / "isolated", max_cost_microusd=1_000,
                kind="segment_3", root_job_id=writer.job_id,
                **common).reason, "weekly_budget_exceeded")
            restarted.close()

    def test_isolated_weekly_authorization_cannot_be_injected_into_regular_reserve(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = Ledger(root)
            base = dict(source_sha256="b" * 64, output_dir=root / "meeting",
                        credential_id="key-1", credential_version=1,
                        workspace_id="workspace-1")
            for number in range(10):
                self.assertEqual(ledger.reserve(semantic_key=f"{number:064x}",
                    max_cost_microusd=100_000, **{**base,
                        "output_dir": root / f"other-{number}"}).kind, "new")
            self.assertEqual(ledger.reserve(semantic_key="a" * 64,
                max_cost_microusd=1_000,
                quality_policy_version="claude_opus_5_5_partitioned_audit_v3",
                **base).reason, "weekly_budget_exceeded")
            with self.assertRaises(TypeError):
                ledger.reserve(semantic_key="b" * 64, max_cost_microusd=1_000,
                               weekly_cap_microusd=2_000_000, **base)
            ledger.close()

    def test_existing_rows_migrate_into_same_billing_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db_path = root / "luna.sqlite3"
            db = sqlite3.connect(db_path)
            db.executescript("""
                CREATE TABLE jobs (
                    id TEXT PRIMARY KEY, semantic_key TEXT NOT NULL UNIQUE,
                    source_sha256 TEXT NOT NULL, output_dir TEXT NOT NULL,
                    artifact_dir TEXT NOT NULL, status TEXT NOT NULL,
                    credential_id TEXT NOT NULL, credential_version INTEGER NOT NULL,
                    workspace_id TEXT, custom_id TEXT NOT NULL UNIQUE,
                    remote_id TEXT UNIQUE, reserved_microusd INTEGER NOT NULL,
                    billed_microusd INTEGER, dispatches INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL, updated_at REAL NOT NULL,
                    next_poll_at REAL, error_code TEXT, accepted_document_path TEXT,
                    generation_id TEXT, kind TEXT NOT NULL DEFAULT 'summary',
                    root_job_id TEXT
                );
            """)
            for job_id, kind in (("old_writer", "summary"), ("old_stage", "segment_1")):
                db.execute("""INSERT INTO jobs
                    (id,semantic_key,source_sha256,output_dir,artifact_dir,status,
                     credential_id,credential_version,custom_id,reserved_microusd,
                     billed_microusd,dispatches,created_at,updated_at,kind,root_job_id)
                    VALUES (?,?,?,?,?,'failed_validation','key',1,?,50000,1000,1,1,1,?,?)""",
                    (job_id, ("a" if kind == "summary" else "b") * 64,
                     "c" * 64, str(root / "meeting"), str(root / "jobs" / job_id),
                     kind + job_id, kind, "old_writer"))
            db.commit()
            db.close()
            ledger = Ledger(root)
            self.assertEqual(ledger.get("old_writer")["billing_group_id"], "old_writer")
            self.assertEqual(ledger.get("old_stage")["billing_group_id"], "old_writer")
            self.assertIsNone(ledger.get("old_writer")["writer_parent_id"])
            ledger.close()

    def test_dispatch_limit_is_rechecked_at_submit_across_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = Ledger(root)
            base = dict(source_sha256="b" * 64, output_dir=root / "meeting",
                        credential_id="key-1", credential_version=1,
                        workspace_id="workspace-1")
            writer = ledger.reserve(semantic_key="a" * 64,
                                    max_cost_microusd=10_000, **base)
            ledger.db.execute("UPDATE jobs SET status='failed_validation',billed_microusd=1000 "
                              "WHERE id=?", (writer.job_id,))
            (Path(ledger.get(writer.job_id)["artifact_dir"]) / "draft_document.json").write_text(
                "{}", encoding="utf-8")
            continuation = ledger.reserve(semantic_key="c" * 64,
                                          max_cost_microusd=0,
                                          continuation_of_job_id=writer.job_id, **base)
            continuation_artifacts = Path(ledger.get(continuation.job_id)["artifact_dir"])
            (continuation_artifacts / "draft_document.json").write_text("{}", encoding="utf-8")
            (continuation_artifacts / "manifest.json").write_text("{}", encoding="utf-8")
            self.assertTrue(ledger.mark_continuation_ready(continuation.job_id))
            first = ledger.reserve(semantic_key="d" * 64, max_cost_microusd=1000,
                                   kind="segment_1", root_job_id=continuation.job_id, **base)
            second = ledger.reserve(semantic_key="e" * 64, max_cost_microusd=1000,
                                    kind="segment_2", root_job_id=continuation.job_id, **base)
            self.assertEqual(first.kind, "new")
            self.assertEqual(second.kind, "new")
            # A restart/concurrent worker may have dispatched earlier reserved
            # work between reservation and this POST. Recheck the whole group.
            ledger.db.execute("UPDATE jobs SET dispatches=5 WHERE id=?", (writer.job_id,))
            self.assertTrue(ledger.mark_submitting(first.job_id))
            self.assertFalse(ledger.mark_submitting(second.job_id))
            self.assertEqual(ledger.get(second.job_id)["status"], "reserved")
            ledger.close()

    def test_exact_batch_shape_and_custom_id_result_mapping(self):
        opener = _Opener()
        reply = BatchClient("fictional-test-token", opener=opener).submit("summary-001", {"messages": [{"role": "user", "content": "тест"}]})
        self.assertEqual(reply.status_code, 202)
        request = opener.calls[0][0]
        payload = json.loads(request.data)
        self.assertEqual(list(payload)[:4], ["endpoint", "model", "provider", "completion_window"])
        self.assertEqual(payload["model"], "openai/gpt-6-luna:batch")
        self.assertEqual(payload["provider"], {"only": ["openai"]})
        self.assertEqual(payload["requests"][0]["custom_id"], "summary-001")
        batch = {"status": "completed", "usage": {"cost": 0.001}, "results": [
            {"custom_id": "someone-else", "response": {"status_code": 200, "body": {"bad": True}}},
            {"custom_id": "summary-001", "response": {"status_code": 200, "body": {"choices": []}}},
        ]}
        body, usage = extract_one_completed(batch, "summary-001")
        self.assertEqual(body, {"choices": []})
        self.assertEqual(usage["cost"], 0.001)


if __name__ == "__main__":
    unittest.main()
