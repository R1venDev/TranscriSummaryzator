"""Focused durable accounting tests for the source-first Batch workflow."""

import hashlib
import json
import os
import tempfile
import time
import unittest
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

from summary.luna_v1.batch import build_batch_payload
from summary.luna_v1.ledger import (
    JOB_CAP_MICROUSD,
    WEEK_SECONDS,
    Ledger,
    batch_item_intent_for_body,
    source_first_job_cap_microusd,
    write_private_json,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _body(text="U00001: тест", cap=1000):
    return {
        "messages": [
            {"role": "developer", "content": "Сверь источник."},
            {"role": "user", "content": text},
        ],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "check", "strict": True,
            "schema": {"type": "object", "additionalProperties": False,
                       "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
        }},
        "max_completion_tokens": cap,
        "reasoning": {"effort": "medium"},
    }


class SourceFirstLedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ledger = Ledger(self.root / "private")
        self.addCleanup(self.ledger.close)
        self.manifest = self.root / "private" / "workflow.json"
        write_private_json(self.manifest, {"source_sha256": "b" * 64,
                                           "judge_workspace_id": "judge-space",
                                           "planned_capacity_microusd": 100_000,
                                           "capacity_basis": {"route": "synthetic"}})
        decision = self.ledger.create_source_first_job(
            semantic_key="a" * 64, source_sha256="b" * 64,
            output_dir=self.root / "output", manifest_path=self.manifest,
            manifest_sha256=_sha(self.manifest), credential_id="writer-key",
            credential_version=1, workspace_id="writer-space")
        self.assertEqual(decision.kind, "new")
        self.workflow_id = decision.job_id
        self.assertEqual(self.ledger.reserve_source_first_plan(
            self.workflow_id, plan_sha256=_sha(self.manifest),
            reserve_microusd=JOB_CAP_MICROUSD).kind, "new")

    def intent(self, index=1, *, stage="audit", cost=10_000, count=1,
               workspace="judge-space", credential="judge-key"):
        entries = [(f"item-{index}-{j}", _body(f"U{index:05d} case {j}"))
                   for j in range(count)]
        payload = self.root / "private" / f"payload-{index}.json"
        write_private_json(payload, build_batch_payload(entries))
        items = [batch_item_intent_for_body(name, body,
                 reserve_microusd=cost) for name, body in entries]
        result = self.ledger.reserve_batch_intent(
            workflow_id=self.workflow_id, intent_key=f"{index:064x}",
            stage=stage, credential_id=credential,
            credential_version=1, workspace_id=workspace,
            payload_path=payload, payload_sha256=_sha(payload), items=items)
        return result, payload, items

    def test_explicit_source_first_cap_is_sealed_without_raising_legacy_cap(self):
        with patch.dict(os.environ, {"TRANSCRI_LUNA_SOURCE_FIRST_JOB_CAP_USD": "0.20"}):
            self.assertEqual(source_first_job_cap_microusd(), 200_000)
            manifest = self.root / "private" / "larger-plan.json"
            write_private_json(manifest, {"source_sha256": "c" * 64,
                "planned_capacity_microusd": 200_000,
                "capacity_basis": {"route": "synthetic"}})
            created = self.ledger.create_source_first_job(
                semantic_key="d" * 64, source_sha256="c" * 64,
                output_dir=self.root / "larger-output", manifest_path=manifest,
                manifest_sha256=_sha(manifest), credential_id="writer-key",
                credential_version=1, workspace_id="writer-space")
            self.assertEqual(self.ledger.reserve_source_first_plan(
                created.job_id, plan_sha256=_sha(manifest),
                reserve_microusd=200_000).kind, "new")
        with patch.dict(os.environ, {"TRANSCRI_LUNA_SOURCE_FIRST_JOB_CAP_USD": "0.10"}):
            self.assertEqual(self.ledger.reserve_source_first_plan(
                created.job_id, plan_sha256=_sha(manifest),
                reserve_microusd=200_000).kind, "pending")
            body = _body("U00002: тест")
            payload = self.root / "private" / "larger-item.json"
            write_private_json(payload, build_batch_payload([("larger-item", body)]))
            decision = self.ledger.reserve_batch_intent(
                workflow_id=created.job_id, intent_key="e" * 64,
                stage="writer", credential_id="writer-key",
                credential_version=1, workspace_id="writer-space",
                payload_path=payload, payload_sha256=_sha(payload),
                items=[batch_item_intent_for_body("larger-item", body,
                                                 reserve_microusd=120_000)])
            self.assertEqual(decision.kind, "new")
            self.assertEqual(JOB_CAP_MICROUSD, 100_000)

    def test_invalid_source_first_cap_rejected(self):
        with patch.dict(os.environ, {"TRANSCRI_LUNA_SOURCE_FIRST_JOB_CAP_USD": "0.25"}):
            self.assertEqual(source_first_job_cap_microusd(), 250_000)
        for value in ("", "nan", "1.01", "1.00", "0.250001", "0", "0.1000001"):
            with patch.dict(os.environ, {"TRANSCRI_LUNA_SOURCE_FIRST_JOB_CAP_USD": value}):
                with self.assertRaisesRegex(ValueError, "invalid_source_first_job_cap"):
                    source_first_job_cap_microusd()

    def test_manifest_and_payload_are_sealed_before_post(self):
        decision, payload, items = self.intent(stage="writer", workspace="writer-space",
                                                credential="writer-key")
        self.assertEqual(decision.kind, "new")
        repeated = self.ledger.reserve_batch_intent(
            workflow_id=self.workflow_id, intent_key=f"{1:064x}", stage="writer",
            credential_id="writer-key", credential_version=1,
            workspace_id="writer-space", payload_path=payload,
            payload_sha256=_sha(payload), items=items)
        self.assertEqual((repeated.kind, repeated.attempt_id),
                         ("pending", decision.attempt_id))
        self.assertEqual(len(self.ledger.list_batch_attempts(self.workflow_id)), 1)
        write_private_json(payload, build_batch_payload(
            [(items[0].custom_id, _body("changed output", cap=2000))]))
        self.assertFalse(self.ledger.mark_batch_submitting(
            decision.attempt_id, _sha(payload)))
        with self.assertRaises(ValueError):
            self.ledger.mark_batch_submitting(
                decision.attempt_id,
                self.ledger.get_batch_attempt(decision.attempt_id)["payload_sha256"])
        self.assertEqual(self.ledger.get_batch_attempt(decision.attempt_id)["post_count"], 0)

    def test_full_plan_hold_survives_restart_and_stage_reserves_do_not_double_count(self):
        self.assertEqual(self.ledger._rolling_spent_microusd(time.time()), 100_000)
        replay = self.ledger.reserve_source_first_plan(
            self.workflow_id, plan_sha256=_sha(self.manifest),
            reserve_microusd=100_000)
        self.assertEqual((replay.kind, replay.reason), ("pending", "plan_reserved"))
        with self.assertRaises(ValueError):
            self.ledger.reserve_source_first_plan(
                self.workflow_id, plan_sha256=_sha(self.manifest),
                reserve_microusd=90_000)
        decision, payload, items = self.intent(cost=30_000)
        self.assertEqual(decision.kind, "new")
        self.assertEqual(self.ledger._rolling_spent_microusd(time.time()), 100_000)
        self.assertTrue(self.ledger.mark_batch_submitting(decision.attempt_id,
                                                          _sha(payload)))
        self.ledger.record_batch_submission(decision.attempt_id,
                                            remote_id="batch_drawdown123")
        self.ledger.close()
        self.ledger = Ledger(self.root / "private")
        self.addCleanup(self.ledger.close)
        self.assertEqual(self.ledger._rolling_spent_microusd(time.time()), 100_000)
        lease = self.ledger.claim_batch_poll(decision.attempt_id,
            "scheduler-12345678", now=time.time()+121)
        self.assertIsNotNone(lease)
        terminal = self.root / "private" / "drawdown-terminal.json"
        write_private_json(terminal, {"id": "batch_drawdown123", "status": "completed"})
        outcomes = {items[0].custom_id: {"status": "completed",
            "billed_microusd": 5000, "prompt_tokens": 100,
            "completion_tokens": 20}}
        self.ledger.record_batch_terminal(decision.attempt_id,
            "scheduler-12345678", "completed", batch_cost_microusd=5000,
            terminal_path=terminal, terminal_sha256=_sha(terminal),
            item_outcomes=outcomes)
        self.assertEqual(self.ledger._rolling_spent_microusd(time.time()), 100_000)
        later, _, _ = self.intent(index=2, cost=90_000)
        self.assertEqual(later.kind, "new")  # $5k actual + $90k next hold
        self.assertEqual(self.ledger._rolling_spent_microusd(time.time()), 100_000)
        self.ledger.cancel_batch_before_submit(later.attempt_id, "not_needed")
        result = self.root / "private" / "drawdown-summary.json"
        write_private_json(result, {"main": []})
        self.ledger.finish_batch_workflow(self.workflow_id, status="accepted",
            result_path=result, result_sha256=_sha(result))
        self.assertEqual(self.ledger._rolling_spent_microusd(time.time()), 5000)

    def test_plan_hold_cannot_exceed_its_sealed_manifest_quote(self):
        manifest = self.root / "private" / "quoted-small.json"
        write_private_json(manifest, {"source_sha256": "e" * 64,
            "planned_capacity_microusd": 20_000,
            "capacity_basis": {"route": "synthetic"}})
        created = self.ledger.create_source_first_job(
            semantic_key="d" * 64, source_sha256="e" * 64,
            output_dir=self.root / "small-output", manifest_path=manifest,
            manifest_sha256=_sha(manifest), credential_id="writer",
            credential_version=1, workspace_id="writer-space")
        with self.assertRaisesRegex(ValueError, "quote differs"):
            self.ledger.reserve_source_first_plan(created.job_id,
                plan_sha256=_sha(manifest), reserve_microusd=30_000)
        self.assertEqual(self.ledger.get_batch_workflow(created.job_id)
                         ["planned_reserve_microusd"], 0)

    def test_unknown_post_holds_money_and_restart_cannot_duplicate(self):
        decision, payload, _ = self.intent(cost=80_000)
        self.assertTrue(self.ledger.mark_batch_submitting(decision.attempt_id,
                                                          _sha(payload)))
        self.assertFalse(self.ledger.mark_batch_submitting(decision.attempt_id,
                                                           _sha(payload)))
        self.ledger.record_batch_submission(decision.attempt_id,
                                            error_code="timeout_after_post")
        self.ledger.close()
        self.ledger = Ledger(self.root / "private")
        self.addCleanup(self.ledger.close)
        row = self.ledger.get_batch_attempt(decision.attempt_id)
        self.assertEqual((row["status"], row["post_count"], row["billed_microusd"]),
                         ("submission_unknown", 1, None))
        second, _, _ = self.intent(index=2, cost=30_000)
        self.assertEqual((second.kind, second.reason),
                         ("blocked", "logical_job_budget_exceeded"))
        self.ledger.defer_batch_unknown(decision.attempt_id,
                                        delay_seconds=60, error_code="reconcile_pending")
        evidence = self.root / "private" / "recovered-remote.json"
        write_private_json(evidence, {"id": "batch_recovered123"})
        self.assertTrue(self.ledger.attach_recovered_batch_remote(
            decision.attempt_id, remote_id="batch_recovered123",
            evidence_path=evidence, evidence_sha256=_sha(evidence)))
        self.assertEqual(self.ledger.get_batch_attempt(decision.attempt_id)["status"],
                         "submitted")

    def test_cancel_before_post_releases_hold_but_not_an_attempted_post(self):
        first, _, _ = self.intent(cost=90_000)
        self.assertTrue(self.ledger.cancel_batch_before_submit(first.attempt_id,
                                                                "wave_not_admitted"))
        self.assertFalse(self.ledger.cancel_batch_before_submit(first.attempt_id,
                                                                 "wave_not_admitted"))
        second, payload, _ = self.intent(index=2, cost=90_000)
        self.assertEqual(second.kind, "new")
        self.assertTrue(self.ledger.mark_batch_submitting(second.attempt_id,
                                                          _sha(payload)))
        with self.assertRaises(ValueError):
            self.ledger.cancel_batch_before_submit(second.attempt_id, "too_late")

    def test_terminal_items_settle_envelope_once_with_lease(self):
        decision, payload, items = self.intent(cost=20_000, count=2)
        self.assertTrue(self.ledger.mark_batch_submitting(decision.attempt_id,
                                                          _sha(payload)))
        self.ledger.record_batch_submission(decision.attempt_id,
                                            remote_id="batch_test123")
        lease = self.ledger.claim_batch_poll(decision.attempt_id,
                                             "scheduler-12345678", now=time.time()+121)
        self.assertEqual(lease["lease_owner"], "scheduler-12345678")
        self.assertIsNone(self.ledger.claim_batch_poll(decision.attempt_id,
                                                        "scheduler-87654321"))
        terminal = self.root / "private" / "terminal.json"
        write_private_json(terminal, {"id": "batch_test123", "status": "completed"})
        raw = self.root / "private" / "item-raw.json"
        write_private_json(raw, {"custom_id": items[0].custom_id, "response": {"ok": True}})
        outcomes = {
            items[0].custom_id: {"status": "completed", "billed_microusd": 3000,
                                 "prompt_tokens": 100, "completion_tokens": 20,
                                 "raw_path": raw, "raw_sha256": _sha(raw)},
            items[1].custom_id: {"status": "missing", "billed_microusd": None,
                                 "prompt_tokens": None, "completion_tokens": None,
                                 "error_code": "missing_in_terminal"},
        }
        self.ledger.record_batch_terminal(decision.attempt_id, "scheduler-12345678",
            "completed", batch_cost_microusd=3500, terminal_path=terminal,
            terminal_sha256=_sha(terminal), item_outcomes=outcomes)
        row = self.ledger.get_batch_attempt(decision.attempt_id)
        self.assertEqual((row["status"], row["billed_microusd"]), ("completed", 3500))
        saved = {item["custom_id"]: item for item in self.ledger.batch_items(decision.attempt_id)}
        self.assertEqual(saved[items[0].custom_id]["billed_microusd"], 3000)
        self.assertEqual(saved[items[1].custom_id]["status"], "missing")
        later, _, _ = self.intent(index=2, cost=90_000)
        self.assertEqual(later.kind, "new")  # envelope charged once, not item + envelope
        with self.assertRaises(ValueError):
            self.ledger.record_batch_terminal(decision.attempt_id, "scheduler-12345678",
                "completed", batch_cost_microusd=3500, terminal_path=terminal,
                terminal_sha256=_sha(terminal), item_outcomes=outcomes)

    def test_twelve_items_and_six_posts_bound_the_workflow(self):
        for index in range(1, 7):
            decision, payload, _ = self.intent(index=index, count=2, cost=1000)
            self.assertEqual(decision.kind, "new")
            self.assertTrue(self.ledger.mark_batch_submitting(decision.attempt_id,
                                                              _sha(payload)))
            self.ledger.record_batch_submission(decision.attempt_id,
                                                error_code="ambiguous")
        extra, _, _ = self.intent(index=7, count=1, cost=1000)
        self.assertEqual((extra.kind, extra.reason),
                         ("blocked", "logical_job_item_limit"))
        self.assertEqual(sum(a["post_count"] for a in
                             self.ledger.list_batch_attempts(self.workflow_id)), 6)

    def test_legacy_weekly_reservation_includes_source_first_holds(self):
        decision, _, _ = self.intent(cost=90_000)
        self.assertEqual(decision.kind, "new")
        legacy = self.ledger.reserve(semantic_key="c" * 64,
            source_sha256="d" * 64, output_dir=self.root / "legacy",
            credential_id="legacy", credential_version=1,
            workspace_id="writer-space", max_cost_microusd=90_000)
        self.assertEqual(legacy.kind, "new")
        # A second independent ledger connection observes both reservations.
        second = Ledger(self.root / "private")
        try:
            self.assertEqual(second._rolling_spent_microusd(time.time()),
                             190_000)
        finally:
            second.close()

    def test_source_first_weekly_cap_includes_legacy_holds_across_workflows(self):
        for index in range(9):
            legacy = self.ledger.reserve(semantic_key=f"{index+30:064x}",
                source_sha256="d" * 64, output_dir=self.root / f"legacy-{index}",
                credential_id="legacy", credential_version=1,
                workspace_id="writer-space", max_cost_microusd=100_000)
            self.assertEqual(legacy.kind, "new")
        first, _, _ = self.intent(cost=90_000)
        self.assertEqual(first.kind, "new")
        second_manifest = self.root / "private" / "second-workflow.json"
        write_private_json(second_manifest, {"source_sha256": "e" * 64,
                                                   "judge_workspace_id": "judge-space",
                                                   "planned_capacity_microusd": 20_000,
                                                   "capacity_basis": {"route": "synthetic"}})
        second = self.ledger.create_source_first_job(
            semantic_key="f" * 64, source_sha256="e" * 64,
            output_dir=self.root / "second-output", manifest_path=second_manifest,
            manifest_sha256=_sha(second_manifest), credential_id="writer-key",
            credential_version=1, workspace_id="writer-space")
        self.assertEqual(second.kind, "new")
        self.assertEqual(self.ledger.reserve_source_first_plan(
            second.job_id, plan_sha256=_sha(second_manifest),
            reserve_microusd=20_000).reason, "weekly_budget_exceeded")
        body = _body("other meeting")
        payload = self.root / "private" / "other-meeting-payload.json"
        write_private_json(payload, build_batch_payload([("other-item", body)]))
        result = self.ledger.reserve_batch_intent(
            workflow_id=second.job_id, intent_key="d" * 64, stage="audit",
            credential_id="judge-key", credential_version=1,
            workspace_id="judge-space", payload_path=payload,
            payload_sha256=_sha(payload),
            items=[batch_item_intent_for_body("other-item", body,
                                              reserve_microusd=20_000)])
        self.assertEqual((result.kind, result.reason),
                         ("blocked", "workflow_plan_not_reserved"))

    def test_legacy_unknown_charge_stays_held_after_rolling_week(self):
        legacy = self.ledger.reserve(semantic_key="e" * 64,
            source_sha256="d" * 64, output_dir=self.root / "old-unknown",
            credential_id="legacy", credential_version=1,
            workspace_id="writer-space", max_cost_microusd=90_000)
        self.assertEqual(legacy.kind, "new")
        self.assertTrue(self.ledger.mark_submitting(legacy.job_id))
        self.ledger.submission_result(legacy.job_id, remote_id=None,
                                      error_code="timeout_after_post")
        old_time = time.time() - WEEK_SECONDS - 60
        self.ledger.db.execute("UPDATE jobs SET created_at=? WHERE id=?",
                               (old_time, legacy.job_id))
        self.assertEqual(self.ledger.get(legacy.job_id)["status"],
                         "submission_unknown")
        self.assertEqual(self.ledger._rolling_spent_microusd(time.time()), 190_000)

    def test_competing_workflow_holds_are_serialized_across_connections(self):
        for index in range(8):
            legacy = self.ledger.reserve(semantic_key=f"{index+80:064x}",
                source_sha256="d" * 64, output_dir=self.root / f"other-{index}",
                credential_id="legacy", credential_version=1,
                workspace_id="writer-space", max_cost_microusd=100_000)
            self.assertEqual(legacy.kind, "new")
        workflows = []
        for index in range(2):
            manifest = self.root / "private" / f"race-{index}.json"
            write_private_json(manifest, {"source_sha256": "e" * 64,
                                          "index": index,
                                          "planned_capacity_microusd": 100_000,
                                          "capacity_basis": {"route": "synthetic"}})
            created = self.ledger.create_source_first_job(
                semantic_key=f"{index+90:064x}", source_sha256="e" * 64,
                output_dir=self.root / f"race-output-{index}",
                manifest_path=manifest, manifest_sha256=_sha(manifest),
                credential_id="writer", credential_version=1,
                workspace_id="writer-space")
            workflows.append((created.job_id, _sha(manifest)))
        barrier = Barrier(2)

        def hold(workflow):
            opened = Ledger(self.root / "private")
            try:
                barrier.wait(timeout=5)
                return opened.reserve_source_first_plan(workflow[0],
                    plan_sha256=workflow[1], reserve_microusd=100_000).kind
            finally:
                opened.close()

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(hold, workflows))
        self.assertCountEqual(results, ["new", "blocked"])
        self.assertEqual(self.ledger._rolling_spent_microusd(time.time()), 1_000_000)

    def test_finish_requires_terminal_attempt_and_sealed_result(self):
        decision, _, _ = self.intent(cost=10_000)
        result = self.root / "private" / "summary.json"
        write_private_json(result, {"main": []})
        with self.assertRaises(ValueError):
            self.ledger.finish_batch_workflow(self.workflow_id, status="accepted",
                result_path=result, result_sha256=_sha(result))
        self.ledger.cancel_batch_before_submit(decision.attempt_id, "not_needed")
        self.assertTrue(self.ledger.finish_batch_workflow(self.workflow_id,
            status="accepted", result_path=result, result_sha256=_sha(result)))
        self.assertFalse(self.ledger.finish_batch_workflow(self.workflow_id,
            status="accepted", result_path=result, result_sha256=_sha(result)))
        self.assertEqual(self.ledger.list_source_first_workflows(states=("accepted",))[0]
                         ["id"], self.workflow_id)


if __name__ == "__main__":
    unittest.main()
