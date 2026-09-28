"""Offline first-wave admission and rolling-budget retry regressions."""
from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from summary.luna_v1 import source_first_runtime as runtime
from summary.luna_v1.batch import BatchError, MODEL, PROVIDER, Reply, SUBMIT_MODEL
from summary.luna_v1.ledger import BatchIntentDecision, Ledger, WEEK_SECONDS
from summary.luna_v1.route import Route, RouteBlocked
from summary.luna_v1.source_first_core import canonical_bytes, load_snapshot, plan_packets


def _route(*, key_remaining: Decimal | None = None) -> Route:
    return Route(SUBMIT_MODEL, PROVIDER, "workspace-1", 1_000_000, 32_000,
                 Decimal("0.000000001"), Decimal("0.000000001"),
                 Decimal("0.000000001"), Decimal(0), key_remaining,
                 batch_endpoint_model=MODEL, provider_endpoint_tag=PROVIDER)


def _transcript(path: Path) -> None:
    path.write_text(json.dumps({
        "source": "01.01.2026 test.mkv", "duration_seconds": 4,
        "speakers": {"s1": "Аня"},
        "utterances": [
            {"start": 0, "end": 4, "speaker": "s1", "text": "Проверим сигнал."},
        ],
    }, ensure_ascii=False), encoding="utf-8")


class WaveOneBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        transcript = self.root / "transcript.json"
        _transcript(transcript)
        self.snapshot = load_snapshot(transcript)
        self.packets = plan_packets(self.snapshot)
        self.ledger = Ledger(self.root / "private")
        self.addCleanup(self.ledger.close)
        manifest = self.root / "private" / "manifest.json"
        manifest_sha = runtime._json_file_once(manifest, {
            "planned_capacity_microusd": 100_000,
            "capacity_basis": {"offline_test": True},
        })
        created = self.ledger.create_source_first_job(
            semantic_key="a" * 64, source_sha256=self.snapshot.source_sha256,
            output_dir=self.root / "output", manifest_path=manifest,
            manifest_sha256=manifest_sha, credential_id="key-1",
            credential_version=1, workspace_id="workspace-1")
        self.workflow = self.ledger.get_batch_workflow(created.job_id)
        self.assertEqual(self.ledger.reserve_source_first_plan(created.job_id,
            plan_sha256=manifest_sha, reserve_microusd=100_000).kind, "new")

    def _wave(self, *, route=None, client=None):
        return runtime._reserve_and_post_wave1(
            ledger=self.ledger, workflow=self.workflow, snapshot=self.snapshot,
            packets=self.packets, route=route or _route(), client=client)

    def test_blocked_peer_reservation_never_posts_or_cancels_held_writer(self):
        original = runtime._batch_intent
        block_extract = True

        def reserve(**kwargs):
            if kwargs["stage"] == "extract" and block_extract:
                return BatchIntentDecision("blocked", None, "offline_block"), None, None
            return original(**kwargs)

        class Client:
            posts = 0

            def submit_prepared(self, *_args, **_kwargs):
                self.posts += 1
                return Reply(202, {"id": f"batch_test{self.posts:03d}"})

        client = Client()
        with patch.object(runtime, "_batch_intent", side_effect=reserve):
            first = self._wave(client=client)
            self.assertEqual(first["status"], "wave_reservation_blocked")
            self.assertEqual(client.posts, 0)
            rows = self.ledger.list_batch_attempts(self.workflow["id"])
            self.assertEqual([(row["stage"], row["status"]) for row in rows],
                             [("writer", "reserved")])
            block_extract = False
            second = self._wave(client=client)
            self.assertEqual(second["status"], "submitted")
            self.assertEqual(client.posts, 2)
            self.assertEqual(self._wave(client=client)["status"], "submitted")
            self.assertEqual(client.posts, 2)

    def test_definite_writer_rejection_cancels_unposted_extraction(self):
        class Client:
            posts = 0

            def submit_prepared(self, *_args, **_kwargs):
                self.posts += 1
                raise BatchError(400, "definite_rejection")

        client = Client()
        first = self._wave(client=client)
        self.assertEqual(first["status"], "wave1_incomplete")
        self.assertEqual(client.posts, 1)
        rows = {row["stage"]: row for row in
                self.ledger.list_batch_attempts(self.workflow["id"])}
        self.assertEqual(rows["writer"]["status"], "rejected_no_charge")
        self.assertEqual(rows["extract"]["status"], "cancelled_before_submit")
        self.assertEqual(rows["extract"]["post_count"], 0)
        self.assertEqual(self._wave(client=client)["status"], "wave1_incomplete")
        self.assertEqual(client.posts, 1)

    def test_restart_prices_only_not_yet_posted_extraction_against_key(self):
        class Client:
            posts = 0

            def submit_prepared(self, *_args, **_kwargs):
                self.posts += 1
                return Reply(202, {"id": f"batch_test{self.posts:03d}"})

        client = Client()
        writer_items = runtime._items_for_wave(self.workflow["id"], "writer",
            [("full", runtime._writer_payload(self.snapshot))])
        decision, payload, payload_sha = runtime._batch_intent(
            ledger=self.ledger, workflow=self.workflow, stage="writer",
            items=writer_items, route=_route())
        self.assertEqual(runtime._post_reserved(
            ledger=self.ledger, attempt_id=decision.attempt_id,
            payload_path=payload, payload_sha=payload_sha,
            client=client)["status"], "submitted")
        extract_cost = _route().reserve_microusd(canonical_bytes(runtime._chat_body(
            "extract", runtime._extraction_payload(self.packets[0]))),
            max_completion_tokens=runtime.STAGE_CAPS["extract"])
        limited = _route(key_remaining=Decimal(extract_cost) / 1_000_000)
        result = self._wave(route=limited, client=client)
        self.assertEqual(result["status"], "submitted")
        self.assertEqual(client.posts, 2)

    def test_key_balance_covers_sum_of_wave_items_before_any_reservation(self):
        route = _route()
        writer = route.reserve_microusd(canonical_bytes(runtime._chat_body(
            "writer", runtime._writer_payload(self.snapshot))),
            max_completion_tokens=runtime.STAGE_CAPS["writer"])
        extract = route.reserve_microusd(canonical_bytes(runtime._chat_body(
            "extract", runtime._extraction_payload(self.packets[0]))),
            max_completion_tokens=runtime.STAGE_CAPS["extract"])
        self.assertGreater(writer + extract, max(writer, extract))
        limited = replace(route,
            key_limit_remaining_usd=Decimal(max(writer, extract)) / 1_000_000)
        with self.assertRaisesRegex(RouteBlocked, "key_budget_insufficient"):
            self._wave(route=limited, client=None)
        self.assertEqual(self.ledger.list_batch_attempts(self.workflow["id"]), [])

    def test_legacy_budget_failure_with_attempt_cannot_be_reactivated(self):
        class Client:
            def submit_prepared(self, *_args, **_kwargs):
                raise BatchError(400, "definite_rejection")

        self._wave(client=Client())
        self.ledger.finish_batch_workflow(self.workflow["id"], status="failed",
                                          error_code="rolling_week_budget_exceeded")
        # Even if an old ledger lost its plan marker, its saved item intents
        # still prove this was more than a zero-POST weekly refusal.
        self.ledger.db.execute("""UPDATE source_first_workflows
            SET plan_sha256=NULL,planned_reserve_microusd=0 WHERE id=?""",
            (self.workflow["id"],))
        self.assertFalse(self.ledger.reactivate_source_first_budget_refusal(
            self.workflow["id"], semantic_key=self.workflow["semantic_key"],
            source_sha256=self.snapshot.source_sha256,
            output_dir=self.root / "output",
            manifest_sha256=self.workflow["manifest_sha256"]))
        self.assertEqual(self.ledger.get_batch_workflow(self.workflow["id"])["status"],
                         "failed")


class WeeklyBudgetRetryTests(unittest.TestCase):
    def test_refused_plan_remains_active_and_dispatches_after_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            private = root / "private"
            transcript = root / "transcript.json"
            _transcript(transcript)
            ledger = Ledger(private)
            try:
                for index in range(10):
                    decision = ledger.reserve(
                        semantic_key=f"{index + 1:064x}",
                        source_sha256="b" * 64,
                        output_dir=root / f"legacy-{index}",
                        credential_id="key-1", credential_version=1,
                        workspace_id="workspace-1", max_cost_microusd=100_000)
                    self.assertEqual(decision.kind, "new")

                class Credential:
                    def __init__(self, path):
                        self.path = Path(path)

                    def dispatch_candidates(self):
                        return [{"id": "key-1", "version": 1,
                                 "workspace_id": "workspace-1"}]

                    def reveal_for_dispatch(self, *_):
                        return "offline-token"

                class Client:
                    posts = 0

                    def submit_prepared(self, *_args, **_kwargs):
                        self.posts += 1
                        return Reply(202, {"id": f"batch_test{self.posts:03d}"})

                client = Client()
                with (patch.object(runtime, "CredentialStore", Credential),
                      patch.object(runtime, "verify_source_first_batch_route",
                                   return_value=_route())):
                    blocked = runtime.submit_source_first(
                        transcript_path=transcript, output_dir=root / "output",
                        private_root=private, client_factory=lambda token: client)
                    self.assertEqual(blocked["status"], "budget_blocked")
                    workflow = ledger.get_batch_workflow(blocked["workflow_id"])
                    self.assertEqual((workflow["status"], workflow["plan_sha256"]),
                                     ("active", None))
                    self.assertEqual(client.posts, 0)
                    self.assertEqual(runtime.submit_source_first(
                        transcript_path=transcript, output_dir=root / "output",
                        private_root=private,
                        client_factory=lambda token: client)["status"], "pending")
                    ledger.db.execute("UPDATE jobs SET created_at=?",
                                      (time.time() - WEEK_SECONDS - 60,))
                    outcomes = runtime.poll_source_first_once(
                        private_root=private, client_factory=lambda token: client)
                    self.assertTrue(any(outcome["status"] == "submitted"
                                        for outcome in outcomes))
                    self.assertEqual(client.posts, 2)
                    attempts = ledger.list_batch_attempts(workflow["id"])
                    self.assertEqual({row["stage"] for row in attempts},
                                     {"writer", "extract"})
                    self.assertTrue(all(row["status"] == "submitted" for row in attempts))
            finally:
                ledger.close()

    def test_preexisting_failed_weekly_refusal_is_reactivated_for_poll(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            private = root / "private"
            transcript = root / "transcript.json"
            _transcript(transcript)

            class Credential:
                def __init__(self, path):
                    self.path = Path(path)

                def dispatch_candidates(self):
                    return [{"id": "key-1", "version": 1,
                             "workspace_id": "workspace-1"}]

                def reveal_for_dispatch(self, *_):
                    return "offline-token"

            class Client:
                posts = 0

                def submit_prepared(self, *_args, **_kwargs):
                    self.posts += 1
                    return Reply(202, {"id": f"batch_test{self.posts:03d}"})

            client = Client()
            ledger = Ledger(private)
            try:
                for index in range(10):
                    self.assertEqual(ledger.reserve(
                        semantic_key=f"{index + 1:064x}",
                        source_sha256="b" * 64,
                        output_dir=root / f"legacy-{index}",
                        credential_id="key-1", credential_version=1,
                        workspace_id="workspace-1",
                        max_cost_microusd=100_000).kind, "new")
                with (patch.object(runtime, "CredentialStore", Credential),
                      patch.object(runtime, "verify_source_first_batch_route",
                                   return_value=_route())):
                    blocked = runtime.submit_source_first(
                        transcript_path=transcript, output_dir=root / "output",
                        private_root=private, client_factory=lambda token: client)
                    workflow_id = blocked["workflow_id"]
                    ledger.finish_batch_workflow(workflow_id, status="failed",
                        error_code="rolling_week_budget_exceeded")
                    self.assertEqual(ledger.get_batch_workflow(workflow_id)["status"],
                                     "failed")
                    ledger.db.execute("UPDATE jobs SET created_at=?",
                                      (time.time() - WEEK_SECONDS - 60,))
                    retry = runtime.submit_source_first(
                        transcript_path=transcript, output_dir=root / "output",
                        private_root=private, client_factory=lambda token: client)
                    self.assertEqual((retry["status"], retry["workflow_id"]),
                                     ("pending", workflow_id))
                    self.assertEqual(client.posts, 0)
                    self.assertEqual(ledger.get_batch_workflow(workflow_id)["status"],
                                     "active")
                    outcomes = runtime.poll_source_first_once(
                        private_root=private, client_factory=lambda token: client)
                    self.assertTrue(any(outcome["status"] == "submitted"
                                        for outcome in outcomes))
                    self.assertEqual(client.posts, 2)
            finally:
                ledger.close()


if __name__ == "__main__":
    unittest.main()
