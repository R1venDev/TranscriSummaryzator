"""Offline transport identity, capacity and lost-POST recovery checks."""
from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from summary.luna_v1.batch import BatchError, Reply, build_batch_payload
from summary.luna_v1.batch import MODEL, PROVIDER, SUBMIT_MODEL
from summary.luna_v1.ledger import Ledger, batch_item_intent_for_body
from summary.luna_v1.route import Route
from summary.luna_v1.source_first_core import SourceSnapshot, load_snapshot, plan_packets
from summary.luna_v1.tasks import ReconciliationConflict
from summary.luna_v1.source_first_runtime import (
    _batch_file_once, _chat_body, _planned_capacity_hold, _post_reserved,
    _extraction_payload, _poll_attempt, _recover_unknown, _writer_payload,
    poll_source_first_once, submit_source_first,
)
from tests.test_luna_source_first_core import _document


class SourceFirstRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "private"
        self.ledger = Ledger(self.root)
        self.addCleanup(self.ledger.close)
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"planned_capacity_microusd": 10_000,
                                        "capacity_basis": {"offline_test": True}}),
                            encoding="utf-8")
        workflow = self.ledger.create_source_first_job(
            semantic_key="a" * 64, source_sha256="b" * 64,
            output_dir=self.root / "output", manifest_path=manifest,
            manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
            credential_id="key-1", credential_version=1,
            workspace_id="workspace-1")
        self.workflow = self.ledger.get_batch_workflow(workflow.job_id)
        self.assertTrue(self.ledger.reserve_source_first_plan(
            workflow.job_id, plan_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
            reserve_microusd=10_000))

    def _reserved(self):
        custom_id = "sf-test12345678-writer-full-a1"
        body = _chat_body("writer", {"SOURCE": {"utterances": []}})
        envelope = build_batch_payload([(custom_id, body)])
        path = self.root / "batch.json"
        digest = _batch_file_once(path, envelope)
        decision = self.ledger.reserve_batch_intent(
            workflow_id=self.workflow["id"], intent_key="c" * 64,
            stage="writer", credential_id="key-1", credential_version=1,
            workspace_id="workspace-1", payload_path=path,
            payload_sha256=digest,
            items=[batch_item_intent_for_body(custom_id, body,
                                               reserve_microusd=1000)])
        self.assertEqual(decision.kind, "new")
        return decision.attempt_id, path, digest, custom_id

    def _source_workflow(self, label):
        transcript = self.root / f"{label}-source.json"
        transcript.write_text('{"utterances":[{"text":"frozen"}]}', encoding="utf-8")
        source_sha = hashlib.sha256(transcript.read_bytes()).hexdigest()
        manifest = {"source_revision": source_sha, "workspace_id": "workspace-1",
                    "planned_capacity_microusd": 10_000,
                    "capacity_basis": {"offline_test": True},
                    "packets": [], "snapshot": {"transcript_path": str(transcript),
                        "source_sha256": source_sha, "source_text": "",
                        "index": {}, "records": []}}
        manifest_path = self.root / f"{label}-manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        semantic_key = hashlib.sha256(label.encode()).hexdigest()
        created = self.ledger.create_source_first_job(
            semantic_key=semantic_key, source_sha256=source_sha,
            output_dir=self.root / f"{label}-output", manifest_path=manifest_path,
            manifest_sha256=manifest_sha, credential_id="key-1",
            credential_version=1, workspace_id="workspace-1")
        self.assertTrue(self.ledger.reserve_source_first_plan(
            created.job_id, plan_sha256=manifest_sha, reserve_microusd=10_000))
        return self.ledger.get_batch_workflow(created.job_id), transcript

    def _reserve_stage(self, workflow, stage):
        custom_id = f"sf-{workflow['id'][:12]}-{stage}-full-a1"
        body = _chat_body(stage, {"SOURCE": {"utterances": []}})
        path = self.root / f"{workflow['id']}-{stage}.json"
        digest = _batch_file_once(path, build_batch_payload([(custom_id, body)]))
        decision = self.ledger.reserve_batch_intent(
            workflow_id=workflow["id"], intent_key=hashlib.sha256(
                (workflow["id"] + stage).encode()).hexdigest(), stage=stage,
            credential_id="key-1", credential_version=1,
            workspace_id="workspace-1", payload_path=path,
            payload_sha256=digest,
            items=[batch_item_intent_for_body(custom_id, body, reserve_microusd=1000)])
        self.assertEqual(decision.kind, "new")
        return decision.attempt_id, digest

    def _list_row(self, attempt_id, batch_id, *, model=SUBMIT_MODEL, total=1):
        return {"id": batch_id, "model": model,
                "endpoint": "/v1/chat/completions", "status": "completed",
                "created_at": int(self.ledger.get_batch_attempt(attempt_id)["created_at"]),
                "request_counts": {"total": total}}

    def test_full_chain_capacity_counts_every_item_cap(self):
        snapshot = SourceSnapshot(Path("unused"), "b" * 64, "{\"utterances\":[]}",
                                  {}, ())
        packets = ({"records": [{"id": "U00001", "text": "тест"}]},) * 3
        low = Route("openai/gpt-6-luna", "openai", "workspace-1", 1_000_000,
                    32_000, Decimal("0.00000001"), Decimal("0.00000001"),
                    Decimal("0.00000001"), Decimal(0), None)
        high = Route("openai/gpt-6-luna", "openai", "workspace-1", 1_000_000,
                     32_000, Decimal("0.000001"), Decimal("0.000001"),
                     Decimal("0.000001"), Decimal(0), None)
        self.assertLess(_planned_capacity_hold(low, snapshot, packets), 100_000)
        self.assertGreater(_planned_capacity_hold(high, snapshot, packets), 100_000)

    def test_post_uses_exact_sealed_ordered_bytes_once(self):
        attempt_id, path, digest, _ = self._reserved()
        wire = path.read_bytes()
        self.assertEqual(list(json.loads(wire)),
                         ["endpoint", "model", "provider", "completion_window", "requests"])

        class Client:
            calls = 0

            def submit_prepared(self, payload_bytes, *, expected_sha256):
                self.calls += 1
                assert payload_bytes == wire
                assert expected_sha256 == digest
                return Reply(202, {"id": "batch_test123"})

        client = Client()
        sent = _post_reserved(ledger=self.ledger, attempt_id=attempt_id,
                              payload_path=path, payload_sha=digest, client=client)
        again = _post_reserved(ledger=self.ledger, attempt_id=attempt_id,
                               payload_path=path, payload_sha=digest, client=client)
        self.assertEqual(sent["status"], "submitted")
        self.assertEqual(again["status"], "pending")
        self.assertEqual(client.calls, 1)
        self.assertEqual(self.ledger.get_batch_attempt(attempt_id)["post_count"], 1)

    def test_lost_post_recovers_only_exact_custom_id(self):
        attempt_id, path, digest, custom_id = self._reserved()
        self.assertTrue(self.ledger.mark_batch_submitting(attempt_id, digest))
        self.ledger.record_batch_submission(attempt_id,
                                            error_code="transport_unknown")
        rows = [self._list_row(attempt_id, "batch_wrong123"),
                self._list_row(attempt_id, "batch_right123"),
                self._list_row(attempt_id, "batch_other123", model="other/model")]
        by_id = {row["id"]: row for row in rows}

        class Client:
            def __init__(self):
                self.get_ids = []

            def list_batches(self, **_):
                return Reply(200, {"data": rows, "has_more": False,
                                   "last_id": rows[-1]["id"]})

            def get(self, batch_id):
                self.get_ids.append(batch_id)
                identity = custom_id if batch_id == "batch_right123" else "different-id"
                return Reply(200, {**by_id[batch_id],
                                   "results": [{"custom_id": identity,
                                                "response": {"status_code": 200,
                                                             "body": {"choices": []}}}]})

        client = Client()
        outcome = _recover_unknown(ledger=self.ledger,
            attempt=self.ledger.get_batch_attempt(attempt_id), client=client,
            private_root=self.root)
        self.assertEqual(outcome["status"], "remote_recovered")
        self.assertEqual(client.get_ids, ["batch_wrong123", "batch_right123"])
        row = self.ledger.get_batch_attempt(attempt_id)
        self.assertEqual((row["status"], row["remote_id"], row["post_count"]),
                         ("submitted", "batch_right123", 1))

    def test_lost_post_with_no_exact_result_keeps_unknown(self):
        attempt_id, path, digest, _ = self._reserved()
        self.ledger.mark_batch_submitting(attempt_id, digest)
        self.ledger.record_batch_submission(attempt_id, error_code="timeout")
        row = self._list_row(attempt_id, "batch_unknown123")

        class Client:
            def list_batches(self, **_):
                return Reply(200, {"data": [row], "has_more": False,
                                   "last_id": row["id"]})

            def get(self, batch_id):
                return Reply(200, {**row,
                                   "results": None})

        outcome = _recover_unknown(ledger=self.ledger,
            attempt=self.ledger.get_batch_attempt(attempt_id), client=Client(),
            private_root=self.root)
        self.assertEqual(outcome["status"], "submission_unknown")
        self.assertEqual(self.ledger.get_batch_attempt(attempt_id)["post_count"], 1)

    def test_lost_post_uses_time_window_and_last_id_to_recover_next_page(self):
        attempt_id, _, digest, custom_id = self._reserved()
        self.ledger.mark_batch_submitting(attempt_id, digest)
        self.ledger.record_batch_submission(attempt_id, error_code="timeout")
        foreign = self._list_row(attempt_id, "batch_foreign123", model="other/model")
        target = self._list_row(attempt_id, "batch_target123")

        class Client:
            def __init__(self):
                self.lists = []
                self.get_ids = []

            def list_batches(self, **kwargs):
                self.lists.append(kwargs)
                if kwargs["after"] is None:
                    return Reply(200, {"data": [foreign], "has_more": True,
                                       "last_id": foreign["id"]})
                return Reply(200, {"data": [target], "has_more": False,
                                   "last_id": target["id"]})

            def get(self, batch_id):
                self.get_ids.append(batch_id)
                return Reply(200, {**target,
                    "results": [{"custom_id": custom_id,
                                 "response": {"status_code": 200,
                                              "body": {"choices": []}}}]})

        client = Client()
        outcome = _recover_unknown(ledger=self.ledger,
            attempt=self.ledger.get_batch_attempt(attempt_id), client=client,
            private_root=self.root)
        created = int(self.ledger.get_batch_attempt(attempt_id)["created_at"])
        self.assertEqual(outcome["status"], "remote_recovered")
        self.assertEqual(client.get_ids, [target["id"]])
        self.assertEqual(client.lists, [
            {"limit": 100, "after": None, "created_after": created - 60,
             "created_before": created + 3601},
            {"limit": 100, "after": foreign["id"], "created_after": created - 60,
             "created_before": created + 3601},
        ])

    def test_incomplete_paginated_recovery_keeps_unknown_without_get(self):
        attempt_id, _, digest, _ = self._reserved()
        self.ledger.mark_batch_submitting(attempt_id, digest)
        self.ledger.record_batch_submission(attempt_id, error_code="timeout")
        rows = [self._list_row(attempt_id, f"batch_page{number}123")
                for number in range(3)]

        class Client:
            def __init__(self):
                self.lists = []

            def list_batches(self, **kwargs):
                self.lists.append(kwargs)
                row = rows[len(self.lists) - 1]
                return Reply(200, {"data": [row], "has_more": True,
                                   "last_id": row["id"]})

            def get(self, _batch_id):
                raise AssertionError("incomplete inventory must not GET candidate results")

        client = Client()
        outcome = _recover_unknown(ledger=self.ledger,
            attempt=self.ledger.get_batch_attempt(attempt_id), client=client,
            private_root=self.root)
        self.assertEqual(outcome["reason"], "recovery_page_limit")
        self.assertEqual([call["after"] for call in client.lists],
                         [None, rows[0]["id"], rows[1]["id"]])
        row = self.ledger.get_batch_attempt(attempt_id)
        self.assertEqual((row["status"], row["remote_id"], row["post_count"]),
                         ("submission_unknown", None, 1))

    def test_too_many_or_ambiguous_remote_candidates_keep_unknown(self):
        attempt_id, _, digest, custom_id = self._reserved()
        self.ledger.mark_batch_submitting(attempt_id, digest)
        self.ledger.record_batch_submission(attempt_id, error_code="timeout")
        rows = [self._list_row(attempt_id, f"batch_many{number}123")
                for number in range(9)]

        class BusyClient:
            def list_batches(self, **_):
                return Reply(200, {"data": rows, "has_more": False,
                                   "last_id": rows[-1]["id"]})

            def get(self, _batch_id):
                raise AssertionError("too many candidates must not trigger GET")

        outcome = _recover_unknown(ledger=self.ledger,
            attempt=self.ledger.get_batch_attempt(attempt_id), client=BusyClient(),
            private_root=self.root)
        self.assertEqual(outcome["reason"], "too_many_recovery_candidates")

        class AmbiguousClient:
            def list_batches(self, **_):
                return Reply(200, {"data": rows[:2], "has_more": False,
                                   "last_id": rows[1]["id"]})

            def get(self, batch_id):
                row = next(row for row in rows[:2] if row["id"] == batch_id)
                return Reply(200, {**row,
                    "results": [{"custom_id": custom_id,
                                 "response": {"status_code": 200,
                                              "body": {"choices": []}}}]})

        outcome = _recover_unknown(ledger=self.ledger,
            attempt=self.ledger.get_batch_attempt(attempt_id), client=AmbiguousClient(),
            private_root=self.root)
        self.assertEqual(outcome["reason"], "ambiguous_remote_candidates")
        row = self.ledger.get_batch_attempt(attempt_id)
        self.assertEqual((row["status"], row["remote_id"], row["post_count"]),
                         ("submission_unknown", None, 1))

    def test_scheduler_recovers_lost_post_without_second_create(self):
        attempt_id, path, digest, custom_id = self._reserved()
        self.ledger.mark_batch_submitting(attempt_id, digest)
        self.ledger.record_batch_submission(attempt_id, error_code="timeout")
        row = self._list_row(attempt_id, "batch_recovered456")

        class Credential:
            def __init__(self, *_):
                pass

            def reveal_for_existing_job(self, *_):
                return "offline-token"

        class Client:
            posts = 0

            def list_batches(self, **_):
                return Reply(200, {"data": [row], "has_more": False,
                                   "last_id": row["id"]})

            def get(self, batch_id):
                return Reply(200, {**row,
                    "results": [{"custom_id": custom_id,
                                 "response": {"status_code": 200,
                                              "body": {"choices": []}}}]})

            def submit_prepared(self, *_args, **_kwargs):
                self.posts += 1
                raise AssertionError("unknown POST replayed")

        client = Client()
        with (patch("summary.luna_v1.source_first_runtime.CredentialStore", Credential),
              patch("summary.luna_v1.source_first_runtime._advance_workflow",
                    return_value={"status": "pending"}),
              patch("summary.luna_v1.ledger.time.time", return_value=time.time() + 121)):
            outcomes = poll_source_first_once(private_root=self.root,
                client_factory=lambda token: client)
        self.assertTrue(any(row.get("status") == "remote_recovered" for row in outcomes))
        self.assertEqual(client.posts, 0)
        self.assertEqual(self.ledger.get_batch_attempt(attempt_id)["remote_id"],
                         "batch_recovered456")

    def test_terminal_reordered_duplicate_missing_items_settle_once(self):
        ids = [f"sf-test12345678-extract-P00{number}-a1" for number in (1, 2, 3)]
        items = [(custom_id, _chat_body("extract", {"CORE_SOURCE": []}))
                 for custom_id in ids]
        path = self.root / "extract.json"
        digest = _batch_file_once(path, build_batch_payload(items))
        decision = self.ledger.reserve_batch_intent(
            workflow_id=self.workflow["id"], intent_key="d" * 64,
            stage="extract", credential_id="key-1", credential_version=1,
            workspace_id="workspace-1", payload_path=path,
            payload_sha256=digest,
            items=[batch_item_intent_for_body(cid, body, reserve_microusd=1000)
                   for cid, body in items])
        self.assertEqual(decision.kind, "new")
        self.ledger.mark_batch_submitting(decision.attempt_id, digest)
        self.ledger.record_batch_submission(decision.attempt_id,
                                            remote_id="batch_items123")

        def raw(custom_id):
            return {"custom_id": custom_id, "response": {"status_code": 200,
                "body": {"id": "gen-test", "choices": [{"finish_reason": "stop",
                    "message": {"role": "assistant", "content": "{}"}}],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 5}}}}

        class Client:
            def get(self, batch_id):
                return Reply(200, {"id": batch_id, "status": "completed",
                    "model": SUBMIT_MODEL, "endpoint": "/v1/chat/completions",
                    "usage": {"cost": "0.001"},
                    "results": [raw(ids[2]), raw(ids[0]), raw(ids[0])]})

        class Credential:
            def __init__(self, *_):
                pass

            def reveal_for_existing_job(self, *_):
                return "offline-token"

        with (patch("summary.luna_v1.source_first_runtime.CredentialStore", Credential),
              patch("summary.luna_v1.ledger.time.time", return_value=time.time() + 121)):
            result = _poll_attempt(ledger=self.ledger,
                attempt=self.ledger.get_batch_attempt(decision.attempt_id),
                workflow=self.workflow, private_root=self.root,
                client_factory=lambda token: Client())
        self.assertEqual(result["status"], "completed")
        saved = {row["custom_id"]: row for row in self.ledger.batch_items(decision.attempt_id)}
        self.assertEqual([saved[cid]["status"] for cid in ids],
                         ["duplicate", "missing", "completed"])
        self.assertEqual(self.ledger.get_batch_attempt(decision.attempt_id)["billed_microusd"],
                         1000)

    def test_failed_batch_without_results_keeps_unknown_cost(self):
        attempt_id, path, digest, custom_id = self._reserved()
        self.ledger.mark_batch_submitting(attempt_id, digest)
        self.ledger.record_batch_submission(attempt_id, remote_id="batch_failed123")

        class Client:
            def get(self, batch_id):
                return Reply(200, {"id": batch_id, "status": "failed",
                    "model": SUBMIT_MODEL, "endpoint": "/v1/chat/completions",
                    "results": None})

        class Credential:
            def __init__(self, *_):
                pass

            def reveal_for_existing_job(self, *_):
                return "offline-token"

        with (patch("summary.luna_v1.source_first_runtime.CredentialStore", Credential),
              patch("summary.luna_v1.ledger.time.time", return_value=time.time() + 121)):
            result = _poll_attempt(ledger=self.ledger,
                attempt=self.ledger.get_batch_attempt(attempt_id),
                workflow=self.workflow, private_root=self.root,
                client_factory=lambda token: Client())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(self.ledger.batch_items(attempt_id)[0]["status"], "unavailable")
        self.assertIsNone(self.ledger.get_batch_attempt(attempt_id)["billed_microusd"])

    def test_terminal_extra_custom_id_invalidates_expected_result(self):
        attempt_id, _, digest, custom_id = self._reserved()
        self.ledger.mark_batch_submitting(attempt_id, digest)
        self.ledger.record_batch_submission(attempt_id, remote_id="batch_extra123")

        def raw(identifier):
            return {"custom_id": identifier, "response": {"status_code": 200,
                "body": {"choices": [{"finish_reason": "stop",
                    "message": {"role": "assistant", "content": "{}"}}],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 5}}}}

        class Client:
            def get(self, batch_id):
                return Reply(200, {"id": batch_id, "status": "completed",
                    "model": SUBMIT_MODEL, "endpoint": "/v1/chat/completions",
                    "request_counts": {"total": 1},
                    "results": [raw(custom_id), raw("foreign-item")]})

        class Credential:
            def __init__(self, *_):
                pass

            def reveal_for_existing_job(self, *_):
                return "offline-token"

        with (patch("summary.luna_v1.source_first_runtime.CredentialStore", Credential),
              patch("summary.luna_v1.ledger.time.time", return_value=time.time() + 121)):
            result = _poll_attempt(ledger=self.ledger,
                attempt=self.ledger.get_batch_attempt(attempt_id),
                workflow=self.workflow, private_root=self.root,
                client_factory=lambda token: Client())
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["item_statuses"][custom_id], "invalid")
        saved = self.ledger.batch_items(attempt_id)[0]
        self.assertEqual(saved["status"], "invalid")
        self.assertIsNotNone(self.ledger.get_batch_attempt(attempt_id)["terminal_path"])

    def test_terminal_route_identity_must_match_sealed_intent(self):
        attempt_id, _, digest, _ = self._reserved()
        self.ledger.mark_batch_submitting(attempt_id, digest)
        self.ledger.record_batch_submission(attempt_id, remote_id="batch_route123")

        class Credential:
            def __init__(self, *_):
                pass

            def reveal_for_existing_job(self, *_):
                return "offline-token"

        base = time.time()
        with patch("summary.luna_v1.source_first_runtime.CredentialStore", Credential):
            for number, identity in enumerate(({},
                    {"model": "other/model", "endpoint": "/v1/chat/completions"},
                    {"model": SUBMIT_MODEL, "endpoint": "/wrong"})):
                class Client:
                    def get(self, batch_id):
                        return Reply(200, {"id": batch_id, "status": "failed",
                            "results": None, **identity})

                with patch("summary.luna_v1.ledger.time.time",
                           return_value=base + 121 + number * 301):
                    result = _poll_attempt(ledger=self.ledger,
                        attempt=self.ledger.get_batch_attempt(attempt_id),
                        workflow=self.workflow, private_root=self.root,
                        client_factory=lambda token: Client())
                self.assertEqual(result["reason"], "terminal_route_identity_unverified")
                self.assertIsNone(self.ledger.get_batch_attempt(attempt_id)["terminal_path"])
                self.assertEqual(self.ledger.batch_items(attempt_id)[0]["status"], "pending")
            class GoodClient:
                def get(self, batch_id):
                    return Reply(200, {"id": batch_id, "status": "failed",
                        "model": SUBMIT_MODEL, "endpoint": "/v1/chat/completions",
                        "results": None})

            with patch("summary.luna_v1.ledger.time.time",
                       return_value=base + 121 + 3 * 301):
                result = _poll_attempt(ledger=self.ledger,
                    attempt=self.ledger.get_batch_attempt(attempt_id),
                    workflow=self.workflow, private_root=self.root,
                    client_factory=lambda token: GoodClient())
        self.assertEqual(result["status"], "failed")

    def test_needs_review_reaches_writer_and_extraction_payloads(self):
        transcript = Path(self.temp.name) / "needs_review.json"
        transcript.write_text(json.dumps({
            "source": "01.01.2026 test.mkv", "duration_seconds": 4,
            "speakers": {}, "utterances": [{"start": 0, "end": 3,
                "speaker": None, "text": "Неясная реплика.",
                "uncertainty": {"needs_review": True}}]}, ensure_ascii=False),
            encoding="utf-8")
        snapshot = load_snapshot(transcript)
        packet = plan_packets(snapshot)[0]
        self.assertTrue(_writer_payload(snapshot)["SOURCE"]["utterances"][0]["needs_review"])
        self.assertTrue(_extraction_payload(packet)["CORE_SOURCE"][0]["needs_review"])

    def test_drift_cancels_reserved_post_and_waits_for_submitted_batch(self):
        self.ledger.finish_batch_workflow(self.workflow["id"], status="failed",
                                          error_code="fixture_closed")
        workflow, transcript = self._source_workflow("drift")
        submitted_id, submitted_sha = self._reserve_stage(workflow, "writer")
        self.ledger.mark_batch_submitting(submitted_id, submitted_sha)
        self.ledger.record_batch_submission(submitted_id, remote_id="batch_drift123")
        reserved_id, _ = self._reserve_stage(workflow, "audit")
        transcript.write_text('{"utterances":[{"text":"changed"}]}', encoding="utf-8")

        class Credential:
            def __init__(self, *_):
                pass

            def reveal_for_existing_job(self, *_):
                return "offline-token"

        class Client:
            def __init__(self):
                self.gets = 0
                self.posts = 0

            def get(self, batch_id):
                self.gets += 1
                if self.gets == 1:
                    return Reply(200, {"id": batch_id, "status": "in_progress"})
                return Reply(200, {"id": batch_id, "status": "failed",
                    "model": SUBMIT_MODEL, "endpoint": "/v1/chat/completions",
                    "results": None})

            def submit_prepared(self, *_args, **_kwargs):
                self.posts += 1
                raise AssertionError("drifted source dispatched")

            def delete(self, batch_id):
                return Reply(200, {"id": batch_id, "deleted": True})

        client = Client()
        base = time.time()
        with patch("summary.luna_v1.source_first_runtime.CredentialStore", Credential):
            with patch("summary.luna_v1.ledger.time.time", return_value=base + 121):
                first = poll_source_first_once(private_root=self.root,
                    client_factory=lambda token: client)
            self.assertEqual(self.ledger.get_batch_attempt(reserved_id)["status"],
                             "cancelled_before_submit")
            self.assertEqual(self.ledger.get_batch_workflow(workflow["id"])["status"],
                             "active")
            self.assertTrue(any(row["status"] == "source_revision_obsolete_pending"
                                for row in first))
            with patch("summary.luna_v1.ledger.time.time", return_value=base + 621):
                second = poll_source_first_once(private_root=self.root,
                    client_factory=lambda token: client)
        self.assertTrue(any(row["status"] == "source_revision_changed" for row in second))
        self.assertEqual(self.ledger.get_batch_attempt(submitted_id)["status"], "failed")
        self.assertEqual(self.ledger.get_batch_workflow(workflow["id"])["status"], "failed")
        self.assertEqual(client.posts, 0)

    def test_reserved_wave_one_peer_failure_blocks_resume_post(self):
        scenarios = (("writer_cancelled", "writer", "extract", "cancel"),
                     ("writer_rejected", "writer", "extract", "reject"),
                     ("extract_cancelled", "extract", "writer", "cancel"))
        for label, peer_stage, waiting_stage, peer_outcome in scenarios:
            workflow, _ = self._source_workflow(label)
            peer_id, peer_sha = self._reserve_stage(workflow, peer_stage)
            waiting_id, _ = self._reserve_stage(workflow, waiting_stage)
            if peer_outcome == "cancel":
                self.ledger.cancel_batch_before_submit(peer_id, "fixture_peer_cancelled")
            else:
                self.ledger.mark_batch_submitting(peer_id, peer_sha)
                self.ledger.record_batch_submission(peer_id,
                    error_code="rejected", definite_rejection=True)
            class NoDispatch:
                def __init__(self, *_):
                    raise AssertionError("peer failure dispatched reserved attempt")

            with (patch("summary.luna_v1.source_first_runtime.CredentialStore", NoDispatch),
                  patch("summary.luna_v1.source_first_runtime._advance_workflow",
                        return_value={"status": "pending"})):
                poll_source_first_once(private_root=self.root)
            self.assertEqual(self.ledger.get_batch_attempt(waiting_id)["status"],
                             "cancelled_before_submit", label)

    def test_reserved_extract_waits_for_writer_remote_acknowledgement(self):
        workflow, _ = self._source_workflow("writer_unacknowledged")
        writer_id, _ = self._reserve_stage(workflow, "writer")
        extract_id, _ = self._reserve_stage(workflow, "extract")

        class Credential:
            def __init__(self, path):
                self.path = Path(path)

            def reveal_for_dispatch(self, *_):
                return "offline-token"

        class Client:
            def __init__(self):
                self.posts = 0

            def submit_prepared(self, *_args, **_kwargs):
                self.posts += 1
                raise BatchError(None, "transport_unknown")

        client = Client()
        with (patch("summary.luna_v1.source_first_runtime.CredentialStore", Credential),
              patch("summary.luna_v1.source_first_runtime._advance_workflow",
                    return_value={"status": "pending"})):
            outcomes = poll_source_first_once(private_root=self.root,
                client_factory=lambda token: client)
        self.assertEqual(client.posts, 1)
        self.assertEqual(self.ledger.get_batch_attempt(writer_id)["status"],
                         "submission_unknown")
        self.assertEqual(self.ledger.get_batch_attempt(extract_id)["status"], "reserved")
        self.assertTrue(any(row.get("status") == "wave1_peer_pending" for row in outcomes))

    def _offline_replay(self, incomplete_inventory: bool = False,
                        uncertain_inventory: bool = False,
                        unresolved_relation: bool = False,
                        additional_unit: bool = False,
                        source_revision_drift: bool = False,
                        verification_new_finding: bool = False,
                        truncated_stage: str | None = None):
        transcript = Path(self.temp.name) / "transcript.json"
        transcript.write_text(json.dumps({
            "source": "01.01.2026 test.mkv", "duration_seconds": 18,
            "speakers": {"s1": "Аня"},
            "utterances": [
                {"start": 0, "end": 4, "speaker": "s1", "text": "Предлагаю проверить сигнал."},
                {"start": 5, "end": 8, "speaker": "s1", "text": "Или взять другую запись."},
                {"start": 9, "end": 12, "speaker": None, "text": "Условие ещё не согласовано."},
            ],
        }, ensure_ascii=False), encoding="utf-8")
        output = Path(self.temp.name) / "output"
        replay_private_root = Path(self.temp.name) / "private_replay"
        route = Route(SUBMIT_MODEL, PROVIDER, "workspace-1", 1_000_000, 32_000,
                      Decimal("0.000000001"), Decimal("0.00000001"),
                      Decimal("0.000000001"), Decimal(0), None,
                      batch_endpoint_model=MODEL, provider_endpoint_tag=PROVIDER)

        class Credential:
            def __init__(self, path):
                self.path = Path(path)

            def dispatch_candidates(self):
                return [{"id": "key-1", "version": 1, "workspace_id": "workspace-1"}]

            def reveal_for_dispatch(self, *_):
                return "offline-token"

            def reveal_for_existing_job(self, *_):
                return "offline-token"

        class Client:
            def __init__(self):
                self.posts = []
                self.batches = {}
                self.deletes = []

            def submit_prepared(self, payload_bytes, *, expected_sha256):
                assert hashlib.sha256(payload_bytes).hexdigest() == expected_sha256
                envelope = json.loads(payload_bytes)
                stage = next(part for part in envelope["requests"][0]["custom_id"].split("-")
                             if part in {"writer", "extract", "audit", "global",
                                         "repair", "verify"})
                self.posts.append(stage)
                batch_id = f"batch_fake{len(self.posts):03d}"
                results = []
                for request in envelope["requests"]:
                    data = json.loads(request["body"]["messages"][1]["content"])
                    if stage == "writer":
                        report = _document()
                    elif stage == "extract":
                        ids = data["EXPECTED_CORE_IDS"]
                        report = {"schema_version": "luna_inventory_v1",
                            "complete": not incomplete_inventory,
                            "evidence": [{"evidence_id": "E1", "u_id": "U00001",
                                          "quote": "Предлагаю проверить сигнал."}],
                            "units": [{"unit_id": "A1", "kind": "action",
                                       "text": "Предложена проверка сигнала.",
                                       "evidence_ids": ["E1"],
                                       "facets": [{"facet_id": "F1", "axis": "status",
                                                   "value": "предложение", "evidence_ids": ["E1"]}]}],
                            "source_accounting": [
                                {"u_id": uid, "disposition": ("content" if uid == "U00001" else
                                    "uncertain" if uncertain_inventory and uid == "U00003" else
                                    "context_only"),
                                 "unit_ids": ["A1"] if uid == "U00001" else [], "note": None}
                                for uid in ids], "open_links": [],
                            "unprocessed_ids": ["U00003"] if incomplete_inventory else []}
                    elif stage == "audit":
                        surfaces = data["DOCUMENT_SURFACES"]
                        main = next(s for s in surfaces if s["field_key"] == "main.0.text")
                        report = {"schema_version": "luna_audit_v1", "complete": True,
                            "evidence": [{"evidence_id": "E1", "u_id": "U00001",
                                          "quote": "Предлагаю проверить сигнал."}],
                            "source_checks": [{"unit_id": "P001:A1",
                                "inventory_verdict": "source_supported", "coverage": "full",
                                "facet_checks": [{"facet_id": "P001:A1:F1", "verdict": "preserved",
                                    "document_evidence": [{"surface_id": main["surface_id"],
                                                           "quote": main["text_or_scalar"]}],
                                    "note": None}], "finding_ids": []}],
                            "document_checks": [{"surface_id": surface["surface_id"],
                                "claims": ([{"claim_text": str(surface["text_or_scalar"]),
                                            "verdict": "supported", "evidence_ids": ["E1"],
                                            "note": None}]
                                           if surface["text_or_scalar"] not in (None, "") else []),
                                "finding_ids": []} for surface in surfaces],
                            "additional_units": ([{"unit_id": "B2", "kind": "statement",
                                "text": "Отдельно обсуждена другая запись.",
                                "evidence_ids": ["E1"], "facets": []}]
                                if additional_unit else []),
                            "findings": ([{"finding_id": "F1", "kind": "omission",
                                "severity": "material", "affected_surface_ids": [main["surface_id"]],
                                "affected_unit_ids": ["P001:A1"], "evidence_ids": ["E1"],
                                "problem": "Проверить выражение предложения",
                                "required_preservation": "Сохранить статус предложения",
                                "proposed_resolution": None}]
                                if verification_new_finding else []),
                            "context_requests": ([{"request_id": "R1",
                                "affected_ids": [main["surface_id"]],
                                "question": "Установлена ли связь?",
                                "known_source_ids": ["U00001"],
                                "reason": "Контекст неясен"}]
                                if unresolved_relation else []),
                            "unprocessed_ids": []}
                    elif stage == "global":
                        report = {"schema_version": "luna_global_v1", "complete": True,
                                  "evidence": ([{"evidence_id": "E1", "u_id": "U00001",
                                                 "quote": "Предлагаю проверить сигнал."}]
                                                if unresolved_relation else []),
                                  "resolutions": ([{"request_id": "P001:audit:R1",
                                      "status": "unresolved", "conclusion": None,
                                      "evidence_ids": ["E1"],
                                      "affected_surface_ids": [
                                          data["CONTEXT_REQUESTS"][0]["affected_ids"][0]]}]
                                      if unresolved_relation else []),
                                  "link_checks": [],
                                  "additional_units": [], "findings": [],
                                  "affected_surfaces": [], "unprocessed_ids": []}
                    elif stage == "repair":
                        finding = data["FINDINGS"][0]
                        target = finding["affected_surface_ids"][0]
                        report = {"schema_version": "luna_patch_plan_v1", "complete": True,
                            "bundles": [{"bundle_id": "B1", "finding_ids": [finding["finding_id"]],
                                "evidence_ids": finding["evidence_ids"],
                                "affected_surface_ids": [target],
                                "operations": [{"kind": "replace_field", "target_id": target,
                                    "field_key": "main.0.text",
                                    "value_json": json.dumps("Обсудили проверку сигнала и условие.",
                                                             ensure_ascii=False),
                                    "position_after_id": None, "temp_id": None,
                                    "lineage_ids": []}],
                                "preservation_notes": "Сохранить предложение.",
                                "dependency_bundle_ids": []}],
                            "unresolved": [], "unprocessed_finding_ids": []}
                    else:
                        target = data["PATCH_BUNDLES"][0]["affected_surface_ids"][0]
                        report = {"schema_version": "luna_verification_v1", "complete": True,
                            "bundle_checks": [{"bundle_id": "B1", "verdict": "accept",
                                "evidence": [{"evidence_id": "E1", "u_id": "U00001",
                                              "quote": "Предлагаю проверить сигнал."}],
                                "preserved_surface_ids": [target], "regressions": [],
                                "note": "Проверено"}],
                            "new_findings": [{"finding_id": "F2", "kind": "new_dispute",
                                "severity": "material", "affected_surface_ids": [target],
                                "affected_unit_ids": [], "evidence_ids": ["E1"],
                                "problem": "Новая спорная формулировка",
                                "required_preservation": "Сохранить исходное предложение",
                                "proposed_resolution": None}],
                            "unprocessed_bundle_ids": []}
                    results.append({"custom_id": request["custom_id"],
                        "response": {"status_code": 200, "request_id": "req-test",
                            "body": {"id": "gen-test", "choices": [{"finish_reason": (
                                "length" if stage == truncated_stage else "stop"),
                                "message": {"role": "assistant", "content": json.dumps(report,
                                                                           ensure_ascii=False)}}],
                                "usage": {"prompt_tokens": 20, "completion_tokens": 5}}}})
                self.batches[batch_id] = {"id": batch_id, "status": "completed",
                    "model": SUBMIT_MODEL, "endpoint": "/v1/chat/completions",
                    "usage": {"cost": "0.001"}, "results": list(reversed(results))}
                return Reply(202, {"id": batch_id})

            def get(self, batch_id):
                return Reply(200, self.batches[batch_id])

            def delete(self, batch_id):
                self.deletes.append(batch_id)
                return Reply(200, {"id": batch_id, "deleted": True})

        client = Client()
        with (patch("summary.luna_v1.source_first_runtime.CredentialStore", Credential),
              patch("summary.luna_v1.source_first_runtime.verify_source_first_batch_route",
                    return_value=route)):
            submitted = submit_source_first(transcript_path=transcript,
                output_dir=output, private_root=replay_private_root,
                client_factory=lambda token: client)
            self.assertEqual(submitted["status"], "submitted")
            if source_revision_drift:
                transcript.write_text(transcript.read_text(encoding="utf-8").replace(
                    "Или взять другую запись.", "Или взять исправленную запись."),
                    encoding="utf-8")
            outcomes = []
            base = time.time()
            for tick in range(1, 6):
                with patch("summary.luna_v1.ledger.time.time", return_value=base + tick * 500):
                    outcomes.extend(poll_source_first_once(private_root=replay_private_root,
                        client_factory=lambda token: client))
                if any(row.get("status") == "accepted" for row in outcomes):
                    break
            if truncated_stage == "writer":
                self.assertTrue(any(row.get("status") == "generation_failed"
                                    for row in outcomes), outcomes)
                self.assertEqual(client.posts, ["writer", "extract"])
                self.assertFalse((output / "summary_current.json").exists())
                replay_ledger = Ledger(replay_private_root)
                try:
                    self.assertEqual(replay_ledger.get_batch_workflow(
                        submitted["workflow_id"])["status"], "failed")
                finally:
                    replay_ledger.close()
                return
            if source_revision_drift:
                self.assertTrue(any(row.get("status") == "source_revision_changed"
                                    for row in outcomes), outcomes)
                self.assertFalse((output / "summary_current.json").exists())
                second = submit_source_first(transcript_path=transcript,
                    output_dir=output, private_root=replay_private_root,
                    client_factory=lambda token: client)
                self.assertEqual(second["status"], "submitted")
                for tick in range(6, 12):
                    with patch("summary.luna_v1.ledger.time.time",
                               return_value=base + tick * 500):
                        outcomes.extend(poll_source_first_once(
                            private_root=replay_private_root,
                            client_factory=lambda token: client))
                    if any(row.get("status") == "accepted" for row in outcomes):
                        break
                self.assertNotEqual(second["workflow_id"], submitted["workflow_id"])
                replay_ledger = Ledger(replay_private_root)
                try:
                    self.assertEqual(replay_ledger.get_batch_workflow(
                        submitted["workflow_id"])["status"], "failed")
                finally:
                    replay_ledger.close()
        expected_posts = ["writer", "extract", "audit", "global"]
        if verification_new_finding:
            expected_posts.extend(["repair", "verify"])
        self.assertEqual(client.posts, (["writer", "extract"] + expected_posts)
                         if source_revision_drift else expected_posts)
        self.assertTrue(any(row.get("status") == "accepted" for row in outcomes), outcomes)
        pointer = json.loads((output / "summary_current.json").read_text(encoding="utf-8"))
        generation = output / "summary_generations" / pointer["generation_id"]
        sidecar = json.loads((generation / "review_sidecar.json").read_text(encoding="utf-8"))
        self.assertEqual(sidecar["review_status"],
                         "review_incomplete" if incomplete_inventory or uncertain_inventory or
                         truncated_stage == "extract" or
                         additional_unit or
                         verification_new_finding else
                         "reviewed_with_uncertainties" if unresolved_relation else
                         "review_completed")
        if incomplete_inventory:
            self.assertIn("U00003", sidecar["unreviewed_ids"])
        if unresolved_relation:
            model_document = json.loads((generation / "model_document.json").read_text(
                encoding="utf-8"))
            self.assertTrue(model_document["main"][0]["text"].startswith(
                "Требует проверки по источнику:"))
            self.assertIn("Связь между репликами", model_document["verification"][-1]["text"])
        if additional_unit:
            model_document = json.loads((generation / "model_document.json").read_text(
                encoding="utf-8"))
            self.assertIn("Обнаруженный при сверке исходный смысл",
                          model_document["verification"][-1]["text"])
        if verification_new_finding:
            self.assertIn("verify:F2", sidecar["model_disagreements"])
            model_document = json.loads((generation / "model_document.json").read_text(
                encoding="utf-8"))
            self.assertTrue(any("Новая спорная формулировка" in item["text"]
                                for item in model_document["verification"]))
        self.assertEqual(sidecar["source_scope_ids"], ["U00001", "U00002", "U00003"])
        self.assertTrue(json.loads((generation / "tasks.json").read_text(encoding="utf-8")))
        self.assertEqual(len(client.deletes), 6 if source_revision_drift else
                         6 if verification_new_finding else 4)
        with (patch("summary.luna_v1.source_first_runtime.CredentialStore",
                    side_effect=AssertionError("cache hit touched key")),
              patch("summary.luna_v1.source_first_runtime.verify_source_first_batch_route",
                    side_effect=AssertionError("cache hit touched provider"))):
            cached = submit_source_first(transcript_path=transcript,
                output_dir=output, private_root=replay_private_root)
        self.assertEqual(cached["status"], "accepted_cache_hit")

    def test_full_offline_wave_replay_publishes_one_generation(self):
        self._offline_replay()

    def test_incomplete_inventory_cannot_be_review_completed(self):
        self._offline_replay(incomplete_inventory=True)

    def test_uncertain_source_accounting_cannot_be_review_completed(self):
        self._offline_replay(uncertain_inventory=True)

    def test_writer_output_limit_fails_without_retry(self):
        self._offline_replay(truncated_stage="writer")

    def test_extract_output_limit_publishes_incomplete_without_retry(self):
        self._offline_replay(truncated_stage="extract")

    def test_unresolved_global_relation_has_local_source_warning(self):
        self._offline_replay(unresolved_relation=True)

    def test_additional_unit_without_finding_is_review_incomplete(self):
        self._offline_replay(additional_unit=True)

    def test_source_revision_drift_fails_old_job_and_allows_new_revision(self):
        self._offline_replay(source_revision_drift=True)

    def test_sixth_stage_new_finding_remains_visible_and_unaccepted(self):
        self._offline_replay(verification_new_finding=True)

    def test_one_reconciliation_conflict_does_not_starve_other_workflow(self):
        class FakeLedger:
            def list_pending_batch_attempts(self):
                return []

            def list_source_first_workflows(self, *, states):
                return [{"id": "first", "semantic_key": "a" * 64},
                        {"id": "second", "semantic_key": "b" * 64}]

            def list_batch_attempts(self, workflow_id):
                return []

            def close(self):
                pass

        with (patch("summary.luna_v1.source_first_runtime.Ledger",
                    return_value=FakeLedger()),
              patch("summary.luna_v1.source_first_runtime._advance_workflow",
                    side_effect=[ReconciliationConflict("ambiguous", ["card-1"]),
                                 {"status": "accepted", "workflow_id": "second"}])):
            results = poll_source_first_once(private_root=self.root)
        self.assertEqual([row["status"] for row in results],
                         ["publication_reconciliation_conflict", "accepted"])


if __name__ == "__main__":
    unittest.main()
