"""End-to-end summary entry with fake transport and artificial source only."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from summary.luna_v1 import SCHEMA_ID, load_source
from summary.luna_v1.audit import AUDIT_SCHEMA_ID
from summary.luna_v1.batch import Reply
from summary.luna_v1.engine import poll_once, submit
from summary.luna_v1.ledger import Ledger
from summary.luna_v1.route import RouteBlocked
from summary.luna_v1.tasks import TaskStore


class FakeStore:
    def dispatch_candidates(self):
        return [{"id": "synthetic-key", "version": 1, "workspace_id": "synthetic-workspace"}]

    def reveal_for_dispatch(self, identifier, version):
        return "synthetic-secret-never-sent"

    def reveal_for_existing_job(self, identifier, version):
        return "synthetic-secret-never-sent"


class FakeClient:
    submit_calls = 0
    request_body = None
    submissions = {}
    report_factory = None

    def __init__(self, token):
        assert token == "synthetic-secret-never-sent"

    def submit(self, custom_id, request_body):
        self.__class__.submit_calls += 1
        self.__class__.custom_id = custom_id
        self.__class__.request_body = request_body
        batch_id = f"batch-synthetic{self.__class__.submit_calls:03d}"
        self.__class__.submissions[batch_id] = (custom_id, request_body)
        return Reply(202, {"id": batch_id, "status": "validating",
                           "model": "openai/gpt-6-luna-20260922"})

    def get(self, batch_id):
        custom_id, request_body = self.__class__.submissions[batch_id]
        document = {
            "schema_version": SCHEMA_ID,
            "meeting": {"topic": "Проверка данных", "project": None},
            "main": [{"text": "Участники предложили проверить один из вариантов.", "source_ids": ["U00001"]}],
            "timecodes": [{"topic": "Варианты", "start_id": "U00001", "end_id": None}],
            "tasks": [{
                "title": "Проверить X или Y", "description": "Проверить один выбранный вариант X или Y, не оба.",
                "discussion_status": "proposed", "assignee": None, "due": None,
                "priority": None, "recipient": None, "source_ids": ["U00001"],
                "field_sources": {"action": ["U00001"], "assignee": [], "due": [],
                                  "priority": [], "recipient": [], "discussion_status": ["U00001"]},
            }],
            "questions": [], "technical": [], "ideas": [], "verification": [],
            "chapters": [{"topic": "Проверка", "start_id": "U00001", "end_id": None,
                          "summary": "Обсудили проверку альтернатив.", "source_ids": ["U00001"],
                          "details": []}],
        }
        if request_body["response_format"]["json_schema"]["name"] == AUDIT_SCHEMA_ID:
            payload = json.loads(request_body["messages"][1]["content"])
            document = {
                "schema_version": AUDIT_SCHEMA_ID,
                "coverage": [{
                    "window_id": window["window_id"], "start_id": window["start_id"],
                    "end_id": window["end_id"], "salient": "Проверена альтернатива X или Y.",
                    "draft_coverage": "covered", "finding_indices": [],
                } for window in payload["SOURCE_WINDOWS"]],
                "findings": [], "patches": [],
            }
            if self.__class__.report_factory is not None:
                document = self.__class__.report_factory(payload, document)
        return Reply(200, {
            "id": batch_id, "status": "completed", "model": "openai/gpt-6-luna-20260922",
            "usage": {"cost": 0.003,
            "prompt_tokens": 300, "completion_tokens": 900},
            "results": [{"custom_id": custom_id,
                         "response": {"status_code": 200, "body": {
                             "model": "openai/gpt-6-luna-20260922",
                             "choices": [{"finish_reason": "stop", "message": {
                                 "content": json.dumps(document, ensure_ascii=False)}}],
                         }}, "error": None}],
        })

    def delete(self, batch_id):
        return Reply(200, {"id": batch_id, "deletion": {"openrouter": "deleted"}})


def arm_poll(private):
    ledger = Ledger(private)
    ledger.db.execute("UPDATE jobs SET next_poll_at=0")
    ledger.close()


def fake_poll(private, route):
    with patch("summary.luna_v1.engine._credential_store", return_value=FakeStore()), \
         patch("summary.luna_v1.engine.verify_batch_route", return_value=route):
        return poll_once(private_root=private, client_factory=FakeClient)


class EngineTests(unittest.TestCase):
    def test_pre_migration_batch_finishes_without_new_audit_post(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "outputs" / "meeting"
            output.mkdir(parents=True)
            source = output / "transcript.json"
            source.write_text(json.dumps({
                "source": "01.01.2030 — Синтетическая встреча.mkv", "duration_seconds": 8,
                "speakers": {"p1": "А"},
                "utterances": [{"start": 0.2, "end": 7.3, "speaker": "p1",
                                "text": "Предлагаю проверить X или Y, не оба."}],
            }, ensure_ascii=False))
            private = root / "state" / "summary_private"
            route = SimpleNamespace(
                reserve_microusd=lambda payload, **kwargs: 20_000,
                workspace_id="synthetic-workspace",
                prompt_usd_per_token="0.00000005", completion_usd_per_token="0.00000025",
                cache_write_usd_per_token="0.0000000625", request_usd="0",
            )
            FakeClient.submit_calls = 0
            FakeClient.report_factory = None
            with patch("summary.luna_v1.engine._credential_store", return_value=FakeStore()), \
                 patch("summary.luna_v1.engine.verify_batch_route", return_value=route):
                started = submit(transcript_path=source, output_dir=output,
                                 private_root=private, client_factory=FakeClient)
            self.assertEqual(started["status"], "submitted")
            ledger = Ledger(private)
            job = ledger.get(started["job_id"])
            manifest_path = Path(job["artifact_dir"]) / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            for field in ("quality_policy_version", "audit_prompt_sha256", "audit_schema_sha256"):
                manifest.pop(field)
            manifest_path.write_text(json.dumps(manifest))
            ledger.db.execute("UPDATE jobs SET next_poll_at=0 WHERE id=?", (job["id"],))
            ledger.close()
            events = fake_poll(private, route)
            self.assertIn("accepted_legacy", {event["status"] for event in events})
            self.assertEqual(FakeClient.submit_calls, 1)
            pointer = json.loads((output / "summary_current.json").read_text())
            generation = output / "summary_generations" / pointer["generation_id"]
            self.assertIsNone(json.loads((generation / "run_manifest.json").read_text())["quality_review"])

    def test_audit_unavailable_still_publishes_with_visible_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "outputs" / "meeting"
            output.mkdir(parents=True)
            source = output / "transcript.json"
            source.write_text(json.dumps({
                "source": "01.01.2030 — Синтетическая встреча.mkv", "duration_seconds": 8,
                "speakers": {"p1": "А"},
                "utterances": [{"start": 0.2, "end": 7.3, "speaker": "p1",
                                "text": "Предлагаю проверить X или Y, не оба."}],
            }, ensure_ascii=False))
            private = root / "state" / "summary_private"
            route = SimpleNamespace(
                reserve_microusd=lambda payload, **kwargs: 20_000,
                workspace_id="synthetic-workspace",
                prompt_usd_per_token="0.00000005", completion_usd_per_token="0.00000025",
                cache_write_usd_per_token="0.0000000625", request_usd="0",
            )
            FakeClient.submit_calls = 0
            FakeClient.report_factory = None
            with patch("summary.luna_v1.engine._credential_store", return_value=FakeStore()), \
                 patch("summary.luna_v1.engine.verify_batch_route", return_value=route):
                submit(transcript_path=source, output_dir=output,
                       private_root=private, client_factory=FakeClient)
            arm_poll(private)
            with patch("summary.luna_v1.engine._credential_store", return_value=FakeStore()), \
                 patch("summary.luna_v1.engine.verify_batch_route",
                       side_effect=RouteBlocked("audit_route_unavailable")):
                events = poll_once(private_root=private, client_factory=FakeClient)
            self.assertIn("accepted", {event["status"] for event in events})
            self.assertEqual(FakeClient.submit_calls, 1)
            pointer = json.loads((output / "summary_current.json").read_text())
            generation = output / "summary_generations" / pointer["generation_id"]
            self.assertIn("Автоматическая смысловая проверка завершилась не полностью",
                          (generation / "summary.md").read_text())
            self.assertEqual(json.loads((generation / "run_manifest.json").read_text())
                             ["quality_review"]["status"], "audit_unavailable")

    def test_inconsistent_audit_coverage_cannot_publish_as_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "outputs" / "meeting"
            output.mkdir(parents=True)
            source = output / "transcript.json"
            source.write_text(json.dumps({
                "source": "01.01.2030 — Синтетическая встреча.mkv", "duration_seconds": 8,
                "speakers": {"p1": "А"},
                "utterances": [{"start": 0.2, "end": 7.3, "speaker": "p1",
                                "text": "Предлагаю проверить X или Y, не оба."}],
            }, ensure_ascii=False))
            private = root / "state" / "summary_private"
            route = SimpleNamespace(
                reserve_microusd=lambda payload, **kwargs: 20_000,
                workspace_id="synthetic-workspace",
                prompt_usd_per_token="0.00000005", completion_usd_per_token="0.00000025",
                cache_write_usd_per_token="0.0000000625", request_usd="0",
            )

            def inconsistent_coverage(_payload, report):
                report["coverage"][0]["draft_coverage"] = "missing"
                return report

            FakeClient.submit_calls = 0
            FakeClient.report_factory = inconsistent_coverage
            with patch("summary.luna_v1.engine._credential_store", return_value=FakeStore()), \
                 patch("summary.luna_v1.engine.verify_batch_route", return_value=route):
                submit(transcript_path=source, output_dir=output,
                       private_root=private, client_factory=FakeClient)
            for expected in ("draft_ready", "stage_complete"):
                arm_poll(private)
                events = fake_poll(private, route)
                self.assertIn(expected, {event["status"] for event in events})
            self.assertEqual(FakeClient.submit_calls, 2)
            pointer = json.loads((output / "summary_current.json").read_text())
            generation = output / "summary_generations" / pointer["generation_id"]
            review = json.loads((generation / "run_manifest.json").read_text())["quality_review"]
            self.assertEqual(review["status"], "coverage_incomplete")
            self.assertEqual(review["coverage_warning_count"], 1)
            self.assertIn("полнота проверки не подтверждена", (generation / "summary.md").read_text())

    def test_omitted_action_is_added_and_verify_downgrades_uncertain_claim_before_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "outputs" / "meeting"
            output.mkdir(parents=True)
            source = output / "transcript.json"
            source.write_text(json.dumps({
                "source": "01.01.2030 — Синтетическая встреча.mkv", "duration_seconds": 12,
                "speakers": {"p1": "А"},
                "utterances": [
                    {"start": 0.2, "end": 5.0, "speaker": "p1", "text": "Предлагаю проверить X или Y, не оба."},
                    {"start": 5.2, "end": 11.0, "speaker": "p1", "text": "Ещё предлагаю записать результат проверки в журнал."},
                ],
            }, ensure_ascii=False))
            private = root / "state" / "summary_private"
            route = SimpleNamespace(
                reserve_microusd=lambda payload, **kwargs: 20_000,
                workspace_id="synthetic-workspace",
                prompt_usd_per_token="0.00000005", completion_usd_per_token="0.00000025",
                cache_write_usd_per_token="0.0000000625", request_usd="0",
            )

            def reports(payload, default):
                if payload["MODE"] == "audit":
                    task = {
                        "title": "Записать результат проверки в журнал",
                        "description": "После проверки выбранного варианта X или Y записать результат в журнал.",
                        "discussion_status": "proposed", "assignee": None, "due": None,
                        "priority": None, "recipient": None, "source_ids": ["U00002"],
                        "field_sources": {"action": ["U00002"], "assignee": [], "due": [],
                                          "priority": [], "recipient": [], "discussion_status": ["U00002"]},
                    }
                    default["coverage"][1]["draft_coverage"] = "missing"
                    default["coverage"][1]["finding_indices"] = [0]
                    default["findings"] = [{
                        "severity": "major", "kind": "omission", "description": "Пропущена запись результата в журнал.",
                        "source_ids": ["U00002"], "status": "repaired", "affected": [], "patch_indices": [0],
                    }]
                    default["patches"] = [{"section": "tasks", "operation": "insert", "index": 1,
                                          "item_json": json.dumps(task, ensure_ascii=False)}]
                else:
                    item = {"text": "Предложена проверка одного из вариантов X или Y; принятие не установлено.",
                            "source_ids": ["U00001"]}
                    default["findings"] = [{
                        "severity": "major", "kind": "modality", "description": "Принятие проверки не подтверждено.",
                        "source_ids": ["U00001"], "status": "unresolved",
                        "affected": [{"section": "main", "index": 0}], "patch_indices": [0],
                    }]
                    default["patches"] = [{"section": "main", "operation": "replace", "index": 0,
                                          "item_json": json.dumps(item, ensure_ascii=False)}]
                return default

            FakeClient.submit_calls = 0
            FakeClient.report_factory = reports
            with patch("summary.luna_v1.engine._credential_store", return_value=FakeStore()), \
                 patch("summary.luna_v1.engine.verify_batch_route", return_value=route):
                started = submit(transcript_path=source, output_dir=output,
                                 private_root=private, client_factory=FakeClient)
            self.assertEqual(started["status"], "submitted")
            for expected in ("draft_ready", "stage_complete", "stage_complete"):
                arm_poll(private)
                events = fake_poll(private, route)
                self.assertIn(expected, {event["status"] for event in events})
            self.assertEqual(FakeClient.submit_calls, 3)
            pointer = json.loads((output / "summary_current.json").read_text())
            generation = output / "summary_generations" / pointer["generation_id"]
            cards = json.loads((generation / "tasks.json").read_text())
            self.assertEqual(len(cards), 2)
            self.assertEqual(cards[1]["assignee"], None)
            markdown = (generation / "summary.md").read_text()
            self.assertIn("записать результат в журнал", markdown)
            self.assertIn("принятие не установлено", markdown)
            self.assertIn("Автоматическая проверка:", markdown)
            self.assertEqual(json.loads((generation / "run_manifest.json").read_text())
                             ["quality_review"]["status"], "postverify_corrected_unchecked")

    def test_one_dispatch_then_terminal_publication_and_zero_call_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "outputs" / "meeting"
            output.mkdir(parents=True)
            source = output / "transcript.json"
            source.write_text(json.dumps({
                "source": "01.01.2030 — Синтетическая встреча.mkv", "duration_seconds": 8,
                "speakers": {"p1": "А"},
                "utterances": [{"start": 0.2, "end": 7.3, "speaker": "p1",
                                "text": "Предлагаю проверить X или Y, не оба."}],
            }, ensure_ascii=False))
            private = root / "state" / "summary_private"
            FakeClient.submit_calls = 0
            FakeClient.report_factory = None
            route = SimpleNamespace(
                reserve_microusd=lambda payload, **kwargs: 20_000,
                workspace_id="synthetic-workspace",
                prompt_usd_per_token="0.00000005", completion_usd_per_token="0.00000025",
                cache_write_usd_per_token="0.0000000625", request_usd="0",
            )
            with patch("summary.luna_v1.engine._credential_store", return_value=FakeStore()), \
                 patch("summary.luna_v1.engine.verify_batch_route", return_value=route):
                started = submit(transcript_path=source, output_dir=output,
                                 private_root=private, client_factory=FakeClient)
                self.assertEqual(started["status"], "submitted")
                self.assertEqual(FakeClient.submit_calls, 1)
                self.assertFalse((output / "summary_current.json").exists())
                pending_output = root / "outputs" / "pending-second-consumer"
                pending_output.mkdir()
                pending_source = pending_output / "transcript.json"
                pending_source.write_bytes(source.read_bytes())
                shared = submit(transcript_path=pending_source, output_dir=pending_output,
                                private_root=private, client_factory=FakeClient)
                self.assertEqual(shared["status"], "submitted")
                self.assertEqual(shared["job_id"], started["job_id"])
                self.assertEqual(FakeClient.submit_calls, 1)
                ledger = Ledger(private)
                self.assertEqual(len(ledger.consumers(shared["semantic_key"])), 2)
                ledger.close()
                arm_poll(private)
                first = fake_poll(private, route)
                self.assertIn("draft_ready", {item["status"] for item in first})
                self.assertFalse((output / "summary_current.json").exists())
                self.assertEqual(FakeClient.submit_calls, 2)
                arm_poll(private)
                terminal = fake_poll(private, route)
                self.assertIn("accepted", {item["status"] for item in terminal})
                pointer = json.loads((output / "summary_current.json").read_text())
                generation = output / "summary_generations" / pointer["generation_id"]
                self.assertIn("**Исполнитель:** Не назначен", (generation / "summary.md").read_text())
                self.assertFalse((generation / "batch_delete.json").exists())
                self.assertEqual(json.loads((generation / "tasks.json").read_text())[0]["assignee"], None)
                pending_pointer = json.loads((pending_output / "summary_current.json").read_text())
                self.assertEqual(pending_pointer["generation_id"], pointer["generation_id"])
                restarted = Ledger(private)
                self.assertEqual({item["status"] for item in restarted.consumer_rows(shared["semantic_key"])}, {"published"})
                # Simulate a restart after the second generation was sealed
                # but before its pointer/consumer state was fully recorded.
                (pending_output / "summary_current.json").unlink()
                restarted.db.execute("UPDATE consumers SET status='pending',generation_id=NULL WHERE semantic_key=? AND output_dir=?",
                    (shared["semantic_key"], str(pending_output)))
                restarted.close()
                recovered_consumer = fake_poll(private, route)
                self.assertEqual(recovered_consumer[0]["status"], "consumer_recovery")
                self.assertEqual(recovered_consumer[0]["consumer_results"][0]["status"], "published")
                self.assertTrue((pending_output / "summary_current.json").is_file())
                self.assertEqual(FakeClient.submit_calls, 2)
                again = submit(transcript_path=source, output_dir=output,
                               private_root=private, client_factory=FakeClient)
                self.assertEqual(again["status"], "accepted_cache_hit")
                self.assertEqual(FakeClient.submit_calls, 2)
                second_output = root / "outputs" / "other-consumer"
                second_output.mkdir()
                second_source = second_output / "transcript.json"
                second_source.write_bytes(source.read_bytes())
                reused = submit(transcript_path=second_source, output_dir=second_output,
                                private_root=private, client_factory=FakeClient)
                self.assertEqual(reused["status"], "accepted_cache_hit")
                self.assertEqual(reused["new_generations"], 0)
                self.assertEqual(FakeClient.submit_calls, 2)
                second_pointer = json.loads((second_output / "summary_current.json").read_text())
                second_generation = second_output / "summary_generations" / second_pointer["generation_id"]
                self.assertEqual(json.loads((second_generation / "tasks.json").read_text())[0]["assignee"], None)
                self.assertEqual(json.loads((second_generation / "tasks.json").read_text())[0]["action_id"],
                                 json.loads((generation / "tasks.json").read_text())[0]["action_id"])

    def test_pointer_io_interruption_retries_saved_terminal_without_new_post(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "outputs" / "meeting"
            output.mkdir(parents=True)
            source = output / "transcript.json"
            source.write_text(json.dumps({
                "source": "01.01.2030 — Синтетическая встреча.mkv", "duration_seconds": 8,
                "speakers": {"p1": "А"},
                "utterances": [{"start": 0.2, "end": 7.3, "speaker": "p1",
                                "text": "Предлагаю проверить X или Y, не оба."}],
            }, ensure_ascii=False))
            private = root / "state" / "summary_private"
            route = SimpleNamespace(
                reserve_microusd=lambda payload, **kwargs: 20_000,
                workspace_id="synthetic-workspace",
                prompt_usd_per_token="0.00000005", completion_usd_per_token="0.00000025",
                cache_write_usd_per_token="0.0000000625", request_usd="0",
            )
            FakeClient.submit_calls = 0
            FakeClient.report_factory = None
            with patch("summary.luna_v1.engine._credential_store", return_value=FakeStore()), \
                 patch("summary.luna_v1.engine.verify_batch_route", return_value=route):
                started = submit(transcript_path=source, output_dir=output,
                                 private_root=private, client_factory=FakeClient)
            self.assertEqual(started["status"], "submitted")
            arm_poll(private)
            self.assertIn("draft_ready", {item["status"] for item in fake_poll(private, route)})
            arm_poll(private)
            from summary.luna_v1 import publication
            real_replace = os.replace

            def fail_pointer(source_path, destination):
                if Path(destination) == output / "summary_current.json":
                    raise OSError("synthetic pointer failure")
                return real_replace(source_path, destination)

            with patch("summary.luna_v1.engine._credential_store", return_value=FakeStore()), \
                 patch("summary.luna_v1.engine.verify_batch_route", return_value=route), \
                 patch.object(publication.os, "replace", side_effect=fail_pointer):
                interrupted = poll_once(private_root=private, client_factory=FakeClient)
            self.assertIn("quality_pending_retry", {item["status"] for item in interrupted})
            self.assertFalse((output / "summary_current.json").exists())
            ledger = Ledger(private)
            self.assertEqual(ledger.get(started["job_id"])["status"], "quality_pending")
            ledger.db.execute("UPDATE jobs SET next_poll_at=0")
            ledger.close()
            recovered = fake_poll(private, route)
            self.assertIn("accepted", {item["status"] for item in recovered})
            self.assertTrue((output / "summary_current.json").is_file())
            self.assertEqual(FakeClient.submit_calls, 2)

    def test_task_edit_race_rebuilds_sealed_projection_from_saved_raw_without_post(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "outputs" / "synthetic-meeting"
            output.mkdir(parents=True)
            source = output / "transcript.json"
            source.write_text(json.dumps({
                "source": "01.01.2030 — Синтетическая встреча.mkv", "duration_seconds": 8,
                "speakers": {"p1": "А"},
                "utterances": [{"start": 0.2, "end": 7.3, "speaker": "p1",
                                "text": "Предлагаю проверить X или Y, не оба."}],
            }, ensure_ascii=False))
            private = root / "state" / "summary_private"
            route = SimpleNamespace(
                reserve_microusd=lambda payload, **kwargs: 20_000,
                workspace_id="synthetic-workspace",
                prompt_usd_per_token="0.00000005", completion_usd_per_token="0.00000025",
                cache_write_usd_per_token="0.0000000625", request_usd="0",
            )
            FakeClient.submit_calls = 0
            FakeClient.report_factory = None
            with patch("summary.luna_v1.engine._credential_store", return_value=FakeStore()), \
                 patch("summary.luna_v1.engine.verify_batch_route", return_value=route):
                started = submit(transcript_path=source, output_dir=output,
                                 private_root=private, client_factory=FakeClient)
            self.assertEqual(started["status"], "submitted")
            raw = FakeClient("synthetic-secret-never-sent").get("batch-synthetic001")
            document = json.loads(raw.body["results"][0]["response"]["body"]["choices"][0]["message"]["content"])
            _, source_index, source_sha = load_source(source)
            store = TaskStore(private / "tasks.sqlite3")
            action_id = store.reconcile(source_sha, document["tasks"])[0]["action_id"]
            arm_poll(private)
            self.assertIn("draft_ready", {item["status"] for item in fake_poll(private, route)})
            arm_poll(private)
            original_commit = TaskStore.commit_reconcile
            edited = []

            def commit_after_edit(task_store, plan):
                if not edited:
                    task_store.update(action_id, 0, {"description": "Вручную уточнённая синтетическая задача."}, "admin")
                    edited.append(True)
                return original_commit(task_store, plan)

            with patch("summary.luna_v1.engine._credential_store", return_value=FakeStore()), \
                 patch("summary.luna_v1.engine.verify_batch_route", return_value=route), \
                 patch.object(TaskStore, "commit_reconcile", commit_after_edit):
                interrupted = poll_once(private_root=private, client_factory=FakeClient)
            self.assertIn("quality_pending_retry", {item["status"] for item in interrupted})
            self.assertIn("RevisionConflict", {item.get("reason") for item in interrupted})
            self.assertFalse((output / "summary_current.json").exists())
            ledger = Ledger(private)
            self.assertEqual(ledger.get(started["job_id"])["status"], "quality_pending")
            ledger.db.execute("UPDATE jobs SET next_poll_at=0")
            ledger.close()
            recovered = fake_poll(private, route)
            self.assertIn("accepted", {item["status"] for item in recovered})
            pointer = json.loads((output / "summary_current.json").read_text())
            sealed = output / "summary_generations" / pointer["generation_id"]
            self.assertIn("Вручную уточнённая синтетическая задача.", (sealed / "summary.md").read_text())
            self.assertEqual(json.loads((sealed / "tasks.json").read_text())[0]["revision"], 1)
            self.assertEqual(len(list((output / "summary_generations").glob(".superseded-*"))), 1)
            self.assertEqual(FakeClient.submit_calls, 2)


if __name__ == "__main__":
    unittest.main()
