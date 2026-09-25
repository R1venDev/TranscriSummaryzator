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
from summary.luna_v1.batch import Reply
from summary.luna_v1.engine import poll_once, submit
from summary.luna_v1.ledger import Ledger
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

    def __init__(self, token):
        assert token == "synthetic-secret-never-sent"

    def submit(self, custom_id, request_body):
        self.__class__.submit_calls += 1
        self.__class__.custom_id = custom_id
        self.__class__.request_body = request_body
        return Reply(202, {"id": "batch-synthetic001", "status": "validating",
                           "model": "openai/gpt-6-luna-20260922"})

    def get(self, batch_id):
        assert batch_id == "batch-synthetic001"
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
        return Reply(200, {
            "id": batch_id, "status": "completed", "model": "openai/gpt-6-luna-20260922",
            "usage": {"cost": 0.003,
            "prompt_tokens": 300, "completion_tokens": 900},
            "results": [{"custom_id": self.__class__.custom_id,
                         "response": {"status_code": 200, "body": {
                             "model": "openai/gpt-6-luna-20260922",
                             "choices": [{"finish_reason": "stop", "message": {
                                 "content": json.dumps(document, ensure_ascii=False)}}],
                         }}, "error": None}],
        })

    def delete(self, batch_id):
        return Reply(200, {"id": batch_id, "deletion": {"openrouter": "deleted"}})


class EngineTests(unittest.TestCase):
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
            route = SimpleNamespace(
                reserve_microusd=lambda payload: 20_000,
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
                ledger.db.execute("UPDATE jobs SET next_poll_at=0")
                ledger.close()
                terminal = poll_once(private_root=private, client_factory=FakeClient)
                self.assertEqual(terminal[0]["status"], "accepted")
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
                recovered_consumer = poll_once(private_root=private, client_factory=FakeClient)
                self.assertEqual(recovered_consumer[0]["status"], "consumer_recovery")
                self.assertEqual(recovered_consumer[0]["consumer_results"][0]["status"], "published")
                self.assertTrue((pending_output / "summary_current.json").is_file())
                self.assertEqual(FakeClient.submit_calls, 1)
                again = submit(transcript_path=source, output_dir=output,
                               private_root=private, client_factory=FakeClient)
                self.assertEqual(again["status"], "accepted_cache_hit")
                self.assertEqual(FakeClient.submit_calls, 1)
                second_output = root / "outputs" / "other-consumer"
                second_output.mkdir()
                second_source = second_output / "transcript.json"
                second_source.write_bytes(source.read_bytes())
                reused = submit(transcript_path=second_source, output_dir=second_output,
                                private_root=private, client_factory=FakeClient)
                self.assertEqual(reused["status"], "accepted_cache_hit")
                self.assertEqual(reused["new_generations"], 0)
                self.assertEqual(FakeClient.submit_calls, 1)
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
                reserve_microusd=lambda payload: 20_000,
                workspace_id="synthetic-workspace",
                prompt_usd_per_token="0.00000005", completion_usd_per_token="0.00000025",
                cache_write_usd_per_token="0.0000000625", request_usd="0",
            )
            FakeClient.submit_calls = 0
            with patch("summary.luna_v1.engine._credential_store", return_value=FakeStore()), \
                 patch("summary.luna_v1.engine.verify_batch_route", return_value=route):
                started = submit(transcript_path=source, output_dir=output,
                                 private_root=private, client_factory=FakeClient)
            self.assertEqual(started["status"], "submitted")
            ledger = Ledger(private); ledger.db.execute("UPDATE jobs SET next_poll_at=0"); ledger.close()
            from summary.luna_v1 import publication
            real_replace = os.replace

            def fail_pointer(source_path, destination):
                if Path(destination) == output / "summary_current.json":
                    raise OSError("synthetic pointer failure")
                return real_replace(source_path, destination)

            with patch("summary.luna_v1.engine._credential_store", return_value=FakeStore()), \
                 patch.object(publication.os, "replace", side_effect=fail_pointer):
                interrupted = poll_once(private_root=private, client_factory=FakeClient)
            self.assertEqual(interrupted[0]["status"], "publication_pending_retry")
            self.assertFalse((output / "summary_current.json").exists())
            ledger = Ledger(private)
            self.assertEqual(ledger.get(started["job_id"])["status"], "completed_raw")
            ledger.db.execute("UPDATE jobs SET next_poll_at=0")
            ledger.close()
            recovered = poll_once(private_root=private, client_factory=FakeClient)
            self.assertEqual(recovered[0]["status"], "accepted")
            self.assertTrue((output / "summary_current.json").is_file())
            self.assertEqual(FakeClient.submit_calls, 1)

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
                reserve_microusd=lambda payload: 20_000,
                workspace_id="synthetic-workspace",
                prompt_usd_per_token="0.00000005", completion_usd_per_token="0.00000025",
                cache_write_usd_per_token="0.0000000625", request_usd="0",
            )
            FakeClient.submit_calls = 0
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
            ledger = Ledger(private); ledger.db.execute("UPDATE jobs SET next_poll_at=0"); ledger.close()
            original_commit = TaskStore.commit_reconcile
            edited = []

            def commit_after_edit(task_store, plan):
                if not edited:
                    task_store.update(action_id, 0, {"description": "Вручную уточнённая синтетическая задача."}, "admin")
                    edited.append(True)
                return original_commit(task_store, plan)

            with patch("summary.luna_v1.engine._credential_store", return_value=FakeStore()), \
                 patch.object(TaskStore, "commit_reconcile", commit_after_edit):
                interrupted = poll_once(private_root=private, client_factory=FakeClient)
            self.assertEqual(interrupted[0]["status"], "publication_pending_retry")
            self.assertEqual(interrupted[0]["reason"], "RevisionConflict")
            self.assertFalse((output / "summary_current.json").exists())
            ledger = Ledger(private)
            self.assertEqual(ledger.get(started["job_id"])["status"], "completed_raw")
            ledger.db.execute("UPDATE jobs SET next_poll_at=0")
            ledger.close()
            recovered = poll_once(private_root=private, client_factory=FakeClient)
            self.assertEqual(recovered[0]["status"], "accepted")
            pointer = json.loads((output / "summary_current.json").read_text())
            sealed = output / "summary_generations" / pointer["generation_id"]
            self.assertIn("Вручную уточнённая синтетическая задача.", (sealed / "summary.md").read_text())
            self.assertEqual(json.loads((sealed / "tasks.json").read_text())[0]["revision"], 1)
            self.assertEqual(len(list((output / "summary_generations").glob(".superseded-*"))), 1)
            self.assertEqual(FakeClient.submit_calls, 1)


if __name__ == "__main__":
    unittest.main()
