"""Selected local cards, manual revision and unified exports; fake source only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from summary.luna_v1 import SCHEMA_ID, load_source
from summary.luna_v1.publication import publish_document
from summary.luna_v1.task_api import TaskViewUnavailable, edit_current, read_current
from summary.luna_v1.tasks import RevisionConflict, TaskStore


def fake_document() -> dict:
    return {
        "schema_version": SCHEMA_ID,
        "meeting": {"topic": "Тестовая встреча", "project": None},
        "main": [{"text": "Обсудили проверку одного варианта.", "source_ids": ["U00001"]}],
        "timecodes": [{"topic": "Выбор варианта", "start_id": "U00001", "end_id": None}],
        "tasks": [{
            "title": "Проверить X или Y",
            "description": "Проверить один из вариантов X или Y после выбора; оба не требуются.",
            "discussion_status": "proposed", "assignee": None, "due": None,
            "priority": None, "recipient": None, "source_ids": ["U00001"],
            "field_sources": {"action": ["U00001"], "assignee": [], "due": [],
                              "priority": [], "recipient": [], "discussion_status": ["U00001"]},
        }],
        "questions": [], "technical": [], "ideas": [], "verification": [],
        "chapters": [{"topic": "Альтернативы", "start_id": "U00001", "end_id": None,
                      "summary": "Предложили проверить одну из альтернатив.",
                      "source_ids": ["U00001"], "details": []}],
    }


def verified_resolver(base: Path) -> Path | None:
    pointer = base / "summary_current.json"
    if not pointer.exists():
        return None
    selected = json.loads(pointer.read_text())
    package = base / "summary_generations" / selected["generation_id"]
    manifest = json.loads((package / "generation_manifest.json").read_text())
    for name, digest in manifest["artifact_sha256"].items():
        if hashlib.sha256((package / name).read_bytes()).hexdigest() != digest:
            return None
    return package


class LocalTaskApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.output = self.root / "meeting-output"
        self.output.mkdir()
        self.source = self.output / "transcript.json"
        self.source.write_text(json.dumps({
            "source": "01.09.2026 — Тест.mkv", "duration_seconds": 8,
            "speakers": {"p1": "А"},
            "utterances": [{"start": 0.4, "end": 7.5, "speaker": "p1",
                            "text": "Предлагаю проверить X или Y, не оба."}],
        }, ensure_ascii=False), encoding="utf-8")
        self.task_db = self.root / "private" / "tasks.sqlite3"
        self.document = fake_document()
        _, self.index, self.source_sha = load_source(self.source)
        store = TaskStore(self.task_db)
        plan = store.preview_reconcile(self.source_sha, self.document["tasks"])
        self.generation_id, self.generation = publish_document(
            document=self.document, source_index=self.index,
            transcript_path=self.source, output_dir=self.output,
            semantic_key="fake-test-key", job_id="fake-job", remote_batch_id="fake-batch",
            credential_id="fake-key", prompt_sha256="a" * 64, schema_sha256="b" * 64,
            effective_tasks=plan.effective_tasks,
            before_pointer=lambda: store.commit_reconcile(plan),
        )

    def test_manual_edit_survives_restart_and_one_effective_export(self):
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in self.generation.iterdir() if p.is_file()}
        selected = read_current(self.output, self.source, self.task_db, verified_resolver)
        task = selected.tasks[0]
        self.assertIsNone(task["assignee"])
        self.assertEqual(task["revision"], 0)
        self.assertEqual(selected.source_refs[task["action_id"]][0]["start_ms"], 400)
        edited = edit_current(
            self.output, self.source, self.task_db, verified_resolver,
            action_id=task["action_id"], expected_generation_id=selected.generation_id,
            expected_revision=task["revision"],
            changes={"description": "Проверить выбранный X либо Y; второй вариант не обязателен.",
                     "assignee": "А", "due": "2026-10-01"}, actor="admin",
        )
        self.assertEqual(edited.tasks[0]["revision"], 1)
        self.assertEqual(edited.tasks[0]["assignee"], "А")
        self.assertIn("**Исполнитель:** А", edited.rendered["summary.md"])
        self.assertIn("Проверить выбранный X либо Y", edited.rendered["summary.html"])
        self.assertEqual(edited.rendered["summary.json"]["tasks"], edited.rendered["tasks.json"])
        self.assertEqual(edited.rendered["tasks.json"], edited.tasks)
        self.assertEqual(json.loads((self.generation / "model_document.json").read_text()), self.document)
        self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in self.generation.iterdir() if p.is_file()})
        restarted = read_current(self.output, self.source, self.task_db, verified_resolver)
        self.assertEqual(restarted.tasks, edited.tasks)
        history = TaskStore(self.task_db).history(task["action_id"])
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["actor"], "admin")

    def test_stale_revision_generation_and_unknown_card_do_not_edit(self):
        selected = read_current(self.output, self.source, self.task_db, verified_resolver)
        action_id = selected.tasks[0]["action_id"]
        edit_current(self.output, self.source, self.task_db, verified_resolver,
                     action_id=action_id, expected_generation_id=selected.generation_id,
                     expected_revision=0, changes={"assignee": "А"}, actor="admin")
        with self.assertRaises(RevisionConflict):
            edit_current(self.output, self.source, self.task_db, verified_resolver,
                         action_id=action_id, expected_generation_id=selected.generation_id,
                         expected_revision=0, changes={"assignee": "Б"}, actor="admin")
        with self.assertRaises(RevisionConflict):
            edit_current(self.output, self.source, self.task_db, verified_resolver,
                         action_id=action_id, expected_generation_id="20000101-000000-aaaaaaaaaaaa",
                         expected_revision=1, changes={"due": "завтра"}, actor="admin")
        with self.assertRaises(TaskViewUnavailable):
            edit_current(self.output, self.source, self.task_db, verified_resolver,
                         action_id="A-" + "f" * 24, expected_generation_id=selected.generation_id,
                         expected_revision=0, changes={"due": "завтра"}, actor="admin")
        self.assertEqual(TaskStore(self.task_db).get(action_id)["assignee"], "А")
        self.assertEqual(len(TaskStore(self.task_db).history(action_id)), 1)

    def test_source_change_and_broken_generation_block_edit(self):
        old = self.source.read_bytes()
        source = json.loads(old)
        source["utterances"][0]["text"] = "Другой разговор."
        self.source.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
        with self.assertRaises(TaskViewUnavailable):
            read_current(self.output, self.source, self.task_db, verified_resolver)
        self.source.write_bytes(old)
        (self.generation / "model_document.json").write_text("{}")
        with self.assertRaises(TaskViewUnavailable):
            read_current(self.output, self.source, self.task_db, verified_resolver)


if __name__ == "__main__":
    unittest.main()
