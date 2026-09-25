"""Local task identity and edit tests use only artificial meeting text."""

from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest

from summary.luna_v1.render import render_document
from summary.luna_v1.tasks import ReconciliationConflict, RevisionConflict, TaskStore


SOURCE_SHA = hashlib.sha256(b"synthetic meeting revision").hexdigest()


def task(title, description, source_id):
    return {
        "title": title, "description": description, "discussion_status": "proposed",
        "assignee": None, "due": None, "priority": None, "recipient": None,
        "source_ids": [source_id],
        "field_sources": {
            "action": [source_id], "assignee": [], "due": [], "priority": [],
            "recipient": [], "discussion_status": [source_id],
        },
    }


def document(tasks):
    return {
        "schema_version": "luna_summary_v1",
        "meeting": {"topic": "Синтетическая встреча", "project": None},
        "main": [{"text": "Обсудили два следующих шага.", "source_ids": ["U00001", "U00002"]}],
        "timecodes": [], "tasks": tasks, "questions": [], "technical": [],
        "ideas": [], "verification": [], "chapters": [],
    }


INDEX = {
    "source_sha256": SOURCE_SHA, "meeting_date": None,
    "participants": ["А"], "unattributed_speech": False,
    "by_id": {
        "U00001": {"start_ms": 1000, "end_ms": 3000, "speaker": "А", "text": "Проверим X или Y."},
        "U00002": {"start_ms": 4000, "end_ms": 6000, "speaker": "Б", "text": "Подготовим данные."},
    },
}


class TaskStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "tasks.sqlite"
        self.store = TaskStore(self.path)
        self.generated = [
            task("Проверить X или Y", "Проверить одну из двух альтернатив после выбора варианта.", "U00001"),
            task("Подготовить данные", "Подготовить материалы для следующего разговора.", "U00002"),
        ]

    def test_unassigned_card_edit_restart_and_one_effective_export(self):
        sealed = deepcopy(self.generated)
        first = self.store.reconcile(SOURCE_SHA, self.generated)
        self.assertEqual(self.generated, sealed)
        self.assertEqual(len(first), 2)
        self.assertEqual(first[0]["revision"], 0)
        self.assertIsNone(first[0]["assignee"])
        updated = self.store.update(first[0]["action_id"], 0, {
            "title": "Проверить выбранную альтернативу",
            "description": "Проверить X либо Y после ручного выбора; оба не требуются.",
            "assignee": "А", "due": "2026-10-01",
        }, actor="local-admin")
        self.assertEqual(updated["revision"], 1)
        self.assertEqual(self.generated, sealed)
        restarted = TaskStore(self.path)
        effective = restarted.reconcile(SOURCE_SHA, self.generated)
        self.assertEqual(effective[0]["action_id"], first[0]["action_id"])
        self.assertEqual(effective[0]["description"], updated["description"])
        rendered = render_document(document(self.generated), INDEX, effective)
        self.assertIn("**Исполнитель:** А", rendered["summary.md"])
        self.assertEqual(rendered["tasks.json"][0]["description"], rendered["summary.json"]["tasks"][0]["description"])
        self.assertEqual(rendered["tasks.json"][0]["action_id"], updated["action_id"])
        self.assertEqual(rendered["tasks.json"][0]["source_ids"], ["U00001"])
        self.assertEqual(len(restarted.history(updated["action_id"])), 1)
        self.assertEqual(restarted.history(updated["action_id"])[0]["actor"], "local-admin")
        self.assertNotIn("action_id", self.generated[0])

    def test_regeneration_reorders_tasks_and_keeps_manual_override(self):
        first = self.store.reconcile(SOURCE_SHA, self.generated)
        chosen = first[0]["action_id"]
        self.store.update(chosen, 0, {"assignee": "А"}, "admin")
        revised = deepcopy(self.generated)
        revised[0]["description"] = "Проверить одну из двух альтернатив после согласования выбора."
        reordered = self.store.reconcile(SOURCE_SHA, list(reversed(revised)))
        self.assertEqual(reordered[1]["action_id"], chosen)
        self.assertEqual(reordered[1]["assignee"], "А")
        self.assertEqual(reordered[1]["revision"], 1)
        self.assertEqual(reordered[1]["description"], revised[0]["description"])

    def test_rejected_candidate_preview_does_not_touch_old_generation_or_db(self):
        accepted = self.store.reconcile(SOURCE_SHA, self.generated)
        ids = [item["action_id"] for item in accepted]
        self.store.update(ids[0], 0, {"assignee": "А"}, "admin")
        old_effective = self.store.effective_for_sealed(SOURCE_SHA, self.generated, ids)
        old_export = render_document(document(self.generated), INDEX, old_effective)
        before = hashlib.sha256(self.path.read_bytes()).hexdigest()
        candidate = deepcopy(self.generated)
        candidate[0]["description"] = "Новая, пока не принятая версия описания задачи."
        plan = self.store.preview_reconcile(SOURCE_SHA, candidate)
        self.assertEqual(plan.effective_tasks[0]["assignee"], "А")
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), before)
        # The candidate is rejected; no commit_reconcile or pointer change.
        after = self.store.effective_for_sealed(SOURCE_SHA, self.generated, ids)
        self.assertEqual(after, old_effective)
        self.assertEqual(render_document(document(self.generated), INDEX, after), old_export)
        self.assertEqual(self.store.get(ids[0])["description"], self.generated[0]["description"])

    def test_sealed_generation_reads_stay_stable_when_commit_precedes_pointer(self):
        accepted = self.store.reconcile(SOURCE_SHA, self.generated)
        ids = [item["action_id"] for item in accepted]
        candidate = deepcopy(self.generated)
        candidate[0]["description"] = "Уточнённый модельный текст в новой запечатанной версии."
        plan = self.store.preview_reconcile(SOURCE_SHA, candidate)
        self.store.commit_reconcile(plan)
        # Simulate failure before switching current pointer: old selected
        # document still provides all model-authored content.
        old = self.store.effective_for_sealed(SOURCE_SHA, self.generated, ids)
        self.assertEqual(old[0]["description"], self.generated[0]["description"])
        newer = self.store.effective_for_sealed(SOURCE_SHA, candidate, ids)
        self.assertEqual(newer[0]["description"], candidate[0]["description"])

    def test_candidate_plan_rejects_concurrent_manual_edit(self):
        accepted = self.store.reconcile(SOURCE_SHA, self.generated)
        candidate = deepcopy(self.generated)
        candidate[0]["description"] = "Проверить одну альтернативу с новым контекстом."
        plan = self.store.preview_reconcile(SOURCE_SHA, candidate)
        self.store.update(accepted[0]["action_id"], 0, {"assignee": "А"}, "admin")
        with self.assertRaises(RevisionConflict):
            self.store.commit_reconcile(plan)
        self.assertEqual(self.store.get(accepted[0]["action_id"])["description"], self.generated[0]["description"])

    def test_explicit_clear_and_stale_revision(self):
        item = self.store.reconcile(SOURCE_SHA, self.generated)[0]
        assigned = self.store.update(item["action_id"], 0, {"assignee": "А"}, "admin")
        cleared = self.store.update(item["action_id"], 1, {"assignee": None}, "admin")
        self.assertIsNone(cleared["assignee"])
        self.assertEqual(cleared["revision"], 2)
        with self.assertRaises(RevisionConflict):
            self.store.update(item["action_id"], 1, {"due": "завтра"}, "stale-browser")
        self.assertEqual(self.store.get(item["action_id"])["revision"], 2)
        later_model = deepcopy(self.generated)
        later_model[0]["assignee"] = "Б"
        later_model[0]["field_sources"]["assignee"] = ["U00001"]
        after = self.store.reconcile(SOURCE_SHA, later_model)
        self.assertIsNone(after[0]["assignee"])
        self.assertEqual([event["revision"] for event in self.store.history(item["action_id"])], [1, 2])

    def test_missing_manually_edited_task_rejected_without_partial_write(self):
        first = self.store.reconcile(SOURCE_SHA, self.generated)
        self.store.update(first[0]["action_id"], 0, {"title": "Человек уточнил задачу"}, "admin")
        with self.assertRaisesRegex(ReconciliationConflict, "missing from regeneration"):
            self.store.reconcile(SOURCE_SHA, [self.generated[1]])
        self.assertEqual(self.store.get(first[0]["action_id"])["title"], "Человек уточнил задачу")
        self.assertEqual(len(self.store.reconcile(SOURCE_SHA, self.generated)), 2)

    def test_different_action_on_same_source_does_not_inherit_edit(self):
        first = self.store.reconcile(SOURCE_SHA, [self.generated[0]])[0]
        self.store.update(first["action_id"], 0, {"assignee": "А"}, "admin")
        changed = [task("Удалить сервер", "Удалить сервер немедленно после встречи.", "U00001")]
        with self.assertRaises(ReconciliationConflict):
            self.store.reconcile(SOURCE_SHA, changed)

    def test_or_to_and_is_not_treated_as_the_same_action(self):
        first = self.store.reconcile(SOURCE_SHA, [self.generated[0]])[0]
        self.store.update(first["action_id"], 0, {"assignee": "А"}, "admin")
        changed = [task(
            "Проверить X и Y", "Проверить обе альтернативы после выбора варианта.", "U00001",
        )]
        with self.assertRaises(ReconciliationConflict):
            self.store.reconcile(SOURCE_SHA, changed)

    def test_two_actions_on_one_source_stay_distinct_or_conflict(self):
        both = [
            task("Проверить X", "Проверить только вариант X.", "U00001"),
            task("Подготовить Y", "Подготовить только вариант Y.", "U00001"),
        ]
        first = self.store.reconcile(SOURCE_SHA, both)
        self.assertNotEqual(first[0]["action_id"], first[1]["action_id"])
        duplicate = [deepcopy(both[0]), deepcopy(both[0])]
        with self.assertRaisesRegex(ReconciliationConflict, "duplicate generated"):
            self.store.reconcile(SOURCE_SHA, duplicate)

    def test_rejects_evidence_edit_and_empty_required_content(self):
        first = self.store.reconcile(SOURCE_SHA, self.generated)[0]
        for changes in ({"source_ids": ["U00002"]}, {"field_sources": {}}, {"title": " "}, {"description": None}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.store.update(first["action_id"], 0, changes, "admin")
        self.assertEqual(self.store.get(first["action_id"])["revision"], 0)


if __name__ == "__main__":
    unittest.main()
