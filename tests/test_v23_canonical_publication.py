import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from semantics.meeting_graph import build_meeting_graph
from summary.planner import plan
from summary.verifier import build_public_items, publication_audit, verify_generated_items


def rec(number, kind, text, act="assert", speaker="@A", **extra):
    return {
        "record_id": f"F{number}", "kind": kind, "statement": text,
        "speech_act": act, "modality": extra.pop("modality", "asserted"),
        "attributed_speakers": extra.pop("attributed_speakers", [speaker]),
        "evidence_ids": [f"U{number}"], "source_word_ids": [f"W{number}"],
        "start": float(number), **extra,
    }


class CanonicalStateTests(unittest.TestCase):
    def test_explicit_commitment_is_required(self):
        explicit = build_meeting_graph([rec(1, "action", "Я отправлю EXE-файл", "commit", assignees=["@A"])])
        proposed = build_meeting_graph([rec(1, "action", "Предлагалось отправить EXE-файл", "commit", assignees=["@A"])])
        self.assertEqual(explicit["task_states"][0]["status"], "self_committed")
        self.assertEqual(explicit["task_states"][0]["commitment_actor"], "@A")
        self.assertEqual(proposed["task_states"][0]["status"], "assigned_pending")

    def test_assignment_requires_local_acceptance(self):
        pending = build_meeting_graph([rec(1, "action", "@B должен отправить отчёт", "commit", assignees=["@B"])])
        accepted = build_meeting_graph([
            rec(1, "action", "@B должен отправить отчёт", "commit", assignees=["@B"]),
            rec(2, "observation", "Да, сделаю", "accept", speaker="@B"),
        ])
        self.assertEqual(pending["task_states"][0]["status"], "assigned_pending")
        self.assertEqual(accepted["task_states"][0]["status"], "accepted")
        self.assertTrue(accepted["task_states"][0]["acceptance_relation_ids"])

    def test_cross_kind_scope_revision_builds_one_current_envelope(self):
        graph = build_meeting_graph([
            rec(1, "action", "Я передам данные Bitcoin за 2021 год", "commit", assignees=["@A"]),
            rec(2, "proposal", "Нужны данные за один месяц", "propose", speaker="@B"),
            rec(3, "constraint", "Для симуляции месяца достаточно", speaker="@B"),
        ])
        task = graph["task_states"][0]
        self.assertEqual(task["current_scope"], "один месяц из 2021 года")
        self.assertIn("2021 год", task["superseded_scopes"])
        self.assertTrue(any(x["type"] == "revises_scope" for x in graph["relations"]))

    def test_similar_atomic_actions_have_one_canonical_task(self):
        graph = build_meeting_graph([
            rec(1, "action", "Я буду размечать и встраивать Order Block", "commit", assignees=["@A"]),
            rec(2, "action", "Я буду параллельно размечать Order Block", "commit", assignees=["@A"]),
        ])
        self.assertEqual(len(graph["task_states"]), 1)
        self.assertEqual(len(graph["task_states"][0]["source_proposition_ids"]), 2)

    def test_upstream_answer_without_answered_slots_is_preserved(self):
        graph = build_meeting_graph([
            rec(1, "question", "Закрывается ли сделка завтра?", "ask", requested_slots=["closing"], question_status="answered", answer_record_ids=["F2"]),
            rec(2, "observation", "Она закрывается автоматически", "answer", speaker="@B"),
        ])
        self.assertEqual(graph["question_states"][0]["status"], "answered")
        p = plan(graph["claims"], graph["episodes"], graph["relations"], lambda _: 1)
        self.assertFalse(any(x["section"] == "questions" for x in build_public_items(graph, p)))

    def test_hard_budgets_and_chronology_order(self):
        graph = build_meeting_graph([rec(i, "observation", f"Технический вывод номер {i}") for i in range(1, 40)])
        result = plan(graph["claims"], graph["episodes"], graph["relations"], lambda _: 1, max_units=5)
        for view in result["view_plans"].values():
            self.assertLessEqual(view["selected_count"], view["budget"])
        items = build_public_items(graph, result)
        starts = [x["start"] for x in items if x["section"] == "minutes"]
        self.assertEqual(starts, sorted(starts))


class PublicationDocumentGateTests(unittest.TestCase):
    def test_cross_item_mutations_are_rejected(self):
        base = {"public_id": "PI1", "section": "minutes", "text": "Проверить Order Block", "claim_ids": ["C1"], "evidence_ids": ["U1"], "source_word_ids": ["W1"], "content_kind": "observation", "social_state": "candidate", "start": 2}
        report = {"audits": [{"passed": True, "errors": []}]}
        duplicate = publication_audit(report, "x", [base, {**base, "public_id": "PI2", "start": 1}], {})
        self.assertFalse(duplicate["passed"])
        self.assertGreater(duplicate["duplicate_items"], 0)
        internal = publication_audit(report, "x", [{**base, "text": "self_committed"}], {})
        self.assertFalse(internal["passed"])
        self.assertEqual(internal["internal_labels_exposed"], 1)

    def test_public_task_matches_canonical_state(self):
        graph = build_meeting_graph([rec(1, "action", "Я отправлю отчёт", "commit", assignees=["@A"])])
        result = plan(graph["claims"], graph["episodes"], graph["relations"], lambda _: 1)
        items = build_public_items(graph, result)
        tasks = [x for x in items if x["section"] == "tasks"]
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["task_state_id"], graph["task_states"][0]["task_id"])
        audit = verify_generated_items(items, result["public_sentence_plans"], graph["claims"])
        self.assertTrue(audit["passed"], audit)


if __name__ == "__main__":
    unittest.main()
