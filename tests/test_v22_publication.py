import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from semantics.meeting_graph import build_meeting_graph
from semantics.propositions import proposition_from_record
from summary.planner import plan
from summary.verifier import build_public_items, can_publish_as_decision, runtime_quality_gates, verify_generated_items
from summary.views import project_views


def record(record_id, kind, statement, act="assert", speaker="@A", **extra):
    return {"record_id": record_id, "kind": kind, "statement": statement, "speech_act": act,
            "modality": extra.pop("modality", "asserted"), "evidence_ids": ["U" + record_id[1:]],
            "source_word_ids": ["W" + record_id[1:]], "attributed_speakers": [speaker],
            "start": float(record_id[1:]), **extra}


class V22PublicationTests(unittest.TestCase):
    def test_unknown_kind_fails_closed(self):
        with self.assertRaises(ValueError):
            proposition_from_record(record("F1", "made_up_kind", "text"))

    def test_rule_or_design_is_not_accepted_decision(self):
        graph = build_meeting_graph([record("F1", "design_choice", "Можно применить TPO", "propose")])
        claim = graph["claims"][0]
        self.assertFalse(can_publish_as_decision(claim))
        self.assertFalse(any(x["status"] == "accepted" for x in graph["decision_states"]))

    def test_short_reply_requires_unique_local_other_speaker(self):
        ambiguous = build_meeting_graph([
            record("F1", "proposal", "Вариант A", "propose", speaker="@A"),
            record("F2", "proposal", "Вариант B", "propose", speaker="@B"),
            record("F3", "observation", "Да", "accept", speaker="@C"),
        ])
        self.assertFalse(any(x["type"] == "accepts" for x in ambiguous["relations"]))

    def test_explicit_revision_supersedes_old_scope(self):
        graph = build_meeting_graph([
            record("F1", "action", "Передать Bitcoin за 2021 год", "commit", scope="2021 год"),
            record("F2", "action", "Передать Bitcoin за месяц", "commit", scope="месяц", supersedes_record_ids=["F1"]),
        ])
        old = next(x for x in graph["claims"] if x["source_record_id"] == "F1")
        new = next(x for x in graph["claims"] if x["source_record_id"] == "F2")
        self.assertEqual(old["lifecycle"], "superseded")
        self.assertEqual(new["lifecycle"], "active")

    def test_automation_unknown_does_not_change_task_status(self):
        graph = build_meeting_graph([record("F1", "action", "Отправить EXE", "commit")])
        task = graph["task_states"][0]
        self.assertEqual(task["status"], "self_committed")
        self.assertEqual(task["automation_eligible"], "unknown")

    def test_partial_question_remains_public(self):
        graph = build_meeting_graph([
            record("F1", "question", "Когда и где встречаемся?", "ask", requested_slots=["day", "time"], answered_slots=["day"], answer_record_ids=["F2"]),
            record("F2", "schedule", "В понедельник", "answer"),
        ])
        self.assertEqual(graph["question_states"][0]["status"], "partially_answered")
        result = plan(graph["claims"], graph["episodes"], graph["relations"], lambda _: 1)
        public = build_public_items(graph, result)
        self.assertTrue(any(x["section"] == "questions" for x in public))

    def test_every_view_obeys_its_plan(self):
        graph = build_meeting_graph([record("F1", "observation", "A"), record("F2", "metric", "B")])
        plans = {"view_plans": {name: {"selected_claim_ids": []} for name in ("executive", "technical", "tasks", "experiments", "questions", "minutes")}}
        self.assertTrue(all(not values for values in project_views(graph, plans).values()))

    def test_orphan_and_mutations_are_rejected(self):
        claim = {"claim_id": "C1", "statement": "Если рынок открыт, нельзя входить на 10%", "lifecycle": "active", "evidence_ids": ["U1"]}
        contract = {"claim_ids": ["C1"], "relation_ids": [], "allowed_numbers": ["10%"], "allowed_relation_markers": [], "allowed_speakers": [], "allowed_assignees": [], "polarity": ["negative"], "modality": ["possible"], "conditions": ["если рынок открыт"], "time_scope": []}
        for text in ("Если рынок открыт, нельзя входить на 20%", "Если рынок открыт, входить на 10%", "Решено: если рынок открыт, нельзя входить на 10%"):
            self.assertFalse(verify_generated_items([{"text": text, "claim_ids": ["C1"]}], [contract], [claim])["passed"])
        orphan = verify_generated_items([{"text": "Unsupported", "claim_ids": []}], [contract], [claim])
        self.assertFalse(orphan["passed"])

    def test_runtime_gate_binds_verified_hash(self):
        report = {"audits": [{"passed": True, "errors": []}]}
        gates = runtime_quality_gates(report, "ok")
        self.assertTrue(gates["passed"])
        self.assertFalse(runtime_quality_gates(report, "changed", gates["verified_artifact_hash"])["passed"])


if __name__ == "__main__":
    unittest.main()
