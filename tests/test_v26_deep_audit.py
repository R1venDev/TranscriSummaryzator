import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from semantics.equivalence import equivalent
from semantics.meeting_graph import build_meeting_graph
from semantics.questions import verify_slot_entailment
from summary.outcomes import build_outcome_cards
from summary.planner import plan
from summary.verifier import build_public_items, publication_audit
from scripts.summary_worker import _document_audit_nodes, retype_rejected_fact, time_link
from contracts import SCHEMA_VERSIONS


def rec(index, kind, statement, act="assert", speaker="@A", **extra):
    return {
        "record_id": f"F{index}", "kind": kind, "content_kind": kind,
        "statement": statement, "speech_act": act, "modality": "asserted",
        "evidence_ids": [f"U{index}"], "source_word_ids": [f"W{index}"],
        "attributed_speakers": [speaker], "start": float(index),
        "verification_status": "supported", **extra,
    }


class DeepAuditV26Tests(unittest.TestCase):
    def test_intervening_question_captures_yes_instead_of_proposal(self):
        graph = build_meeting_graph([
            rec(1, "proposal", "Предлагаю изменить схему", "propose", "@A"),
            rec(2, "question", "Экран видно?", "ask", "@B", requested_slots=["boolean"]),
            rec(3, "observation", "Да", "answer", "@A"),
        ])
        proposal = next(x for x in graph["claims"] if x["source_record_id"] == "F1")
        question = next(x for x in graph["claims"] if x["source_record_id"] == "F2")
        self.assertFalse(any(x["type"] == "accepts" and x["target_claim_id"] == proposal["claim_id"] for x in graph["relations"]))
        self.assertTrue(any(x["type"] == "answers" and x["target_claim_id"] == question["claim_id"] for x in graph["relations"]))

    def test_multiple_actions_keep_independent_roles_and_lineage(self):
        graph = build_meeting_graph([rec(
            1, "action", "Я подготовлю отчёт и отправлю файл", "commit", "@A",
            origin_id="OR1", assignees=["@A"], actions=[
                {"action_id": "A01", "actor": "@A", "predicate": "подготовлю", "object": "отчёт", "recipient": None,
                 "temporal_state": "planned", "commitment_state": "explicit_commitment", "evidence_ids": ["U1"],
                 "field_evidence": {"actor": ["U1"], "predicate": ["U1"], "object": ["U1"], "recipient": []}},
                {"action_id": "A02", "actor": "@A", "predicate": "отправлю", "object": "файл", "recipient": "@B",
                 "temporal_state": "planned", "commitment_state": "explicit_commitment", "evidence_ids": ["U1"],
                 "field_evidence": {"actor": ["U1"], "predicate": ["U1"], "object": ["U1"], "recipient": ["U1"]}},
            ],
        )])
        self.assertEqual(len(graph["task_states"]), 2)
        self.assertEqual({x["action_id"] for x in graph["task_states"]}, {"A01", "A02"})
        self.assertEqual({tuple(x["origin_ids"]) for x in graph["task_states"]}, {("OR1:A01",), ("OR1:A02",)})
        self.assertTrue(all(x["source_record_id"] == "F1" for x in graph["task_states"]))

    def test_semantic_equivalence_preserves_polarity_numbers_conditions_and_actor(self):
        base = {"text": "@A проверит 2 файла", "polarity": "positive", "assignee": "@A",
                "quantities": [{"value": 2, "dimension": "count", "object_binding": "файлы"}], "conditions": []}
        self.assertFalse(equivalent(base, {**base, "polarity": "negative", "text": "@A не проверит 2 файла"}))
        self.assertFalse(equivalent(base, {**base, "text": "@A проверит 3 файла", "quantities": [{"value": 3, "dimension": "count", "object_binding": "файлы"}]}))
        self.assertFalse(equivalent(base, {**base, "conditions": [{"predicate": "если получит доступ"}]}))
        self.assertFalse(equivalent(base, {**base, "assignee": "@B", "text": "@B проверит 2 файла"}))

    def test_structured_answer_cannot_override_opposite_source_text(self):
        result = verify_slot_entailment(
            ["boolean"],
            {"statement": "Нет.", "speech_act": "answer", "slots": {"boolean": "да"}},
            {"statement": "Продолжаем?"},
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["verification_status"], "contradicted")

    def test_unverifiable_claim_is_only_in_quarantine(self):
        graph = build_meeting_graph([rec(1, "decision", "Используем неизвестный метод", "decide", verification_status="insufficient_evidence")])
        planned = plan(graph["claims"], graph["episodes"], graph["relations"], lambda _: 10)
        public = build_public_items(graph, planned)
        self.assertTrue(public)
        self.assertEqual({item["section"] for item in public}, {"requires_verification"})
        self.assertFalse(any(item["section"] in {"overview", "decisions", "minutes"} for item in public))

    def test_resource_is_not_labeled_as_work_result(self):
        graph = build_meeting_graph([rec(1, "resource", "Упомянут набор данных")])
        cards = build_outcome_cards(graph)
        self.assertTrue(cards)
        self.assertIsNotNone(cards[0]["fields"]["mentioned_resource"])
        self.assertIsNone(cards[0]["fields"]["work_result"])

    def test_past_attempt_and_intent_are_preserved_as_states(self):
        graph = build_meeting_graph([
            rec(1, "action", "Я пробовал проверить отчёт", "assert", temporal_state="past_attempt", assignees=["@A"], commitment_actor="@A"),
            rec(2, "action", "Я попробую проверить файл", "commit", temporal_state="planned", commitment_state="intent_to_attempt", assignees=["@A"], commitment_actor="@A"),
        ])
        self.assertEqual({item["status"] for item in graph["task_states"]}, {"past_attempt", "intent_to_attempt"})

    def test_final_audit_inventory_includes_chronology_items(self):
        document = {
            "title": {"text": "Итоги", "claim_ids": ["C1"], "evidence_ids": ["U1"]},
            "overview": [], "navigation": [], "sections": {}, "outcome_cards": [],
            "chronology": [{"items": [{"text": "Факт", "claim_ids": ["C1"], "evidence_ids": ["U1"]}]}],
        }
        self.assertIn("chronology:1:1", {item["node_id"] for item in _document_audit_nodes(document)})

    def test_retyping_keeps_reported_work_instead_of_deleting_it(self):
        fact = {"fact_id": "F1", "type": "decision", "statement": "Я попробую проверить отчёт", "certainty": "explicit",
                "evidence_ids": ["U1"], "speaker_refs": ["@A"], "owner_refs": [],
                "evidence": [{"id": "U1", "speaker": "@A", "text": "Я попробую проверить отчёт"}]}
        repaired = retype_rejected_fact(fact, "Это не решение, а намерение попробовать")
        self.assertEqual(repaired["type"], "action")
        self.assertEqual(repaired["commitment_state"], "intent_to_attempt")
        self.assertEqual(repaired["reported_content_support"], "supported")

    def test_topic_entities_are_not_empty_when_topic_is_known(self):
        graph = build_meeting_graph([rec(1, "problem", "Сервис задерживает ответы", topic="Очередь обработки")])
        claim = graph["claims"][0]
        self.assertIn("Очередь обработки", claim["topic_entities"])

    def test_standalone_time_is_plain_and_absolute_application_link_is_supported(self):
        self.assertEqual(time_link(10.2, job_id=7), "00:00:10")
        self.assertEqual(time_link(10.2, job_id=7, base_url="https://example.test"), "[00:00:10](https://example.test/result?id=7#t-10200)")

    def test_exported_schema_versions_match_registry(self):
        graph = build_meeting_graph([rec(1, "observation", "Сервис работает")])
        self.assertEqual(graph["schema_version"], SCHEMA_VERSIONS["MeetingGraphSchema"])
        audit = publication_audit({"audits": [{"passed": True}]}, "# документ", [], {})
        self.assertEqual(audit["schema_version"], SCHEMA_VERSIONS["PublicationAuditSchema"])


if __name__ == "__main__":
    unittest.main()
