import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from meeting_intelligence import (  # noqa: E402
    apply_question_resolutions, build_summary_plan, consolidate_tasks,
    question_candidate_bundles, valid_hypothesis,
)


def record(record_id, kind, statement, start, **extra):
    return {"record_id": record_id, "kind": kind, "statement": statement, "start": start, **extra}


def fact(fact_id, kind, statement, start, topic="Тема"):
    return {
        "fact_id": fact_id, "type": kind, "statement": statement,
        "start": start, "end": start + 2, "topic": topic,
        "evidence_ids": ["U" + fact_id[1:]], "evidence": [],
        "uncertainty": {"needs_review": False},
    }


class MeetingIntelligenceTests(unittest.TestCase):
    def test_global_candidates_do_not_require_topic_equality(self):
        records = [
            record("F00001", "question", "Какие зоны интереса?", 10, topic="Вопрос"),
            record("F00002", "definition", "Это imbalance, blocks, Breaker и OTE", 13, topic="POI"),
        ]
        bundles = question_candidate_bundles(records)
        self.assertEqual(bundles[0]["candidates"][0]["record_id"], "F00002")

    def test_partial_answer_is_not_open(self):
        records = [
            record("F00001", "question", "Что делать?", 10),
            record("F00002", "proposal", "Сначала проверить фильтр", 12),
        ]
        resolved = apply_question_resolutions(records, [{
            "question_record_id": "F00001", "status": "partially_answered",
            "answer_record_ids": ["F00002"], "confidence": 0.9,
        }])
        self.assertEqual(resolved[0]["question_status"], "partially_answered")
        self.assertEqual(resolved[0]["answer_relation"], "partially_answers")

    def test_answer_may_come_directly_from_immutable_utterance(self):
        records = [
            record("F00001", "question", "Какой диапазон AM?", 10),
            record("F00099", "observation", "Поздняя несвязанная реплика", 800, evidence_ids=["U00099"]),
        ]
        turns = [{"id": "U00002", "start": 12, "end": 15, "speaker": "@A", "text": "С 17 до 18", "source_word_ids": ["W1"]}]
        resolved = apply_question_resolutions(records, [{
            "question_record_id": "F00001", "status": "answered",
            "answer_record_ids": ["F00099"], "answer_evidence_ids": ["U00002"],
            "confidence": 0.95,
        }], utterances=turns)
        self.assertEqual(resolved[0]["question_status"], "answered")
        self.assertEqual(resolved[0]["answer_evidence_ids"], ["U00002"])
        self.assertEqual(resolved[0]["answer_record_ids"], [])
        self.assertEqual(resolved[0]["answer_spans"][0]["source_word_ids"], ["W1"])

    def test_banter_is_not_hypothesis(self):
        item = fact("F00001", "hypothesis", "Когда Dow Jones появился, начало XX века?", 10)
        self.assertFalse(valid_hypothesis(item))

    def test_tasks_with_same_owner_and_deliverable_are_consolidated(self):
        tasks = [
            {"task_id": "T1", "source_record_id": "F1", "description": "Передать реализацию для симуляции", "assignees": ["@Yachoy"], "start": 10, "evidence_ids": ["U1"]},
            {"task_id": "T2", "source_record_id": "F2", "description": "Подготовить TradingView или EXE для симуляции", "assignees": ["@Yachoy"], "start": 30, "evidence_ids": ["U2"]},
        ]
        consolidated = consolidate_tasks(tasks)
        self.assertEqual(len(consolidated), 1)
        self.assertEqual(consolidated[0]["source_record_ids"], ["F1", "F2"])

    def test_summary_plan_is_selective_and_suppresses_acknowledgement(self):
        facts = [
            fact("F00001", "problem", "BOS без HTF-контекста даёт ложные входы", 10),
            fact("F00002", "observation", "Угу. Угу. Понятно.", 20),
            fact("F00003", "action", "Проверить HTF-фильтр в симуляции", 30),
        ]
        events = [
            {"event_id": "E1", "source_record_id": item["fact_id"], "content_kind": item["type"]}
            for item in facts
        ]
        state = {"events": events, "relations": [], "views": {"questions": []}}
        plan = build_summary_plan(facts, state)
        self.assertIn("F00001", plan["selected_fact_ids"])
        self.assertIn("F00003", plan["selected_fact_ids"])
        self.assertNotIn("F00002", plan["selected_fact_ids"])


if __name__ == "__main__":
    unittest.main()
