import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("quality_schema", ROOT / "scripts" / "quality_schema.py")
quality = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(quality)


class QualitySchemaTests(unittest.TestCase):
    def test_first_person_work_onset_survives_proposal_retyping_and_model_failure(self):
        fact = {
            "fact_id": "F-onset", "type": "proposal",
            "statement": "Обсуждалась структура, после чего участник начал её делать.",
            "speaker_refs": ["@A"], "owner_refs": [], "evidence_ids": ["U1"],
            "evidence": [{"id": "U1", "speaker": "@A", "text": "Я структуру пошёл делать."}],
            "uncertainty": {"needs_review": False},
        }
        record = quality.normalize_semantic_record({}, fact)
        self.assertEqual(record["assignees"], ["@A"])
        self.assertEqual(record["assignment_status"], "unconfirmed")
        self.assertEqual(len(record["actions"]), 1)
        self.assertEqual(record["actions"][0]["actor"], "@A")
        self.assertEqual(record["actions"][0]["predicate"].casefold(), "делать")
        self.assertEqual(record["actions"][0]["object"].casefold(), "структуру")
        self.assertEqual(record["actions"][0]["temporal_state"], "in_progress")
        self.assertEqual(record["actions"][0]["commitment_state"], "unknown")

    def test_work_onset_about_another_person_does_not_assign_the_speaker(self):
        fact = {
            "fact_id": "F-third-person", "type": "proposal",
            "statement": "Обсуждалось начало работы над панелью.",
            "speaker_refs": ["@A"], "owner_refs": [], "evidence_ids": ["U1"],
            "evidence": [{"id": "U1", "speaker": "@A", "text": "Я думаю, он начал делать панель."}],
            "uncertainty": {"needs_review": False},
        }
        record = quality.normalize_semantic_record({}, fact)
        self.assertEqual(record["assignees"], [])
        self.assertEqual(record["actions"], [])

    def test_source_first_person_commitment_restores_owner_after_editorial_rewrite(self):
        fact = {
            "fact_id": "F-demo", "type": "action",
            "statement": "Участник подготовит демонстрацию или отправит файл.",
            "speaker_refs": ["@A"], "owner_refs": [], "evidence_ids": ["U1"],
            "evidence": [{"id": "U1", "speaker": "@A",
                          "text": "К следующему разу я подготовлю демонстрацию или отправлю файл."}],
            "uncertainty": {"needs_review": False},
        }
        record = quality.normalize_semantic_record({"actions": [{
            "predicate": "prepare_or_send", "temporal_state": "unknown",
            "commitment_state": "unknown", "evidence_ids": ["U1"],
        }]}, fact)
        self.assertEqual(record["speech_act"], "commit")
        self.assertEqual(record["assignees"], ["@A"])
        self.assertEqual(record["commitment_actor"], "@A")
        self.assertEqual(len(record["actions"]), 1)
        self.assertEqual(record["actions"][0]["temporal_state"], "planned")
        self.assertEqual(record["actions"][0]["commitment_state"], "explicit_commitment")

    def test_source_past_attempt_overrides_model_in_progress_label(self):
        fact = {
            "fact_id": "F-past", "type": "action", "statement": "Проверка через модель.",
            "speaker_refs": ["@A"], "owner_refs": [], "evidence_ids": ["U1"],
            "evidence": [{"id": "U1", "speaker": "@A",
                          "text": "Я попробовал решить это через модель; это то, чем я занимался."}],
            "uncertainty": {"needs_review": False},
        }
        record = quality.normalize_semantic_record({"actions": [{
            "predicate": "проверить", "temporal_state": "in_progress",
            "commitment_state": "explicit_commitment", "evidence_ids": ["U1"],
        }]}, fact)
        self.assertEqual(record["assignees"], ["@A"])
        self.assertEqual(record["actions"][0]["temporal_state"], "past_attempt")
        self.assertEqual(record["actions"][0]["commitment_state"], "none")

    def test_structured_time_expression_is_normalized_to_text(self):
        fact = {
            "fact_id": "F00001", "type": "action", "statement": "Сделать отчёт",
            "speaker_refs": ["@Alpha"], "owner_refs": ["@Alpha"],
            "evidence_ids": ["U00001"],
            "evidence": [{"id": "U00001", "speaker": "@Alpha", "text": "Я сделаю отчёт"}],
            "uncertainty": {"needs_review": False},
        }
        record = quality.normalize_semantic_record(
            {"time_expression": {"text": "к следующему разу"}}, fact
        )
        self.assertEqual(record["time_expression"], "к следующему разу")

    def test_missing_asr_confidence_stays_unavailable(self):
        result = quality.word_uncertainty({"speaker": "A", "speaker_confidence": .8, "flags": []})
        self.assertIsNone(result["recognition"]["confidence"])
        self.assertEqual(result["recognition"]["confidence_source"], "unavailable")

    def test_inferred_speaker_reaches_utterance(self):
        result = quality.utterance_uncertainty([
            {"speaker": "A", "speaker_confidence": .75, "flags": ["speaker_context"]},
            {"speaker": "A", "speaker_confidence": .9, "flags": []},
        ])
        self.assertTrue(result["needs_review"])
        self.assertEqual(result["speaker"]["confidence"], .75)
        self.assertEqual(result["speaker"]["inferred_word_ratio"], .5)

    def test_assignee_requires_known_speaker_and_confirmation(self):
        fact = {
            "fact_id": "F00001", "type": "action", "statement": "Сделать отчёт",
            "speaker_refs": ["@Beta"], "owner_refs": ["@Beta"], "evidence_ids": ["U00001", "U00002"],
            "evidence": [{"id": "U00001", "speaker": "@Beta"}, {"id": "U00002", "speaker": "@Alpha"}],
            "uncertainty": {"needs_review": False},
        }
        record = quality.normalize_semantic_record({
            "assignees": ["@Beta", "@Invented"],
            "confirmation_evidence_ids": ["U00002", "U99999"],
            "conditions": [{"text": "после проверки", "evidence_ids": ["U00001"]}],
        }, fact)
        self.assertEqual(record["assignees"], ["@Beta"])
        self.assertEqual(record["confirmation_evidence_ids"], [])
        self.assertEqual(record["confirmation_utterances"], [])
        self.assertEqual(record["assignment_status"], "unconfirmed")
        self.assertEqual(record["conditions"][0]["evidence_ids"], ["U00001"])

    def test_quantity_without_valid_evidence_is_rejected(self):
        fact = {
            "fact_id": "F00001", "type": "observation", "statement": "Рост 10%",
            "speaker_refs": ["A"], "evidence_ids": ["U1"],
            "evidence": [{"id": "U1", "speaker": "A", "text": "Рост десять процентов"}],
            "uncertainty": {"needs_review": False},
        }
        record = quality.normalize_semantic_record({"quantities": [
            {"value": "10", "unit": "%", "evidence_ids": ["U1"]},
            {"value": "20", "unit": "%", "evidence_ids": ["U9"]},
        ]}, fact)
        self.assertEqual(record["quantities"][0]["value"], "10")
        self.assertEqual(record["quantities"][0]["unit"], "%")
        self.assertEqual(record["quantities"][0]["source_span"], "Рост десять процентов")

    def test_non_condition_and_non_numeric_quantity_are_rejected(self):
        fact = {
            "fact_id": "F00001", "type": "observation", "statement": "Есть небольшой аномалия",
            "speaker_refs": ["A"], "evidence_ids": ["U1"],
            "evidence": [{"id": "U1", "speaker": "A", "text": "Есть небольшой аномалия"}],
            "uncertainty": {"needs_review": False},
        }
        record = quality.normalize_semantic_record({
            "conditions": [{"text": "небольшой аномалия", "evidence_ids": ["U1"]}],
            "quantities": [{"value": "small", "evidence_ids": ["U1"]}],
            "proposed_by": ["A"],
        }, fact)
        self.assertEqual(record["conditions"], [])
        self.assertEqual(record["quantities"], [])
        self.assertEqual(record["proposed_by"], [])

    def test_time_range_is_not_misclassified_as_condition(self):
        fact = {
            "fact_id": "F00001", "type": "schedule", "statement": "Окно с 17 до 18",
            "speaker_refs": ["A"], "evidence_ids": ["U1"],
            "evidence": [{"id": "U1", "speaker": "A", "text": "Окно с 17 до 18"}],
            "uncertainty": {"needs_review": False},
        }
        record = quality.normalize_semantic_record({
            "conditions": [{"text": "окно с 17 до 18", "evidence_ids": ["U1"]}],
        }, fact)
        self.assertEqual(record["conditions"], [])

    def test_legacy_audited_owner_is_safe_fallback(self):
        fact = {
            "fact_id": "F00001", "type": "action", "statement": "A даст данные",
            "speaker_refs": ["A"], "owner_refs": ["A"], "evidence_ids": ["U1"],
            "evidence": [{"id": "U1", "speaker": "A", "text": "Я дам данные"}],
            "uncertainty": {"needs_review": False},
        }
        record = quality.normalize_semantic_record({"confirmation_evidence_ids": ["U1"]}, fact)
        self.assertEqual(record["assignees"], ["A"])
        self.assertEqual(record["assignment_status"], "confirmed")

    def test_unconfirmed_assignee_is_visible(self):
        fact = {
            "fact_id": "F00001", "type": "action", "statement": "Сделать отчёт",
            "speaker_refs": ["@Beta"], "evidence_ids": ["U00001"],
            "evidence": [{"speaker": "@Beta"}], "uncertainty": {"needs_review": False},
        }
        record = quality.normalize_semantic_record({"assignees": ["@Beta"]}, fact)
        self.assertEqual(record["assignees"], [])
        self.assertEqual(record["assignment_status"], "unknown")
        self.assertTrue(record["uncertainty"]["needs_review"])

    def test_nearby_speaker_cannot_replace_audited_task_owner(self):
        fact = {
            "fact_id": "F00126", "type": "action", "statement": "Помощник сделает разметчик",
            "speaker_refs": ["@Beta", "@Delta"], "owner_refs": ["@Beta"],
            "evidence_ids": ["U1", "U2"],
            "evidence": [{"id": "U1", "speaker": "@Beta"}, {"id": "U2", "speaker": "@Delta"}],
            "uncertainty": {"needs_review": False},
        }
        record = quality.normalize_semantic_record({
            "assignees": ["@Delta"], "confirmation_evidence_ids": ["U2"],
        }, fact)
        self.assertEqual(record["assignees"], ["@Beta"])
        self.assertEqual(record["assignment_status"], "unconfirmed")

    def test_two_source_actions_survive_one_opaque_model_frame(self):
        fact = {
            "fact_id": "F-order", "type": "action",
            "statement": "Создать разметчик и затем провести эксперименты.",
            "speaker_refs": ["@Beta"], "owner_refs": ["@Beta"],
            "evidence_ids": ["U1"],
            "evidence": [{
                "id": "U1", "speaker": "@Beta",
                "text": "Мне сделать тебе разметчик Event Mark и пойти экспериментировать с аномалиями?",
            }],
            "uncertainty": {"needs_review": False},
        }
        record = quality.normalize_semantic_record({"actions": [{
            "action_id": "opaque", "actor": "@Beta",
            "predicate": "create_event_marker_tool", "object": None,
            "recipient": None, "temporal_state": "planned",
            "commitment_state": "unknown", "evidence_ids": ["U1"],
            "actor_evidence_ids": ["U1"], "predicate_evidence_ids": ["U1"],
        }]}, fact)
        self.assertEqual(len(record["actions"]), 2)
        self.assertEqual(
            [item["predicate"].casefold() for item in record["actions"]],
            ["сделать", "экспериментировать"],
        )
        self.assertTrue(all(item["actor"] == "@Beta" for item in record["actions"]))

    def test_single_source_action_replaces_extra_opaque_model_frames(self):
        fact = {
            "fact_id": "F-demo", "type": "action",
            "statement": "Подготовить демонстрацию результата.",
            "speaker_refs": ["@Gamma"], "owner_refs": ["@Gamma"],
            "evidence_ids": ["U1"],
            "evidence": [{
                "id": "U1", "speaker": "@Gamma",
                "text": "Я подготовлю это в демо-панель.",
            }],
            "uncertainty": {"needs_review": False},
        }
        record = quality.normalize_semantic_record({"actions": [
            {"predicate": "commit_prepare_demo_panel", "actor": "@Gamma", "evidence_ids": ["U1"]},
            {"predicate": "show_demo_to_team", "actor": "@Gamma", "evidence_ids": ["U1"]},
        ]}, fact)
        self.assertEqual(len(record["actions"]), 1)
        self.assertEqual(record["actions"][0]["predicate"].casefold(), "подготовлю")

    def test_delivery_alternatives_remain_one_task_clause(self):
        fact = {
            "fact_id": "F-alternative", "type": "action",
            "statement": "Подготовить демонстрацию или передать файл.",
            "speaker_refs": ["@Gamma"], "owner_refs": ["@Gamma"],
            "evidence_ids": ["U1"],
            "evidence": [{
                "id": "U1", "speaker": "@Gamma",
                "text": "Я подготовлю в демо-панель или отправлю PDF-файл.",
            }],
            "uncertainty": {"needs_review": False},
        }
        record = quality.normalize_semantic_record({"actions": [{
            "predicate": "prepare_demo", "actor": "@Gamma", "evidence_ids": ["U1"],
        }]}, fact)
        self.assertEqual(len(record["actions"]), 1)
        self.assertIn("или отправлю PDF-файл", record["actions"][0]["object"])

    def test_model_action_type_does_not_turn_an_assertion_into_commitment(self):
        fact = {
            "fact_id": "F-state", "type": "action",
            "statement": "Система добавляет отметку на график.",
            "speaker_refs": ["@A"], "owner_refs": ["@A"],
            "evidence_ids": ["U1"],
            "evidence": [{"id": "U1", "speaker": "@A", "text": "Система добавляет отметку на график."}],
            "uncertainty": {"needs_review": False},
        }
        record = quality.normalize_semantic_record({"speech_act": "commit", "actions": [{
            "predicate": "добавляет", "actor": "@A", "temporal_state": "in_progress",
            "commitment_state": "unknown", "evidence_ids": ["U1"],
        }]}, fact)
        self.assertEqual(record["speech_act"], "assert")

    def test_conditional_instruction_is_typed_as_rule_without_domain_vocabulary(self):
        fact = {
            "fact_id": "F-rule", "type": "proposal",
            "statement": "Если запись просрочена, уведомление отправляется автоматически.",
            "speaker_refs": ["@A"], "owner_refs": [], "evidence_ids": ["U1"],
            "evidence": [{"id": "U1", "speaker": "@A", "text": "Если запись просрочена, уведомление отправляется автоматически."}],
            "uncertainty": {"needs_review": False},
        }
        record = quality.normalize_semantic_record({}, fact)
        self.assertEqual(record["content_kind"], "system_rule")

    def test_task_is_separate_record(self):
        record = {
            "record_id": "F00007", "kind": "action", "statement": "Проверить данные",
            "proposed_by": ["A"], "assignees": ["B"], "assignment_status": "confirmed",
            "conditions": [], "time_expression": None, "evidence_ids": ["U1"],
            "confirmation_evidence_ids": ["U2"], "uncertainty": {},
            "confirmation_utterances": [{"evidence_id": "U2", "speaker": "B", "text": "Да"}],
        }
        task = quality.task_records([record])[0]
        self.assertEqual(task["task_id"], "T00007")
        self.assertEqual(task["assignees"], ["B"])
        self.assertTrue(task["automation_eligible"])
        self.assertEqual(task["automation_status"], "eligible")

    def test_tasks_are_sorted_by_source_time(self):
        def record(record_id, start):
            return {
                "record_id": record_id, "kind": "action", "statement": record_id, "start": start,
                "proposed_by": [], "assignees": ["A"], "assignment_status": "confirmed",
                "conditions": [], "time_expression": None, "evidence_ids": ["U1"],
                "confirmation_evidence_ids": ["U1"], "uncertainty": {},
                "confirmation_utterances": [],
            }
        tasks = quality.task_records([record("F00002", 20), record("F00001", 10)])
        self.assertEqual([item["source_record_id"] for item in tasks], ["F00001", "F00002"])

    def test_uncertain_task_is_not_eligible_for_automatic_creation(self):
        record = {
            "record_id": "F00008", "kind": "action", "statement": "Проверить данные",
            "proposed_by": [], "assignees": ["B"], "assignment_status": "confirmed",
            "conditions": [], "time_expression": None, "evidence_ids": ["U1"],
            "confirmation_evidence_ids": ["U2"], "uncertainty": {"needs_review": True},
            "confirmation_utterances": [{"evidence_id": "U2", "speaker": "B", "text": "Да"}],
        }
        task = quality.task_records([record])[0]
        self.assertFalse(task["automation_eligible"])
        self.assertEqual(task["automation_status"], "unknown")

    def test_question_status_requires_grounded_answer(self):
        fact = {
            "fact_id": "F00009", "type": "question", "statement": "Нужно уточнить точный период.",
            "speaker_refs": ["A"], "evidence_ids": ["U1"],
            "evidence": [{"id": "U1", "speaker": "A", "text": "Какой именно период?"}],
            "uncertainty": {"needs_review": False},
        }
        record = quality.normalize_semantic_record({"question_status": "resolved"}, fact)
        self.assertEqual(record["question_status"], "unresolved")
        record = quality.normalize_semantic_record({"answer_evidence_ids": ["U1", "U9"]}, fact)
        self.assertEqual(record["question_status"], "resolved")
        self.assertEqual(record["answer_evidence_ids"], ["U1"])

    def test_transcript_verification_question_is_separate_from_discussion(self):
        fact = {
            "fact_id": "F00010", "type": "question",
            "statement": "Окончание фразы в стенограмме оборвано и требует проверки по аудио.",
            "speaker_refs": ["A"], "evidence_ids": ["U1"],
            "evidence": [{"id": "U1", "speaker": "A", "text": "неразборчиво"}],
            "uncertainty": {"needs_review": True},
        }
        record = quality.normalize_semantic_record({}, fact)
        self.assertEqual(record["question_kind"], "transcript_verification")

    def test_unconfirmed_meeting_time_remains_discussion_question(self):
        fact = {
            "fact_id": "F00011", "type": "question",
            "statement": "В распознанной реплике смешаны 20:00 и 19:00; подтверждения времени следующего созвона нет.",
            "speaker_refs": ["A"], "evidence_ids": ["U1"],
            "evidence": [{"id": "U1", "speaker": "A", "text": "Во вторник в семь?"}],
            "uncertainty": {"needs_review": True},
        }
        record = quality.normalize_semantic_record({}, fact)
        self.assertEqual(record["question_kind"], "discussion")
        self.assertEqual(record["question_status"], "unresolved")

    def test_meeting_state_has_stable_claims_relations_and_all_views(self):
        base = {
            "subject": "порог", "predicate": "установить", "object": "значение",
            "polarity": "positive", "conditions": [], "time_expression": None,
            "attributed_speakers": ["A"], "proposed_by": [], "assignees": [],
            "assignment_status": "not_applicable", "confirmation_evidence_ids": [],
            "confirmation_utterances": [], "question_status": "not_applicable",
            "answer_record_ids": [], "answer_evidence_ids": [], "uncertainty": {},
            "semantic_risks": ["quantity"], "risk_level": "HIGH", "source_word_ids": ["W1"],
            "topic": "Порог",
        }
        first = {**base, "record_id": "F00001", "kind": "proposal", "statement": "Предложен порог 0.5", "start": 1, "evidence_ids": ["U1"], "quantities": [{"value": "0.5", "evidence_ids": ["U1"]}], "modality": "proposed"}
        second = {**base, "record_id": "F00002", "kind": "decision", "statement": "Согласован порог 0.7", "start": 20, "evidence_ids": ["U2"], "quantities": [{"value": "0.7", "evidence_ids": ["U2"]}], "modality": "committed"}
        state = quality.meeting_state([first, second], provenance={"audio_sha256": "a" * 64})
        self.assertEqual(state["schema_version"], 4)
        self.assertIn("timeline", state["views"])
        self.assertIn("full_timeline", state["views"])
        self.assertIn("summary", state["views"])
        self.assertTrue(any(item["relation"] == "conflicts_with" for item in state["relations"]))
        self.assertTrue(all(item.get("decision_basis", {}).get("rule") for item in state["relations"]))
        self.assertEqual(state["views"]["summary"], [])
        self.assertTrue(all(item["provenance"]["source_word_ids"] for item in state["events"]))

    def test_explicit_correction_supersedes_low_overlap_claim(self):
        base = {
            "kind": "observation", "polarity": "positive", "modality": "asserted",
            "conditions": [], "quantities": [], "time_expression": None,
            "attributed_speakers": ["@Beta"], "proposed_by": [], "assignees": [],
            "assignment_status": "not_applicable", "confirmation_evidence_ids": [],
            "confirmation_utterances": [], "question_status": "not_applicable",
            "answer_record_ids": [], "answer_evidence_ids": [], "uncertainty": {},
            "risk_level": "HIGH", "source_word_ids": ["W1"], "topic": "Точка входа",
        }
        stale = {**base, "record_id": "F1", "statement": "Точка возникает после слома M15", "start": 10, "evidence_ids": ["U1"], "semantic_risks": []}
        corrected = {**base, "record_id": "F2", "statement": "Точка возникает после пересечения нарисованной линии", "start": 20, "evidence_ids": ["U2"], "semantic_risks": ["correction"], "revision_cue": True, "speech_act": "correct"}
        state = quality.meeting_state([stale, corrected], provenance={"audio_sha256": "a" * 64})
        self.assertTrue(any(item["relation"] == "corrects" for item in state["relations"]))
        self.assertEqual([item["source_record_id"] for item in state["views"]["summary"]], ["F2"])

    def test_signed_index_quantity_is_not_a_timeframe(self):
        fact = {
            "fact_id": "F1", "type": "metric", "statement": "Используется индекс -5",
            "speaker_refs": ["A"], "evidence_ids": ["U1"],
            "evidence": [{"id": "U1", "speaker": "A", "text": "берём позицию окна с индексом -5"}],
            "uncertainty": {"needs_review": False},
        }
        record = quality.normalize_semantic_record({"quantities": [{"value": "-5", "unit": "таймфрейм", "evidence_ids": ["U1"]}]}, fact)
        self.assertEqual(record["quantities"][0]["entity"], "window_index")
        self.assertEqual(record["quantities"][0]["role"], "index_offset")
        self.assertIsNone(record["quantities"][0]["unit"])

    def test_risk_scheduler_spends_audio_compute_only_on_critical(self):
        self.assertEqual(quality.adaptive_compute_plan({"risk_level": "LOW", "kind": "observation"})["passes"], ["deterministic"])
        self.assertIn("audio_repair", quality.adaptive_compute_plan({"risk_level": "CRITICAL", "kind": "action"})["passes"])


if __name__ == "__main__":
    unittest.main()
