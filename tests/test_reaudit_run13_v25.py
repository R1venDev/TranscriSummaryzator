import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from contracts.meeting import PublicItemContract
from pipeline import REQUIRED_GENERATION_FILES, current_summary_generation_id, current_summary_output
from scripts.speech_acts import primary_speech_act
from scripts.summary_worker import _chapter_label, build_public_document, deterministic_fact_check, render_public_document
from semantics.entities import EntityRegistry
from semantics.meeting_graph import build_meeting_graph
from semantics.propositions import proposition_from_record
from summary.outcomes import build_outcome_cards
from summary.planner import plan
from summary.verifier import build_public_items, verify_generated_items, verify_public_document


def rec(number, kind, statement, act="assert", **extra):
    return {
        "record_id": f"F{number}", "kind": kind, "statement": statement,
        "speech_act": act, "modality": extra.pop("modality", "asserted"),
        "attributed_speakers": extra.pop("attributed_speakers", ["@A"]),
        "evidence_ids": [f"U{number}"], "source_word_ids": [f"W{number}"],
        "start": float(number), "end": float(number) + 1, **extra,
    }


class Run13SemanticRegressions(unittest.TestCase):
    def test_contextual_commitment_and_acceptance_are_detected(self):
        self.assertEqual(primary_speech_act("Потом встрою это в методичку"), "commit")
        self.assertEqual(primary_speech_act("А, месяца. Ну ладно"), "accept")
        self.assertEqual(primary_speech_act("Да-да-да. Дальше этап апробации"), "accept")

    def test_ambiguous_alias_does_not_resolve_last_write_wins(self):
        registry = EntityRegistry()
        registry.register("Максим", "person", entity_id="E1")
        registry.register("Максим", "person", entity_id="E2")
        self.assertIsNone(registry.resolve("Максим", "person"))
        self.assertEqual({x["entity_id"] for x in registry.resolve_candidates("Максим")}, {"E1", "E2"})

    def test_direction_is_part_of_proposition_identity(self):
        base = {"kind": "metric", "statement": "Конверсия изменилась на 10%", "evidence_ids": ["U1"], "quantities": [{"value": 10, "unit": "%"}]}
        up = proposition_from_record({**base, "quantities": [{"value": 10, "unit": "%", "direction": "increase"}]})
        down = proposition_from_record({**base, "quantities": [{"value": 10, "unit": "%", "direction": "decrease"}]})
        self.assertNotEqual(up["proposition_id"], down["proposition_id"])

    def test_question_event_survives_shared_proposition(self):
        graph = build_meeting_graph([
            rec(1, "current_state", "Сервис готов."),
            rec(2, "question", "Сервис готов?", "ask", requested_slots=["yes_no"]),
        ])
        self.assertEqual(len(graph["propositions"]), 1)
        self.assertEqual(len(graph["question_states"]), 1)
        self.assertEqual(graph["question_states"][0]["status"], "unanswered")

    def test_immediate_explicit_recommendation_closes_yes_no_question(self):
        question = rec(
            1, "question", "Существует ли дневное тренд-направление на индексах?", "ask",
            requested_slots=["index_trend_daily"], question_status="answered",
            dialogue_evidence=[
                {"id": "U1", "start": 1, "text": "Существует ли дневное тренд-направление на индексах?"},
                {"id": "U2", "start": 10, "text": "Не следует смотреть: между минутным и дневным слишком большая разница."},
            ],
        )
        graph = build_meeting_graph([question])
        self.assertEqual(graph["question_states"][0]["status"], "answered")
        self.assertIn("U2", graph["question_states"][0]["answer_evidence_ids"])

    def test_delay_status_question_infers_slot_and_closes_on_status_answer(self):
        question = rec(
            1, "question", "Есть ли проблема с задержкой в алгоразметке?", "ask",
            question_status="partially_answered", answer_record_ids=["F2"],
            dialogue_evidence=[
                {"id": "U1", "start": 1, "text": "Есть ли проблема с задержкой в алгоразметке?"},
                {"id": "U2", "start": 2, "text": "Задержку можно снять до одной свечи, но качество просядет."},
            ],
        )
        answer = rec(2, "current_state", "Задержку можно снять до одной свечи, но качество просядет.", "answer")
        graph = build_meeting_graph([question, answer])
        state = next(item for item in graph["question_states"] if "задерж" in item["original_question"].casefold())
        self.assertEqual(state["requested_slots"], ["implementation_status"])
        self.assertEqual(state["status"], "answered")
        self.assertIsNone(state["remaining_unknown"])

    def test_semantic_revision_changes_generation_id(self):
        first = build_meeting_graph([rec(1, "observation", "Сервис работает", verification_status="supported")], provenance={"recording_id": "R"})
        second = build_meeting_graph([rec(1, "observation", "Сервис работает", modality="tentative", verification_status="insufficient_evidence")], provenance={"recording_id": "R"})
        self.assertNotEqual(first["generation_id"], second["generation_id"])

    def test_argument_negation_is_not_predicate_contradiction(self):
        graph = build_meeting_graph([
            rec(1, "current_state", "Сервис работает"),
            rec(2, "current_state", "Сервис работает без ошибок"),
        ])
        self.assertFalse(any(x["type"] == "contradicts" for x in graph["relations"]))

    def test_number_time_and_profile_normalization(self):
        fact = {"statement": "@Misha. проверит четыре сигнала после 00:00", "evidence": [{"speaker": "@Misha", "text": "Проверю 4 сигнала после 00"}]}
        self.assertEqual(deterministic_fact_check(fact), (True, None))


class Run13PublicationRegressions(unittest.TestCase):
    @staticmethod
    def sentence_plan(claim_id, **extra):
        value = {
            "claim_ids": [claim_id], "relation_ids": [], "allowed_numbers": [],
            "allowed_relation_markers": [], "allowed_speakers": [],
            "allowed_assignees": [], "polarity": ["positive"],
            "modality": ["certain"], "conditions": [], "time_scope": [],
        }
        value.update(extra)
        return value

    def test_public_contract_matches_runtime_fields(self):
        graph = build_meeting_graph([rec(1, "action", "Я подготовлю отчёт", "commit", assignees=["@A"])])
        result = plan(graph["claims"], graph["episodes"], graph["relations"], lambda _item: 1)
        item = next(x for x in build_public_items(graph, result) if x["section"] == "tasks")
        PublicItemContract.model_validate(item)

    def test_unknown_evidence_and_unrendered_card_are_rejected(self):
        item = {"public_id": "PI1", "section": "overview", "text": "Сервис работает", "claim_ids": ["C1"], "evidence_ids": ["U1"], "source_word_ids": ["W1"], "content_kind": "current_state", "social_state": "candidate", "start": 1, "end": 2}
        document = build_public_document([item])
        artifact = render_public_document(document)
        document["title"]["evidence_ids"] = ["U999"]
        self.assertFalse(verify_public_document(document, artifact, [item])["passed"])
        document = build_public_document([item])
        document["outcome_cards"][0]["fields"]["current_state"]["value"] = "Выдуманное состояние"
        self.assertFalse(verify_public_document(document, artifact, [item])["passed"])

    def test_contextual_sections_render_evidence_bound_explanations(self):
        task = {"public_id": "PI1", "section": "tasks", "text": "@A подготовит демонстрацию", "claim_ids": ["C1"], "evidence_ids": ["U1"], "source_word_ids": ["W1"], "content_kind": "action", "social_state": "self_committed", "start": 10, "end": 11, "episode_id": "E1"}
        hypothesis = {"public_id": "PI2", "section": "experiments", "text": "Демонстрация нужна для проверки точки входа", "claim_ids": ["C2"], "evidence_ids": ["U2"], "source_word_ids": ["W2"], "content_kind": "hypothesis", "social_state": "candidate", "start": 9, "end": 10, "episode_id": "E1"}
        document = build_public_document([task, hypothesis])
        self.assertEqual(document["sections"]["tasks"][0]["context"][0]["text"], hypothesis["text"])
        self.assertEqual(document["sections"]["experiments"][0]["context"][0]["text"], task["text"])
        artifact = render_public_document(document)
        self.assertIn("  - **Зачем это нужно:**", artifact)
        self.assertIn("  - **Что именно проверяют:**", artifact)
        self.assertTrue(verify_public_document(document, artifact, [task, hypothesis])["passed"])

    def test_context_prefers_near_topic_claim_and_verifies_graph_provenance(self):
        graph = build_meeting_graph([
            rec(1, "observation", "Предлагается использовать старший таймфрейм для сопротивления"),
            rec(80, "observation", "Создана задача на разметчик трёх свечных паттернов"),
            rec(100, "constraint", "Для разметки трёх свечных паттернов используется тот же свинг-маркер"),
        ])
        planned = plan(graph["claims"], graph["episodes"], graph["relations"], lambda _item: 1)
        items = build_public_items(graph, planned)
        document = build_public_document(items, graph=graph)
        target = next(item for item in document["sections"]["technical"] if "тот же свинг-маркер" in item["text"])
        self.assertIn("задача на разметчик трёх свечных паттернов", target["context"][0]["text"])
        artifact = render_public_document(document)
        errors = verify_public_document(document, artifact, items, graph)["errors"]
        self.assertNotIn("unsupported_section_context", errors)
        self.assertNotIn("section_context_evidence_outside_closure", errors)

    def test_technical_context_prefers_explanatory_constraint_over_nearby_topic(self):
        graph = build_meeting_graph([
            rec(1, "constraint", "Всё упирается в зону интереса старшего таймфрейма, которую не видно."),
            rec(2, "definition", "Зоны интереса — это имбалансы, блоки, Breaker и зона OTE."),
            rec(3, "proposal", "Существуют проекции внутридневной амплитуды цены."),
        ])
        planned = plan(graph["claims"], graph["episodes"], graph["relations"], lambda _item: 1)
        items = build_public_items(graph, planned)
        document = build_public_document(items, graph=graph)
        target = next(item for item in document["sections"]["technical"] if "Зоны интереса" in item["text"])
        self.assertIn("старшего таймфрейма", target["context"][0]["text"])
        self.assertEqual(target["context"][0]["role"], "importance")

    def test_chapter_label_never_hides_mid_sentence_truncation(self):
        first = "Предлагается провести дополнительный бэктест для анализа тех. причин проигрышных сделок и исключения их из массива"
        second = "Для разметки трёх свечных паттернов можно использовать тот же самый свинг-маркер, что и для пяти свечных"
        self.assertEqual(_chapter_label(first), "Предлагается провести дополнительный бэктест для анализа технических причин проигрышных сделок и исключения их из массива")
        self.assertEqual(_chapter_label(second), second)

    def test_goal_only_claim_is_not_labeled_as_experiment(self):
        graph = build_meeting_graph([rec(1, "hypothesis", "Discussed potential goal: creating a baseline solution with winrate around 30–40%.")])
        result = plan(graph["claims"], graph["episodes"], graph["relations"], lambda _item: 1)
        self.assertFalse(any(item["section"] == "experiments" for item in build_public_items(graph, result)))

    def test_outcome_card_does_not_repeat_resolution_as_next_step(self):
        graph = {
            "claims": [{"claim_id": "C1", "proposition_id": "P1", "content_kind": "proposal", "statement": "Можно проверить точку входа", "publication_text": "Можно проверить точку входа", "decision_status": "accepted", "evidence_ids": ["U1"]}],
            "dialogue_bundles": [{"bundle_id": "DB1", "topic": "Точка входа", "claim_ids": ["C1"], "ranges": [{"start": 1, "end": 2}]}],
            "task_states": [], "question_states": [],
        }
        card = build_outcome_cards(graph)[0]
        self.assertEqual(card["fields"]["resolution"]["value"], "Можно проверить точку входа")
        self.assertIsNone(card["fields"]["next_step"])

    def test_open_question_cannot_fill_state_or_next_step(self):
        graph = {
            "claims": [{"claim_id": "C1", "proposition_id": "P1", "content_kind": "observation",
                        "statement": "Есть ли задержка?", "publication_text": "Есть ли задержка?",
                        "evidence_ids": ["U1"]}],
            "dialogue_bundles": [{"bundle_id": "DB1", "topic": "Задержка",
                                  "claim_ids": ["C1"], "ranges": [{"start": 1, "end": 2}]}],
            "task_states": [],
            "question_states": [{"proposition_id": "P1", "status": "unanswered",
                                 "remaining_unknown": "Есть ли задержка?"}],
        }
        card = build_outcome_cards(graph)[0]
        self.assertIsNone(card["fields"]["current_state"])
        self.assertIsNone(card["fields"]["next_step"])
        self.assertEqual(card["fields"]["remaining_unknown"][0]["value"], "Есть ли задержка?")

    def test_chronology_deduplicates_fields_across_chapters(self):
        def card(number):
            return {"outcome_id": f"OC{number}", "fields": {
                "current_state": {"value": "Общий подтверждённый факт", "claim_ids": [f"C{number}"], "evidence_ids": [f"U{number}"]},
                "constraint": None, "resolution": None, "work_result": None,
                "next_step": None, "remaining_unknown": [],
            }}
        document = {
            "title": {"text": "Итоги", "claim_ids": ["C1"], "evidence_ids": ["U1"]},
            "overview": [], "sections": {}, "navigation": [], "metadata": {},
            "outcome_cards": [card(1), card(2)],
            "chronology": [
                {"label": "Первая тема", "start": 1, "end": 2, "outcome_ids": ["OC1"], "items": []},
                {"label": "Вторая тема", "start": 3, "end": 4, "outcome_ids": ["OC2"], "items": []},
            ],
        }
        artifact = render_public_document(document)
        self.assertEqual(artifact.count("Общий подтверждённый факт."), 1)

    def test_question_projection_does_not_overwrite_answer_claim_in_outcome(self):
        items = [
            {"public_id": "PIQ", "section": "questions", "text": "@A спрашивает: нужен ли фильтр?",
             "claim_ids": ["CQ", "CA"], "evidence_ids": ["UQ", "UA"], "source_word_ids": ["WQ"],
             "content_kind": "question", "social_state": "unanswered", "start": 1, "end": 2, "episode_id": "E1",
             "question_state": {"answer_record_ids": ["FA"]}},
            {"public_id": "PIA", "section": "minutes", "text": "Предложено проверить дополнительный фильтр.",
             "claim_ids": ["CA"], "evidence_ids": ["UA"], "source_word_ids": ["WA"],
             "content_kind": "proposal", "social_state": "candidate", "start": 2, "end": 3, "episode_id": "E1"},
        ]
        graph = {
            "claims": [
                {"claim_id": "CQ", "proposition_id": "PQ", "source_record_id": "FQ", "content_kind": "question",
                 "statement": "Нужен ли фильтр?", "evidence_ids": ["UQ"], "lifecycle": "active", "verification_status": "supported"},
                {"claim_id": "CA", "proposition_id": "PA", "source_record_id": "FA", "content_kind": "proposal",
                 "statement": "Предложено проверить дополнительный фильтр.", "evidence_ids": ["UA"], "lifecycle": "active", "verification_status": "supported"},
            ],
            "dialogue_bundles": [{"bundle_id": "DB1", "topic": "Фильтр", "claim_ids": ["CQ", "CA"],
                                  "ranges": [{"start": 1, "end": 3}]}],
            "task_states": [],
            "question_states": [{"proposition_id": "PQ", "status": "unanswered", "remaining_unknown": "Нужен ли фильтр?"}],
        }
        document = build_public_document(items, graph=graph)
        card = document["outcome_cards"][0]
        self.assertEqual(card["fields"]["next_step"]["value"], "Предложено проверить дополнительный фильтр.")
        self.assertNotIn("спрашивает", card["fields"]["next_step"]["value"])

    def test_chapter_uses_episode_end(self):
        item = {"public_id": "PI1", "section": "minutes", "text": "Сервис работает", "claim_ids": ["C1"], "evidence_ids": ["U1"], "source_word_ids": ["W1"], "content_kind": "current_state", "social_state": "candidate", "start": 10, "end": 18, "episode_id": "E1"}
        chapter = build_public_document([item])["chronology"][0]
        self.assertEqual(chapter["end"], 18)
        self.assertGreater(chapter["end"], chapter["start"])

    def test_incomplete_generation_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); gid = "20260915-120000-abcdef123456"
            target = base / "summary_generations" / gid; target.mkdir(parents=True)
            (target / "summary.md").write_text("ok")
            digest = hashlib.sha256(b"ok").hexdigest()
            (target / "generation_manifest.json").write_text(json.dumps({"generation_id": gid, "artifact_sha256": {"summary.md": digest}}))
            (base / "summary_current.json").write_text(json.dumps({"generation_id": gid}))
            self.assertTrue(REQUIRED_GENERATION_FILES - {"summary.md"})
            self.assertIsNone(current_summary_output(base))

    def test_generation_pointer_is_visible_to_status_polling(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); gid = "20260915-120000-abcdef123456"
            (base / "summary_current.json").write_text(json.dumps({"generation_id": gid}))
            self.assertEqual(current_summary_generation_id(base), gid)
            (base / "summary_current.json").write_text(json.dumps({"generation_id": "../unsafe"}))
            self.assertIsNone(current_summary_generation_id(base))

    def test_verbatim_source_negation_is_not_rejected(self):
        text = "Если сделка не дошла до Take Profit, она закрывается автоматически после 00."
        item = {"section": "technical", "text": text, "claim_ids": ["C1"], "evidence_ids": ["U1"]}
        claim = {"claim_id": "C1", "statement": text, "evidence_ids": ["U1"], "lifecycle": "active"}
        result = verify_generated_items([item], [self.sentence_plan("C1")], [claim])
        self.assertTrue(result["passed"], result)

    def test_exact_evidence_surface_numbers_and_negation_are_source_backed(self):
        text = "На минутке задержка в 2 мин — это не страшно, а на старшем ждать 8 часов больно."
        item = {"section": "overview", "text": text, "claim_ids": ["C1"], "evidence_ids": ["U1"]}
        claim = {
            "claim_id": "C1", "statement": "На минутном графике можно получить полную структуру с BOS.",
            "evidence_ids": ["U1"], "lifecycle": "active",
            "dialogue_evidence": [{"id": "U1", "text": text}],
        }
        result = verify_generated_items([item], [self.sentence_plan("C1")], [claim])
        self.assertTrue(result["passed"], result)
        concise = {**item, "text": "На минутном графике можно получить полную структуру с BOS."}
        result = verify_generated_items([concise], [self.sentence_plan("C1")], [claim])
        self.assertTrue(result["passed"], result)

    def test_sanitized_translation_preserves_source_negation(self):
        source = "@Yachoy suggests returning to algorithmic thinking and potentially incorporating higher timeframes if the current approach does not yield results."
        text = "Если текущий подход не даст результата, @Yachoy предлагает вернуться к алгоритмическому подходу и, возможно, подключить старшие таймфреймы."
        item = {"section": "minutes", "text": text, "claim_ids": ["C1"], "evidence_ids": ["U1"]}
        claim = {"claim_id": "C1", "statement": source, "evidence_ids": ["U1"], "lifecycle": "active", "speaker_refs": ["@Yachoy"]}
        result = verify_generated_items([item], [self.sentence_plan("C1", allowed_speakers=["@Yachoy"])], [claim])
        self.assertTrue(result["passed"], result)

    def test_repeated_task_actor_metadata_is_not_a_role_swap(self):
        text = "@Yachoy подготовит TradingView. — исполнитель: @Yachoy — статус: участник взял на себя"
        item = {"section": "tasks", "text": text, "claim_ids": ["C1"], "evidence_ids": ["U1"],
                "task_state_id": "T1", "social_state": "self_committed",
                "task_state": {"assignee": "@Yachoy", "evidence_ids": ["U1"],
                               "action_frame": {"state": "self_committed"}}}
        claim = {"claim_id": "C1", "statement": "@Yachoy подготовит TradingView.",
                 "evidence_ids": ["U1"], "lifecycle": "active", "canonical_task_state_id": "T1"}
        plan = self.sentence_plan("C1", allowed_speakers=["@Yachoy"], allowed_assignees=["@Yachoy"])
        result = verify_generated_items([item], [plan], [claim])
        self.assertTrue(result["passed"], result)

    def test_outcome_card_retains_bundle_claims_and_field_evidence_closure(self):
        graph = {
            "claims": [
                {"claim_id": "C1", "proposition_id": "P1", "content_kind": "current_state",
                 "statement": "Сервис работает", "evidence_ids": ["U1"]},
                {"claim_id": "C2", "proposition_id": "P2", "content_kind": "action",
                 "statement": "Проверить задержку", "evidence_ids": ["U2"]},
            ],
            "dialogue_bundles": [{"bundle_id": "DB1", "topic": "Сервис",
                                  "claim_ids": ["C1", "C2"], "ranges": [{"start": 1, "end": 2}]}],
            "question_states": [{"proposition_id": "P1", "status": "partial",
                                 "remaining_unknown": "Когда исправят задержку?",
                                 "answer_evidence_ids": ["U9"]}],
            "task_states": [],
        }
        card = build_outcome_cards(graph)[0]
        self.assertEqual(set(card["claim_ids"]), {"C1", "C2"})
        self.assertEqual(set(card["evidence_ids"]), {"U1", "U2"})
        self.assertEqual(card["fields"]["remaining_unknown"][0]["evidence_ids"], ["U1"])


if __name__ == "__main__":
    unittest.main()
