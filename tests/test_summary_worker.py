import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "summary_worker.py"
SPEC = importlib.util.spec_from_file_location("summary_worker", MODULE_PATH)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


def utterance(index, start, end, speaker="@Riven", text="Текст"):
    return {"id": f"U{index:05d}", "start": start, "end": end, "speaker": speaker, "speaker_id": speaker, "text": text, "flags": []}


def fact(kind="proposal", statement="Предложено проверить BOS", evidence=None):
    evidence = evidence or [utterance(1, 10.125, 12.75, text="Предлагаю проверить BOS")]
    return {
        "fact_id": "F00001", "type": kind, "topic": "BOS", "statement": statement,
        "certainty": "explicit", "evidence_ids": [item["id"] for item in evidence],
        "speaker_refs": [evidence[0]["speaker"]], "start": evidence[0]["start"],
        "end": evidence[-1]["end"], "evidence": evidence, "source_chunks": [1],
    }


class SummaryWorkerTests(unittest.TestCase):
    def test_evidence_repair_uses_only_work_tree_for_download_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = summary.evidence_repair_environment(root)
            expected_cache = root / "work" / "cache"
            self.assertEqual(environment["XDG_CACHE_HOME"], str(expected_cache))
            self.assertEqual(environment["HF_HOME"], str(expected_cache / "huggingface"))
            self.assertEqual(environment["HF_XET_CACHE"], str(expected_cache / "huggingface" / "xet"))
            self.assertEqual(environment["HF_HUB_DISABLE_XET"], "1")
            self.assertEqual(environment["TMPDIR"], str(root / "work" / "tmp" / "evidence-repair"))
            for key in ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE", "TMPDIR"):
                self.assertTrue(Path(environment[key]).is_dir())

    def test_semantic_chapter_batches_follow_planner_anchors(self):
        facts = []
        for index in range(8):
            item = fact(statement=f"Тезис {index}")
            item.update({"fact_id": f"F{index:05d}", "start": index * 100})
            facts.append(item)
        batches = summary.semantic_chapter_batches(
            facts, ["F00000", "F00002", "F00004", "F00006"],
        )
        self.assertEqual(len(batches), 4)
        self.assertEqual(
            {item["fact_id"] for batch in batches for item in batch},
            {item["fact_id"] for item in facts},
        )
        self.assertTrue(all(batch for batch in batches))

    def test_clustered_chapter_anchors_are_rebalanced(self):
        facts = []
        for index in range(168):
            item = fact(statement=f"Тезис {index}")
            item.update({"fact_id": f"F{index:05d}", "start": index * 10})
            facts.append(item)
        batches = summary.semantic_chapter_batches(facts, [f"F{index:05d}" for index in range(12)])
        self.assertEqual(len(batches), 12)
        self.assertLessEqual(max(map(len, batches)), 14)
        self.assertEqual(sum(map(len, batches)), 168)

    def test_empty_writer_chapter_has_one_deterministic_topic(self):
        facts = [dict(fact(statement=f"Тезис {index}"), fact_id=f"F{index:05d}", topic="Имбалансы") for index in range(1, 6)]
        topic = summary.deterministic_chapter(facts, 3)
        self.assertEqual(topic["title"], "Имбалансы")
        self.assertEqual(len(topic["items"]), 5)

    def test_rejected_revision_is_terminal_despite_origin_rewrite(self):
        denials = [
            {"fact": {"fact_id": "F00181", "origin_id": "ORafter", "revision_id": "RV-after",
                      "evidence_ids": ["U00302"]}, "reason": "Нет доказательств решения."},
            {"fact": {"fact_id": "F00181", "origin_id": "ORother", "revision_id": "RV-other",
                      "evidence_ids": ["U00327"]}, "reason": "Метафора, не факт."},
        ]
        reasons = summary.rejection_reason_by_revision(denials)
        self.assertEqual(reasons["RV-after"], "Нет доказательств решения.")
        self.assertEqual(reasons["RV-other"], "Метафора, не факт.")
        self.assertNotIn("F00181", reasons)

    def test_section_and_detailed_views_are_not_limited_to_executive_facts(self):
        core = fact(statement="Основной результат встречи")
        hypothesis = dict(
            fact(kind="hypothesis", statement="Старший таймфрейм может повысить качество фильтрации"),
            fact_id="F00002", start=20, end=22, evidence_ids=["U00002"],
            evidence=[utterance(2, 20, 22, text="Старший таймфрейм может повысить качество фильтрации")],
        )
        question = dict(
            fact(kind="question", statement="Какой фильтр использовать?"),
            fact_id="F00003", start=30, end=32, evidence_ids=["U00003"],
            evidence=[utterance(3, 30, 32, text="Какой фильтр использовать?")],
        )
        detail = dict(
            fact(statement="Дополнительное подробное объяснение механизма фильтрации"),
            fact_id="F00004", start=40, end=44, evidence_ids=["U00004"],
            evidence=[utterance(4, 40, 44, text="Дополнительное подробное объяснение механизма фильтрации")],
        )
        state = {"views": {"questions": [{
            "source_record_id": "F00003", "state": "answered",
            "answer_spans": [{"text": "Использовать контекст старшего таймфрейма."}],
            "answer_record_ids": [],
        }]}}
        rendered = summary.render_markdown(
            {"main_topic": {"text": "Проверка фильтра", "fact_ids": ["F00001"]},
             "objective": None, "overview": [], "chronology": [], "topics": [],
             "decisions": [], "actions": [], "open_questions": []},
            [core], {"total_seconds": 60},
            semantic_registry={"records": [], "tasks": []}, meeting_state_document=state,
            section_facts=[hypothesis, question], detailed_facts=[core, detail],
        )
        self.assertIn("## Ответы и уточнения", rendered)
        self.assertIn("## Открытые вопросы и гипотезы", rendered)
        self.assertIn("Старший таймфрейм может повысить качество", rendered)
        self.assertNotIn("Открытых вопросов не обнаружено", rendered)
        self.assertIn("Дополнительное подробное объяснение", rendered)

    def test_chunks_cover_every_utterance(self):
        items = [utterance(index, index * 50, index * 50 + 4) for index in range(1, 20)]
        chunks = summary.make_chunks(items, seconds=180, overlap=30)
        covered = {item["id"] for chunk in chunks for item in chunk["utterances"]}
        self.assertEqual(covered, {item["id"] for item in items})

    def test_rejects_number_not_in_evidence(self):
        item = fact(statement="Качество упало на 10%")
        okay, reason = summary.deterministic_fact_check(item)
        self.assertFalse(okay)
        self.assertIn("10%", reason)

    def test_rejects_invented_profile_tag(self):
        item = fact(statement="@Maxim должен проверить BOS")
        okay, reason = summary.deterministic_fact_check(item)
        self.assertFalse(okay)
        self.assertIn("@Maxim", reason)

    def test_rejects_technical_marker_missing_from_evidence(self):
        item = fact(statement="Точка BOS отображается на M4")
        okay, reason = summary.deterministic_fact_check(item)
        self.assertFalse(okay)
        self.assertIn("m4", reason)

    def test_accepts_technical_marker_present_in_evidence(self):
        item = fact(
            statement="Проверить структуру на M15",
            evidence=[utterance(1, 1, 3, text="Нужно проверить структуру на M15")],
        )
        okay, reason = summary.deterministic_fact_check(item)
        self.assertTrue(okay, reason)

    def test_proposal_cannot_enter_decisions(self):
        item = fact(kind="proposal")
        document = {"decisions": [{"text": "Решили проверить BOS", "fact_ids": ["F00001"]}]}
        clean, rejected = summary.sanitize_structured(document, [item])
        self.assertEqual(clean["decisions"], [])
        self.assertEqual(len(rejected), 1)

    def test_renderer_uses_evidence_timestamps(self):
        item = fact(
            statement="Предложено проверить работу фильтра BOS на текущей реализации",
            evidence=[utterance(1, 10.125, 12.75,
                                text="Предложено проверить работу фильтра BOS на текущей реализации")],
        )
        document = {
            "main_topic": {"text": "Обсуждение BOS", "fact_ids": ["F00001"]},
            "objective": None, "overview": [],
            "chronology": [{"text": "Предложили проверку", "fact_ids": ["F00001"]}],
            "topics": [], "decisions": [], "actions": [], "open_questions": [],
        }
        rendered = summary.render_markdown(document, [item], {"covered_seconds": 100, "total_seconds": 100})
        self.assertIn("00:00:10", rendered)
        self.assertNotIn("transcript.html", rendered)
        self.assertNotIn("F00001", rendered)

    def test_public_renderer_has_prose_overview_timecode_index_and_detailed_chronology(self):
        base = {"claim_ids": ["C1"], "evidence_ids": ["U1"], "source_word_ids": ["W1"], "content_kind": "observation", "social_state": "candidate", "lifecycle": "active", "relation_ids": [], "topic_entities": ["Order Block", "Bitcoin"]}
        items = [
            dict(base, public_id="PI1", section="overview", text="Обсудили фильтрацию сигналов", start=10),
            dict(base, public_id="PI2", section="overview", text="Зафиксировали следующий шаг", start=20),
            dict(base, public_id="PI3", section="minutes", text="Разобрали текущую реализацию", start=10),
            dict(base, public_id="PI4", section="minutes", text="Согласовали дальнейшую проверку", start=20),
        ]
        rendered = summary.render_public_items(items, {"source": "12.07.2026.mkv", "project": "Aurion", "job_id": 8})
        self.assertIn("Bitcoin", rendered.splitlines()[0])
        self.assertIn("Order Block", rendered.splitlines()[0])
        self.assertNotIn("результаты, ограничения и следующие шаги", rendered.splitlines()[0])
        self.assertNotIn("торговой системы", rendered.splitlines()[0])
        overview = rendered.split("## Главное", 1)[1].split("## Таймкоды", 1)[0]
        self.assertNotIn("\n- ", overview)
        self.assertNotIn("/result?", overview)
        self.assertIn("## Таймкоды", rendered)
        self.assertIn("## Подробная хронология встречи", rendered)
        self.assertNotIn("transcript.html", rendered)

    def test_compact_renderer_has_required_sections_and_no_empty_optional_sections(self):
        item = fact()
        document = {
            "main_topic": {"text": "Встреча была посвящена проверке BOS.", "fact_ids": ["F00001"]},
            "objective": None, "overview": [],
            "chronology": [{"text": "Проверили логику BOS.", "fact_ids": ["F00001"]}],
            "topics": [], "decisions": [], "actions": [], "open_questions": [],
        }
        rendered = summary.render_markdown(
            document, [item], {"covered_seconds": 100, "total_seconds": 100},
            metadata={"source": "12.07.2026 — встреча.mp4", "project": "Aurion", "duration_seconds": 100},
        )
        self.assertTrue(rendered.startswith("# Дата не указана | Aurion — проверке BOS"))
        for heading in ("## Краткое описание", "## Участники", "## Таймкоды", "## Подробное описание встречи"):
            self.assertIn(heading, rendered)
        self.assertNotIn("## Решения", rendered)
        self.assertNotIn("## Задачи", rendered)
        self.assertNotIn("Покрытие:", rendered)

    def test_tasks_render_from_structured_registry(self):
        item = fact(kind="action", statement="Проверить BOS")
        item["uncertainty"] = {"needs_review": False}
        document = {
            "main_topic": {"text": "Проверка BOS", "fact_ids": ["F00001"]},
            "objective": None, "overview": [], "chronology": [], "topics": [],
            "decisions": [], "actions": [{"text": "Проверить BOS", "fact_ids": ["F00001"]}],
            "open_questions": [],
        }
        registry = {"tasks": [{
            "description": "Проверить BOS", "title": "Проверка сигналов BOS",
            "details": "Сопоставить ложные сигналы BOS с условиями входа.",
            "assignees": ["@Riven"],
            "assignment_status": "confirmed", "due": "до пятницы",
            "conditions": [], "source_record_id": "F00001", "evidence_ids": ["U00001"],
            "uncertainty": {"needs_review": False},
        }]}
        rendered = summary.render_markdown(
            document, [item], {"covered_seconds": 100, "total_seconds": 100},
            semantic_registry=registry,
        )
        self.assertIn("## Задачи и следующие шаги", rendered)
        self.assertIn("**T-01. Проверка сигналов BOS**", rendered)
        self.assertIn("Сопоставить ложные сигналы BOS", rendered)
        self.assertIn("**@Riven** (подтверждено)", rendered)

    def test_hypothesis_renders_author(self):
        item = fact(
            kind="hypothesis",
            statement="Старший таймфрейм может отфильтровать ложные сигналы",
            evidence=[utterance(1, 10, 12, speaker="@Yachoy",
                                text="Старший таймфрейм может отфильтровать ложные сигналы")],
        )
        item["speaker_refs"] = ["@Yachoy"]
        rendered = summary.render_markdown(
            {"main_topic": {"text": "Фильтрация", "fact_ids": ["F00001"]},
             "objective": None, "overview": [], "chronology": [], "topics": [],
             "decisions": [], "actions": [], "open_questions": []},
            [item], {"total_seconds": 60}, semantic_registry={"records": [], "tasks": []},
        )
        self.assertIn("автор: **@Yachoy**", rendered)

    def test_unreliable_task_is_published_as_review_candidate(self):
        item = fact(kind="action", statement="Неясное обещание что-то попробовать")
        item["uncertainty"] = {"needs_review": True, "reasons": ["overlap"]}
        registry = {"records": [], "tasks": [{
            "description": item["statement"], "assignees": ["@Riven"],
            "assignment_status": "confirmed", "automation_eligible": False,
            "due": None, "conditions": [], "source_record_id": "F00001",
            "uncertainty": item["uncertainty"],
        }]}
        rendered = summary.render_markdown(
            {"main_topic": {"text": "Тема", "fact_ids": []}, "objective": None,
             "overview": [], "chronology": [], "topics": [], "decisions": [],
             "actions": [], "open_questions": []},
            [item], {"total_seconds": 60}, semantic_registry=registry,
        )
        self.assertIn("## Задачи и следующие шаги", rendered)
        self.assertIn("### Требуют подтверждения", rendered)
        self.assertIn("Неясное обещание", rendered)
        self.assertIn("потенциальная задача; требуется подтверждение", rendered)

    def test_legacy_unclear_question_is_not_published_without_global_resolution(self):
        item = fact(kind="question", statement="Какой диапазон имеет сессия AM?")
        item["uncertainty"] = {"needs_review": False, "reasons": []}
        registry = {"records": [{
            "record_id": "F00001", "question_status": "unclear", "question_kind": "discussion",
        }], "tasks": []}
        rendered = summary.render_markdown(
            {"main_topic": {"text": "Тема", "fact_ids": ["F00001"]}, "objective": None,
             "overview": [], "chronology": [], "topics": [], "decisions": [],
             "actions": [], "open_questions": []},
            [item], {"total_seconds": 60}, semantic_registry=registry,
        )
        self.assertNotIn("### Нерешённые вопросы встречи", rendered)
        self.assertNotIn("Какой диапазон имеет сессия AM?", rendered)

    def test_speaker_only_uncertainty_keeps_anonymous_content_but_not_attribution(self):
        item = fact(kind="problem", statement="Система создаёт лишние сигналы рядом с паттерном",
                    evidence=[utterance(1, 10, 12,
                                        text="Система создаёт лишние сигналы рядом с паттерном")])
        item["uncertainty"] = {"needs_review": True, "reasons": ["speaker_smoothed"]}
        self.assertTrue(summary.fact_is_reliable_for_main(item))
        self.assertFalse(summary.fact_has_safe_attribution(item))
        self.assertTrue(summary.navigation_label(item))

    def test_overlap_uncertainty_blocks_main_content(self):
        item = fact(kind="problem", statement="Система создаёт лишние сигналы рядом с паттерном")
        item["uncertainty"] = {"needs_review": True, "reasons": ["overlap"]}
        self.assertFalse(summary.fact_is_reliable_for_main(item))
        self.assertEqual(summary.navigation_label(item), "")

    def test_malformed_asr_term_is_kept_out_of_public_summary(self):
        item = fact(
            kind="observation",
            statement="Трёхсычный (трёхсочный) паттерн имеет маленький имбаланс",
        )
        self.assertTrue(summary.fact_needs_transcript_review(item))
        self.assertFalse(summary.fact_is_reliable_for_main(item))
        self.assertEqual(summary.navigation_label(item), "")

    def test_editorial_review_note_is_not_published_as_question(self):
        item = fact(
            kind="question",
            statement="Как включить объёмы? (требует проверки: фраза искажена)",
        )
        self.assertTrue(summary.fact_needs_transcript_review(item))

    def test_publication_cleanup_turns_literal_fragments_into_standalone_text(self):
        imbalance = fact(
            kind="observation",
            statement="Может быть там три или четыре точки на одном имбалансе линии.",
        )
        action = fact(
            kind="action",
            statement="@Yachoy говорит, что подготовит TradingView к следующему разу или отправит EXE-файл.",
        )
        self.assertEqual(summary.clean_publication_statement(imbalance), imbalance["statement"])
        self.assertEqual(summary.clean_publication_statement(action), action["statement"])

    def test_only_explicit_handles_are_rendered_as_bold_people(self):
        rendered = summary.canonicalize_people("@A спросил Николая, @B ответил")
        self.assertIn("**@A**", rendered)
        self.assertIn("**@B**", rendered)
        self.assertIn("Николая", rendered)

    def test_people_are_canonical_in_structured_text_without_markdown(self):
        rendered = summary.canonicalize_people_plain("Николай спросил Мишу, Максим ответил Yachoy")
        self.assertEqual(rendered, "Николай спросил Мишу, Максим ответил Yachoy")
        self.assertNotIn("**", rendered)

    def test_all_unresolved_questions_are_retained_in_human_summary(self):
        facts = []
        questions = []
        for index in range(12):
            item = dict(
                fact(kind="question", statement=f"Вопрос {index + 1}"),
                fact_id=f"F{index + 1:05d}", start=index * 60, end=index * 60 + 2,
            )
            item["uncertainty"] = {"needs_review": False}
            facts.append(item)
            questions.append({"text": item["statement"], "fact_ids": [item["fact_id"]]})
        document = {
            "main_topic": {"text": "Вопросы встречи", "fact_ids": [item["fact_id"] for item in facts]},
            "objective": None, "overview": [], "chronology": [], "topics": [],
            "decisions": [], "actions": [], "open_questions": questions,
        }
        registry = {"records": [
            {"record_id": item["fact_id"], "question_status": "unresolved"}
            for item in facts
        ], "tasks": []}
        rendered = summary.render_markdown(
            document, facts, {"covered_seconds": 720, "total_seconds": 720},
            semantic_registry=registry,
        )
        self.assertEqual(rendered.count("**Q-"), 12)
        self.assertNotIn("Ещё 4 пунктов", rendered)

    def test_personal_opinion_is_not_a_decision(self):
        evidence = [utterance(1, 1, 3, text="Я думаю, не стоит пытаться отыграть позицию")]
        item = fact(kind="decision", statement="Не следует отыгрывать позицию", evidence=evidence)
        updated = summary.enforce_fact_policy(item)
        self.assertEqual(updated["type"], "proposal")
        self.assertEqual(updated["certainty"], "tentative")

    def test_detailed_chronology_covers_distant_parts_of_meeting(self):
        first = fact(
            statement="Обсудили работу фильтра BOS на текущей реализации",
            evidence=[utterance(1, 10, 14, text="Обсудили работу фильтра BOS на текущей реализации")],
        )
        second = dict(
            fact(statement="Проверили задержку подтверждения структуры на M15",
                 evidence=[utterance(2, 620, 625,
                                     text="Проверили задержку подтверждения структуры на M15")]),
            fact_id="F00002", start=620, end=625,
        )
        groups = summary.detailed_chronology_points([first, second], 700)
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0][0]["fact_id"], "F00001")
        self.assertEqual(groups[1][0]["fact_id"], "F00002")

    def test_detail_keeps_material_before_first_salient_anchor(self):
        opening = fact(
            kind="current_state", statement="Имбалансы пока только подсвечиваются на постобработке",
            evidence=[utterance(1, 10, 14, text="Имбалансы пока только подсвечиваются на постобработке")],
        )
        anchor = dict(
            fact(kind="action", statement="Подготовить размеченные данные для симуляции на M15",
                 evidence=[utterance(2, 620, 625, text="Подготовить размеченные данные для симуляции на M15")]),
            fact_id="F00002", start=620, end=625,
        )
        groups = summary.detailed_chronology_points([opening, anchor], 700)
        self.assertEqual(groups[0][0]["fact_id"], "F00001")
        self.assertEqual(
            [item["fact_id"] for group in groups for item in group],
            ["F00001", "F00002"],
        )

    def test_resolved_question_renders_even_when_public_fact_type_is_observation(self):
        question = fact(kind="observation", statement="Какой диапазон нужен для симуляции?")
        answer = dict(
            fact(kind="constraint", statement="Для симуляции достаточно месяца данных на M15–H4"),
            fact_id="F00002", start=20, end=22,
        )
        state = {"views": {"questions": [{
            "source_record_id": "F00001", "state": "answered",
            "question_kind": "discussion", "answer_record_ids": ["F00002"],
        }]}}
        rendered = summary.render_markdown(
            {"main_topic": {"text": "Симуляция", "fact_ids": ["F00001"]},
             "objective": None, "overview": [], "chronology": [], "topics": [],
             "decisions": [], "actions": [], "open_questions": []},
            [question], {"covered_seconds": 30, "total_seconds": 30},
            meeting_state_document=state, section_facts=[question, answer],
        )
        self.assertIn("## Ответы и уточнения", rendered)
        self.assertIn("Для симуляции достаточно месяца данных", rendered)

    def test_resolved_question_does_not_render_unrelated_late_answer(self):
        question = fact(kind="question", statement="Какой диапазон используется для открытия сделок?")
        unrelated = dict(
            fact(kind="observation", statement="Последняя загрузка содержала данные с Binance"),
            fact_id="F00002", start=400, end=402,
        )
        state = {"views": {"questions": [{
            "source_record_id": "F00001", "state": "answered",
            "question_kind": "discussion", "answer_record_ids": ["F00002"],
        }]}}
        rendered = summary.render_markdown(
            {"main_topic": {"text": "Сделки", "fact_ids": ["F00001"]},
             "objective": None, "overview": [], "chronology": [], "topics": [],
             "decisions": [], "actions": [], "open_questions": []},
            [question], {"covered_seconds": 500, "total_seconds": 500},
            meeting_state_document=state, section_facts=[question, unrelated],
        )
        self.assertNotIn("## Ответы и уточнения", rendered)

    def test_resolved_question_rejects_editorial_fallback_without_question_text(self):
        question = fact(kind="observation", statement="Обсуждалась альтернатива с дневными свечами")
        answer = dict(
            fact(kind="proposal", statement="Предложено рассмотреть дневные свечи"),
            fact_id="F00002", start=20, end=22,
        )
        registry = {"records": [{
            "record_id": "F00001", "kind": "question", "question_status": "answered",
            "statement": "Миха спрашивает о дневных свечах как альтернативе",
            "answer_record_ids": ["F00002"], "start": 10,
        }]}
        rendered = summary.render_markdown(
            {"main_topic": {"text": "Свечи", "fact_ids": ["F00001"]},
             "objective": None, "overview": [], "chronology": [], "topics": [],
             "decisions": [], "actions": [], "open_questions": []},
            [question], {"covered_seconds": 30, "total_seconds": 30},
            semantic_registry=registry, section_facts=[question, answer],
        )
        self.assertNotIn("## Ответы и уточнения", rendered)

    def test_resolved_question_is_shortened_and_has_single_question_mark(self):
        question = fact(kind="question", statement=("Как проверить длинный набор данных " * 20) + "??")
        answer = dict(
            fact(kind="observation", statement="Нужно открыть файл и проверить конец набора данных"),
            fact_id="F00002", start=20, end=22,
        )
        state = {"views": {"questions": [{
            "source_record_id": "F00001", "state": "answered",
            "question_kind": "discussion", "answer_record_ids": ["F00002"],
        }]}}
        rendered = summary.render_markdown(
            {"main_topic": {"text": "Данные", "fact_ids": ["F00001"]},
             "objective": None, "overview": [], "chronology": [], "topics": [],
             "decisions": [], "actions": [], "open_questions": []},
            [question], {"covered_seconds": 30, "total_seconds": 30},
            meeting_state_document=state, section_facts=[question, answer],
        )
        line = next(line for line in rendered.splitlines() if line.startswith("- **Q-"))
        self.assertLess(len(line.split("?**", 1)[0]), 285)
        self.assertNotIn(". ?**", line)

    def test_navigation_excludes_admin_schedule_and_audio_review(self):
        schedule = fact(kind="schedule", statement="Обсуждалось время созвона во вторник, 19:00 или 20:00")
        schedule["start"] = 100
        uncertain = dict(
            fact(statement="Неясная техническая формулировка требует проверки по аудио"),
            fact_id="F00002", start=200,
        )
        useful = dict(
            fact(kind="problem", statement="Некорректная точка входа приводит к слишком большому Stop Loss",
                 evidence=[utterance(3, 250, 255, text="Некорректная точка входа приводит к слишком большому Stop Loss")]),
            fact_id="F00003", start=250,
        )
        points = summary.navigation_points({}, [schedule, uncertain, useful], 300)
        self.assertEqual([item["fact_id"] for item in points], ["F00003"])

    def test_navigation_label_is_standalone_and_not_a_short_fragment(self):
        self.assertEqual(summary.navigation_label(fact(statement="Нужен BOS")), "")
        item = fact(
            kind="proposal",
            statement="Обсуждалась возможность использовать M15 для подтверждения точки входа",
            evidence=[utterance(1, 10, 12, text="Можно использовать M15 для подтверждения точки входа")],
        )
        self.assertEqual(
            summary.navigation_label(item),
            "Возможность использовать M15 для подтверждения точки входа",
        )

    def test_navigation_label_does_not_invent_domain_context_for_hours(self):
        item = fact(
            kind="proposal",
            statement="Первый подход — с 17 до 18, второй подход — с 16:30 до 18.",
        )
        item["topic"] = "сессия AM для индексов"
        self.assertEqual(summary.navigation_label(item), "")

    def test_navigation_rejects_editorial_doubt_and_embedded_question(self):
        doubtful = fact(statement="Факт требует перепроверки из-за контекста разговора")
        question = fact(kind="problem", statement="Возникает вопрос: куда фиксироваться, если точка не отображается?")
        self.assertEqual(summary.navigation_label(doubtful), "")
        self.assertEqual(summary.navigation_label(question), "")

    def test_navigation_audit_uses_evidence_start_and_has_no_noise(self):
        item = fact(
            kind="problem",
            statement="Некорректная точка входа приводит к слишком большому Stop Loss",
            evidence=[utterance(1, 10.125, 12.75, text="Некорректная точка входа приводит к слишком большому Stop Loss")],
        )
        item["start"] = 25
        report = summary.navigation_quality([item], 60)
        self.assertTrue(report["passed"])
        self.assertEqual(report["entries"][0]["timestamp"], "00:00:10")
        self.assertEqual(report["entries"][0]["evidence_ids"], ["U00001"])

    def test_compact_overview_prefers_validated_executive_summary(self):
        document = {"executive_summary": [
            {"text": "Связный первый абзац о цели и основной проблеме встречи.", "fact_ids": ["F00001"]},
            {"text": "Связный второй абзац о подходах и следующих шагах.", "fact_ids": ["F00002"]},
        ]}
        self.assertEqual(summary.compact_overview(document, {}), [
            "Связный первый абзац о цели и основной проблеме встречи.",
            "Связный второй абзац о подходах и следующих шагах.",
        ])

    def test_executive_summary_rejects_new_number(self):
        item = fact(statement="Проверили качество торговых сигналов")
        paragraphs = [
            {"text": "Обсудили качество торговых сигналов и основные проблемы реализации. "
                     "Отдельно рассмотрели причины ошибок и ограничения текущего подхода.",
             "fact_ids": ["F00001"]},
            {"text": "Модель показала точность 95 процентов на проверочном наборе. "
                     "После этого участники определили направления дальнейшей работы.",
             "fact_ids": ["F00001"]},
        ]
        self.assertIsNone(summary.validate_executive_paragraphs(paragraphs, {"F00001": item}))

    def test_executive_summary_rejects_uncited_sentence(self):
        first = fact(statement="Обсудили качество торговых сигналов и ограничения текущей реализации")
        second = dict(fact(statement="Подключить старшие таймфреймы для дополнительной проверки"), fact_id="F00002")
        paragraphs = [
            {"text": "Обсудили качество торговых сигналов и ограничения текущей реализации. "
                     "Затем сравнили параметры работы проверяемого алгоритма.",
             "fact_ids": ["F00001"]},
            {"text": "Подключить старшие таймфреймы для дополнительной проверки. "
                     "Отдельно решили полностью заменить модель распознавания речи.",
             "fact_ids": ["F00002"]},
        ]
        self.assertIsNone(summary.validate_executive_paragraphs(
            paragraphs, {"F00001": first, "F00002": second}
        ))

    def test_executive_summary_repairs_omitted_fact_link(self):
        facts = {
            "F00001": fact(statement="Задержки ещё не решены на старших таймфреймах"),
            "F00002": dict(fact(statement="Подключить другие таймфреймы к анализу"), fact_id="F00002"),
        }
        repaired = summary.repair_executive_fact_ids([
            {"text": "Сначала обсудили задержки на старших таймфреймах. "
                     "Затем рассмотрели подключение других таймфреймов к анализу.",
             "fact_ids": ["F00001"]},
        ], facts)
        self.assertEqual(set(repaired[0]["fact_ids"]), {"F00001", "F00002"})

    def test_cross_fact_answer_must_be_later_and_nearby(self):
        facts = [
            dict(fact(kind="question", statement="Как округляется стоп?"), fact_id="F00001", start=100),
            dict(fact(statement="Правила задаёт брокер"), fact_id="F00002", start=120),
            dict(fact(statement="Старый тезис"), fact_id="F00003", start=80),
        ]
        records = [
            {"record_id": "F00001", "kind": "question", "question_status": "unclear",
             "answer_evidence_ids": [], "answer_record_ids": ["F00002", "F00003", "F99999"]},
            {"record_id": "F00002", "kind": "observation", "answer_record_ids": []},
            {"record_id": "F00003", "kind": "observation", "answer_record_ids": []},
        ]
        updated = summary.validate_question_links(records, facts)
        self.assertEqual(updated[0]["answer_record_ids"], ["F00002"])
        self.assertEqual(updated[0]["question_status"], "resolved")

    def test_grounded_within_fact_answer_survives_cross_fact_validation(self):
        facts = [dict(fact(kind="question", statement="Спросили о размере; в ответ речь шла о ширине."),
                      fact_id="F00001", start=100)]
        records = [{"record_id": "F00001", "kind": "question", "question_status": "resolved",
                    "answer_resolution_basis": "within_fact_statement", "answer_evidence_ids": [],
                    "answer_record_ids": []}]
        updated = summary.validate_question_links(records, facts)
        self.assertEqual(updated[0]["question_status"], "resolved")

    def test_conflicting_schedule_becomes_question(self):
        evidence = [utterance(1, 1, 3, text="Давайте во вторник, в 20:00, в 19:00")]
        item = fact(kind="schedule", statement="Созвон будет во вторник в 19:00", evidence=evidence)
        updated = summary.enforce_fact_policy(item)
        self.assertEqual(updated["type"], "question")
        self.assertIn("19:00 или 20:00", updated["statement"])
        self.assertIn("не подтверждено", updated["statement"])

    def test_first_person_commitment_becomes_action(self):
        evidence = [utterance(1, 1, 3, speaker="@Yachoy", text="К следующему разу я подготовлю TradingView и скину файл")]
        item = fact(kind="proposal", statement="Необходимо подготовить TradingView и отправить файл", evidence=evidence)
        updated = summary.enforce_fact_policy(item)
        self.assertEqual(updated["type"], "action")
        self.assertEqual(updated["certainty"], "explicit")

    def test_single_evidence_speaker_is_restored_and_invalid_reference_removed(self):
        item = fact(statement="Сообщил о результате", evidence=[utterance(1, 1, 3, speaker="@Yachoy")])
        item["speaker_refs"] = ["@Riven"]
        updated = summary.repair_fact_attribution(item)
        self.assertEqual(updated["speaker_refs"], ["@Yachoy"])

    def test_action_owner_is_derived_from_first_person_commitment(self):
        item = fact(kind="action", statement="Подготовить демонстрацию",
                    evidence=[utterance(1, 1, 3, speaker="@Yachoy", text="Я подготовлю демонстрацию")])
        updated = summary.repair_fact_attribution(item)
        self.assertEqual(updated["owner_refs"], ["@Yachoy"])

    def test_bare_proposal_does_not_create_action_owner(self):
        item = fact(kind="action", statement="Подготовить демонстрацию",
                    evidence=[utterance(1, 1, 3, speaker="@Yachoy", text="Нужно подготовить демонстрацию")])
        updated = summary.repair_fact_attribution(item)
        self.assertEqual(updated["owner_refs"], [])

    def test_bare_maxim_is_marked_ambiguous(self):
        item = fact(statement="Макс посмотрел результат", evidence=[utterance(1, 1, 3, speaker="@Yachoy")])
        item["ambiguous_person_mentions"] = ["Макс"]
        updated = summary.repair_fact_attribution(item)
        self.assertTrue(updated["uncertainty"]["needs_review"])
        self.assertIn("ambiguous_mentioned_person", updated["uncertainty"]["reasons"])

    def test_non_question_asr_problem_stays_out_of_public_summary(self):
        item = fact(kind="problem", statement="Термин распознан неоднозначно и требует проверки по аудио")
        item["uncertainty"] = {"needs_review": True}
        rendered = summary.render_markdown(
            {"main_topic": {"text": "Тема", "fact_ids": [item["fact_id"]]}, "objective": None,
             "overview": [], "chronology": [], "topics": [], "decisions": [], "actions": [], "open_questions": []},
            [item], {"total_seconds": 60}, semantic_registry={"records": [], "tasks": []},
        )
        self.assertNotIn("### Требует сверки с аудио", rendered)
        self.assertNotIn("**A-01.**", rendered)
        self.assertNotIn("Термин распознан", rendered)

    def test_unclear_audio_fragment_is_not_used_as_participant_contribution(self):
        bad = fact(kind="problem", statement="Окончание фразы оборвано и требует проверки по аудио",
                   evidence=[utterance(1, 1, 3, speaker="@Riven")])
        bad["speaker_refs"] = ["@Riven"]
        good = dict(fact(kind="proposal", statement="Предложил проверить реализацию",
                         evidence=[utterance(2, 5, 7, speaker="@Riven",
                                             text="Предложил проверить реализацию")]), fact_id="F00002")
        good["speaker_refs"] = ["@Riven"]
        rendered = "\n".join(summary.participant_lines([bad, good]))
        self.assertEqual(rendered, "")
        self.assertNotIn("Окончание фразы", rendered)

    def test_reviewed_non_fact_is_counted_as_accounted_coverage(self):
        utterances = [
            {"id": "U1", "text": "Это достаточно длинная содержательная реплика о проверке алгоритма и его дальнейшей настройке."},
            {"id": "U2", "text": "Это ещё одна достаточно длинная реплика, которая является повтором уже сохранённого ответа."},
        ]
        facts = [{"evidence_ids": ["U1"]}]
        result = summary.evidence_coverage(utterances, facts, {"U2"})
        self.assertEqual(result["material_coverage_ratio"], 1.0)
        self.assertEqual(result["reviewed_non_fact_utterances"], 1)

    def test_non_active_lifecycle_evidence_is_counted_as_reviewed(self):
        state = {"events": [
            {"lifecycle": "active", "evidence_ids": ["U00001"]},
            {"lifecycle": "superseded", "evidence_ids": ["U00002", "U00003"]},
            {"lifecycle": "resolved", "evidence_ids": ["U00004"]},
            {"lifecycle": "conflicting", "evidence_ids": ["U00003", "U00005"]},
        ]}
        self.assertEqual(
            summary.lifecycle_reviewed_evidence_ids(state),
            ["U00002", "U00003", "U00004", "U00005"],
        )

    def test_first_person_intent_cannot_be_silently_classified_as_context(self):
        source = utterance(1, 1, 3, speaker="@Yachoy", text="Я сейчас, наверное, сделаю упор на доработку этого способа")
        focused = {"index": 0, "utterances": [source], "start": 1, "end": 3}
        facts, resolved, nonfacts = summary.apply_resolution_response(
            {"facts": [], "non_facts": [{"id": "U00001", "class": "context", "reason": "контекст"}]},
            focused, ["U00001"], [], Path(tempfile.mkdtemp()),
        )
        self.assertEqual(facts, [])
        self.assertEqual(resolved, set())
        self.assertEqual(nonfacts, set())

    def test_omitted_tentative_intent_is_recovered_as_action(self):
        source = utterance(
            1, 10, 14, speaker="@Yachoy",
            text="Скорее, возможно, мне нужно ещё над этим посидеть и меньше часть ИИ пилить",
        )
        facts, recovered = summary.recover_omitted_intent("U00001", [source], [])
        self.assertTrue(recovered)
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["type"], "action")
        self.assertEqual(facts[0]["certainty"], "tentative")
        self.assertEqual(facts[0]["owner_refs"], ["@Yachoy"])

    def test_followup_intent_is_attached_to_nearby_action(self):
        first = utterance(1, 10, 14, speaker="@Yachoy", text="Мне нужно ещё над этим посидеть")
        second = utterance(2, 30, 34, speaker="@Yachoy", text="Мне надо ещё раз это попробовать")
        existing = fact(kind="action", statement="@Yachoy намерен ещё раз проверить подход", evidence=[first])
        existing["speaker_refs"] = ["@Yachoy"]
        existing["owner_refs"] = ["@Yachoy"]
        facts, recovered = summary.recover_omitted_intent("U00002", [first, second], [existing])
        self.assertTrue(recovered)
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["evidence_ids"], ["U00001", "U00002"])

    def test_first_person_future_with_details_becomes_action(self):
        evidence = [utterance(1, 1, 3, speaker="@Yachoy", text="Я Bitcoin 2021 года тебе дам для симуляции")]
        item = fact(kind="proposal", statement="Предоставить данные Bitcoin за 2021 год", evidence=evidence)
        updated = summary.enforce_fact_policy(item)
        self.assertEqual(updated["type"], "action")

    def test_promise_does_not_promote_neighboring_predicate(self):
        evidence = [utterance(1, 1, 4, speaker="@A", text="Я передам отчёт. Проверка сервера пока не завершена.")]
        proposal = fact(kind="proposal", statement="Проверить сервер", evidence=evidence)
        self.assertEqual(summary.enforce_fact_policy(proposal)["type"], "proposal")
        commitment = fact(kind="proposal", statement="Передать отчёт", evidence=evidence)
        self.assertEqual(summary.enforce_fact_policy(commitment)["type"], "action")

    def test_promoted_action_survives_second_policy_pass(self):
        evidence = [utterance(1, 1, 3, speaker="@HoTTaBbicH", text="Order Block я буду параллельно размечивать")]
        item = fact(kind="proposal", statement="Параллельно размечать Order Block", evidence=evidence)
        promoted = summary.enforce_fact_policy(item)
        self.assertEqual(promoted["type"], "action")
        self.assertEqual(summary.enforce_fact_policy(promoted)["type"], "action")

    def test_action_statement_removes_editorial_wrapper(self):
        text = "Обсуждалась необходимость участник Yachoy предлагает предоставить данные Bitcoin"
        self.assertEqual(summary.concise_action_statement(text), "Предоставить данные Bitcoin")

    def test_action_statement_removes_explanatory_lead_in(self):
        text = "Поскольку данных уже хватит, необходимо параллельно размечать Order Blocks"
        self.assertEqual(summary.concise_action_statement(text), "Параллельно размечать Order Blocks")

    def test_action_statement_removes_declared_intention_wrapper(self):
        text = "Спикер заявил о намерении подготовить TradingView и отправить EXE-файл"
        self.assertEqual(
            summary.concise_action_statement(text),
            "Подготовить TradingView и отправить EXE-файл",
        )

    def test_recovered_action_is_made_reusable(self):
        item = fact(
            kind="action",
            statement="необходимо ещё раз попробовать работу в тех реализациях, которых я пытался.",
        )
        self.assertEqual(
            summary.clean_publication_statement(item),
            "Ещё раз проверить предыдущие реализации.",
        )

    def test_publication_statement_removes_internal_editor_note(self):
        item = fact(
            statement="Найти точку входа на нужном активе (исправлено: термин требует проверки)."
        )
        self.assertEqual(
            summary.clean_publication_statement(item),
            "Найти точку входа на нужном активе.",
        )

    def test_long_title_ends_at_a_complete_topic(self):
        document = {"main_topic": {"text": (
            "Статус имбалансов, анализ разворотов после пробоя барьеров, "
            "поведение цены после снятия минимума Лондона, тактика максимизации прибыли, "
            "стратегии входа и выхода с дополнительными фильтрами"
        )}}
        title = summary.concise_title(document)
        self.assertLessEqual(len(title), 140)
        self.assertNotIn("…", title)
        self.assertIn(title, document["main_topic"]["text"])
        self.assertNotRegex(title, r"[,;:]$")

    def test_task_registry_removes_condition_that_repeats_task(self):
        registry = {"schema_version": 3, "records": [], "tasks": [{
            "description": "Подключить другие таймфреймы к анализу.",
            "conditions": [
                {"text": "при подключении других таймфреймов к анализу", "evidence_ids": ["U00001"]},
                {"text": "после завершения базовой логики", "evidence_ids": ["U00002"]},
                {"text": "до следующего раза (срок выполнения)", "evidence_ids": ["U00003"]},
            ],
        }]}
        cleaned = summary.clean_task_registry(registry)
        self.assertEqual(
            [item["text"] for item in cleaned["tasks"][0]["conditions"]],
            ["после завершения базовой логики"],
        )
        self.assertEqual(cleaned["tasks"][0]["due"], "к следующему разу")

    def test_conflicting_percent_estimates_are_not_reconciled_by_writer(self):
        first = fact(kind="hypothesis", statement="При одной свече качество проседает на 8%")
        second = dict(
            fact(kind="observation", statement="При сокращении до одной свечи качество просядет на 10%"),
            fact_id="F00002",
            evidence_ids=["U00002"],
            evidence=[utterance(2, 20, 24, text="При одной свече качество проседает на 10%")],
            start=20,
            end=24,
        )
        document = {
            "chronology": [{
                "text": "При одной свече качество падает на 8%, а без задержки — на 10%",
                "fact_ids": ["F00001", "F00002"],
            }]
        }
        clean, _ = summary.sanitize_structured(document, [first, second])
        result = clean["chronology"][0]["text"]
        # Rendering uses verbatim audited registry statements; arbitrary paraphrases
        # must pass through grounding before publication.
        chapter = {"topics": [{"title": "Оценки", "items": document["chronology"]}]}
        grounded = summary.ground_chapter_items(chapter, [first, second])
        result = grounded["topics"][0]["items"][0]["text"]
        self.assertIn("8%", result)
        self.assertIn("10%", result)
        self.assertNotIn("без задержки", result)
        self.assertNotIn("противореч", result)

    def test_conflicting_percent_facts_do_not_replace_broad_heading(self):
        first = fact(kind="hypothesis", statement="При одной свече качество проседает на 8%")
        second = dict(
            fact(kind="observation", statement="При сокращении до одной свечи качество просядет на 10%"),
            fact_id="F00002",
            evidence_ids=["U00002"],
            evidence=[utterance(2, 20, 24, text="При одной свече качество проседает на 10%")],
            start=20,
            end=24,
        )
        document = {
            "main_topic": {
                "text": "Обсуждение алгоритма входа и задержек на таймфреймах",
                "fact_ids": ["F00001", "F00002"],
            },
            "overview": [{
                "text": "Методика предсказания тренда и оптимизация симуляции данных.",
                "fact_ids": ["F00001", "F00002"],
            }],
        }
        clean, _ = summary.sanitize_structured(document, [first, second])
        self.assertEqual(clean["main_topic"]["text"], document["main_topic"]["text"])
        self.assertEqual(clean["overview"][0]["text"], document["overview"][0]["text"])

    def test_writer_temporal_links_without_relation_are_replaced_by_claims(self):
        facts = []
        statements = [
            ("hypothesis", "При одной свече качество проседает на 8%"),
            ("observation", "При сокращении до одной свечи качество просядет на 10%"),
            ("observation", "Обсудили направление тренда"),
            ("proposal", "Предложено проверить Bitcoin"),
            ("action", "Параллельно размечать Order Blocks"),
        ]
        for index, (kind, statement) in enumerate(statements, 1):
            facts.append(dict(
                fact(kind=kind, statement=statement),
                fact_id=f"F{index:05d}",
                evidence_ids=[f"U{index:05d}"],
                evidence=[utterance(index, index * 10, index * 10 + 2, text=statement)],
                start=index * 10,
                end=index * 10 + 2,
            ))
        text = (
            "Сначала сравнили оценки 8% и 10%, затем обсудили направление тренда, "
            "проверку Bitcoin и разметку Order Blocks."
        )
        clean, _ = summary.sanitize_structured({
            "chronology": [{"text": text, "fact_ids": [item["fact_id"] for item in facts]}],
        }, facts)
        self.assertNotEqual(clean["chronology"][0]["text"], text)
        self.assertIn("При одной свече качество проседает на 8%.", clean["chronology"][0]["text"])

    def test_evidence_containment_does_not_remove_additional_action(self):
        first = fact(kind="action", statement="Параллельно размечать Order Block")
        second = dict(
            fact(kind="action", statement="Order Block нужно параллельно размечать и встраивать"),
            fact_id="F00002",
            evidence_ids=["U00001", "U00002"],
            evidence=[utterance(1, 10.125, 12.75, text="Я буду размечать Order Block"), utterance(2, 13, 14, text="Да")],
        )
        self.assertEqual(len(summary.deduplicate([first, second])), 2)

    def test_same_text_with_different_modality_is_preserved(self):
        first = fact(kind="observation", statement="Цена перешла в боковое движение.")
        second = dict(
            fact(kind="hypothesis", statement="Цена перешла в боковое движение"),
            fact_id="F00002",
            evidence_ids=["U00002"],
            evidence=[utterance(2, 20, 24, text="Цена перешла в боковое движение")],
            start=20,
            end=24,
        )
        merged = summary.deduplicate([first, second])
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["evidence_ids"], ["U00001"])

    def test_owner_question_plus_local_yes_confirms_assignment(self):
        question = utterance(1, 1, 3, speaker="@Misha", text="Мне сделать тебе разметчик Order Block?")
        confirmation = utterance(2, 3.1, 4, speaker="@HoTTaBbicH", text="Да, дальше этап апробации")
        item = fact(kind="proposal", statement="Misha должен сделать разметчик Order Block", evidence=[question])
        updated = summary.resolve_dialogue_commitments([item], [question, confirmation])[0]
        self.assertEqual(updated["type"], "action")
        self.assertEqual(updated.get("policy_note"), "confirmed_owner_question_promoted")
        self.assertEqual(updated["evidence_ids"], ["U00001", "U00002"])

    def test_recognition_review_cannot_turn_commitment_into_question(self):
        item = fact(kind="action", statement="Я передам выгрузку", evidence=[utterance(1, 1, 2, text="Я передам выгрузку")])
        accepted, rejected = summary.apply_reviews([item], [{"fact_id": "F00001", "verdict": "corrected", "type": "question", "statement": "Нужно ли передать выгрузку?", "confidence": .7}], enforce_policy=False)
        self.assertFalse(rejected)
        self.assertEqual(accepted[0]["type"], "action")
        self.assertEqual(accepted[0]["statement"], "Я передам выгрузку")
        self.assertEqual(accepted[0]["review_patch"]["reason"], "speech_act_guard")

    def test_cache_is_invalidated_when_prompt_changes(self):
        class Client:
            def __init__(self): self.calls = 0
            def chat(self, *args, **kwargs):
                self.calls += 1
                return json.dumps({"facts": [], "no_material": True}), {}
        client = Client()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            summary.call_json_with_retries(client, "model", "system", "first", path)
            summary.call_json_with_retries(client, "model", "system", "first", path)
            summary.call_json_with_retries(client, "model", "system", "second", path)
        self.assertEqual(client.calls, 2)

    def test_material_coverage_detects_long_unreferenced_turn(self):
        items = [
            utterance(1, 1, 3, text="Короткое приветствие"),
            utterance(2, 4, 12, text="Нужно перенести текущую реализацию в симуляцию и собрать статистику работы алгоритма за месяц."),
        ]
        report = summary.evidence_coverage(items, [])
        self.assertEqual(report["material_utterances"], 1)
        self.assertEqual(report["covered_material_utterances"], 0)
        self.assertEqual(report["missing_material_ids"], ["U00002"])

    def test_reviewed_context_counts_without_becoming_fact(self):
        items = [
            utterance(1, 1, 8, text="А можешь подробно объяснить, почему программа сейчас не открывает сделку, хотя все показанные условия выглядят выполненными?"),
            utterance(2, 9, 18, text="Проблема находится в последней проверке условия входа, её нужно отдельно диагностировать."),
        ]
        supported = fact(
            kind="observation",
            statement="Проблема находится в последней проверке условия входа",
            evidence=[items[1]],
        )
        report = summary.evidence_coverage(items, [supported], reviewed_non_fact_ids={"U00001"})
        self.assertEqual(report["material_coverage_ratio"], 1.0)
        self.assertEqual(report["fact_material_utterances"], 1)
        self.assertEqual(report["reviewed_non_fact_utterances"], 1)

    def test_focused_context_keeps_neighbors_and_targets(self):
        items = [utterance(i, i * 10, i * 10 + 2, text=f"Реплика {i} с достаточно подробным содержанием") for i in range(1, 8)]
        focused = summary.focused_context(items, ["U00003", "U00006"])
        self.assertEqual(
            [item["id"] for item in focused["utterances"]],
            ["U00002", "U00003", "U00004", "U00005", "U00006", "U00007"],
        )

    def test_resolution_accepts_combined_non_fact_class(self):
        item = utterance(1, 1, 8, text="Это уточняющий вопрос, смысл которого раскрывается в следующем содержательном ответе участника.")
        focused = summary.focused_context([item], [item["id"]])
        response = {"facts": [], "non_facts": [{"id": item["id"], "class": "context|question", "reason": "уточнение"}]}
        with tempfile.TemporaryDirectory() as directory:
            facts, resolved, reviewed = summary.apply_resolution_response(
                response, focused, [item["id"]], [], Path(directory)
            )
        self.assertEqual(facts, [])
        self.assertEqual(resolved, set())
        self.assertEqual(reviewed, {item["id"]})

    def test_resolution_promotes_fact_misplaced_in_non_facts(self):
        item = utterance(1, 1, 8, text="У других наблюдателей этот вариант может работать стабильнее, но это пока предположение.")
        focused = summary.focused_context([item], [item["id"]])
        response = {"facts": [], "non_facts": [{"id": item["id"], "class": "hypothesis", "reason": "Другой вариант может работать стабильнее"}]}
        with tempfile.TemporaryDirectory() as directory:
            facts, resolved, reviewed = summary.apply_resolution_response(
                response, focused, [item["id"]], [], Path(directory)
            )
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["type"], "hypothesis")
        self.assertEqual(resolved, {item["id"]})
        self.assertEqual(reviewed, set())

    def test_backfill_guarantees_every_fact_is_used(self):
        first = fact(kind="proposal", statement="Предложено проверить BOS")
        second = dict(
            fact(kind="action", statement="Подготовить TradingView"),
            fact_id="F00002",
            evidence_ids=["U00002"],
            evidence=[utterance(2, 20, 24, text="Я подготовлю TradingView")],
            start=20,
            end=24,
        )
        document = {
            "main_topic": {"text": "Обсуждение BOS", "fact_ids": ["F00001"]},
            "objective": None,
            "overview": [], "chronology": [], "topics": [],
            "decisions": [], "actions": [], "open_questions": [],
        }
        completed, report = summary.backfill_missing_facts(document, [first, second])
        self.assertEqual(report["fact_coverage_ratio"], 1.0)
        self.assertEqual(completed["actions"][0]["fact_ids"], ["F00002"])

    def test_detailed_topic_coverage_is_not_masked_by_main_heading(self):
        facts = [fact(), dict(fact(statement="Второй факт"), fact_id="F00002")]
        document = summary.empty_document()
        document["main_topic"] = {"text": "Общая тема", "fact_ids": ["F00001", "F00002"]}
        document["topics"] = [{
            "title": "BOS", "items": [{"text": facts[0]["statement"], "fact_ids": ["F00001"]}],
        }]
        repaired = summary.backfill_topic_facts(document, facts)
        self.assertEqual(summary.topic_fact_ids(repaired), {"F00001", "F00002"})
        self.assertEqual(len(repaired["topics"]), 1)
        report = summary.structural_quality(repaired, facts, chapters=1)
        self.assertEqual(report["topic_fact_coverage_ratio"], 1.0)
        self.assertEqual(report["missing_topic_fact_ids"], [])

    def test_structure_gate_requires_detailed_topic_coverage(self):
        facts = [
            fact(statement="Первый тезис"),
            dict(fact(statement="Второй тезис"), fact_id="F00002", evidence_ids=["U00002"]),
        ]
        document = summary.empty_document()
        document["main_topic"] = {"text": "Общая тема", "fact_ids": ["F00002"]}
        document["chronology"] = [
            {"text": "Первый тезис", "fact_ids": ["F00001"]},
            {"text": "Второй тезис", "fact_ids": ["F00002"]},
            {"text": "Обзор", "fact_ids": ["F00001", "F00002"]},
        ]
        document["topics"] = [
            {"title": "Тема 1", "items": [{"text": "Первый тезис", "fact_ids": ["F00001"]}]},
            {"title": "Тема 2", "items": []},
            {"title": "Тема 3", "items": []},
            {"title": "Тема 4", "items": []},
        ]
        report = summary.structural_quality(document, facts, chapters=1)
        self.assertEqual(report["fact_coverage_ratio"], 1.0)
        self.assertEqual(report["topic_fact_coverage_ratio"], 0.5)
        self.assertEqual(report["missing_topic_fact_ids"], ["F00002"])
        # Accounting remains observable, but public summaries no longer need
        # to dump every fact into every thematic view.
        summary.require_structural_quality({**report, "topics": 1, "largest_topic_ratio": 0.4})

    def test_structure_gate_rejects_empty_writer_output(self):
        report = summary.structural_quality(summary.empty_document(), [fact()], chapters=1, repaired_facts=1)
        with self.assertRaises(RuntimeError):
            summary.require_structural_quality(report, final=True)

    def test_named_topic_repair_avoids_catch_all(self):
        item = fact(statement="Предложено проверить BOS")
        repaired = summary.add_missing_to_named_topics(summary.empty_document(), [item])
        self.assertEqual(repaired["topics"][0]["title"], "BOS")
        self.assertNotEqual(repaired["topics"][0]["title"], "Дополнительные подтверждённые детали")

    def test_rejected_chapter_chronology_gets_evidence_safe_fallback(self):
        chapter = summary.empty_document()
        chapter["topics"] = [{
            "title": "Структура",
            "items": [
                {"text": "Сначала обсудили BOS", "fact_ids": ["F00001"]},
                {"text": "Затем договорились проверить результат", "fact_ids": ["F00002"]},
            ],
        }]
        repaired = summary.ensure_chapter_chronology(chapter)
        self.assertEqual(len(repaired["chronology"]), 1)
        self.assertEqual(repaired["chronology"][0]["fact_ids"], ["F00001", "F00002"])
        self.assertIn("Сначала обсудили BOS", repaired["chronology"][0]["text"])

    def test_generated_chronology_is_replaced_by_all_sanitized_topic_items(self):
        chapter = summary.empty_document()
        chapter["chronology"] = [{"text": "Придуманная связка", "fact_ids": ["F00001"]}]
        chapter["topics"] = [{
            "title": "Структура",
            "items": [
                {"text": "Первый проверенный тезис", "fact_ids": ["F00001"]},
                {"text": "Второй проверенный тезис", "fact_ids": ["F00002"]},
                {"text": "Третий проверенный тезис", "fact_ids": ["F00003"]},
            ],
        }]
        repaired = summary.ensure_chapter_chronology(chapter)
        self.assertNotIn("Придуманная связка", repaired["chronology"][0]["text"])
        self.assertEqual(repaired["chronology"][0]["fact_ids"], ["F00001", "F00002", "F00003"])
        self.assertIn("Третий проверенный тезис", repaired["chronology"][0]["text"])

    def test_published_chronology_is_concise_but_topics_keep_full_coverage(self):
        facts = []
        for index in range(1, 13):
            kind = "action" if index in (4, 9) else "observation"
            facts.append(dict(
                fact(kind=kind, statement=f"Проверенный тезис {index}"),
                fact_id=f"F{index:05d}",
                evidence_ids=[f"U{index:05d}"],
                evidence=[utterance(index, index, index + 1, text=f"Проверенный тезис {index}")],
                start=index, end=index + 1,
            ))
        chapter = summary.empty_document()
        chapter["topics"] = [{
            "title": "Тема",
            "items": [{"text": item["statement"], "fact_ids": [item["fact_id"]]} for item in facts],
        }]
        repaired = summary.ensure_chapter_chronology(chapter, facts, max_facts=6)
        self.assertEqual(len(repaired["chronology"][0]["fact_ids"]), 6)
        self.assertIn("F00004", repaired["chronology"][0]["fact_ids"])
        self.assertIn("F00009", repaired["chronology"][0]["fact_ids"])
        self.assertEqual(len(repaired["topics"][0]["items"]), 12)

    def test_unanswered_question_about_completed_work_stays_question(self):
        item = fact(
            kind="observation",
            statement="Миша уже сделал имбалансы.",
            evidence=[utterance(1, 1, 4, text="Ты сделал уже имбалансы, да? Если я ничего не путаю.")],
        )
        class Client:
            def chat(self, *args, **kwargs):
                return json.dumps({"reviews": [{
                    "fact_id": "F00001", "verdict": "corrected", "type": "question",
                    "statement": "У Миши уточнили, завершена ли разметка имбалансов.", "evidence_ids": ["U00001"],
                    "confidence": 0.9, "reason": "",
                }]}), {}
        with tempfile.TemporaryDirectory() as directory:
            kept, _ = summary.prepare_publishable_facts(Client(), "model", [item], Path(directory))
        self.assertEqual(kept[0]["type"], "question")
        self.assertIn("уточнили", kept[0]["statement"].casefold())

    def test_distinct_delay_scenarios_are_not_forced_into_conflict(self):
        first = fact(
            kind="observation", statement="Задержка в 2 минуты на минутном таймфрейме не является критичной.",
            evidence=[utterance(1, 1, 4, text="На одной минутке задержка в 2 минуты вообще не страшно")],
        )
        second = dict(
            fact(
                kind="observation", statement="В показанном сценарии модель M1 без задержек.",
                evidence=[utterance(2, 5, 8, text="Модель M1, на которой нет никаких задержек")],
            ),
            fact_id="F00002", evidence_ids=["U00002"], start=5, end=8,
        )
        class Client:
            def chat(self, *args, **kwargs):
                return json.dumps({"reviews": [
                    {"fact_id": "F00001", "verdict": "supported", "type": "observation", "statement": first["statement"], "evidence_ids": ["U00001"], "confidence": .9, "reason": ""},
                    {"fact_id": "F00002", "verdict": "supported", "type": "observation", "statement": second["statement"], "evidence_ids": ["U00002"], "confidence": .9, "reason": ""},
                ]}), {}
        with tempfile.TemporaryDirectory() as directory:
            kept, _ = summary.prepare_publishable_facts(Client(), "model", [first, second], Path(directory))
        conflict = next(item for item in kept if item["fact_id"] == "F00002")
        self.assertEqual(conflict["type"], "observation")
        self.assertEqual(conflict["evidence_ids"], ["U00002"])
        self.assertEqual(conflict["statement"], second["statement"])

    def test_publishable_review_splits_a_batch_when_output_is_truncated(self):
        facts = []
        for index in range(1, 9):
            facts.append(dict(
                fact(statement=f"Проверенный тезис {index}"),
                fact_id=f"F{index:05d}", evidence_ids=[f"U{index:05d}"],
                evidence=[utterance(index, index, index + 1, text=f"Проверенный тезис {index}")],
            ))

        class Client:
            def chat(self, _model, _system, prompt, **_kwargs):
                payload = json.loads(prompt.split("\n", 1)[1].rsplit("\nФормат:", 1)[0])
                if len(payload) > 4:
                    raise RuntimeError("ответ оборван или достигнут лимит вывода")
                return json.dumps({"reviews": [{
                    "fact_id": item["fact_id"], "verdict": "supported",
                    "type": item["type"], "statement": item["statement"],
                    "evidence_ids": item["evidence_ids"], "confidence": .9, "reason": "",
                } for item in payload]}), {}

        with tempfile.TemporaryDirectory() as directory:
            kept, rejected = summary.prepare_publishable_facts(Client(), "model", facts, Path(directory))
            cache_files = sorted((Path(directory) / "publishable").glob("facts-*.json"))
        self.assertEqual(len(kept), 8)
        self.assertEqual(rejected, [])
        self.assertEqual([path.name for path in cache_files], [
            "facts-00001-00004.json", "facts-00005-00008.json",
        ])

    def test_publishable_review_does_not_hide_single_fact_failure(self):
        class Client:
            def chat(self, *_args, **_kwargs):
                raise RuntimeError("ответ оборван или достигнут лимит вывода")

        with tempfile.TemporaryDirectory() as directory:
            kept, rejected = summary.prepare_publishable_facts(Client(), "model", [fact()], Path(directory))
            self.assertFalse(rejected)
            self.assertEqual(kept[0]["verification_status"], "verification_unavailable")
            self.assertEqual(kept[0]["verification_failure_stage"], "editorial_review")

    def test_final_audit_splits_when_model_omits_reviews(self):
        facts = [
            fact(statement="Первый тезис"),
            dict(fact(statement="Второй тезис"), fact_id="F00002"),
        ]

        class Client:
            def chat(self, _model, _system, prompt, **_kwargs):
                payload = json.loads(prompt.split("\n", 1)[1].rsplit("\nФормат:", 1)[0])
                selected = payload[:1] if len(payload) > 1 else payload
                return json.dumps({"reviews": [{
                    "fact_id": item["fact_id"], "verdict": "supported",
                    "type": item["type"], "statement": item["statement"],
                    "evidence_ids": item["evidence_ids"], "confidence": .9, "reason": "",
                } for item in selected]}), {}

        with tempfile.TemporaryDirectory() as directory:
            accepted, rejected = summary.audit_final_facts(Client(), "model", facts, Path(directory))
        self.assertEqual([item["fact_id"] for item in accepted], ["F00001", "F00002"])
        self.assertEqual(rejected, [])

    def test_large_public_auditor_removes_unsupported_visible_fact(self):
        first = fact(kind="action", statement="Проверить BOS в симуляции")
        first["owner_refs"] = ["@Riven"]
        second = dict(
            fact(kind="action", statement="Отправить неподтверждённый отчёт"),
            fact_id="F00002", evidence_ids=["U00002"],
            evidence=[utterance(2, 20, 22, text="Такого действия не было")],
            start=20, end=22, owner_refs=["@Riven"],
        )

        class Client:
            def chat(self, *_args, **_kwargs):
                return json.dumps({
                    "supported_ids": ["F00001"],
                    "changes": [{"fact_id": "F00002", "verdict": "reject"}],
                }), {}

        with tempfile.TemporaryDirectory() as directory:
            kept, rejected, details = summary.audit_public_surface_facts(
                Client(), "large-model", [first, second], Path(directory), 60
            )
        self.assertEqual([item["fact_id"] for item in kept], ["F00001"])
        self.assertEqual(len(rejected), 1)
        self.assertEqual(details["degraded_batches"], [])

    def test_large_public_auditor_failure_quarantines_critical_fact(self):
        item = fact(kind="action", statement="Проверить BOS в симуляции")
        item["owner_refs"] = ["@Riven"]

        class Client:
            def chat(self, *_args, **_kwargs):
                raise RuntimeError("large model temporarily unavailable")

        with tempfile.TemporaryDirectory() as directory:
            kept, rejected, details = summary.audit_public_surface_facts(
                Client(), "large-model", [item], Path(directory), 60
            )
            self.assertEqual(len(kept), 1)
            self.assertEqual(kept[0]["verification_status"], "verification_unavailable")
            self.assertEqual(rejected, [])
        self.assertEqual(len(details["degraded_batches"]), 1)

    def test_overview_uses_real_chapter_evidence(self):
        document = summary.empty_document()
        document["topics"] = [{
            "title": "Точки входа и BOS",
            "items": [
                {"text": "Первый тезис", "fact_ids": ["F00001"]},
                {"text": "Второй тезис", "fact_ids": ["F00002", "F00001"]},
            ],
        }]
        overview = summary.overview_from_chapters(document)
        self.assertEqual(overview, [{"text": "Точки входа и BOS.", "fact_ids": ["F00001", "F00002"]}])

    def test_chapter_items_use_exact_validated_facts_not_model_paraphrase(self):
        facts = [
            fact(statement="Было предложено, чтобы Максим разметил размер имбалансов."),
            dict(fact(statement="Миша предложил показать точки на демо."), fact_id="F00002"),
        ]
        chapter = summary.empty_document()
        chapter["topics"] = [{
            "title": "Имбалансы",
            "items": [{
                "text": "Максим предложил разметить имбалансы и показать демо.",
                "fact_ids": ["F00001", "F00002"],
            }],
        }]
        grounded = summary.ground_chapter_items(chapter, facts)
        text = grounded["topics"][0]["items"][0]["text"]
        self.assertIn("Было предложено, чтобы Максим", text)
        self.assertNotIn("Максим предложил разметить", text)

    def test_conflicting_percent_facts_are_split_from_neighboring_claims(self):
        facts = [
            fact(statement="Алгоритм имеет задержку в две свечи."),
            dict(fact(kind="hypothesis", statement="Качество проседает на 8%."), fact_id="F00002"),
            dict(fact(kind="observation", statement="Качество просядет на 10%."), fact_id="F00003"),
        ]
        chapter = summary.empty_document()
        chapter["topics"] = [{
            "title": "Задержка",
            "items": [{"text": "Свободный пересказ", "fact_ids": ["F00001", "F00002", "F00003"]}],
        }]
        grounded = summary.ground_chapter_items(chapter, facts)
        items = grounded["topics"][0]["items"]
        self.assertEqual([item["fact_ids"] for item in items], [["F00001"], ["F00002", "F00003"]])

    def test_main_topic_lists_every_chapter(self):
        document = summary.empty_document()
        document["topics"] = [
            {"title": "Имбалансы", "items": []},
            {"title": "Стоп-лоссы", "items": []},
            {"title": "Симуляция", "items": []},
        ]
        result = summary.main_topic_from_chapters(document, [fact()])
        self.assertIn("имбалансы", result["text"])
        self.assertIn("стоп-лоссы", result["text"])
        self.assertIn("симуляция", result["text"])

    def test_main_topic_does_not_assign_higher_timeframe_delay_to_m1(self):
        facts = [
            fact(statement="На старших таймфреймах есть задержка."),
            dict(fact(statement="В показанном сценарии использовалась модель M1 без задержек."), fact_id="F00002"),
        ]
        result = summary.normalize_main_topic("Обсуждение задержек на M1.", facts)
        self.assertEqual(result, "Обсуждение задержек на M1.")

    def test_chapter_title_does_not_assign_delay_only_to_m1(self):
        result = summary.normalize_topic_title("Анализ задержек на M1 и работа со сломов структуры")
        self.assertEqual(result, "Анализ задержек на M1 и работа со сломами структуры")

    def test_primary_validation_splits_incomplete_response(self):
        facts = [fact(), dict(fact(statement="Второй подтверждённый тезис"), fact_id="F00002")]

        class Client:
            def chat(self, _model, _system, prompt, **_kwargs):
                payload = json.loads(prompt.split("\n", 1)[1].split("\n\nФормат:", 1)[0])
                chosen = payload[:1] if len(payload) > 1 else payload
                return json.dumps({"reviews": [{
                    "fact_id": item["fact_id"], "verdict": "supported",
                    "type": item["type"], "statement": item["statement"],
                    "evidence_ids": item["evidence_ids"], "confidence": .9,
                } for item in chosen]}), {}

        with tempfile.TemporaryDirectory() as directory:
            accepted, rejected = summary.validate_facts_adaptive(
                Client(), "model", facts, Path(directory)
            )
        self.assertEqual([item["fact_id"] for item in accepted], ["F00001", "F00002"])
        self.assertEqual(rejected, [])

    def test_primary_validation_bisects_a_zero_progress_batch(self):
        facts = [fact(), dict(fact(statement="Второй подтверждённый тезис"), fact_id="F00002")]

        class Client:
            def chat(self, _model, _system, prompt, **_kwargs):
                payload = json.loads(prompt.split("\n", 1)[1].split("\n\nФормат:", 1)[0])
                if len(payload) > 1:
                    return json.dumps({"reviews": []}), {}
                item = payload[0]
                return json.dumps({"reviews": [{
                    "fact_id": item["fact_id"], "verdict": "supported",
                    "type": item["type"], "statement": item["statement"],
                    "evidence_ids": item["evidence_ids"], "confidence": .9,
                }]}), {}

        with tempfile.TemporaryDirectory() as directory:
            accepted, rejected = summary.validate_facts_adaptive(
                Client(), "model", facts, Path(directory)
            )
        self.assertEqual([item["fact_id"] for item in accepted], ["F00001", "F00002"])
        self.assertEqual(rejected, [])

    def test_arbitration_bisects_a_zero_progress_batch(self):
        facts = [fact(), dict(fact(statement="Второй подтверждённый тезис"), fact_id="F00002")]
        original = summary.call_json_with_retries

        def response(_client, _model, _system, prompt, _cache_path, **_kwargs):
            payload = json.loads(
                prompt.split("Независимо перепроверь критичные и сомнительные факты:\n", 1)[1]
                .split("\nВерни строго", 1)[0]
            )
            reviews = [] if len(payload) > 1 else [{
                "fact_id": payload[0]["fact_id"], "verdict": "supported",
                "type": payload[0]["type"], "statement": payload[0]["statement"],
                "evidence_ids": payload[0]["evidence_ids"], "confidence": .9,
            }]
            return {"response": {"reviews": reviews}}

        summary.call_json_with_retries = response
        try:
            accepted, rejected = summary.arbitrate(
                object(), "model", facts, Path("batch.json")
            )
        finally:
            summary.call_json_with_retries = original
        self.assertEqual([item["fact_id"] for item in accepted], ["F00001", "F00002"])
        self.assertEqual(rejected, [])

    def test_semantic_registry_splits_incomplete_response(self):
        facts = [fact(), dict(fact(statement="Второй подтверждённый тезис"), fact_id="F00002")]

        class Client:
            def chat(self, _model, _system, prompt, **_kwargs):
                payload = json.loads(prompt.split("ТЕЗИСЫ:\n", 1)[1])
                chosen = payload[:1] if len(payload) > 1 else payload
                return json.dumps({"records": [{"record_id": item["fact_id"]} for item in chosen]}), {}

        with tempfile.TemporaryDirectory() as directory:
            registry = summary.build_semantic_registry(Client(), "model", facts, Path(directory))
        self.assertEqual([item["record_id"] for item in registry["records"]], ["F00001", "F00002"])

    def test_critical_consensus_accepts_two_matching_supported_verdicts(self):
        original = dict(fact(kind="decision"), confidence=.8)
        primary = dict(original, validation="supported", confidence=.91)
        secondary = dict(original, validation="supported", confidence=.84)
        accepted, rejected = summary.critical_verifier_consensus(
            [original], [primary], [secondary], "ministral", "gemma"
        )
        self.assertEqual(rejected, [])
        self.assertEqual(accepted[0]["validation"], "dual_supported")
        self.assertEqual(accepted[0]["confidence"], .84)
        self.assertEqual(accepted[0]["critical_consensus"]["models"], ["ministral", "gemma"])

    def test_critical_consensus_fails_closed_on_disagreement(self):
        original = dict(fact(kind="decision"), confidence=.8)
        primary = dict(original, validation="supported", confidence=.91)
        accepted, rejected = summary.critical_verifier_consensus(
            [original], [primary], [], "ministral", "gemma"
        )
        self.assertFalse(rejected)
        self.assertEqual(accepted[0]["verification_status"], "verification_unavailable")
        self.assertEqual(accepted[0]["critical_consensus"]["reason"], "critical_verifier_disagreement")

    def test_critical_consensus_requires_identical_corrections(self):
        original = dict(fact(kind="decision"), confidence=.8)
        primary = dict(original, validation="corrected", statement="Предложено проверить BOS", confidence=.9)
        secondary = dict(original, validation="corrected", statement="Решено проверить BOS", confidence=.9)
        accepted, rejected = summary.critical_verifier_consensus(
            [original], [primary], [secondary], "ministral", "gemma"
        )
        self.assertFalse(rejected)
        self.assertEqual(accepted[0]["verification_status"], "verification_unavailable")
        self.assertEqual(accepted[0]["critical_consensus"]["reason"], "critical_correction_mismatch")

    def test_question_metrics_use_canonical_final_states(self):
        metrics = summary.canonical_question_metrics([
            {"status": "answered"}, {"status": "answered"},
            {"status": "partially_answered"}, {"status": "unanswered"},
        ])
        self.assertEqual(metrics["questions_resolved"], 2)
        self.assertEqual(metrics["questions_partially_answered"], 1)
        self.assertEqual(metrics["questions_unresolved"], 1)
        self.assertEqual(metrics["question_status_counts"], {"answered": 2, "partially_answered": 1, "unanswered": 1})


if __name__ == "__main__":
    unittest.main()
