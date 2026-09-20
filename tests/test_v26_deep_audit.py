import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from semantics.equivalence import equivalent
from semantics.meeting_graph import _expand_action_records, build_meeting_graph
from semantics.questions import verify_slot_entailment
from semantics.reducers import reduce_questions
from semantics.relation_resolver import is_explicit_acceptance_reply
from scripts.quality_schema import normalize_semantic_record, source_action_clauses
import scripts.summary_worker as summary
from summary.outcomes import build_outcome_cards
from summary.planner import plan
from summary.verifier import build_public_items, can_publish_as_decision, canonicalize_public_item_order, navigation_label_needs_repair, protected_public_item_abstentions, publication_audit, public_surface_text, sanitize_public_surface, task_surface_text, verify_generated_items, verify_public_document
from scripts.summary_worker import _best_navigation_item, _document_audit_nodes, _has_unresolved_title_reference, _title_repair_candidates, apply_bounded_document_writer, build_public_document, ensure_late_technical_outcomes, preflight_candidate_lineage, reconcile_final_document_audit, render_public_document, retype_rejected_fact, time_link
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
    def test_candidate_preflight_rejects_a_rebuild_without_audio_provenance(self):
        record = rec(1, "observation", "Проверяется состояние системы.")
        with self.assertRaisesRegex(ValueError, "audio_sha256"):
            preflight_candidate_lineage([], [record], [], [], [])

    def test_candidate_preflight_repairs_source_grounded_work_before_planning(self):
        turn = {
            "id": "U1", "speaker": "@A", "start": 7.0,
            "text": "Я схему пошёл делать.", "source_word_ids": ["W1"],
        }
        fact = {
            "fact_id": "F1", "type": "proposal",
            "statement": "Участник начал делать схему.",
            "speaker_refs": ["@A"], "owner_refs": [],
            "evidence_ids": ["U1"], "evidence": [turn],
            "uncertainty": {"needs_review": False},
            "verification_status": "supported", "origin_id": "OR1",
        }
        stale_record = {
            **rec(1, "proposal", fact["statement"], "propose", speaker="@A"),
            "origin_id": "OR1", "actions": [], "dialogue_evidence": [turn],
            "evidence_ids": ["U1"], "source_word_ids": ["W1"],
        }
        records, graph, report = preflight_candidate_lineage(
            [{
                "origin_id": "OR1:A01", "revision_id": "RV1",
                "fact_id": "F1", "kind": "action", "statement": fact["statement"],
                "evidence_ids": ["U1"], "source_word_ids": ["W1"],
            }],
            [stale_record], [fact], [turn], [],
            provenance={"audio_sha256": "audio-hash"},
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["dispositions"][0]["status"], "canonical_task")
        self.assertEqual(report["repairs"][0]["status"], "repaired_task")
        self.assertEqual(len(graph["task_states"]), 1)
        self.assertEqual(graph["task_states"][0]["status"], "in_progress")
        self.assertEqual(records[0]["actions"][0]["object"], "схему")
        compatible = summary.compatibility_state(graph)
        self.assertTrue(compatible["events"])
        self.assertTrue(all(
            event["provenance"]["audio_sha256"] == "audio-hash"
            and event["provenance"]["source_word_ids"]
            for event in compatible["events"]
        ))

    def test_candidate_preflight_quarantines_unprovable_work_instead_of_failing_late(self):
        turn = {
            "id": "U2", "speaker": "@B", "start": 9.0,
            "text": "Может быть, этот материал пригодится.", "source_word_ids": ["W2"],
        }
        fact = {
            "fact_id": "F2", "type": "proposal",
            "statement": "Обсуждался материал.",
            "speaker_refs": ["@B"], "owner_refs": [],
            "evidence_ids": ["U2"], "evidence": [turn],
            "uncertainty": {"needs_review": False},
            "verification_status": "supported", "origin_id": "OR2",
        }
        stale_record = {
            **rec(2, "proposal", fact["statement"], "propose", speaker="@B"),
            "origin_id": "OR2", "actions": [], "dialogue_evidence": [turn],
            "evidence_ids": ["U2"], "source_word_ids": ["W2"],
        }
        records, graph, report = preflight_candidate_lineage(
            [{
                "origin_id": "OR2:A01", "revision_id": "RV2",
                "fact_id": "F2", "kind": "action", "statement": fact["statement"],
                "evidence_ids": ["U2"], "source_word_ids": ["W2"],
            }],
            [stale_record], [fact], [turn], [],
            provenance={"audio_sha256": "audio-hash"},
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["dispositions"][0]["status"], "requires_verification")
        self.assertEqual(graph["task_states"], [])
        self.assertEqual(records[0]["verification_status"], "insufficient_evidence")
        self.assertIn("candidate_lineage_requires_review", records[0]["semantic_risks"])
        compatible = summary.compatibility_state(graph)
        self.assertTrue(all(
            event["provenance"]["audio_sha256"] == "audio-hash"
            and event["provenance"]["source_word_ids"]
            for event in compatible["events"]
        ))

    def test_output_limit_fallback_routes_first_person_work_onset_to_task_state(self):
        fact = {
            "fact_id": "F-onset", "type": "proposal",
            "statement": "Обсуждалась панель, после чего участник начал её собирать.",
            "speaker_refs": ["@A"], "owner_refs": [], "evidence_ids": ["U1"],
            "evidence": [{
                "id": "U1", "speaker": "@A", "start": 12.0,
                "text": "Я панель начал собирать.", "source_word_ids": ["W1"],
            }],
            "uncertainty": {"needs_review": False},
            "verification_status": "supported", "origin_id": "OR-onset",
        }
        record = normalize_semantic_record({}, fact)
        graph = build_meeting_graph([record])
        self.assertEqual(len(graph["task_states"]), 1)
        self.assertEqual(graph["task_states"][0]["assignee"], "@A")
        self.assertEqual(graph["task_states"][0]["status"], "in_progress")
        self.assertEqual(graph["task_states"][0]["origin_ids"], ["OR-onset:A01"])

    def test_single_action_expansion_keeps_reviewed_public_statement(self):
        record = {
            **rec(1, "action", "@Yachoy подготовит TradingView или отправит EXE-файл в ближайшее время.", "commit", speaker="@Yachoy"),
            "actions": [{
                "predicate": "подготовлю", "object": "TradingView, собственно, работы. Чё даю EXE-файл",
                "actor": "@Yachoy", "temporal_state": "planned",
                "commitment_state": "explicit_commitment", "evidence_ids": ["U1"],
            }],
        }
        expanded = _expand_action_records([record])
        self.assertEqual(len(expanded), 1)
        self.assertEqual(expanded[0]["statement"], record["statement"])
        self.assertNotIn("собственно", expanded[0]["statement"])

    def test_deictic_methodology_task_inherits_adjacent_antecedent(self):
        record = {
            **rec(2, "action", "Потом встрою это в методичку.", "commit"),
            "actions": [{
                "predicate": "встрою", "object": "это в методичку", "actor": "@A",
                "temporal_state": "planned", "commitment_state": "explicit_commitment",
                "evidence_ids": ["U2"],
            }],
            "dialogue_evidence": [
                {"id": "U1", "speaker": "@A", "start": 1, "text": "Это все типы имбалансов, их четыре.", "source_word_ids": ["W1"]},
                {"id": "U2", "speaker": "@A", "start": 2, "text": "Ну, не типы, а варианты. Потом встрою это в методичку.", "source_word_ids": ["W2"]},
            ],
        }
        expanded = _expand_action_records([record])[0]
        self.assertEqual(expanded["evidence_ids"], ["U1", "U2"])
        self.assertEqual(expanded["reference_resolution"]["antecedent_evidence_ids"], ["U1"])
        self.assertEqual(set(expanded["source_word_ids"]), {"W1", "W2"})

    def test_task_surfaces_resolve_real_recording_fragments(self):
        cases = [
            (
                {"statement": "Потом встрою это в методичку.", "dialogue_evidence": [
                    {"text": "Это все типы имбалансов, их четыре."},
                    {"text": "Ну, не типы, а варианты. Потом встрою это в методичку."},
                ]},
                {"deliverable": "Потом встрою это в методичку."},
                "Встроить в методичку: варианты имбалансов; их четыре",
            ),
            (
                {"statement": "@Yachoy подготовит TradingView или отправит EXE-файл.", "dialogue_evidence": [
                    {"text": "Я подготовлю TradingView, собственно, работы. Чё даю EXE-файл."},
                ]},
                {"assignee": "@Yachoy", "deliverable": "@Yachoy подготовит TradingView или отправит EXE-файл."},
                "Подготовить TradingView или передать EXE-файл",
            ),
            (
                {"statement": "Я Bitcoin 2021 года тебе дам.", "dialogue_evidence": [
                    {"text": "Я Bitcoin 2021 года тебе дам."}, {"text": "Там достаточно месяца."},
                ]},
                {"deliverable": "Предоставлю данные Bitcoin 2021 года, достаточно месяца."},
                "Предоставить данные Bitcoin; период источника — 2021 год, объём выборки — 1 месяц",
            ),
        ]
        for claim, state, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(task_surface_text(claim, state), expected)

    def test_second_person_marker_and_imbalance_experiment_remain_separate_tasks(self):
        claim = {
            "statement": "Мне сделать тебе разметчик Order Block и пойти экспериментировать с имбалансами?",
            "dialogue_evidence": [{"text": "Мне сделать тебе разметчик Order Block?"}],
        }
        marker = task_surface_text(claim, {
            "deliverable": "Сделать тебе разметчик Order Block",
            "action_frame": {"explicit_acceptance_actor": "@HoTTaBbicH"},
        })
        experiment = task_surface_text(claim, {"deliverable": "Пойти экспериментировать с имбалансами"})
        self.assertEqual(marker, "Сделать разметчик Order Block для @HoTTaBbicH")
        self.assertEqual(experiment, "Экспериментировать с имбалансами")

    def test_task_surface_rules_generalize_to_unrelated_entities_and_artifacts(self):
        documentation = task_surface_text({
            "statement": "Затем внесу это в wiki.",
            "dialogue_evidence": [
                {"text": "Есть категории уведомлений, их три."},
                {"text": "Не категории, а формы. Затем внесу это в wiki."},
            ],
        }, {"deliverable": "Затем внесу это в wiki."})
        resource = task_surface_text({
            "statement": "Я передам данные.",
            "dialogue_evidence": [
                {"text": "Логи 2024 года передам."},
                {"text": "Для проверки достаточно недели."},
            ],
        }, {"deliverable": "Предоставлю данные за неделю."})
        recipient = task_surface_text({"statement": "Мне сделать тебе прототип отчёта?", "dialogue_evidence": []}, {
            "deliverable": "Сделать А, ну то есть тебе прототип отчёта",
            "action_frame": {"explicit_acceptance_actor": "@Reviewer"},
        })
        self.assertEqual(documentation, "Внести в wiki: формы уведомлений; их три")
        self.assertEqual(resource, "Предоставить данные Логи; период источника — 2024 год, объём выборки — 1 неделя")
        self.assertEqual(recipient, "Сделать прототип отчёта для @Reviewer")

    def test_deontic_reference_resolution_is_domain_independent(self):
        result = task_surface_text({
            "statement": "Категории уведомлений нужно будет внести в wiki.",
            "dialogue_evidence": [
                {"text": "Есть категории уведомлений, их три."},
                {"text": "Не категории, а формы. Их нужно будет внести в wiki."},
            ],
        }, {"deliverable": "Категории уведомлений нужно будет внести в wiki."})
        self.assertEqual(result, "Внести в wiki: формы уведомлений; их три")

    def test_reference_resolution_survives_graph_and_every_public_task_projection(self):
        record = {
            **rec(20, "action", "Категории уведомлений нужно будет внести в wiki.", "commit"),
            "commitment_strength": "explicit", "commitment_actor": "@A",
            "assignees": ["@A"], "assignment_status": "confirmed",
            "actions": [{
                "predicate": "внести", "object": "это в wiki", "actor": "@A",
                "temporal_state": "planned", "commitment_state": "explicit_commitment",
                "evidence_ids": ["U20"],
                "field_evidence": {
                    "actor": ["U20"], "predicate": ["U20"],
                    "object": ["U20"], "recipient": [],
                },
            }],
            "dialogue_evidence": [
                {"id": "U19", "speaker": "@A", "start": 19, "end": 19.5,
                 "text": "Есть категории уведомлений, их три.", "source_word_ids": ["W19"]},
                {"id": "U20", "speaker": "@A", "start": 20, "end": 21,
                 "text": "Не категории, а формы. Их нужно будет внести в wiki.", "source_word_ids": ["W20"]},
            ],
        }
        graph = build_meeting_graph([record])
        claim_id = graph["claims"][0]["claim_id"]
        summary_plan = {
            "view_plans": {
                view: {"selected_claim_ids": [claim_id]}
                for view in ("executive", "tasks", "minutes")
            },
            "public_sentence_plans": [],
        }
        items = build_public_items(graph, summary_plan)
        projected = [item for item in items if item.get("task_state_id")]
        self.assertEqual({item["section"] for item in projected}, {"overview", "tasks", "minutes"})
        self.assertTrue(all("их три" in item["text"] for item in projected))
        self.assertTrue(all(item["evidence_ids"] == ["U19", "U20"] for item in projected))

    def test_post_render_verifier_accepts_task_projection_in_every_view(self):
        claim = {
            "claim_id": "C1", "statement": "@Misha сделать разметчик тебе Order Block",
            "evidence_ids": ["U1", "U2"], "speaker_refs": ["@Misha"],
            "canonical_task_state_id": "T1", "lifecycle": "active",
            "dialogue_evidence": [
                {"id": "U1", "speaker": "@Misha", "text": "Мне сделать тебе разметчик Order Block?"},
                {"id": "U2", "speaker": "@Reviewer", "text": "Да, сделай."},
            ],
        }
        state = {
            "task_id": "T1", "assignee": "@Misha",
            "description": "Сделать тебе разметчик Order Block",
            "deliverable": "Сделать тебе разметчик Order Block",
            "status": "accepted", "evidence_ids": ["U1", "U2"],
            "acceptance_evidence_ids": ["U2"],
            "action_frame": {"explicit_acceptance_actor": "@Reviewer"},
            "field_support": {"actor": ["U1"], "predicate": ["U1"], "object": ["U1"]},
        }
        text = "Сделать разметчик Order Block для @Reviewer"
        plan = {
            "claim_ids": ["C1"], "relation_ids": [], "allowed_numbers": [],
            "allowed_relation_markers": [], "allowed_speakers": ["@Misha"],
            "allowed_assignees": ["@Misha"], "polarity": [], "modality": [],
            "conditions": [], "time_scope": [],
        }
        for section in ("overview", "tasks", "minutes"):
            suffix = " — исполнитель: @Misha — статус: согласовано" if section == "tasks" else ""
            item = {
                "section": section, "text": text + suffix, "claim_ids": ["C1"],
                "evidence_ids": ["U1", "U2"], "task_state_id": "T1",
                "task_state": state, "social_state": "accepted",
            }
            result = verify_generated_items([item], [plan], [claim])
            self.assertTrue(result["passed"], (section, result))

    def test_structured_resource_field_labels_do_not_create_added_clause(self):
        text = (
            "Предоставить данные Bitcoin; период источника — 2021 год, "
            "объём выборки — 1 месяц — исполнитель: @Owner — "
            "статус: участник взял на себя"
        )
        claim = {
            "claim_id": "C1", "statement": "Готов предоставить данные Bitcoin.",
            "evidence_ids": ["U1", "U2"], "speaker_refs": ["@Owner"],
            "canonical_task_state_id": "T1", "lifecycle": "active",
            "dialogue_evidence": [
                {"id": "U1", "text": "Я Bitcoin 2021 года тебе дам."},
                {"id": "U2", "text": "Там достаточно 1 месяца."},
            ],
        }
        state = {
            "task_id": "T1", "assignee": "@Owner", "status": "self_committed",
            "description": "Предоставить данные Bitcoin",
            "deliverable": "Предоставить данные Bitcoin",
            "current_scope": "1 месяц", "data_origin": "2021 год",
            "evidence_ids": ["U1", "U2"], "action_frame": {},
        }
        plan = {
            "claim_ids": ["C1"], "relation_ids": [],
            "allowed_numbers": ["2021", "1"], "allowed_relation_markers": [],
            "allowed_speakers": ["@Owner"], "allowed_assignees": ["@Owner"],
            "polarity": [], "modality": [], "conditions": [],
            "time_scope": ["1 месяц"],
        }
        item = {
            "section": "tasks", "text": text, "claim_ids": ["C1"],
            "evidence_ids": ["U1", "U2"], "task_state_id": "T1",
            "task_state": state, "social_state": "self_committed",
        }
        result = verify_generated_items([item], [plan], [claim])
        self.assertTrue(result["passed"], result)

    def test_protected_task_abstention_cannot_be_silently_filtered(self):
        items = [
            {"section": "tasks", "social_state": "accepted", "text": "Подтверждённая задача"},
            {"section": "technical", "social_state": "observation", "text": "Слабая деталь"},
        ]
        report = {"audits": [
            {"passed": False, "errors": ["unsupported_added_clause"]},
            {"passed": False, "errors": ["unsupported_added_clause"]},
        ]}
        protected = protected_public_item_abstentions(items, report)
        self.assertEqual(len(protected), 1)
        self.assertEqual(protected[0]["public_item"]["section"], "tasks")

    def test_task_recipient_requires_structured_evidence(self):
        claim = {
            "claim_id": "C1", "statement": "Мне сделать тебе прототип отчёта?",
            "evidence_ids": ["U1", "U2"], "speaker_refs": ["@Author"],
            "canonical_task_state_id": "T1", "lifecycle": "active",
            "dialogue_evidence": [
                {"id": "U1", "speaker": "@Author", "text": "Мне сделать тебе прототип отчёта?"},
                {"id": "U2", "speaker": "@Reviewer", "text": "Да, сделай."},
            ],
        }
        state = {
            "assignee": "@Author", "description": "Сделать тебе прототип отчёта",
            "evidence_ids": ["U1", "U2"], "acceptance_evidence_ids": ["U2"],
            "action_frame": {"explicit_acceptance_actor": "@Reviewer"},
        }
        item = {
            "section": "tasks", "text": "Сделать прототип отчёта для @Reviewer",
            "claim_ids": ["C1"], "evidence_ids": ["U1", "U2"],
            "task_state_id": "T1", "task_state": state,
        }
        sentence_plan = {
            "claim_ids": ["C1"], "relation_ids": [], "allowed_numbers": [],
            "allowed_relation_markers": [], "allowed_speakers": ["@Author"],
            "allowed_assignees": ["@Author"], "polarity": [], "modality": [],
            "conditions": [], "time_scope": [],
        }
        accepted = verify_generated_items([item], [sentence_plan], [claim])
        self.assertTrue(accepted["passed"], accepted)
        unsupported = verify_generated_items([{**item, "text": item["text"].replace("@Reviewer", "@Other")}], [sentence_plan], [claim])
        self.assertFalse(unsupported["passed"])
        self.assertIn("speaker_or_assignee_not_preserved", unsupported["audits"][0]["errors"])

    def test_ambiguous_clock_preserves_exact_source_wording(self):
        fact = {
            "fact_id": "F-time", "type": "system_rule",
            "statement": "Операция закрывается автоматически после 00:00.",
            "speaker_refs": ["@A"], "owner_refs": [], "evidence_ids": ["U1"],
            "source_word_ids": ["W1"], "evidence": [{"id": "U1", "speaker": "@A", "text": "Она закроется сама после 00."}],
            "uncertainty": {"needs_review": False}, "verification_status": "supported",
        }
        record = normalize_semantic_record({"time_expression": {"raw_text": "после 00:00"}}, fact)
        self.assertEqual(record["time_contract"]["raw"], "после 00")
        self.assertEqual(record["time_contract"]["resolution_status"], "ambiguous_clock")
        self.assertFalse(record["time_contract"]["execution_safe"])
        self.assertEqual(public_surface_text(record), "Операция закрывается автоматически после 00.")

    def test_incomplete_source_fragment_abstains_before_document_render(self):
        text = "То есть, если не получится реализовать, э-э-э..."
        claim = {
            "claim_id": "C1", "statement": text, "evidence_ids": ["U1"],
            "lifecycle": "active", "polarity": "negative",
            "dialogue_evidence": [{"id": "U1", "text": text}],
        }
        plan = {
            "claim_ids": ["C1"], "relation_ids": [], "allowed_numbers": [],
            "allowed_relation_markers": [], "allowed_speakers": [],
            "allowed_assignees": [], "polarity": ["negative"], "modality": [],
            "conditions": [], "time_scope": [],
        }
        item = {"section": "overview", "text": text, "claim_ids": ["C1"], "evidence_ids": ["U1"]}
        result = verify_generated_items([item], [plan], [claim])
        self.assertFalse(result["passed"])
        self.assertIn("incomplete_public_fragment", result["audits"][0]["errors"])

    def test_preverbal_object_and_parallel_modifier_are_recovered_generically(self):
        clauses = source_action_clauses({
            "evidence": [{
                "id": "U1", "speaker": "@A",
                "text": "Контрольный набор тогда буду размечать, и параллельно мы его встроим.",
            }],
        }, ["@A"])
        self.assertEqual(len(clauses), 1)
        self.assertEqual(clauses[0]["object"], "Контрольный набор")
        self.assertTrue(clauses[0]["parallel"])

    def test_proposal_cannot_supply_its_own_acceptance_evidence(self):
        base = {
            "lifecycle": "active", "decision_status": "accepted",
            "content_kind": "proposal", "speech_act": "propose",
            "evidence_ids": ["U1"], "decision_evidence_ids": ["U1"],
            "acceptance_check": "entailed", "acceptance_relation_ids": ["R1"],
            "acceptance_evidence_ids": ["U1"], "accepted_by": "@A",
            "dialogue_evidence": [{"id": "U1", "text": "Предлагаю провести дополнительный бэктест."}],
        }
        self.assertFalse(can_publish_as_decision(base))
        accepted = dict(base, evidence_ids=["U1", "U2"], acceptance_evidence_ids=["U2"],
                        dialogue_evidence=base["dialogue_evidence"] + [{"id": "U2", "text": "Согласен, делаем."}])
        self.assertTrue(can_publish_as_decision(accepted))

    def test_unrelated_yes_preface_cannot_accept_an_earlier_proposal(self):
        self.assertFalse(is_explicit_acceptance_reply(
            "Да, Анна, прогноз на неделю оказался неточным.",
            "Предлагаю провести аудит доступа.",
        ))
        self.assertFalse(is_explicit_acceptance_reply(
            "Да, показатели росли и начали взаимодействовать между собой, поэтому я сменил модель.",
            "Предлагаю провести дополнительную проверку.",
        ))
        self.assertTrue(is_explicit_acceptance_reply(
            "Да-да. Дальше этап апробации.",
            "Предлагаю подготовить прототип.",
        ))

    def test_late_technical_pass_recovers_full_path_correction_and_objection(self):
        turns = [
            {"id": "U1", "start": 100, "end": 101, "speaker": "@A", "text": "Берём движение внутри четырёхчасового диапазона.", "flags": [], "source_word_ids": ["W1"]},
            {"id": "U2", "start": 110, "end": 111, "speaker": "@A", "text": "На M15 ждём слом структуры и отмечаем зону.", "flags": [], "source_word_ids": ["W2"]},
            {"id": "U3", "start": 120, "end": 121, "speaker": "@A", "text": "На M1 ищем модель, не дожидаясь точки старшего таймфрейма.", "flags": [], "source_word_ids": ["W3"]},
            {"id": "U4", "start": 130, "end": 131, "speaker": "@A", "text": "Точка показывается раньше пересечения линии.", "flags": [], "source_word_ids": ["W4"]},
            {"id": "U5", "start": 132, "end": 133, "speaker": "@A", "text": "А нет, подожди.", "flags": [], "source_word_ids": ["W5"]},
            {"id": "U6", "start": 134, "end": 135, "speaker": "@A", "text": "Когда линия пересекается, тогда точка отрисуется на графике.", "flags": [], "source_word_ids": ["W6"]},
            {"id": "U7", "start": 140, "end": 141, "speaker": "@B", "text": "Так не получится: возникает круговая зависимость старшего и младшего таймфрейма.", "flags": [], "source_word_ids": ["W7"]},
        ]
        facts, report = ensure_late_technical_outcomes([], turns, window_seconds=300)
        text = " ".join(item["statement"] for item in facts)
        self.assertRegex(text, r"четырёхчас\w*.*M15.*M1")
        self.assertIn("Когда линия пересекается", text)
        self.assertIn("круговая зависимость", text)
        self.assertEqual({item["kind"] for item in report["candidates"]}, {
            "multi_stage_path", "explicit_correction", "late_counterargument",
        })
        self.assertTrue(all(item.get("protected_outcome") for item in facts))

    def test_late_path_recovery_is_not_tied_to_specific_timeframes(self):
        turns = [
            {"id": "U1", "start": 1, "end": 2, "speaker": "@A", "text": "На H1 фиксируем исходное состояние.", "flags": [], "source_word_ids": ["W1"]},
            {"id": "U2", "start": 5, "end": 6, "speaker": "@A", "text": "После этого на M5 проверяем условие.", "flags": [], "source_word_ids": ["W2"]},
            {"id": "U3", "start": 9, "end": 10, "speaker": "@A", "text": "Затем на M2 выполняем действие.", "flags": [], "source_word_ids": ["W3"]},
            {"id": "U4", "start": 12, "end": 13, "speaker": "@B", "text": "Так не сработает: условие конфликтует с исходным состоянием.", "flags": [], "source_word_ids": ["W4"]},
        ]
        facts, report = ensure_late_technical_outcomes([], turns, window_seconds=60)
        text = " ".join(item["statement"] for item in facts)
        self.assertRegex(text, r"H1.*M5.*M2")
        self.assertIn("не сработает", text)
        self.assertIn("multi_stage_path", {item["kind"] for item in report["candidates"]})

    def test_incomplete_objection_cannot_steal_the_protected_counterargument(self):
        turns = [
            {"id": "U1", "start": 1, "end": 2, "speaker": "@A", "text": "На H2 фиксируем структуру.", "flags": [], "source_word_ids": ["W1"]},
            {"id": "U2", "start": 5, "end": 6, "speaker": "@A", "text": "После этого на M10 проверяем слом структуры.", "flags": [], "source_word_ids": ["W2"]},
            {"id": "U3", "start": 9, "end": 10, "speaker": "@A", "text": "Затем на M3 пытаемся реализовать вход.", "flags": [], "source_word_ids": ["W3"]},
            {"id": "U4", "start": 11, "end": 12, "speaker": "@B", "text": "То есть, если не получится реализовать, э-э-э...", "flags": [], "source_word_ids": ["W4"]},
            {"id": "U5", "start": 14, "end": 15, "speaker": "@B", "text": "Так не получится: для определения структуры уже нужен слом, поэтому получается замкнутый круг.", "flags": [], "source_word_ids": ["W5"]},
        ]
        facts, report = ensure_late_technical_outcomes([], turns, window_seconds=60)
        constraint = next(item for item in facts if item["type"] == "constraint")
        self.assertIn("замкнутый круг", constraint["statement"])
        self.assertNotIn("э-э-э", constraint["statement"])
        counter = next(item for item in report["candidates"] if item["kind"] == "late_counterargument")
        self.assertEqual(counter["evidence_ids"], ["U5"])

    def test_unsupported_model_rewrite_keeps_checked_protected_original(self):
        turns = [
            {"id": "U1", "start": 1, "end": 2, "speaker": "@A", "text": "На H2 фиксируем исходный диапазон.", "flags": [], "source_word_ids": ["W1"]},
            {"id": "U2", "start": 5, "end": 6, "speaker": "@A", "text": "После этого на M10 подтверждаем слом.", "flags": [], "source_word_ids": ["W2"]},
            {"id": "U3", "start": 9, "end": 10, "speaker": "@A", "text": "Затем на M3 ищем точку входа.", "flags": [], "source_word_ids": ["W3"]},
        ]
        facts, _report = ensure_late_technical_outcomes([], turns, window_seconds=60)
        original = next(item for item in facts if item["type"] == "design_choice")
        accepted, rejected = summary.apply_reviews([original], [{
            "fact_id": original["fact_id"], "verdict": "corrected",
            "type": "design_choice", "statement": "Путь H9 → M10 → M3 подтверждён.",
            "evidence_ids": original["evidence_ids"], "confidence": .8,
        }], enforce_policy=False)
        self.assertFalse(rejected)
        self.assertEqual(accepted[0]["statement"], original["statement"])
        self.assertEqual(accepted[0]["validation"], "protected_original")
        self.assertEqual(accepted[0]["review_patch"]["reason"], "unsupported_model_correction_fallback")

    def test_coherent_late_path_beats_an_earlier_compact_duration_list(self):
        turns = [
            {"id": "U1", "start": 1, "end": 2, "speaker": "@A", "text": "Сравнивали H4, H8 и M20 как варианты длительности.", "flags": [], "source_word_ids": ["W1"]},
            {"id": "U2", "start": 100, "end": 101, "speaker": "@B", "text": "Сначала на H2 фиксируем исходный диапазон.", "flags": [], "source_word_ids": ["W2"]},
            {"id": "U3", "start": 110, "end": 111, "speaker": "@B", "text": "После этого на M10 подтверждаем слом.", "flags": [], "source_word_ids": ["W3"]},
            {"id": "U4", "start": 120, "end": 121, "speaker": "@B", "text": "Дальше на M3 ищем точку входа.", "flags": [], "source_word_ids": ["W4"]},
        ]
        facts, _report = ensure_late_technical_outcomes([], turns, window_seconds=300)
        path = next(item["statement"] for item in facts if item["type"] == "design_choice")
        self.assertRegex(path, r"H2.*M10.*M3")
        self.assertNotIn("H8", path)

    def test_scattered_existing_stage_mentions_do_not_suppress_the_relationship(self):
        existing = {
            "fact_id": "F-old", "type": "observation", "topic": "сравнение",
            "statement": "В разных частях обсуждения упомянуты H2, M10 и M3.",
            "evidence_ids": ["U0"], "speaker_refs": ["@A"], "certainty": "explicit",
            "start": 0.0, "end": .5, "source_chunks": [], "source_word_ids": ["W0"],
            "evidence": [{"id": "U0", "start": 0.0, "end": .5, "speaker": "@A",
                          "text": "H2, M10 и M3 упоминались отдельно.", "flags": [],
                          "uncertainty": {}, "source_word_ids": ["W0"]}],
            "uncertainty": {}, "semantic_risks": [], "risk_level": "LOW",
        }
        turns = [
            {"id": "U1", "start": 10, "end": 11, "speaker": "@A", "text": "Сначала на H2 фиксируем диапазон.", "flags": [], "source_word_ids": ["W1"]},
            {"id": "U2", "start": 20, "end": 21, "speaker": "@A", "text": "После этого на M10 проверяем условие.", "flags": [], "source_word_ids": ["W2"]},
            {"id": "U3", "start": 30, "end": 31, "speaker": "@A", "text": "Затем на M3 выполняем действие.", "flags": [], "source_word_ids": ["W3"]},
        ]
        facts, report = ensure_late_technical_outcomes([existing], turns, window_seconds=60)
        protected = [item for item in facts if item.get("protected_outcome")]
        self.assertEqual(len(protected), 1)
        self.assertRegex(protected[0]["statement"], r"H2.*M10.*M3")
        self.assertIn("multi_stage_path", {item["kind"] for item in report["candidates"]})

    def test_title_fragment_rejects_pronouns_and_dialogue_acknowledgements(self):
        self.assertTrue(_has_unresolved_title_reference("Пока что они подсвечиваются"))
        self.assertTrue(_has_unresolved_title_reference("Угу. Есть инструмент"))
        self.assertFalse(_has_unresolved_title_reference("Инструмент ТПО и проверка задержки"))

    def test_title_rejects_low_information_clause_and_surface_deduplicates_status(self):
        self.assertTrue(summary._title_surface_needs_repair(
            "Задержка старшего диапазона; Есть обилие квадратов в строку"
        ))
        cleaned = sanitize_public_surface(
            "Подготовить отчёт — статус: участник взял на себя "
            "(статус: участник взял на себя)"
        )
        self.assertEqual(cleaned, "Подготовить отчёт — статус: участник взял на себя")

    def test_outcome_card_does_not_append_an_existing_identical_task_status(self):
        graph = {
            "claims": [{
                "claim_id": "C1", "proposition_id": "P1",
                "content_kind": "action", "lifecycle": "active",
                "verification_status": "supported", "task_status": "self_committed",
                "temporal_state": "planned", "evidence_ids": ["U1"],
                "publication_text": (
                    "Подготовить отчёт — исполнитель: @A — "
                    "статус: участник взял на себя"
                ),
            }],
            "task_states": [{
                "task_id": "T1", "proposition_id": "P1",
                "source_proposition_ids": ["P1"], "status": "self_committed",
            }],
            "question_states": [], "relations": [],
            "dialogue_bundles": [{
                "bundle_id": "DB1", "topic": "Отчёт", "claim_ids": ["C1"],
                "ranges": [{"start": 1, "end": 2}],
            }],
        }
        card = build_outcome_cards(graph)[0]
        value = card["fields"]["next_step"]["value"]
        self.assertEqual(value.casefold().count("статус:"), 1)

    def test_asserted_action_frame_does_not_become_a_task(self):
        fact = {
            "fact_id": "F-state", "type": "action",
            "statement": "Система добавляет отметку на график.",
            "speaker_refs": ["@A"], "owner_refs": ["@A"],
            "evidence_ids": ["U1"], "source_word_ids": ["W1"],
            "evidence": [{"id": "U1", "speaker": "@A", "text": "Система добавляет отметку на график."}],
            "uncertainty": {"needs_review": False}, "verification_status": "supported",
        }
        record = normalize_semantic_record({"speech_act": "commit", "actions": [{
            "predicate": "добавляет", "actor": "@A", "temporal_state": "in_progress",
            "commitment_state": "unknown", "evidence_ids": ["U1"],
        }]}, fact)
        self.assertEqual(build_meeting_graph([record])["task_states"], [])

    def test_source_intent_to_try_an_extra_timeframe_is_retained(self):
        fact = {
            "fact_id": "F-timeframe", "type": "action",
            "statement": "Подключить другие таймфреймы и анализировать последние свинги.",
            "speaker_refs": ["@Yachoy"], "owner_refs": ["@Yachoy"],
            "evidence_ids": ["U1"], "source_word_ids": ["W1"],
            "evidence": [{"id": "U1", "speaker": "@Yachoy", "start": 1,
                          "text": "Я ещё попробую подключить другие таймфреймы, не только минутный."}],
            "uncertainty": {"needs_review": False}, "verification_status": "supported",
        }
        record = normalize_semantic_record({"speech_act": "answer"}, fact)
        states = build_meeting_graph([record])["task_states"]
        self.assertEqual(len(states), 1)
        self.assertEqual(states[0]["status"], "intent_to_attempt")
        self.assertEqual(states[0]["owner"], "@Yachoy")

    def test_one_acceptance_applies_to_both_actions_in_the_same_turn(self):
        fact = {
            "fact_id": "F-marker", "type": "action",
            "statement": "Создать разметчик Order Block и экспериментировать с имбалансами.",
            "speaker_refs": ["@Misha"], "owner_refs": ["@Misha"],
            "evidence_ids": ["U1", "U2"], "source_word_ids": ["W1"],
            "evidence": [
                {"id": "U1", "speaker": "@Misha", "start": 1,
                 "text": "Мне сделать тебе разметчик Order Block и пойти экспериментировать с имбалансами?"},
                {"id": "U2", "speaker": "@Recipient", "start": 2,
                 "text": "Да-да-да. Дальше этап апробации."},
            ],
            "uncertainty": {"needs_review": False}, "verification_status": "supported",
        }
        record = normalize_semantic_record({"actions": [{
            "predicate": "create_order_block_marker_tool", "actor": "@Misha",
            "temporal_state": "planned", "commitment_state": "unknown",
            "evidence_ids": ["U1", "U2"],
        }]}, fact)
        states = build_meeting_graph([record])["task_states"]
        self.assertEqual(len(states), 2)
        self.assertEqual({state["status"] for state in states}, {"accepted"})

    def test_context_record_cannot_steal_acknowledgement_from_direct_proposal(self):
        proposal_turn = {
            "id": "U2", "speaker": "@Misha", "start": 2,
            "text": "Мне сделать разметчик Order Block и пойти экспериментировать с имбалансами?",
        }
        acceptance_turn = {
            "id": "U3", "speaker": "@Recipient", "start": 3,
            "text": "Да-да-да. Дальше этап апробации.",
        }
        broad_context_record = {
            **rec(1, "observation", "Имбалансов пока достаточно.", "assert", speaker="@Misha"),
            "actions": [{
                "predicate": "обсуждать имбалансы", "actor": "@Misha",
                "temporal_state": "unknown", "commitment_state": "unknown",
                "evidence_ids": ["U1"],
            }],
            "dialogue_evidence": [
                {"id": "U1", "speaker": "@Misha", "start": 1, "text": "Имбалансов пока достаточно."},
                proposal_turn, acceptance_turn,
            ],
        }
        fact = {
            "fact_id": "F2", "type": "action",
            "statement": "Создать разметчик Order Block и экспериментировать с имбалансами.",
            "speaker_refs": ["@Misha"], "owner_refs": ["@Misha"],
            "evidence_ids": ["U2", "U3"], "source_word_ids": ["W2"],
            "evidence": [proposal_turn, acceptance_turn],
            "dialogue_evidence": [proposal_turn, acceptance_turn],
            "uncertainty": {"needs_review": False}, "verification_status": "supported",
        }
        proposal_record = normalize_semantic_record({"actions": [{
            "predicate": "create_order_block_marker_tool", "actor": "@Misha",
            "temporal_state": "planned", "commitment_state": "unknown",
            "evidence_ids": ["U2"],
        }]}, fact)
        states = [
            state for state in build_meeting_graph([broad_context_record, proposal_record])["task_states"]
            if state.get("source_record_id") == "F2"
        ]
        self.assertEqual(len(states), 2)
        self.assertEqual({state["status"] for state in states}, {"accepted"})

    def test_same_timestamp_fact_cannot_steal_compound_proposal_acceptance(self):
        proposal_turn = {
            "id": "U1", "speaker": "@Misha", "start": 1,
            "text": "Мне сделать разметчик Order Block и пойти экспериментировать с имбалансами?",
        }
        acceptance_turn = {
            "id": "U2", "speaker": "@Recipient", "start": 2,
            "text": "Да-да-да. Дальше этап апробации.",
        }
        fact = {
            "fact_id": "F-work", "type": "proposal",
            "statement": "Создать разметчик Order Block и экспериментировать с имбалансами.",
            "speaker_refs": ["@Misha"], "owner_refs": ["@Misha"],
            "evidence_ids": ["U1"], "source_word_ids": ["W1"],
            "evidence": [proposal_turn], "dialogue_evidence": [proposal_turn, acceptance_turn],
            "uncertainty": {"needs_review": False}, "verification_status": "supported",
        }
        proposal = normalize_semantic_record({"speech_act": "propose", "actions": [
            {"predicate": "создать", "object": "разметчик Order Block", "actor": "@Misha",
             "temporal_state": "planned", "commitment_state": "unknown", "evidence_ids": ["U1"]},
            {"predicate": "экспериментировать", "object": "с имбалансами", "actor": "@Misha",
             "temporal_state": "planned", "commitment_state": "unknown", "evidence_ids": ["U1"]},
        ]}, fact)
        competing = rec(2, "proposal", "Следующий этап работы — апробация.", "propose", speaker="@Recipient")
        competing.update({"evidence_ids": ["U2"], "start": 2.0})
        states = [
            state for state in build_meeting_graph([proposal, competing])["task_states"]
            if state.get("source_record_id") == "F-work"
        ]
        self.assertEqual(len(states), 2)
        self.assertEqual({state["status"] for state in states}, {"accepted"})
        self.assertTrue(all(state["acceptance_evidence_ids"] == ["U2"] for state in states))

    def test_equivalent_quarantine_items_merge_without_losing_provenance(self):
        graph = build_meeting_graph([
            rec(1, "proposal", "Проверить один и тот же подход", "propose",
                verification_status="insufficient_evidence", episode_id="E1"),
            rec(2, "action", "Проверить один и тот же подход", "propose",
                verification_status="insufficient_evidence", episode_id="E1"),
        ])
        claim_ids = [claim["claim_id"] for claim in graph["claims"]]
        summary_plan = {
            "view_plans": {"requires_verification": {"selected_claim_ids": claim_ids, "dispositions": {}}},
            "public_sentence_plans": [],
        }
        items = build_public_items(graph, summary_plan)
        quarantined = [item for item in items if item["section"] == "requires_verification"]
        self.assertEqual(len(quarantined), 1)
        self.assertEqual(set(quarantined[0]["claim_ids"]), set(claim_ids))

    def test_equivalent_accepted_proposal_and_action_merge_across_kinds(self):
        graph = {
            "claims": [
                {
                    "claim_id": "C-proposal", "content_kind": "proposal",
                    "statement": "Предлагается провести дополнительный бэктест для анализа причин проигрышных сделок.",
                    "evidence_ids": ["U1"], "source_word_ids": ["W1"],
                    "verification_status": "supported", "lifecycle": "active",
                    "social_state": "accepted", "episode_id": "E1",
                    "proposition_id": "P-proposal",
                    "start": 1.0, "end": 2.0,
                },
                {
                    "claim_id": "C-action", "content_kind": "action",
                    "statement": "@HoTTaBbicH провести дополнительный бэктест для анализа причин проигрышных сделок",
                    "evidence_ids": ["U1"], "source_word_ids": ["W1"],
                    "verification_status": "supported", "lifecycle": "active",
                    "social_state": "accepted", "episode_id": "E1",
                    "proposition_id": "P-action",
                    "start": 1.0, "end": 2.0,
                },
            ],
            "task_states": [], "question_states": [], "relations": [],
        }
        summary_plan = {
            "view_plans": {
                "minutes": {
                    "selected_claim_ids": ["C-proposal", "C-action"],
                    "dispositions": {},
                },
            },
            "public_sentence_plans": [],
        }
        minutes = [
            item for item in build_public_items(graph, summary_plan)
            if item["section"] == "minutes"
        ]
        self.assertEqual(len(minutes), 1)
        self.assertEqual(set(minutes[0]["claim_ids"]), {"C-proposal", "C-action"})

    def test_multi_episode_document_title_uses_two_supported_claims(self):
        items = []
        for index, (kind, section, text) in enumerate([
            ("constraint", "technical", "Модель не показывает устойчивый результат"),
            ("action", "tasks", "@A подготовит проверочный файл"),
            ("observation", "minutes", "Проверка выполнена на истории"),
            ("hypothesis", "experiments", "Сравнить результат на другом периоде"),
        ], 1):
            items.append({
                "public_id": f"PI{index}", "section": section, "text": text,
                "claim_ids": [f"C{index}"], "evidence_ids": [f"U{index}"],
                "source_word_ids": [f"W{index}"], "content_kind": kind,
                "social_state": "self_committed" if section == "tasks" else "candidate",
                "lifecycle": "active", "episode_id": f"E{index}", "start": index,
                "verification_status": "supported", "task_state": {
                    "status": "self_committed", "commitment_strength": "explicit",
                    "deliverable": text, "evidence_ids": [f"U{index}"],
                } if section == "tasks" else {},
            })
        document = build_public_document(items)
        self.assertGreaterEqual(len(document["title"]["claim_ids"]), 2)
        self.assertIn(";", document["title"]["text"])
        self.assertNotIn("@A", document["title"]["text"])

    def test_document_title_rejects_fragment_with_unresolved_pronoun(self):
        items = []
        for index, (kind, section, text) in enumerate([
            ("constraint", "technical", "Разметка имеет задержку в две свечи"),
            ("observation", "experiments", "Определить теоретические сроки его появления"),
            ("experimental_result", "experiments", "Имбалансы подсвечиваются после закрытия свечи"),
        ], 1):
            items.append({
                "public_id": f"PI{index}", "section": section, "text": text,
                "claim_ids": [f"C{index}"], "evidence_ids": [f"U{index}"],
                "source_word_ids": [f"W{index}"], "content_kind": kind,
                "social_state": "asserted", "lifecycle": "active",
                "episode_id": f"E{index}", "start": index,
                "verification_status": "supported", "task_state": {},
            })
        title = build_public_document(items)["title"]
        self.assertNotIn("его появления", title["text"].casefold())
        self.assertEqual(set(title["claim_ids"]), {"C1", "C3"})

    def test_title_repair_uses_supported_themes_not_raw_tasks(self):
        items = []
        for index, (kind, section, text) in enumerate([
            ("constraint", "technical", "Разметка имеет задержку в две свечи"),
            ("observation", "overview", "Имбалансы подсвечиваются на постобработке"),
            ("action", "tasks", "@Misha сделать разметчик Order Block"),
        ], 1):
            items.append({
                "public_id": f"PI{index}", "section": section, "text": text,
                "claim_ids": [f"C{index}"], "evidence_ids": [f"U{index}"],
                "source_word_ids": [f"W{index}"], "content_kind": kind,
                "social_state": "accepted" if section == "tasks" else "asserted",
                "lifecycle": "active", "episode_id": f"E{index}", "start": index,
                "verification_status": "supported", "task_state": {
                    "status": "accepted", "commitment_strength": "implicit",
                    "deliverable": text, "evidence_ids": [f"U{index}"],
                } if section == "tasks" else {},
            })
        document = build_public_document(items)
        nodes = _document_audit_nodes(document)
        audit = {
            "status": "failed", "total_nodes": len(nodes),
            "reviews": [
                {"node_id": node["node_id"], "verdict": "insufficient_evidence" if node["node_id"] == "title" else "supported"}
                for node in nodes
            ],
        }
        candidates = _title_repair_candidates(document, audit, items)
        self.assertTrue(candidates)
        self.assertNotIn("@Misha", candidates[0]["text"])
        self.assertEqual(set(candidates[0]["claim_ids"]), {"C1", "C2"})
        repaired, reconciled = reconcile_final_document_audit(document, audit, items)
        self.assertEqual(repaired["title"], candidates[0])
        self.assertEqual(reconciled["unresolved_node_ids"], ["title"])

    def test_compact_stage_path_title_stays_inside_source_lexical_closure(self):
        items = [
            {
                "public_id": "PI1", "section": "technical",
                "text": "Модель входа отображается некорректно",
                "claim_ids": ["C1"], "evidence_ids": ["U1"],
                "source_word_ids": ["W1"], "content_kind": "problem",
                "social_state": "asserted", "lifecycle": "active",
                "episode_id": "E1", "start": 1,
                "verification_status": "supported", "task_state": {},
            },
            {
                "public_id": "PI2", "section": "technical",
                "text": (
                    "Обсуждена последовательность по таймфреймам H4 → M15 → M1: "
                    "на H4 фиксируется структура, на M15 подтверждается слом, "
                    "на M1 выбирается вход."
                ),
                "claim_ids": ["C2"], "evidence_ids": ["U2"],
                "source_word_ids": ["W2"], "content_kind": "design_choice",
                "social_state": "asserted", "lifecycle": "active",
                "episode_id": "E2", "start": 2,
                "verification_status": "supported", "task_state": {},
            },
        ]
        document = build_public_document(items)
        nodes = _document_audit_nodes(document)
        audit = {
            "status": "failed",
            "reviews": [
                {"node_id": node["node_id"], "verdict": "supported"}
                for node in nodes if node["node_id"] != "title"
            ],
        }
        candidate = _title_repair_candidates(document, audit, items)[0]
        self.assertIn("Последовательность H4 → M15 → M1", candidate["text"])
        document["title"] = candidate
        errors = verify_public_document(
            document, render_public_document(document), items,
        )["errors"]
        self.assertNotIn("title_semantic_drift", errors)

    def test_capability_idea_is_not_promoted_to_current_task(self):
        fact = {
            "fact_id": "F-capability", "type": "action",
            "statement": "Добавить дополнительные инструменты.",
            "speaker_refs": ["@Yachoy"], "owner_refs": ["@Yachoy"],
            "evidence_ids": ["U1"], "source_word_ids": ["W1"],
            "evidence": [{
                "id": "U1", "speaker": "@Yachoy", "start": 1,
                "text": "Я думаю, что могу добавить дополнительные инструменты.",
            }],
            "uncertainty": {"needs_review": False}, "verification_status": "supported",
        }
        record = normalize_semantic_record({"speech_act": "commit", "actions": [{
            "predicate": "добавить", "actor": "@Yachoy",
            "temporal_state": "in_progress", "commitment_state": "explicit_commitment",
            "evidence_ids": ["U1"],
        }]}, fact)
        self.assertNotEqual(record["speech_act"], "commit")
        self.assertEqual(build_meeting_graph([record])["task_states"], [])

    def test_bounded_writer_uses_exact_public_item_evidence_closure(self):
        items = [
            {"public_id": "PI1", "section": "technical", "text": "Остаётся задержка.",
             "claim_ids": ["C1"], "evidence_ids": ["U1"], "episode_id": "E1",
             "content_kind": "problem", "social_state": "asserted"},
            {"public_id": "PI2", "section": "technical", "text": "Проверяется фильтр.",
             "claim_ids": ["C2"], "evidence_ids": ["U2"], "episode_id": "E2",
             "content_kind": "observation", "social_state": "asserted"},
        ]
        document = {
            "title": {"text": "Остаётся задержка", "claim_ids": ["C1"], "evidence_ids": ["U1", "U2"]},
            "overview": [{"text": "Остаётся задержка.", "claim_ids": ["C1"], "evidence_ids": ["U1"]}],
            "sections": {"technical": items}, "chronology": [], "navigation": [],
            "outcome_cards": [], "source_items": items, "metadata": {},
        }
        graph = {"claims": [
            {"claim_id": "C1", "statement": "Остаётся задержка.", "lifecycle": "active", "verification_status": "supported"},
            {"claim_id": "C2", "statement": "Проверяется фильтр.", "lifecycle": "active", "verification_status": "supported"},
        ], "relations": []}
        response = {"response": {
            "title": {"text": "Остаётся задержка", "claim_ids": ["C1"]},
            "overview": [{"text": "Остаётся задержка.", "claim_ids": ["C1"]}],
            "section_edits": [], "chapter_edits": [],
        }}
        with tempfile.TemporaryDirectory() as directory, patch(
            "scripts.summary_worker.call_json_with_retries", return_value=response
        ):
            edited = apply_bounded_document_writer(object(), "writer", document, graph, pathlib.Path(directory))
        self.assertEqual(edited["title"]["evidence_ids"], ["U1"])
        self.assertEqual(edited["overview"][0]["evidence_ids"], ["U1"])

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

    def test_internal_action_identifier_keeps_human_task_and_counterparty_acceptance(self):
        graph = build_meeting_graph([rec(
            1, "action",
            "Создать разметчик для Order Block, чтобы участник мог экспериментировать с имбалансами.",
            "commit", "@Misha", origin_id="OR-order-block", assignees=["@Misha"],
            dialogue_evidence=[
                {"id": "U1", "start": 1, "speaker": "@Misha", "text": "Мне сделать тебе разметчик Order Block?"},
                {"id": "U2", "start": 2, "speaker": "@Recipient", "text": "Да-да-да. Дальше этап апробации."},
            ],
            actions=[{
                "action_id": "A01", "actor": "@Misha",
                "predicate": "create_order_block_marker_tool", "object": None,
                "recipient": "@Recipient", "temporal_state": "planned",
                "commitment_state": "unknown", "evidence_ids": ["U1", "U2"],
                "field_evidence": {"actor": ["U1"], "predicate": ["U1"], "recipient": ["U1"]},
            }],
        )])
        self.assertEqual(len(graph["task_states"]), 1)
        task = graph["task_states"][0]
        self.assertNotIn("create_order_block_marker_tool", task["description"])
        self.assertIn("Создать разметчик", task["description"])
        self.assertEqual(task["assignee"], "@Misha")
        self.assertEqual(task["status"], "accepted")

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

    def test_rules_have_a_dedicated_protected_reader_view(self):
        graph = build_meeting_graph([rec(1, "system_rule", "Позиция закрывается после подтверждённого сигнала")])
        planned = plan(graph["claims"], graph["episodes"], graph["relations"], lambda _: 1)
        self.assertTrue(planned["view_plans"]["rules"]["selected_claim_ids"])
        public = build_public_items(graph, planned)
        self.assertEqual(len([item for item in public if item["section"] == "rules"]), 1)

    def test_unverified_candidate_answer_is_not_reopened_as_raw_question(self):
        graph = build_meeting_graph([
            rec(1, "question", "Почему сервис задерживает данные?", "ask", requested_slots=["explanation"], answer_record_ids=["F2"]),
            rec(2, "observation", "Сервис включён.", "answer"),
        ])
        state = graph["question_states"][0]
        self.assertEqual(state["status"], "answer_not_verified")
        planned = plan(graph["claims"], graph["episodes"], graph["relations"], lambda _: 1)
        self.assertFalse(any(item["section"] == "questions" for item in build_public_items(graph, planned)))

    def test_chapter_count_is_adaptive_below_configured_ceiling(self):
        items = [{
            "public_id": f"PI{index}", "section": "minutes", "text": f"Проверен этап {index}",
            "claim_ids": [f"C{index}"], "evidence_ids": [f"U{index}"],
            "source_word_ids": [f"W{index}"], "content_kind": "observation",
            "social_state": "candidate", "start": index * 10, "end": index * 10 + 5,
            "episode_id": f"E{index}", "verification_status": "supported",
        } for index in range(1, 21)]
        document = build_public_document(items, {"max_chapters": 12})
        self.assertLess(len(document["chronology"]), 12)

    def test_navigation_selector_prefers_topic_over_weak_transcript_fragments(self):
        candidates = [
            {"text": "Есть такой инструмент Альфа.", "content_kind": "observation", "evidence_ids": ["U1"], "start": 1},
            {"text": "Марина решила проверить отчёт.", "content_kind": "action", "evidence_ids": ["U2"], "start": 2},
            {"text": "Отчёт теряет строки при повторной загрузке.", "content_kind": "problem", "evidence_ids": ["U3"], "start": 3},
            {"text": "Данные следует обновить после 2028 года.", "content_kind": "action", "evidence_ids": ["U4"], "start": 4},
            {"text": "Изменения не были внесены.", "content_kind": "observation", "evidence_ids": ["U5"], "start": 5},
        ]
        self.assertTrue(navigation_label_needs_repair(candidates[0]["text"]))
        self.assertTrue(navigation_label_needs_repair(candidates[1]["text"]))
        self.assertTrue(navigation_label_needs_repair(candidates[3]["text"]))
        self.assertTrue(navigation_label_needs_repair(candidates[4]["text"]))
        self.assertEqual(_best_navigation_item(candidates), candidates[2])
        correction = {"text": "Уточнено: лимит относится к позиции, а не к зоне.",
                      "content_kind": "correction", "evidence_ids": ["U6"], "start": 6}
        colloquial_problem = {"text": "Если нет этих квадратов в строчку, это означает слабость стороны.",
                              "content_kind": "problem", "evidence_ids": ["U7"], "start": 7}
        self.assertEqual(_best_navigation_item([colloquial_problem, correction]), correction)

    def test_navigation_reconciliation_does_not_choose_shortest_weak_item(self):
        weak = {"public_id": "PI1", "text": "Есть такой модуль.", "content_kind": "observation",
                "claim_ids": ["C1"], "evidence_ids": ["U1"], "start": 1}
        useful = {"public_id": "PI2", "text": "Модуль теряет записи при повторной загрузке.", "content_kind": "problem",
                  "claim_ids": ["C2"], "evidence_ids": ["U2"], "start": 2}
        document = {
            "title": {"text": useful["text"], "claim_ids": ["C2"], "evidence_ids": ["U2"]},
            "overview": [], "sections": {}, "outcome_cards": [], "metadata": {},
            "navigation": [{"chapter_id": "CH1", "label": "Недоказанная редактура",
                            "claim_ids": ["C1", "C2"], "evidence_ids": ["U1", "U2"],
                            "start": 1, "end": 3}],
            "chronology": [{"chapter_id": "CH1", "label": "Недоказанная редактура",
                            "claim_ids": ["C1", "C2"], "evidence_ids": ["U1", "U2"],
                            "start": 1, "end": 3, "items": [weak, useful], "summary": []}],
        }
        audit = {"status": "failed", "total_nodes": 2, "reviews": [
            {"node_id": "title", "verdict": "supported"},
            {"node_id": "navigation:1", "verdict": "insufficient_evidence"},
        ]}
        repaired, reconciled = reconcile_final_document_audit(document, audit, [weak, useful])
        self.assertEqual(repaired["navigation"][0]["label"], useful["text"])
        self.assertEqual(reconciled["status"], "passed")

    def test_semantically_supported_weak_navigation_is_still_repaired(self):
        weak = {
            "public_id": "PI1", "text": "@Ирина указывает, что проблема в загрузке данных.",
            "content_kind": "problem", "claim_ids": ["C1"],
            "evidence_ids": ["U1"], "start": 1,
        }
        useful = {
            "public_id": "PI2", "text": "Повторная загрузка теряет часть записей.",
            "content_kind": "observation", "claim_ids": ["C2"],
            "evidence_ids": ["U2"], "start": 2,
        }
        document = {
            "title": {"text": useful["text"], "claim_ids": ["C2"], "evidence_ids": ["U2"]},
            "overview": [], "sections": {}, "outcome_cards": [], "metadata": {},
            "navigation": [{"chapter_id": "CH1", "label": weak["text"],
                            "claim_ids": ["C1"], "evidence_ids": ["U1"],
                            "start": 1, "end": 3}],
            "chronology": [{"chapter_id": "CH1", "label": weak["text"],
                            "claim_ids": ["C1"], "evidence_ids": ["U1"],
                            "start": 1, "end": 3, "items": [weak, useful], "summary": []}],
        }
        audit = {"status": "passed", "total_nodes": 2, "reviews": [
            {"node_id": "title", "verdict": "supported"},
            {"node_id": "navigation:1", "verdict": "supported"},
        ]}

        repaired, reconciled = reconcile_final_document_audit(
            document, audit, [weak, useful],
        )

        self.assertEqual(repaired["navigation"][0]["label"], useful["text"])
        self.assertEqual(repaired["chronology"][0]["label"], useful["text"])
        self.assertEqual(reconciled["status"], "passed")
        self.assertEqual(reconciled["abstention_count"], 0)

    def test_navigation_quality_uses_reader_visible_markdown_surface(self):
        self.assertTrue(navigation_label_needs_repair(
            "**@Ирина** указывает, что проблема заключается в загрузке данных."
        ))

    def test_runtime_gate_rejects_weak_navigation_even_when_grounded(self):
        document = {
            "navigation": [{"label": "Есть такой инструмент Альфа."}],
            "semantic_audit": {"status": "passed", "reviews": []},
        }
        report = publication_audit({"audits": []}, "# Итоги\n", [], {}, document=document)
        self.assertEqual(report["weak_navigation_labels"], 1)
        self.assertEqual(report["reports"]["utility"]["status"], "failed")

    def test_evidence_anchor_reordering_keeps_final_chronology_monotonic(self):
        task = {"public_id": "PI1", "section": "tasks", "start": 8, "end": 9}
        correction = {
            "public_id": "PI2", "section": "minutes", "start": 3329.62,
            "end": 3332.78, "text": "Поправка к условию.",
        }
        preceding_state = {
            "public_id": "PI3", "section": "minutes", "start": 3310.44,
            "end": 3314.68, "text": "Система восстанавливается после сбоя.",
        }
        question = {"public_id": "PI4", "section": "questions", "start": 3392.86, "end": 3437.96}

        ordered = canonicalize_public_item_order(
            [task, correction, preceding_state, question]
        )

        self.assertEqual(
            [item["public_id"] for item in ordered],
            ["PI1", "PI3", "PI2", "PI4"],
        )
        self.assertIs(ordered[0], task)
        self.assertIs(ordered[-1], question)
        report = publication_audit(
            {"audits": []}, "# Итоги\n", ordered, {}, document=None,
        )
        self.assertEqual(report["chronology_inversions"], 0)

    def test_schedule_residual_names_source_time_alternatives(self):
        statement = (
            "Созвон предложен на четверг; варианты времени: 09:30 или 11:00. "
            "Точное время не подтверждено."
        )
        states = reduce_questions(
            [{"proposition_id": "P1", "content_kind": "schedule", "statement": statement,
              "evidence_ids": ["U1"]}],
            [{"event_id": "EV1", "proposition_id": "P1", "source_record_id": "F1",
              "speech_act": "ask", "timestamp": 1}],
            [],
            [{"record_id": "F1", "statement": statement, "start": 1,
              "requested_slots": ["free_text", "time"], "answered_slots": ["free_text"],
              "question_status": "partially_answered", "answer_evidence_ids": ["U1"],
              "evidence_ids": ["U1"]}],
        )
        self.assertEqual(states[0]["missing_slot_labels"], ["выбрать точное время: 09:30 или 11:00"])
        self.assertEqual(
            states[0]["remaining_question"],
            "Созвон предложен на четверг. Какое точное время выбрать: 09:30 или 11:00?",
        )
        self.assertNotIn("или период", states[0]["remaining_question"])

    def test_condensed_chronology_replaces_mechanical_outcome_dump(self):
        document = {
            "title": {"text": "Проверка", "claim_ids": ["C1"], "evidence_ids": ["U1"]},
            "overview": [], "navigation": [], "sections": {}, "metadata": {},
            "outcome_cards": [{"outcome_id": "OC1", "fields": {"current_state": {"value": "Сырой факт", "claim_ids": ["C1"], "evidence_ids": ["U1"]}}}],
            "chronology": [{"chapter_id": "CH1", "label": "Этап", "start": 1, "end": 2,
                            "outcome_ids": ["OC1"], "items": [],
                            "summary": [{"text": "Краткий итог этапа", "claim_ids": ["C1"], "evidence_ids": ["U1"]}]}],
        }
        document["chronology"][0]["items"] = [
            {"text": "Сырой факт", "claim_ids": ["C1"], "evidence_ids": ["U1"], "start": 1}
        ]
        rendered = render_public_document(document)
        self.assertIn("Краткий итог этапа.", rendered)
        self.assertIn("<summary>Подтверждённые детали</summary>", rendered)
        self.assertNotIn("**Состояние:** Сырой факт", rendered)
        node_ids = {node["node_id"] for node in _document_audit_nodes(document)}
        self.assertIn("chronology_summary:1:1", node_ids)
        self.assertNotIn("outcome:1:current_state:1", node_ids)

    def test_condensed_chronology_retains_unrepresented_typed_fields(self):
        document = {
            "title": {"text": "Проверка", "claim_ids": ["C1"], "evidence_ids": ["U1"]},
            "overview": [], "navigation": [], "sections": {}, "metadata": {},
            "outcome_cards": [{
                "outcome_id": "OC1", "claim_ids": ["C1", "C2"],
                "evidence_ids": ["U1", "U2"],
                "fields": {
                    "current_state": {"value": "Основной факт", "claim_ids": ["C1"], "evidence_ids": ["U1"]},
                    "constraint": {"value": "Отдельное ограничение", "claim_ids": ["C2"], "evidence_ids": ["U2"]},
                },
            }],
            "chronology": [{
                "chapter_id": "CH1", "label": "Этап", "start": 1, "end": 2,
                "outcome_ids": ["OC1"],
                "items": [{"text": "Основной факт", "claim_ids": ["C1"], "evidence_ids": ["U1"], "start": 1}],
                "summary": [{"text": "Основной факт", "claim_ids": ["C1"], "evidence_ids": ["U1"]}],
            }],
        }
        rendered = render_public_document(document)
        self.assertIn("**Ограничение:** Отдельное ограничение.", rendered)
        node_ids = {node["node_id"] for node in _document_audit_nodes(document)}
        self.assertIn("outcome:1:constraint:1", node_ids)
        self.assertNotIn("outcome:1:current_state:1", node_ids)

    def test_failed_chronology_summary_is_replaced_by_exact_public_item(self):
        item = {
            "public_id": "PI1", "section": "minutes", "text": "Точная проверенная формулировка",
            "claim_ids": ["C1"], "evidence_ids": ["U1"], "content_kind": "observation",
            "start": 1,
        }
        document = {
            "title": dict(item), "overview": [], "navigation": [], "sections": {},
            "outcome_cards": [], "metadata": {},
            "chronology": [{
                "chapter_id": "CH1", "label": "Этап", "items": [dict(item)],
                "summary": [{"text": "Недоказанное обобщение", "claim_ids": ["C1"], "evidence_ids": ["U1"]}],
            }],
        }
        audit = {
            "status": "failed", "total_nodes": 2,
            "reviews": [
                {"node_id": "title", "verdict": "supported"},
                {"node_id": "chronology_summary:1:1", "verdict": "insufficient_evidence"},
            ],
        }
        repaired, reconciled = reconcile_final_document_audit(document, audit, [item])
        self.assertEqual(repaired["chronology"][0]["summary"][0]["text"], item["text"])
        self.assertEqual(reconciled["status"], "passed")
        self.assertEqual(reconciled["unresolved_node_ids"], [])

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

    def test_force_policy_separates_view_rebuild_from_fresh_model_calls(self):
        self.assertEqual(summary.resolve_replay_mode(False, "semantics", "fresh_models"), "semantics")
        self.assertEqual(summary.resolve_replay_mode(True, "cached", "rebuild"), "views")
        self.assertEqual(summary.resolve_replay_mode(True, "cached", "fresh_models"), "fresh")

    def test_release_manifest_ignores_operating_system_sidecars(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "scripts").mkdir()
            (root / "scripts" / "worker.py").write_text("VALUE = 1\n", encoding="utf-8")
            (root / "scripts" / "._worker.py").write_bytes(b"appledouble")
            (root / "scripts" / "__pycache__").mkdir()
            (root / "scripts" / "__pycache__" / "cached.py").write_text("x=1\n", encoding="utf-8")
            with patch.object(summary, "APPLICATION_ROOT", root), \
                    patch.object(summary, "release_commit", return_value="a" * 40):
                manifest = summary.build_release_manifest({}, {})
        self.assertEqual(set(manifest["files"]), {"scripts/worker.py"})

    def test_meeting_date_prefers_explicit_timestamp_and_rejects_partial_dates(self):
        self.assertEqual(summary.meeting_date("archive-01.02.2020.wav", "2026-09-17T12:30:00+03:00"), "17.09.2026")
        self.assertEqual(summary.meeting_date("meeting-17-09.wav"), "Дата не указана")


if __name__ == "__main__":
    unittest.main()
