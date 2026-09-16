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
    def test_resource_commitment_is_still_a_task(self):
        graph = build_meeting_graph([rec(1, "resource", "Я передам выгрузку Bitcoin", "commit", assignees=["@A"])])
        self.assertEqual(len(graph["task_states"]), 1)
        planned = plan(graph["claims"], graph["episodes"], graph["relations"], lambda _: 1)
        self.assertIn(graph["claims"][0]["claim_id"], planned["view_plans"]["tasks"]["selected_claim_ids"])

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
        self.assertEqual(task["current_scope"], "1 месяц")
        self.assertEqual(task["data_origin"], "2021")
        self.assertEqual(task["superseded_scopes"], [])
        self.assertTrue(all(value.startswith("R") for value in task["scope_relation_ids"]))
        self.assertTrue(any(x["type"] == "revises_scope" for x in graph["relations"]))

    def test_bare_month_revision_wins_over_later_four_hour_constraint(self):
        graph = build_meeting_graph([
            rec(1, "resource", "@A предоставит данные Bitcoin за 2021 год", "commit", assignees=["@A"]),
            rec(2, "constraint", "Уточнение требования к объему данных: месяц вместо года", "correct", speaker="@B"),
            rec(3, "constraint", "Для симуляции достаточно целых минут за период 15–4 часа", "assert", speaker="@B"),
        ])
        self.assertEqual(graph["task_states"][0]["current_scope"], "1 месяц")

    def test_exact_dialogue_recovers_personal_commitments_and_merges_same_promise(self):
        promise = {"id": "U36", "speaker": "@B", "text": "Order Block я буду параллельно размечивать, и параллельно мы их будем встраивать."}
        graph = build_meeting_graph([
            rec(1, "action", "Участники договорятся о параллельной разметке Order Block", "commit", speaker="@B", assignees=["@B"], commitment_strength="implicit", verification_status="verification_unavailable", dialogue_evidence=[promise]),
            rec(2, "action", "Order Block будут размечать параллельно с встраиванием имбалансов", "commit", speaker="@B", assignees=["@B"], commitment_strength="implicit", verification_status="verification_unavailable", dialogue_evidence=[promise]),
            rec(3, "action", "Варианты имбалансов нужно встроить в методичку", "commit", speaker="@B", assignees=["@B"], commitment_strength="implicit", verification_status="verification_unavailable", dialogue_evidence=[{"id": "U31", "speaker": "@B", "text": "Потом встрою это в методичку."}]),
        ])
        self.assertEqual(len(graph["task_states"]), 2)
        self.assertTrue(all(item["status"] == "self_committed" for item in graph["task_states"]))
        anchors = [claim for claim in graph["claims"] if claim.get("canonical_task_anchor")]
        self.assertTrue(all(claim["verification_status"] == "supported" for claim in anchors))

    def test_resource_commitment_is_recovered_without_model_assignee(self):
        graph = build_meeting_graph([
            rec(1, "resource", "@A предложил предоставить данные по Bitcoin за 2021 год", "assert", speaker="@A", assignees=[], commitment_strength="none", dialogue_evidence=[
                {"id": "U1", "speaker": "@A", "text": "Я Bitcoin 2021 года тебе дам.", "start": 1},
                {"id": "U2", "speaker": "@B", "text": "Мне хватит месяца.", "start": 2},
                {"id": "U3", "speaker": "@A", "text": "А, месяца. Ну ладно.", "start": 3},
            ]),
            rec(2, "constraint", "Уточнение требования к объему данных: месяц вместо года", "correct", speaker="@B"),
        ])
        task = graph["task_states"][0]
        self.assertEqual((task["status"], task["assignee"], task["current_scope"], task["scope_confidence"]), ("self_committed", "@A", "1 месяц", "accepted"))
        self.assertEqual(task["deliverable"], "@A предоставит данные по Bitcoin за 2021 год")

    def test_vague_focus_intention_is_not_a_task(self):
        graph = build_meeting_graph([rec(1, "action", "Меньше условно часть ИИ пилить и сделать упор на это", "commit", assignees=["@A"], commitment_strength="implicit")])
        self.assertEqual(graph["task_states"], [])

    def test_cross_day_rule_closes_question(self):
        graph = build_meeting_graph([
            rec(1, "question", "Может ли сделка закрыться на следующий день?", "ask", requested_slots=["cross_day_closure_feasibility"], answer_record_ids=["F2"]),
            rec(2, "system_rule", "Если сделка не дошла до Take Profit, она закрывается автоматически после 00:00", "answer", speaker="@B"),
        ])
        self.assertEqual(graph["question_states"][0]["status"], "answered")
        short_midnight = build_meeting_graph([
            rec(1, "question", "Может ли сделка закрыться на следующий день?", "ask", requested_slots=["cross_day_closure_feasibility"], answer_record_ids=["F2"]),
            rec(2, "system_rule", "Если сделка не дошла до Take Profit, она закрывается автоматически после 00.", "answer", speaker="@B"),
        ])
        self.assertEqual(short_midnight["question_states"][0]["status"], "answered")

    def test_repeated_acknowledgement_does_not_span_the_meeting(self):
        graph = build_meeting_graph([
            rec(1, "observation", "Сервис работает"),
            rec(2, "observation", "Угу.", "accept", speaker="@B"),
            rec(200, "observation", "Угу.", "accept", speaker="@A"),
            rec(201, "observation", "Проверка завершена"),
        ])
        acknowledgement = next(claim for claim in graph["claims"] if claim["statement"] == "Угу.")
        self.assertTrue(acknowledgement["dialogue_only"])
        self.assertFalse(any(acknowledgement["claim_id"] in episode["claim_ids"] for episode in graph["episodes"]))

    def test_proposal_with_local_other_speaker_yes_is_accepted(self):
        graph = build_meeting_graph([rec(1, "proposal", "@A предложил сделать разметчик для @B", "propose", speaker="@A", dialogue_evidence=[
            {"id": "U0", "speaker": "@B", "text": "Да, другой план подходит.", "start": 0},
            {"id": "U1", "speaker": "@A", "text": "Мне сделать разметчик тебе?", "start": 1},
            {"id": "U2", "speaker": "@B", "text": "Да-да-да. Дальше этап апробации.", "start": 2},
        ], evidence_ids=["U1", "U2"])])
        decision = graph["decision_states"][0]
        self.assertEqual(decision["status"], "accepted")
        self.assertEqual(decision["acceptance_evidence_ids"], ["U2"])
        self.assertEqual(decision["accepted_by"], ["@B"])
        self.assertTrue(decision["acceptance_check"] == "entailed")

    def test_neighboring_commitment_does_not_promote_confirmed_plan(self):
        graph = build_meeting_graph([rec(1, "proposal", "@A предложил создать разметчик для @B, чтобы экспериментировать с имбалансами", "propose", speaker="@A", dialogue_evidence=[
            {"id": "U0", "speaker": "@B", "text": "Order Block я буду параллельно размечивать, и параллельно мы их будем встраивать.", "start": 0},
            {"id": "U1", "speaker": "@A", "text": "Мне сделать разметчик тебе Order Block и пойти экспериментировать с имбалансами?", "start": 1},
            {"id": "U2", "speaker": "@B", "text": "Да-да-да. Дальше этап апробации.", "start": 2},
        ], evidence_ids=["U1", "U2"])])
        self.assertEqual(graph["decision_states"][0]["status"], "accepted")
        self.assertEqual(graph["task_states"], [])

    def test_reported_plan_does_not_assign_the_reporter(self):
        graph = build_meeting_graph([rec(1, "action", "@B / @C проанализирует рынок и разметит данные", "commit", speaker="@A", assignees=["@A"], commitment_strength="implicit", verification_status="verification_unavailable")])
        task = graph["task_states"][0]
        self.assertIsNone(task["owner"])
        self.assertEqual(task["assignees"], [])
        self.assertEqual(task["action_frame"]["state"], "reported_plan")
        planned = plan(graph["claims"], graph["episodes"], graph["relations"], lambda _: 1)
        item = next(value for value in build_public_items(graph, planned) if value["section"] == "requires_verification")
        self.assertTrue(item["text"].startswith("По словам @A, один из @B или @C"))

    def test_structure_status_answer_closes_question_and_minutes(self):
        graph = build_meeting_graph([
            rec(1, "question", "Со структурами полностью закончили?", "ask", speaker="@A", requested_slots=["статус работы со структурами"], answer_record_ids=["F2", "F3"], question_status="answered"),
            rec(2, "problem", "На старших таймфреймах остаётся задержка", "answer", speaker="@B"),
            rec(3, "observation", "На минутном таймфрейме алгоритм работал корректно", "answer", speaker="@B"),
        ])
        question = graph["question_states"][0]
        self.assertEqual(question["status"], "answered")
        planned = plan(graph["claims"], graph["episodes"], graph["relations"], lambda _: 1)
        items = build_public_items(graph, planned)
        self.assertFalse(any(item["content_kind"] == "question" for item in items))

    def test_unrelated_month_does_not_revise_task_scope(self):
        graph = build_meeting_graph([
            rec(1, "resource", "Я передам выгрузку Bitcoin", "commit", assignees=["@A"]),
            rec(2, "proposal", "Нужна подписка на два месяца", "propose", speaker="@B"),
        ])
        self.assertFalse(any(x["type"] == "revises_scope" for x in graph["relations"]))
        self.assertIsNone(graph["task_states"][0]["current_scope"])

    def test_every_selected_technical_claim_has_a_public_disposition(self):
        graph = build_meeting_graph([
            rec(1, "definition", "Order Block определяется по импульсу"),
            rec(2, "dependency", "Фильтр зависит от таймфрейма"),
            rec(3, "system_rule", "Нельзя входить без подтверждения"),
        ])
        planned = plan(graph["claims"], graph["episodes"], graph["relations"], lambda _: 1)
        items = build_public_items(graph, planned)
        published = {claim_id for item in items if item["section"] in {"technical", "rules"} for claim_id in item["claim_ids"]}
        selected = set(planned["view_plans"]["technical"]["selected_claim_ids"])
        self.assertEqual(selected, published)
        self.assertTrue(all(planned["view_plans"]["technical"]["dispositions"][value]["status"] == "published" for value in selected))

    def test_distinct_annotation_and_integration_deliverables_do_not_merge(self):
        graph = build_meeting_graph([
            rec(1, "action", "Я буду размечать и встраивать Order Block", "commit", assignees=["@A"]),
            rec(2, "action", "Я буду параллельно размечать Order Block", "commit", assignees=["@A"]),
        ])
        self.assertEqual(len(graph["task_states"]), 2)

    def test_upstream_answer_without_answered_slots_is_preserved(self):
        graph = build_meeting_graph([
            rec(1, "question", "Закрывается ли сделка завтра?", "ask", requested_slots=["closing"], question_status="answered", answer_record_ids=["F2"]),
            rec(2, "observation", "Она закрывается автоматически", "answer", speaker="@B"),
        ])
        self.assertEqual(graph["question_states"][0]["status"], "answered")
        p = plan(graph["claims"], graph["episodes"], graph["relations"], lambda _: 1)
        self.assertFalse(any(x["section"] == "questions" for x in build_public_items(graph, p)))

    def test_open_question_names_its_known_asker(self):
        graph = build_meeting_graph([rec(1, "question", "Удалось реализовать вход?", "ask", speaker="@Misha", requested_slots=["result"])])
        planned = plan(graph["claims"], graph["episodes"], graph["relations"], lambda _: 1)
        question = next(x for x in build_public_items(graph, planned) if x["section"] == "questions")
        self.assertTrue(question["text"].startswith("@Misha спрашивает:"))

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

    def test_overview_must_expose_readiness_constraint_and_next_step(self):
        report = {"audits": [{"passed": True, "errors": []}]}
        base = {"claim_ids": ["C1"], "evidence_ids": ["U1"], "source_word_ids": ["W1"], "start": 1}
        items = [
            {**base, "public_id": "PI1", "section": "overview", "text": "На минутном таймфрейме алгоритм работал корректно", "content_kind": "observation", "social_state": "asserted"},
            {**base, "public_id": "PI2", "section": "overview", "text": "На старших таймфреймах остаётся задержка", "content_kind": "problem", "social_state": "asserted"},
            {**base, "public_id": "PI3", "section": "tasks", "text": "@A подготовит демонстрацию в TradingView", "content_kind": "action", "social_state": "self_committed", "task_state_id": "T1", "task_state": {"status": "self_committed", "deliverable": "демонстрация в TradingView", "commitment_strength": "explicit"}},
        ]
        incomplete = publication_audit(report, "# Встреча — Структуры\n\n## Главное\n- Обсудили структуры.\n\n## Задачи и следующие шаги\n- @A подготовит демонстрацию в TradingView", items, {})
        self.assertEqual(incomplete["overview_missing_htf_readiness"], 1)
        self.assertEqual(incomplete["overview_missing_htf_constraint"], 1)
        self.assertEqual(incomplete["overview_missing_committed_next_step"], 1)
        complete = publication_audit(report, "# Встреча — Структуры\n\n## Главное\n- На минутном таймфрейме алгоритм работал корректно.\n- На старших таймфреймах остаётся задержка.\n- @A подготовит демонстрацию в TradingView.\n\n## Задачи и следующие шаги\n- @A подготовит демонстрацию в TradingView", items, {})
        self.assertEqual(complete["overview_missing_htf_readiness"], 0)
        self.assertEqual(complete["overview_missing_htf_constraint"], 0)
        self.assertEqual(complete["overview_missing_committed_next_step"], 0)


if __name__ == "__main__":
    unittest.main()
