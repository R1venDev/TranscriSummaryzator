import pathlib, tempfile, unittest
ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from evidence.refinement import asr_lattice, calibrate_speaker_probability, schedule_repairs, target_speaker_windows
from evaluation.semantic_metrics import architecture_metrics, release_gate
from pipeline_core.dag import MEETING_DAG
from pipeline_core.models import cache_identity, model_identity
from project_memory.graph_store import ProjectGraphStore, apply_meeting, empty
from project_memory.retrieval import retrieve
from semantics.meeting_graph import build_meeting_graph, compatibility_state
from semantics.entities import EntityRegistry
from semantics.questions import retrieve_answer_candidates, verify_slot_entailment
from semantics.propositions import proposition_from_record
from summary.planner import plan
from summary.verifier import audit_realization, source_aware_plan, verify_generated_items


def record(record_id, kind, statement, act="assert", **extra):
    value = {"record_id": record_id, "kind": kind, "statement": statement, "speech_act": act, "modality": extra.pop("modality", "asserted"), "evidence_ids": ["U" + record_id[1:]], "attributed_speakers": ["@A"], "start": float(record_id[1:]), **extra}
    return value


class V21ArchitectureTests(unittest.TestCase):
    def test_proposition_identity_is_normalized_and_condition_is_structural(self):
        a = record("F1", "trading_rule", "На M15 подтверждаем BOS", subject="entry", predicate="requires", object="BOS", scope={"timeframe": "M15"}, conditions=[{"antecedent": "M15 confirmed"}])
        b = {**a, "record_id": "F2", "statement": "Слом подтверждается пятнадцатиминутной структурой"}
        self.assertEqual(proposition_from_record(a)["proposition_id"], proposition_from_record(b)["proposition_id"])
        self.assertEqual(proposition_from_record(a)["content_kind"], "rule")

    def test_meeting_graph_is_authoritative_and_decision_requires_acceptance(self):
        records = [record("F1", "proposal", "Предлагаю использовать M15", "propose"), record("F2", "observation", "Да", "accept")]
        graph = build_meeting_graph(records, {"audio_sha256": "abc"})
        self.assertTrue(graph["authoritative"]); self.assertEqual(graph["schema"], "MeetingGraphSchema")
        self.assertTrue(any(x["type"] == "accepts" for x in graph["relations"]))
        self.assertTrue(any(x["status"] == "accepted" for x in graph["decision_states"]))
        self.assertEqual(compatibility_state(graph)["authoritative_source"], "meeting_graph.json")

    def test_question_slots_do_not_accept_unrelated_answer(self):
        records = [record("F1", "question", "Одна или две сделки?", "ask", requested_slots=["number_of_trades"], answered_slots=[], answer_record_ids=["F2"]), record("F2", "observation", "Стоп за максимумом", "answer")]
        graph = build_meeting_graph(records)
        self.assertEqual(graph["question_states"][0]["status"], "unanswered")
        self.assertEqual(graph["question_states"][0]["missing_slots"], ["number_of_trades"])

    def test_entity_registry_and_answer_entailment_are_separate(self):
        registry = EntityRegistry([{"canonical_name": "Bitcoin", "type": "asset", "aliases": ["BTC"]}])
        self.assertEqual(registry.resolve("btc")["canonical_name"], "Bitcoin")
        candidates = retrieve_answer_candidates({"statement": "Сколько сделок?", "start": 0}, [{"statement": "Будет 2 сделки", "timestamp": 2, "speech_act": "answer"}])
        self.assertEqual(len(candidates), 1)
        self.assertEqual(verify_slot_entailment(["number_of_trades"], candidates[0])["entailed_slots"]["number_of_trades"], "2")

    def test_task_state_does_not_invent_owner(self):
        graph = build_meeting_graph([record("F1", "action", "Наверное, посмотрим другой подход", "propose", modality="tentative")])
        self.assertEqual(graph["task_states"][0]["status"], "proposed")
        self.assertIsNone(graph["task_states"][0]["owner"])
        self.assertFalse(graph["task_states"][0]["automation_eligible"])

    def test_planner_builds_view_specific_subgraphs_and_contracts(self):
        graph = build_meeting_graph([record("F1", "problem", "BOS даёт ложные входы"), record("F2", "trading_rule", "Если M15 подтверждён, искать M1", "propose", conditions=["если M15 подтверждён"], quantities=[{"value": 15, "unit": "minute", "entity": "timeframe"}])])
        result = plan(graph["claims"], graph["episodes"], graph["relations"], lambda _: 1)
        self.assertIn("technical", result["view_plans"]); self.assertIn("executive", result["view_plans"])
        contract = next(x for x in result["sentence_plans"] if x["conditions"])
        self.assertTrue(contract["allowed_quantities"]); self.assertIn("condition_drop", contract["forbidden_inferences"])

    def test_actual_output_verifier_preserves_modality_condition_and_negation(self):
        plan_contract = {"claim_ids": ["C1"], "relation_ids": [], "allowed_numbers": [], "allowed_relation_markers": [], "polarity": ["negative"], "modality": ["possible"], "conditions": ["если рынок открыт"], "allowed_speakers": [], "allowed_assignees": []}
        bad = audit_realization("Решено использовать вход.", plan_contract)
        self.assertIn("negation_not_preserved", bad["errors"]); self.assertIn("modality_upgraded", bad["errors"]); self.assertIn("condition_not_preserved", bad["errors"])
        claim = {"claim_id": "C1"}
        verified = verify_generated_items([{"text": "Если рынок открыт, нельзя входить.", "fact_ids": ["F1"], "claim_ids": ["C1"]}], [plan_contract], [claim])
        self.assertTrue(verified["passed"])

    def test_navigation_and_composite_people_are_verified_by_role(self):
        contract = {"claim_ids": ["C1"], "relation_ids": [], "allowed_numbers": [], "allowed_relation_markers": [], "polarity": ["positive"], "modality": ["certain"], "conditions": [], "allowed_speakers": ["@Yachoy / @HoTTaBbicH"], "allowed_assignees": []}
        claim = {"claim_id": "C1"}
        navigation = verify_generated_items([{"text": "Обсуждение торгового подхода", "claim_ids": ["C1"], "_semantic_role": "overview"}], [contract], [claim])
        self.assertTrue(navigation["passed"]); self.assertEqual(navigation["audits"][0]["status"], "NAVIGATION")
        statement = verify_generated_items([{"text": "@Yachoy / @HoTTaBbicH должен проверить подход.", "claim_ids": ["C1"]}], [contract], [claim])
        self.assertTrue(statement["passed"])

        source_backed = verify_generated_items(
            [{"text": "@Yachoy / @HoTTaBbicH должен разметить данные.", "claim_ids": ["C1"]}],
            [{**contract, "allowed_speakers": ["@Misha"]}],
            [{"claim_id": "C1", "statement": "@Yachoy / @HoTTaBbicH должен разметить данные."}],
        )
        self.assertTrue(source_backed["passed"])
        enriched = source_aware_plan({**contract, "allowed_speakers": ["@Misha"]}, [{"statement": "@Yachoy / @HoTTaBbicH размечает данные."}])
        self.assertTrue(audit_realization("@Yachoy / @HoTTaBbicH размечает данные.", enriched)["passed"])

    def test_project_graph_is_persistent_lineage_not_literal_document(self):
        meeting = build_meeting_graph([record("F1", "trading_rule", "Используем M15", subject="entry", predicate="requires", object="BOS")], meeting_id="M1")
        with tempfile.TemporaryDirectory() as root:
            store = ProjectGraphStore(root, "Aurion"); graph, delta = store.publish(meeting)
            self.assertEqual(graph["revision"], 1); self.assertEqual(delta["changes"][0]["status"], "NEW")
            self.assertEqual(store.load()["project_graph_id"], graph["project_graph_id"])
            self.assertTrue(retrieve("BOS entry", graph))

    def test_speech_refinement_lattice_and_calibration(self):
        spans = [{"start": 1, "risk": 1, "semantic_importance": 1, "publication_probability": 1}, {"start": 2, "risk": .2, "semantic_importance": 1, "publication_probability": 1}]
        self.assertEqual(schedule_repairs(spans, 1)[0]["start"], 1)
        self.assertFalse(asr_lattice("U1", [{"text": "один к трём", "score": .6}, {"text": "один к двум", "score": .4}])["publication_allowed"])
        self.assertTrue(target_speaker_windows([{"start": 0, "end": 2, "speaker": "A", "overlap": True}], [], ["A"]))
        self.assertGreater(calibrate_speaker_probability({"agreement": 1, "similarity": 1, "margin": 1}), .5)

    def test_models_are_immutable_and_dag_has_all_19_stages(self):
        model = model_identity("repo", "commit", "sha256:x", "runtime", "q4", {})
        self.assertEqual(cache_identity("v1", {"a": 1}, [model]), cache_identity("v1", {"a": 1}, [model]))
        self.assertEqual(len(MEETING_DAG.order("19_publish")), 19)
        with self.assertRaises(ValueError): model_identity("repo", "", "", "runtime")

    def test_benchmark_is_executable_release_policy(self):
        metrics = architecture_metrics({"atomic_claims": ["A"], "importance": {"A": 3}}, {"atomic_claims": ["A"], "usefulness": .9})
        self.assertEqual(metrics["public_factual_precision"], 1)
        self.assertTrue(release_gate({**metrics, "decision_false_positive": 0, "question_false_resolution": 0})["passed"])


if __name__ == "__main__": unittest.main()
