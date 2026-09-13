import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from semantics.episodes import build_episodes, build_threads
from semantics.graph import cross_episode_allowed, reconcile_latest_state
from semantics.ontology import CLAIM_KINDS, ClaimKind
from summary.planner import adaptive_budget, plan
from summary.verifier import audit_realization
from scripts.consensus import duration, intersect_duration


class V20SemanticCoreTests(unittest.TestCase):
    def test_canonical_ontology_includes_rich_types(self):
        self.assertEqual(set(CLAIM_KINDS), {x.value for x in ClaimKind})
        self.assertIn("experimental_result", CLAIM_KINDS)
        self.assertIn("trading_rule", CLAIM_KINDS)

    def test_interval_measure_never_double_counts(self):
        left = [{"start": 0, "end": 10}, {"start": 5, "end": 12}]
        right = [{"start": 3, "end": 11}, {"start": 7, "end": 13}]
        intersection = intersect_duration(left, right)
        self.assertLessEqual(intersection, min(duration(left), duration(right)))
        self.assertEqual(duration(left), 12)

    def test_correction_supersedes_prior_claim(self):
        claims = [{"claim_id": "C1"}, {"claim_id": "C2"}]
        reconcile_latest_state(claims, [{"relation_id": "R1", "type": "corrects", "source_claim_id": "C2", "target_claim_id": "C1"}])
        self.assertEqual(claims[0]["lifecycle"], "superseded")
        self.assertEqual(claims[1]["lifecycle"], "active")

    def test_cross_episode_merge_requires_relation(self):
        claims = [{"claim_id": "C1", "episode_id": "E1"}, {"claim_id": "C2", "episode_id": "E2"}]
        self.assertFalse(cross_episode_allowed(["C1", "C2"], [], claims, []))
        relations = [{"relation_id": "R1", "source_claim_id": "C1", "target_claim_id": "C2"}]
        self.assertTrue(cross_episode_allowed(["C1", "C2"], ["R1"], claims, relations))

    def test_planner_is_mandatory_first_and_adaptive(self):
        claims = [
            {"claim_id": "C1", "kind": "action", "task_status": "accepted", "statement": "Сделать тест", "episode_id": "E1", "lifecycle": "active", "start": 1},
            {"claim_id": "C2", "kind": "observation", "statement": "Наблюдение", "episode_id": "E2", "lifecycle": "active", "start": 2},
        ]
        episodes = [{"episode_id": "E1"}, {"episode_id": "E2"}]
        result = plan(claims, episodes, [], lambda _: 1)
        self.assertIn("C1", result["channels"]["mandatory"])
        self.assertEqual(set(result["episode_coverage"]), {"E1", "E2"})
        self.assertGreaterEqual(adaptive_budget(claims, episodes), 20)

    def test_surface_guard_rejects_new_number_and_causality(self):
        audit = audit_realization("Из-за этого результат вырос на 20%", {"claim_ids": ["C1"], "relation_ids": [], "allowed_numbers": ["10%"]})
        self.assertFalse(audit["passed"])
        self.assertIn("unsupported_relation_language", audit["errors"])

    def test_planner_allows_relation_wording_already_present_in_atomic_claim(self):
        claim = {
            "claim_id": "C1", "kind": "observation", "statement": "Есть проблема из-за ложных срабатываний.",
            "episode_id": "E1", "lifecycle": "active", "start": 1,
        }
        result = plan([claim], [{"episode_id": "E1"}], [], lambda _: 1)
        sentence_plan = result["sentence_plans"][0]
        self.assertEqual(sentence_plan["allowed_relation_markers"], ["из-за"])
        self.assertTrue(audit_realization(claim["statement"], sentence_plan)["passed"])
        self.assertFalse(audit_realization("Поэтому появилась проблема.", sentence_plan)["passed"])


if __name__ == "__main__":
    unittest.main()
