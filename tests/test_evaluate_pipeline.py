import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("evaluate", ROOT / "scripts" / "evaluate_pipeline.py")
evaluate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate)


class EvaluationTests(unittest.TestCase):
    def test_text_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ref.txt").write_text("один два три")
            (root / "hyp.txt").write_text("один два")
            result = evaluate.text_score(root / "ref.txt", root / "hyp.txt")
        self.assertAlmostEqual(result["WER"], 1 / 3, places=5)

    def test_semantics_scores_claims_owners_and_conditions_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold = {"records": [{"record_id": "G1", "kind": "action", "statement": "Не проверить 10", "assignees": ["A"], "conditions": [{"text": "после теста"}], "evidence_ids": ["U1"]}]}
            hyp = {"records": [{"record_id": "H1", "kind": "action", "statement": "Проверить 20", "assignees": ["B"], "conditions": [], "evidence_ids": ["U2"]}], "relations": [{"relation": "causes", "source_event": "X", "target_event": "Y"}]}
            alignment = {"matches": [{"reference_id": "G1", "hypothesis_id": "H1", "supported": True}]}
            for name, value in (("gold.json", gold), ("hyp.json", hyp), ("alignment.json", alignment)):
                (root / name).write_text(json.dumps(value))
            result = evaluate.semantic_score(root / "gold.json", root / "hyp.json", root / "alignment.json")
        self.assertEqual(result["claim_F1"], 1)
        self.assertEqual(result["type_accuracy"], 1)
        self.assertEqual(result["assignee_F1"], 0)
        self.assertEqual(result["condition_F1"], 0)
        self.assertEqual(result["number_accuracy"], 0)
        self.assertEqual(result["negation_accuracy"], 0)
        self.assertEqual(result["citation_precision"], 0)
        self.assertEqual(result["unsupported_relation_rate"], 1)

    def test_non_gold_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps({"cases": [{"id": "draft", "reference_status": "automatic"}]}))
            with self.assertRaisesRegex(ValueError, "gold"):
                evaluate.evaluate(path)


if __name__ == "__main__":
    unittest.main()
