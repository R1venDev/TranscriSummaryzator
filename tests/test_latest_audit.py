import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from semantics.questions import normalize_slot, verify_slot_entailment
from summary.verifier import publication_audit, verify_generated_items
from scripts.diagnostics import _safe, summarize


class LatestAuditRegressionTests(unittest.TestCase):
    def test_typed_slots_close_direct_answers(self):
        yes = {"text": "Да.", "speech_act": "answer"}
        self.assertTrue(verify_slot_entailment(["should_Misha_make_OrderBlock_labeler"], yes)["passed"])
        window = {"text": "Рабочее окно с 16:30 до 18:00", "speech_act": "answer"}
        self.assertTrue(verify_slot_entailment(["диапазон времени"], window, {"text": "Какой рабочий диапазон?"})["passed"])
        self.assertEqual(normalize_slot("result on higher timeframes"), "implementation_status")

    def test_actor_swap_is_rejected_outside_task_view(self):
        claim = {"claim_id": "C1", "statement": "@A должен доставить документ для @B", "speaker_refs": ["@A", "@B"]}
        plan = {"claim_ids": ["C1"], "relation_ids": [], "allowed_numbers": [], "allowed_relation_markers": [], "allowed_speakers": ["@A", "@B"], "allowed_assignees": ["@A"], "polarity": ["positive"], "modality": ["certain"], "conditions": []}
        item = {"public_id": "PI1", "section": "minutes", "text": "@B должен доставить документ для @A", "claim_ids": ["C1"]}
        result = verify_generated_items([item], [plan], [claim])
        self.assertFalse(result["passed"])
        self.assertIn("actor_recipient_swap", result["audits"][0]["errors"])

    def test_cross_view_state_conflict_requires_aspects(self):
        items = [
            {"section": "decisions", "text": "Согласована параллельная работа", "claim_ids": ["C1"], "evidence_ids": ["U1"], "source_word_ids": ["W1"], "social_state": "accepted"},
            {"section": "tasks", "text": "Предложена параллельная работа", "claim_ids": ["C1"], "evidence_ids": ["U1"], "source_word_ids": ["W1"], "social_state": "assigned_pending", "task_state_id": "T1", "task_state": {"status": "assigned_pending", "deliverable": "работа"}},
        ]
        audit = publication_audit({"audits": [{"passed": True, "errors": []}] * 2}, "# d — x\n", items, {"view_plans": {}})
        self.assertEqual(audit["cross_view_state_conflicts"], 1)

    def test_safe_diagnostics_keep_counts_and_hashes(self):
        value = _safe({"utterances": 4, "transcript_sha256": "a" * 64, "transcript": "private"})
        self.assertEqual(value["utterances"], 4)
        self.assertEqual(value["transcript_sha256"], "a" * 64)
        self.assertEqual(value["transcript"], "[REDACTED]")

    def test_recovered_terminal_is_not_reported_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "d.jsonl"
            rows = [
                {"attempt_id": "A", "timestamp": "1", "severity": "ERROR", "outcome": "failed_terminal", "name": "summary_job", "component": "x", "category": "stage"},
                {"attempt_id": "A", "timestamp": "2", "severity": "INFO", "outcome": "completed", "name": "summary_job", "component": "x", "category": "stage"},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.assertIsNone(summarize(path)["last_fatal_error"])

    def test_reader_hash_validation(self):
        try:
            from pipeline import current_summary_output
        except ModuleNotFoundError:
            self.skipTest("local interpreter lacks production dependencies")
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); gid = "20260915-120000-abcdef123456"
            target = base / "summary_generations" / gid; target.mkdir(parents=True)
            body = b"ok"; (target / "summary.md").write_bytes(body)
            (target / "generation_manifest.json").write_text(json.dumps({"generation_id": gid, "artifact_sha256": {"summary.md": hashlib.sha256(body).hexdigest()}}))
            (base / "summary_current.json").write_text(json.dumps({"generation_id": gid}))
            self.assertEqual(current_summary_output(base), target)
            (target / "summary.md").write_text("corrupt")
            self.assertIsNone(current_summary_output(base))


if __name__ == "__main__":
    unittest.main()
