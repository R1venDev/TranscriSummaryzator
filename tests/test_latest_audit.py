import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from semantics.questions import normalize_slot, verify_slot_entailment
from summary.verifier import has_english_prose, partition_verified_public_items, public_surface_text, publication_audit, sanitize_public_surface, substantive_unverified_surface, verify_generated_items
from scripts.diagnostics import _safe, summarize


class LatestAuditRegressionTests(unittest.TestCase):
    def test_typed_slots_close_direct_answers(self):
        yes = {"text": "Да.", "speech_act": "answer"}
        self.assertTrue(verify_slot_entailment(["should_Misha_make_OrderBlock_labeler"], yes)["passed"])
        window = {"text": "Рабочее окно с 16:30 до 18:00", "speech_act": "answer"}
        self.assertTrue(verify_slot_entailment(["диапазон времени"], window, {"text": "Какой рабочий диапазон?"})["passed"])
        self.assertEqual(normalize_slot("result on higher timeframes"), "implementation_state")
        cross_day = {"text": "Она закрывается автоматически после 00:00.", "speech_act": "answer"}
        self.assertTrue(verify_slot_entailment(["cross_day_closure_feasibility"], cross_day)["passed"])

    def test_publication_gate_rejects_surface_regressions(self):
        base = {"section": "minutes", "text": "Order Block размечается.", "claim_ids": ["C1"], "evidence_ids": ["U1"], "source_word_ids": ["W1"]}
        report = {"audits": [{"passed": True, "errors": []}]}
        artifact = "# Итоги\n\n### 00:01:00–00:01:00 — SM (Structure Maker)\nDiscussed potential goal for future work\nassigned_pending\n"
        audit = publication_audit(report, artifact, [base], {"view_plans": {}})
        self.assertGreater(audit["rendered_english_prose"], 0)
        self.assertGreater(audit["invented_acronym_expansions"], 0)
        self.assertGreater(audit["internal_labels_exposed"], 0)
        self.assertGreater(audit["zero_duration_chapters"], 0)
        self.assertFalse(audit["passed"])

    def test_handles_are_not_mistaken_for_english_prose(self):
        self.assertFalse(has_english_prose("@Yachoy подготовит TradingView — исполнитель: @Yachoy"))
        source = "Discussed potential goal: creating a baseline solution with winrate around 30–40%."
        self.assertEqual(sanitize_public_surface(source), source)
        self.assertTrue(has_english_prose(source))

    def test_internal_labels_and_bare_acknowledgements_are_not_public_surfaces(self):
        claim = {"statement": "@A link_stop_loss_to_projection_extremes", "evidence_ids": []}
        self.assertEqual(public_surface_text(claim), "")
        self.assertFalse(substantive_unverified_surface("Угу."))
        self.assertFalse(substantive_unverified_surface("Да, на индексах в AM."))
        self.assertTrue(substantive_unverified_surface("Размер имбаланса в источнике не подтверждён."))

    def test_rejected_public_items_are_quarantined_with_full_audit(self):
        items = [{"public_id": "PI1"}, {"public_id": "PI2"}]
        report = {"audits": [
            {"passed": True, "errors": []},
            {"passed": False, "errors": ["qa_slot_failure"]},
        ]}
        retained, rejected = partition_verified_public_items(items, report)
        self.assertEqual([item["public_id"] for item in retained], ["PI1"])
        self.assertEqual(rejected[0]["public_item"]["public_id"], "PI2")
        self.assertEqual(rejected[0]["audit"]["errors"], ["qa_slot_failure"])

    def test_group_plan_time_scope_does_not_leak_between_claims(self):
        plan = {
            "claim_ids": ["C1", "C2"], "relation_ids": [], "allowed_numbers": ["1"],
            "allowed_relation_markers": [], "allowed_speakers": [], "allowed_assignees": [],
            "polarity": ["positive", "positive"], "modality": ["certain", "certain"],
            "conditions": [], "time_scope": ["1 месяц"],
        }
        claims = [
            {"claim_id": "C1", "statement": "Можно проверить спотовую стратегию.", "lifecycle": "active"},
            {"claim_id": "C2", "statement": "Нужны данные за месяц.", "time_scope": "1 месяц", "lifecycle": "active"},
        ]
        item = {"section": "minutes", "text": claims[0]["statement"], "claim_ids": ["C1"]}
        result = verify_generated_items([item], [plan], claims)
        self.assertTrue(result["passed"], result)
        month_item = {"section": "minutes", "text": claims[1]["statement"], "claim_ids": ["C2"]}
        result = verify_generated_items([month_item], [plan], claims)
        self.assertTrue(result["passed"], result)

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
            from pipeline import REQUIRED_GENERATION_FILES, current_summary_output
        except ModuleNotFoundError:
            self.skipTest("local interpreter lacks production dependencies")
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); gid = "20260915-120000-abcdef123456"
            target = base / "summary_generations" / gid; target.mkdir(parents=True)
            body = b"ok"; (target / "summary.md").write_bytes(body)
            for name in REQUIRED_GENERATION_FILES - {"summary.md"}:
                path = target / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("{}")
            digests = {name: hashlib.sha256((target / name).read_bytes()).hexdigest() for name in REQUIRED_GENERATION_FILES}
            (target / "generation_manifest.json").write_text(json.dumps({"generation_id": gid, "artifact_sha256": digests}))
            (base / "summary_current.json").write_text(json.dumps({"generation_id": gid}))
            self.assertEqual(current_summary_output(base), target)
            (target / "summary.md").write_text("corrupt")
            self.assertIsNone(current_summary_output(base))


if __name__ == "__main__":
    unittest.main()
