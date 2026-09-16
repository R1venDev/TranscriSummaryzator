"""Release checks for public generations, links and stuck stage processes."""
import hashlib
import json
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path

from pipeline import REQUIRED_GENERATION_FILES, current_summary_output, find_existing_job, run_command, submission_fingerprint
from scripts.summary_worker import build_public_document, render_public_document, time_link
from summary.planner import plan
from summary.verifier import build_public_items, publication_audit, verify_generated_items, verify_public_document


class GenerationTests(unittest.TestCase):
    def test_quarantined_claim_always_has_public_sentence_plan(self):
        claim = {
            "claim_id": "C1", "content_kind": "resource", "kind": "resource",
            "statement": "Участник предоставит файл.", "lifecycle": "active",
            "verification_status": "verification_unavailable", "start": 10, "end": 11,
            "evidence_ids": ["U1"], "risk": {}, "speaker_refs": ["@A"],
            "polarity": "positive", "modality": "certain", "conditions": [],
            "quantities": [], "entities": [],
        }
        result = plan([claim], [], [], lambda _: 1, max_units=0)
        self.assertIn("C1", result["view_plans"]["requires_verification"]["selected_claim_ids"])
        self.assertTrue(any("C1" in item["claim_ids"] for item in result["public_sentence_plans"]))

    def test_merged_task_cites_all_canonical_source_claims(self):
        claims = []
        for cid in ("C1", "C2"):
            claims.append({"claim_id": cid, "content_kind": "action", "kind": "action",
                           "statement": "Размечать Order Block параллельно.", "lifecycle": "active",
                           "verification_status": "supported", "canonical_task_state_id": "TS1",
                           "evidence_ids": ["U1"], "source_word_ids": ["W1"], "start": 10,
                           "speaker_refs": ["@A"], "social_state": "assigned_pending"})
        graph = {"claims": claims, "task_states": [{"task_id": "TS1", "status": "assigned_pending",
                 "deliverable": "Размечать Order Block параллельно.", "assignee": "@A",
                 "evidence_ids": ["U1"], "source_word_ids": ["W1"]}]}
        summary_plan = {"view_plans": {"tasks": {"selected_claim_ids": ["C1", "C2"]}},
                        "public_sentence_plans": [{"claim_ids": ["C1", "C2"], "relation_ids": []}]}
        task_items = [item for item in build_public_items(graph, summary_plan) if item["section"] == "tasks"]
        self.assertEqual(len(task_items), 1)
        self.assertEqual(set(task_items[0]["claim_ids"]), {"C1", "C2"})

    def test_residual_question_does_not_inherit_answer_or_schedule_negation(self):
        plan = {"claim_ids": ["C1"], "relation_ids": [], "allowed_numbers": [],
                "allowed_relation_markers": [], "allowed_speakers": ["@Riven"],
                "allowed_assignees": [], "polarity": ["negative"],
                "modality": ["possible"], "conditions": [], "time_scope": []}
        claim = {"claim_id": "C1", "statement": "Точное время не подтверждено.",
                 "speaker_refs": ["@Riven"], "lifecycle": "active"}
        question = {"text": "@Riven спрашивает: подтвердить точное время",
                    "claim_ids": ["C1"], "section": "questions",
                    "question_state": {"status": "partially_answered",
                                       "original_question": claim["statement"],
                                       "remaining_question": "подтвердить точное время"}}
        assertion = {"text": "Точное время подтверждено.",
                     "claim_ids": ["C1"], "section": "minutes"}
        self.assertTrue(verify_generated_items([question], [plan], [claim])["passed"])
        self.assertFalse(verify_generated_items([assertion], [plan], [claim])["passed"])

    def test_public_quality_distinguishes_cross_view_reuse_from_duplicates(self):
        item = {"text": "Подготовить TradingView.", "claim_ids": ["C1"],
                "evidence_ids": ["U1"], "source_word_ids": ["W1"]}
        report = publication_audit({"audits": []}, "# Итоги\n", [
            {**item, "section": "overview"}, {**item, "section": "minutes"}], {})
        self.assertEqual(report["duplicate_items"], 0)
        self.assertEqual(report["cross_view_repetitions"], 1)
        repeated = publication_audit({"audits": []}, "# Итоги\n", [
            {**item, "section": "minutes"}, {**item, "section": "minutes"}], {})
        self.assertEqual(repeated["duplicate_items"], 1)

    def test_pending_task_removes_conflicting_agreement_wording(self):
        claim = {"claim_id": "C1", "content_kind": "action", "kind": "action",
                 "statement": "Предлагалось участники договорились параллельно размечать имбалансы и Order Block.",
                 "lifecycle": "active", "verification_status": "supported",
                 "canonical_task_state_id": "TS1", "evidence_ids": ["U1"],
                 "source_word_ids": ["W1"], "start": 10,
                 "speaker_refs": ["@A"], "social_state": "candidate"}
        state = {"task_id": "TS1", "status": "assigned_pending",
                 "deliverable": claim["statement"], "assignee": "@A",
                 "evidence_ids": ["U1"], "source_word_ids": ["W1"]}
        items = build_public_items(
            {"claims": [claim], "task_states": [state]},
            {"view_plans": {"tasks": {"selected_claim_ids": ["C1"]}}},
        )
        task = next(item for item in items if item["section"] == "tasks")
        self.assertIn("Предлагалось параллельно размечать", task["text"])
        self.assertNotIn("участники договорились", task["text"])

    def test_confirmed_hypothesis_constraint_does_not_use_technical_budget(self):
        items = [{"section": "technical", "content_kind": "observation",
                  "social_state": "observation", "text": f"Тезис {index}."}
                 for index in range(6)]
        items.append({"section": "technical", "content_kind": "hypothesis",
                      "social_state": "constraint", "text": "Техническое ограничение."})
        report = publication_audit({"audits": []}, "# Итоги\n", items, {})
        self.assertEqual(report["excessive_technical_items"], 0)

    def test_watcher_does_not_reenqueue_web_upload_under_storage_name(self):
        db = sqlite3.connect(":memory:")
        db.row_factory = sqlite3.Row
        db.execute("""CREATE TABLE jobs (
            id INTEGER PRIMARY KEY, fingerprint TEXT, content_sha256 TEXT,
            source_path TEXT, original_name TEXT)""")
        digest = "a" * 64
        source = "/inbox/meeting-deadbeef.mkv"
        original = "Встреча команды.mkv"
        db.execute("INSERT INTO jobs VALUES (1, ?, ?, ?, ?)",
                   (submission_fingerprint(digest, original), digest, source, original))
        found = find_existing_job(db, digest, "meeting-deadbeef.mkv", source)
        self.assertEqual(found["id"], 1)

    def test_only_committed_generation_is_public(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            generation = "20260915-120000-abcdef123456"
            pending = base / "summary_generations" / (generation + ".pending")
            pending.mkdir(parents=True)
            (pending / "summary.md").write_text("partial", encoding="utf-8")
            self.assertIsNone(current_summary_output(base))
            final = pending.with_name(generation)
            pending.rename(final)
            for name in REQUIRED_GENERATION_FILES - {"summary.md"}:
                path = final / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("{}")
            digests = {name: hashlib.sha256((final / name).read_bytes()).hexdigest() for name in REQUIRED_GENERATION_FILES}
            (final / "generation_manifest.json").write_text(json.dumps({
                "generation_id": generation, "artifact_sha256": digests}), encoding="utf-8")
            (base / "summary_current.json").write_text(json.dumps({"generation_id": generation}), encoding="utf-8")
            self.assertEqual(current_summary_output(base), final)
            (base / "summary_current.json").write_text(json.dumps({"generation_id": "../unsafe"}), encoding="utf-8")
            self.assertIsNone(current_summary_output(base))

    def test_subprocess_partial_line_cannot_stall_watchdog(self):
        with tempfile.TemporaryDirectory() as directory:
            start = time.monotonic()
            command = [sys.executable, "-c", "import sys,time;sys.stdout.write('partial');sys.stdout.flush();time.sleep(8)"]
            with self.assertRaises(TimeoutError):
                run_command(command, Path(directory) / "stage.log", deadline_seconds=2, idle_seconds=.5)
            self.assertLess(time.monotonic() - start, 4)

    def test_portable_transcript_link_without_private_server_address(self):
        self.assertEqual(time_link(10.125, job_id=9), "[00:00:10](transcript.html#t-10125)")

    def test_final_document_mutations_are_rejected(self):
        item = {"public_id": "PI00001", "section": "overview", "text": "Нужно проверить качество сигналов.",
                "claim_ids": ["C1"], "evidence_ids": ["U1"], "start": 10.125}
        document = build_public_document([item], {"job_id": 9, "project": "Проект"})
        artifact = render_public_document(document)
        self.assertTrue(verify_public_document(document, artifact, [item])["passed"])
        invented = json.loads(json.dumps(document))
        invented["title"]["text"] = "Подписана сделка без согласования"
        self.assertFalse(verify_public_document(invented, artifact, [item])["passed"])
        invented = json.loads(json.dumps(document))
        invented["overview"][0]["text"] += " Сервер уничтожен."
        self.assertFalse(verify_public_document(invented, artifact, [item])["passed"])


if __name__ == "__main__":
    unittest.main()
