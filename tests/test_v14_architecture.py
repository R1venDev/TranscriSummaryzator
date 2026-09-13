import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from unittest.mock import patch

from scripts.config_schema import PipelineConfig, load_config
from scripts.evidence_ledger import attach_word_ids, ledger_document, risk_level, semantic_risks
from scripts.quality_schema import meeting_state
import pipeline


class V14ArchitectureTests(unittest.TestCase):
    def test_short_confirmation_is_semantically_critical(self):
        risks = semantic_risks("Да.")
        self.assertIn("agreement", risks)
        self.assertEqual(risk_level(risks, "decision"), "HIGH")

    def test_phrase_normalization_keeps_all_source_ids(self):
        raw = attach_word_ids([
            {"text": "ордер", "start": 0, "end": .2},
            {"text": "блок", "start": .2, "end": .4},
        ])
        normalized = [{**raw[0], "text": "Order Block", "end": .4, "source_word_ids": ["W00000001", "W00000002"], "speaker": "P1"}]
        ledger = ledger_document(raw, normalized, [{"start": 0, "end": .4, "speaker": "P1"}])
        self.assertEqual(ledger["normalized_tokens"][0]["source_word_ids"], ["W00000001", "W00000002"])

    def test_revision_supersedes_older_quantity(self):
        def record(number, value):
            return {"record_id": f"F{number:05d}", "kind": "proposal", "statement": str(value), "start": number,
                    "subject": "stop", "predicate": "set", "object": str(value), "polarity": "positive",
                    "modality": "proposed", "quantities": [{"value": str(value)}], "conditions": [],
                    "attributed_speakers": ["P1"], "assignees": [], "evidence_ids": [f"E{number}"], "uncertainty": {}}
        newer = record(2, .7)
        newer.update({"speech_act": "correct", "revision_cue": True})
        state = meeting_state([record(1, .5), newer])
        self.assertEqual(state["relations"][0]["relation"], "supersedes")

    def test_config_rejects_unknown_keys(self):
        with self.assertRaises(Exception):
            PipelineConfig.model_validate({"unknown": True})

    def test_concurrent_config_resolution_uses_unique_atomic_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "config.json"
            resolved = root / "state" / "config.resolved.json"
            source.write_text("{}\n", encoding="utf-8")
            with ThreadPoolExecutor(max_workers=12) as pool:
                results = list(pool.map(lambda _: load_config(source, resolved), range(100)))
            self.assertTrue(all(result == results[0] for result in results))
            self.assertEqual(json.loads(resolved.read_text(encoding="utf-8")), results[0])
            self.assertEqual(list(resolved.parent.glob(".config.resolved.json.*.tmp")), [])

    def test_same_content_with_different_name_is_a_new_submission(self):
        digest = "a" * 64
        first = pipeline.submission_fingerprint(digest, "meeting-one.wav")
        second = pipeline.submission_fingerprint(digest, "meeting-two.wav")
        self.assertNotEqual(first, second)
        self.assertEqual(first, pipeline.submission_fingerprint(digest, "meeting-one.wav"))

    def test_early_upload_lookup_uses_content_and_name(self):
        import sqlite3
        db = sqlite3.connect(":memory:")
        db.row_factory = sqlite3.Row
        db.execute("create table jobs (id integer, fingerprint text, content_sha256 text, original_name text)")
        digest = "b" * 64
        db.execute("insert into jobs values (2, ?, ?, ?)", (pipeline.submission_fingerprint(digest, "old.mkv"), digest, "old.mkv"))
        self.assertIsNotNone(pipeline.find_existing_job(db, digest, "old.mkv"))
        self.assertIsNone(pipeline.find_existing_job(db, digest, "renamed.mkv"))

    def test_legacy_job_is_backfilled_and_same_name_remains_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(pipeline, "STATE", root / "state"), patch.object(pipeline, "DB_PATH", root / "state/jobs.sqlite3"):
                db = pipeline.connect()
                db.execute(
                    "INSERT INTO jobs (fingerprint,source_path,original_name,status,stage,job_dir,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?)",
                    ("a" * 64, "/tmp/old.wav", "old.wav", "done", "done", "/tmp/job", "now", "now"),
                )
                db.commit()
                db.execute("UPDATE jobs SET content_sha256=NULL")
                db.commit(); db.close()
                migrated = pipeline.connect()
                row = migrated.execute("SELECT content_sha256 FROM jobs").fetchone()
                self.assertEqual(row["content_sha256"], "a" * 64)

    def test_resolved_profile_is_not_marked_for_review_by_stale_boundary_flags(self):
        turn = {"speaker": "profile:known", "flags": ["ambiguous", "no_diarization"]}
        self.assertFalse(pipeline.turn_needs_speaker_review(turn, {"profile:known": "@Misha"}))

    def test_unresolved_speaker_still_requires_review(self):
        self.assertTrue(pipeline.turn_needs_speaker_review({"speaker": None, "flags": []}, {}))
        self.assertTrue(pipeline.turn_needs_speaker_review({"speaker": "UNKNOWN_1", "flags": []}, {}))


if __name__ == "__main__":
    unittest.main()
