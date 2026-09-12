import tempfile
import unittest
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
        state = meeting_state([record(1, .5), record(2, .7)])
        self.assertEqual(state["relations"][0]["relation"], "supersedes")

    def test_config_rejects_unknown_keys(self):
        with self.assertRaises(Exception):
            PipelineConfig.model_validate({"unknown": True})

    def test_same_content_with_different_name_is_a_new_submission(self):
        digest = "a" * 64
        first = pipeline.submission_fingerprint(digest, "meeting-one.wav")
        second = pipeline.submission_fingerprint(digest, "meeting-two.wav")
        self.assertNotEqual(first, second)
        self.assertEqual(first, pipeline.submission_fingerprint(digest, "meeting-one.wav"))

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


if __name__ == "__main__":
    unittest.main()
