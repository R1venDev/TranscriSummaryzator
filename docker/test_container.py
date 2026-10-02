"""Offline checks for persistence, secret handling, and transcript import."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "docker" / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


entrypoint = module("container_entrypoint", "entrypoint.py")
importer = module("container_importer", "import-transcript.py")


class ContainerPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.environment = mock.patch.dict(os.environ, {
            "TRANSCRI_DATA_DIR": str(self.root / "data"),
            "TRANSCRI_CONFIG_FILE": str(self.root / "data" / "config.json"),
            "TRANSCRI_SECRETS_DIR": str(self.root / "secrets"),
            "TRANSCRI_IMAGE_VARIANT": "summary",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def bootstrap(self, **kwargs):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            entrypoint.bootstrap(**kwargs)
        return output.getvalue()

    def test_bootstrap_keeps_keys_and_custom_config(self):
        initial = self.bootstrap()
        self.assertIn("Password (shown once):", initial)
        master = self.root / "secrets/summary-master.key"
        admin = self.root / "secrets/admin.scrypt"
        saved = master.read_bytes(), admin.read_bytes()
        self.assertEqual(master.stat().st_mode & 0o777, 0o600)
        self.assertEqual(admin.stat().st_mode & 0o777, 0o600)
        config_path = self.root / "data/config.json"
        config = json.loads(config_path.read_text())
        config["summary_project_name"] = "Custom project"
        config_path.write_text(json.dumps(config))
        repeated = self.bootstrap()
        self.assertNotIn("Password (shown once):", repeated)
        self.assertEqual((master.read_bytes(), admin.read_bytes()), saved)
        self.assertEqual(json.loads(config_path.read_text())["summary_project_name"], "Custom project")
        self.assertFalse(json.loads(config_path.read_text())["processing_enabled"])

    def test_lost_master_is_not_silently_replaced(self):
        self.bootstrap()
        credential_db = self.root / "data/state/summary_private/credentials.sqlite3"
        credential_db.touch()
        master = self.root / "secrets/summary-master.key"
        master.unlink()
        with self.assertRaisesRegex(RuntimeError, "restore its master key"):
            self.bootstrap()
        self.assertFalse(master.exists())

    def test_speech_update_preserves_paused_processing(self):
        self.bootstrap()
        config_path = self.root / "data/config.json"
        self.assertFalse(json.loads(config_path.read_text())["processing_enabled"])
        with mock.patch.dict(os.environ, {"TRANSCRI_IMAGE_VARIANT": "speech"}):
            self.bootstrap()
            self.assertFalse(json.loads(config_path.read_text())["processing_enabled"])
            self.bootstrap(enable_speech=True)
            self.assertTrue(json.loads(config_path.read_text())["processing_enabled"])

    def test_reset_admin_keeps_encryption_key(self):
        self.bootstrap()
        master = self.root / "secrets/summary-master.key"
        before = master.read_bytes()
        admin = self.root / "secrets/admin.scrypt"
        prior = admin.read_bytes()
        self.bootstrap(reset_admin=True)
        self.assertEqual(before, master.read_bytes())
        self.assertNotEqual(prior, admin.read_bytes())

    def test_import_preserves_source_and_does_not_enqueue_paid_work(self):
        self.bootstrap()
        data = self.root / "data"
        raw = json.dumps({"source": "test.wav", "duration_seconds": 2,
            "speakers": {"S1": "Test"}, "utterances": [
                {"speaker": "S1", "start": 0, "end": 2, "text": "Synthetic test."}]}).encode()
        with mock.patch.multiple(importer.pipeline, STATE=data / "state",
                DB_PATH=data / "state/queue.sqlite3", JOBS=data / "work/jobs",
                OUTPUTS=data / "outputs", INBOX=data / "inbox"):
            first = importer.import_transcript(raw, "Test meeting")
            second = importer.import_transcript(raw, "Test meeting")
            self.assertTrue(first["imported"])
            self.assertFalse(second["imported"])
            self.assertEqual(first["job_id"], second["job_id"])
            db = importer.pipeline.connect()
            try:
                row = db.execute("SELECT * FROM jobs").fetchone()
            finally:
                db.close()
            self.assertEqual(row["stage"], "imported_transcript")
            self.assertEqual(row["summary_status"], "not_started")
            self.assertEqual((Path(row["output_dir"]) / "transcript.json").read_bytes(), raw)
            with self.assertRaises(ValueError):
                importer.import_transcript(b'{"utterances": []}', "Invalid")


if __name__ == "__main__":
    unittest.main()
