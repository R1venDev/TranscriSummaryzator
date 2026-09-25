"""Synthetic-only checks for the production record-delete wrapper."""

import base64
from contextlib import closing
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest

import record_dashboard_wrapper as delete


ORIGIN = "http://127.0.0.1:8765"
PASSWORD = "synthetic-test-password"


class Handler:
    def __init__(self, payload, *, authorization=True, origin=ORIGIN, host="127.0.0.1:8765"):
        self.client_address = ("127.0.0.1", 54321)
        self.path = delete.ROUTE
        self.headers = {
            "Host": host, "Origin": origin, "X-Requested-With": delete.REQUEST_HEADER,
            "Content-Type": "application/json", "Content-Length": str(len(payload)),
        }
        if authorization:
            token = base64.b64encode(b"admin:" + PASSWORD.encode()).decode("ascii")
            self.headers["Authorization"] = "Basic " + token
        self.rfile = BytesIO(payload)
        self.wfile = BytesIO()
        self.status = None
        self.response_headers = {}

    def send_response(self, status):
        self.status = status

    def send_header(self, name, value):
        self.response_headers[name] = value

    def end_headers(self):
        pass

    def body(self):
        return json.loads(self.wfile.getvalue())


class DeleteWrapperTests(unittest.TestCase):
    def setUp(self):
        delete.AUTH_FAILURES.clear()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for relative in ("inbox", "work/jobs", "outputs", "state"):
            (self.root / relative).mkdir(parents=True)
        self.pipeline = SimpleNamespace(
            ROOT=self.root, INBOX=self.root / "inbox", JOBS=self.root / "work/jobs",
            OUTPUTS=self.root / "outputs", STATE=self.root / "state",
            DB_PATH=self.root / "state/queue.sqlite3",
        )

        def snapshot(db):
            count = db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            (self.root / "state/progress.json").write_text(json.dumps({"count": count}))

        self.pipeline.write_status_snapshot = snapshot
        db = sqlite3.connect(self.pipeline.DB_PATH)
        db.execute("""CREATE TABLE jobs (
            id INTEGER PRIMARY KEY, fingerprint TEXT UNIQUE, source_path TEXT,
            original_name TEXT, status TEXT, stage TEXT, job_dir TEXT,
            output_dir TEXT, summary_status TEXT, created_at TEXT, updated_at TEXT)""")
        db.commit(); db.close()
        salt = b"synthetic-salt!!"
        verifier = "scrypt$16384$8$1${}${}".format(
            base64.urlsafe_b64encode(salt).decode().rstrip("="),
            base64.urlsafe_b64encode(hashlib.scrypt(PASSWORD.encode(), salt=salt, n=16384, r=8, p=1)).decode().rstrip("="),
        )
        admin = self.root / "state/record_delete_admin.json"
        admin.write_text(json.dumps({"version": 1, "origin": ORIGIN, "verifier": verifier}))
        admin.chmod(0o600)
        self.source = self.root / "inbox/record.mkv"
        self.source.write_bytes(b"synthetic media")
        self.work = self.root / "work/jobs/job-1"
        self.work.mkdir()
        (self.work / "job.json").write_text("{}")
        self.output = self.root / "outputs/record-result"
        self.output.mkdir()
        (self.output / "transcript.json").write_text("{}")
        self.created_at = "2026-09-15T12:00:00+00:00"
        self._insert()

    def tearDown(self):
        self.temp.cleanup()

    def _insert(self, status="done", summary_status="failed", source=None):
        with closing(sqlite3.connect(self.pipeline.DB_PATH)) as db:
            with db:
                db.execute("""INSERT INTO jobs
                (id,fingerprint,source_path,original_name,status,stage,job_dir,
                 output_dir,summary_status,created_at,updated_at)
                 VALUES (1,?,?,?,?,?,?,?,?,?,?)""",
                    ("f" * 64, str(source or self.source), "record.mkv", status, status,
                     str(self.work), str(self.output), summary_status, self.created_at, self.created_at))

    def _payload(self, **changes):
        value = {"id": 1, "name": "record.mkv", "created_at": self.created_at}
        value.update(changes)
        return json.dumps(value).encode()

    def _row_count(self):
        with closing(sqlite3.connect(self.pipeline.DB_PATH)) as db:
            return db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]

    def test_requires_admin_and_exact_origin_before_deletion(self):
        unauthenticated = Handler(self._payload(), authorization=False)
        delete.handle_delete(unauthenticated, self.pipeline)
        self.assertEqual(unauthenticated.status, 401)
        self.assertIn("WWW-Authenticate", unauthenticated.response_headers)
        bad_origin = Handler(self._payload(), origin="https://foreign.example")
        delete.handle_delete(bad_origin, self.pipeline)
        self.assertEqual(bad_origin.status, 403)
        self.assertEqual(self._row_count(), 1)
        self.assertTrue(self.source.exists())

    def test_deletes_only_named_terminal_record_and_updates_snapshot(self):
        stale = Handler(self._payload(name="other.mkv"))
        delete.handle_delete(stale, self.pipeline)
        self.assertEqual(stale.status, 409)
        handler = Handler(self._payload())
        delete.handle_delete(handler, self.pipeline)
        self.assertEqual(handler.status, 200, handler.body())
        self.assertFalse(handler.body()["cleanup_pending"])
        self.assertEqual(self._row_count(), 0)
        self.assertFalse(self.source.exists())
        self.assertFalse(self.work.exists())
        self.assertFalse(self.output.exists())
        self.assertEqual(json.loads((self.root / "state/progress.json").read_text())["count"], 0)
        receipts = list((self.root / "state/record_delete/receipts").glob("*.json"))
        self.assertEqual(len(receipts), 1)
        self.assertEqual(json.loads(receipts[0].read_text())["status"], "deleted")
        repeated = Handler(self._payload())
        delete.handle_delete(repeated, self.pipeline)
        self.assertEqual(repeated.status, 404)

    def test_active_and_external_paths_are_refused(self):
        with closing(sqlite3.connect(self.pipeline.DB_PATH)) as db:
            with db:
                db.execute("UPDATE jobs SET summary_status='pending_batch'")
        active = Handler(self._payload())
        delete.handle_delete(active, self.pipeline)
        self.assertEqual(active.status, 409)
        with closing(sqlite3.connect(self.pipeline.DB_PATH)) as db:
            with db:
                db.execute("UPDATE jobs SET summary_status='failed', source_path=?", (str(self.root / "elsewhere.mkv"),))
        external = Handler(self._payload())
        delete.handle_delete(external, self.pipeline)
        self.assertEqual(external.status, 409)
        self.assertEqual(self._row_count(), 1)
        self.assertTrue(self.work.exists())

    def test_running_speech_and_short_body_are_refused(self):
        with closing(sqlite3.connect(self.pipeline.DB_PATH)) as db:
            with db:
                db.execute("UPDATE jobs SET status='running'")
        running = Handler(self._payload())
        delete.handle_delete(running, self.pipeline)
        self.assertEqual(running.status, 409)
        short = Handler(self._payload())
        short.headers["Content-Length"] = str(len(self._payload()) + 7)
        delete.handle_delete(short, self.pipeline)
        self.assertEqual(short.status, 400)
        self.assertEqual(self._row_count(), 1)

    def test_recovers_staged_files_if_database_transaction_rolled_back(self):
        delete._private_dir(self.root / "state/record_delete")
        delete._private_dir(self.root / "state/record_delete/staging")
        staging = self.root / "state/record_delete/staging" / ("ab" * 16)
        staging.mkdir(parents=True, mode=0o700)
        manifest = {"version": 1, "transaction_id": staging.name, "job_id": 1,
                    "fingerprint": "f" * 64, "name": "record.mkv", "created_at": self.created_at,
                    "paths": [{"slot": "source", "original": str(self.source), "existed": True}]}
        (staging / "manifest.json").write_text(json.dumps(manifest))
        os.replace(self.source, staging / "source")
        outcomes = delete.recover_staged_deletions(self.pipeline)
        self.assertIn("1:restored", outcomes)
        self.assertTrue(self.source.exists())
        self.assertFalse(staging.exists())

    def test_committed_staging_is_purged_even_if_job_id_is_reused(self):
        staging = delete._private_dir(self.root / "state/record_delete/staging") / ("cd" * 16)
        staging.mkdir(mode=0o700)
        manifest = {"version": 1, "transaction_id": staging.name, "job_id": 1,
                    "fingerprint": "f" * 64, "name": "record.mkv", "created_at": self.created_at,
                    "paths": [{"slot": "source", "original": str(self.source), "existed": True}]}
        (staging / "manifest.json").write_text(json.dumps(manifest))
        os.replace(self.source, staging / "source")
        with closing(sqlite3.connect(self.pipeline.DB_PATH)) as db:
            with db:
                db.execute("""CREATE TABLE record_deletions (
                    transaction_id TEXT PRIMARY KEY, job_id INTEGER NOT NULL, fingerprint TEXT NOT NULL,
                    created_at TEXT NOT NULL, committed_at TEXT NOT NULL)""")
                db.execute("INSERT INTO record_deletions VALUES (?,?,?,?,?)",
                           (staging.name, 1, "f" * 64, self.created_at, self.created_at))
                db.execute("DELETE FROM jobs WHERE id=1")
                db.execute("""INSERT INTO jobs (id,fingerprint,source_path,original_name,status,stage,job_dir,
                           output_dir,summary_status,created_at,updated_at) VALUES (1,?,?,?,?,?,?,?,?,?,?)""",
                           ("g" * 64, str(self.source), "new.mkv", "done", "done", str(self.work),
                            str(self.output), "failed", "2026-09-16T12:00:00+00:00", self.created_at))
        outcomes = delete.recover_staged_deletions(self.pipeline)
        self.assertIn("1:purged", outcomes)
        self.assertFalse(staging.exists())
        self.assertFalse(self.source.exists())
        self.assertEqual(self._row_count(), 1)


if __name__ == "__main__":
    unittest.main()
