"""Real dashboard admin routes with synthetic credentials; no provider inference."""

import base64
from email.message import Message
from io import BytesIO
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from cryptography.fernet import Fernet

import pipeline
from summary_credentials import CredentialError, CredentialStore, credential_dispatch_guard, make_admin_password_hash


FAKE_KEY_A = "sk-or-v1-fake-credential-0001"
FAKE_KEY_B = "sk-or-v1-fake-credential-0002"
FAKE_GOOGLE_KEY = "synthetic-google-api-key-0001"


class FakeKeyResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, _):
        return json.dumps({"data": {"workspace_id": "0df9e665-d932-5740-b2c7-b52af166bc11", "limit": 1.0,
                                    "limit_remaining": 0.8, "limit_reset": "weekly",
                                    "usage_weekly": 0.2}}).encode()


class FakeGoogleModelsResponse(FakeKeyResponse):
    def read(self, _):
        return json.dumps({"models": [{"name": "models/synthetic-gemini"}]}).encode()


class CredentialStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = CredentialStore(self.root / "summary_private" / "credentials.sqlite3", Fernet.generate_key())

    def tearDown(self):
        self.temp.cleanup()

    def test_encrypted_store_rotation_order_and_pending_poll(self):
        a = self.store.add("Основной", FAKE_KEY_A)
        b = self.store.add("Резерв", FAKE_KEY_B)
        raw = self.store.path.read_bytes()
        self.assertNotIn(FAKE_KEY_A.encode(), raw)
        self.assertNotIn(FAKE_KEY_B.encode(), raw)
        self.assertNotIn("ciphertext", json.dumps(self.store.list()))
        self.assertNotIn(FAKE_KEY_A, json.dumps(self.store.list()))
        self.store.check(a["id"], opener=lambda *_args, **_kwargs: FakeKeyResponse())
        self.assertEqual(self.store.dispatch_candidates()[0]["id"], a["id"])
        self.assertEqual(self.store.reveal_for_dispatch(a["id"], 1), FAKE_KEY_A)
        self.store.set_enabled(a["id"], False)
        self.assertEqual(self.store.reveal_for_existing_job(a["id"], 1), FAKE_KEY_A)
        self.assertEqual(self.store.dispatch_candidates(), [])
        with self.assertRaises(CredentialError):
            self.store.delete(a["id"], active_jobs=1)
        with self.assertRaises(CredentialError):
            self.store.replace(a["id"], FAKE_KEY_B, active_jobs=1)
        self.store.set_order([b["id"], a["id"]])
        self.assertEqual(self.store.list()["keys"][0]["id"], b["id"])
        self.store.replace(a["id"], FAKE_KEY_B)
        with self.assertRaises(CredentialError):
            self.store.reveal_for_existing_job(a["id"], 1)
        self.assertFalse(self.store.delete(a["id"], active_jobs=0)["revoked_upstream"])

    def test_master_and_database_permissions(self):
        self.assertEqual(self.store.path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.store.path.parent.stat().st_mode & 0o777, 0o700)
        with sqlite3.connect(self.store.path) as db:
            events = db.execute("SELECT operation FROM credential_events").fetchall()
        self.assertEqual(events, [])

    def test_judge_key_isolated_and_paid_project_attestation_required(self):
        writer = self.store.add("Автор", FAKE_KEY_A)
        judge = self.store.add("Проверяющий", FAKE_GOOGLE_KEY, role="judge")
        requests = []

        def google_open(request, **_kwargs):
            requests.append(request)
            return FakeGoogleModelsResponse()

        checked = self.store.check(judge["id"], opener=google_open)
        self.assertEqual(checked["status"], "ready")
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].get_method(), "GET")
        self.assertEqual(requests[0].full_url,
                         "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1")
        self.assertEqual(dict((name.lower(), value) for name, value in requests[0].header_items())["x-goog-api-key"],
                         FAKE_GOOGLE_KEY)
        self.assertNotIn(FAKE_GOOGLE_KEY, requests[0].full_url)
        self.assertEqual(self.store.dispatch_candidates(role="judge"), [])
        with self.assertRaises(CredentialError):
            self.store.reveal_for_dispatch(judge["id"], 1, role="judge")
        with self.assertRaises(CredentialError):
            self.store.reveal_for_dispatch(judge["id"], 1)
        self.assertEqual(self.store.dispatch_candidates(), [])  # writer not checked yet
        self.store.check(writer["id"], opener=lambda *_args, **_kwargs: FakeKeyResponse())
        self.assertEqual([item["id"] for item in self.store.dispatch_candidates()], [writer["id"]])
        self.assertEqual(self.store.dispatch_candidates(role="judge"), [])
        with self.assertRaises(CredentialError):
            self.store.set_judge_policy(writer["id"], paid_tier_confirmed=True,
                                        project_scope="paid-project-1")
        confirmed = self.store.set_judge_policy(judge["id"], paid_tier_confirmed=True,
                                                project_scope="paid-project-1")
        self.assertTrue(confirmed["paid_tier_confirmed"])
        self.assertIsNotNone(confirmed["paid_tier_confirmed_at"])
        self.assertEqual(self.store.dispatch_candidates(role="judge")[0]["id"], judge["id"])
        self.assertEqual([item["id"] for item in self.store.dispatch_candidates()], [writer["id"]])
        with self.assertRaises(CredentialError):
            self.store.set_order([writer["id"]], role="judge")
        self.assertEqual(self.store.reveal_for_dispatch(judge["id"], 1, role="judge"), FAKE_GOOGLE_KEY)
        self.assertNotIn(FAKE_GOOGLE_KEY.encode(), self.store.path.read_bytes())
        self.assertNotIn(FAKE_GOOGLE_KEY, json.dumps(self.store.list()))
        self.store.set_judge_policy(judge["id"], paid_tier_confirmed=False, project_scope=None)
        self.assertEqual(self.store.dispatch_candidates(role="judge"), [])
        self.store.set_judge_policy(judge["id"], paid_tier_confirmed=True,
                                    project_scope="paid-project-1")
        self.store.replace(judge["id"], FAKE_GOOGLE_KEY)
        self.assertFalse(next(k for k in self.store.list()["keys"] if k["id"] == judge["id"])["paid_tier_confirmed"])

    def test_existing_database_migrates_to_writer_without_changing_ciphertext(self):
        legacy_path = self.root / "legacy" / "credentials.sqlite3"
        legacy_path.parent.mkdir(mode=0o700)
        master = Fernet.generate_key()
        ciphertext = Fernet(master).encrypt(FAKE_KEY_A.encode())
        with sqlite3.connect(legacy_path) as db:
            db.execute("""CREATE TABLE credentials (
                id TEXT PRIMARY KEY, version INTEGER NOT NULL, label TEXT NOT NULL UNIQUE COLLATE NOCASE,
                ciphertext BLOB NOT NULL, mask TEXT NOT NULL, enabled INTEGER NOT NULL,
                priority INTEGER NOT NULL, status TEXT NOT NULL, checked_at TEXT,
                workspace_id TEXT, limit_amount REAL, limit_remaining REAL,
                limit_reset TEXT, usage_weekly REAL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)""")
            db.execute("""INSERT INTO credentials
                (id,version,label,ciphertext,mask,enabled,priority,status,created_at,updated_at)
                VALUES(?,1,'Старый',?,'••••0001',1,0,'ready','old','old')""",
                ("a" * 32, ciphertext))
        os.chmod(legacy_path, 0o600)
        migrated = CredentialStore(legacy_path, master)
        self.assertEqual(migrated.dispatch_candidates()[0]["role"], "writer")
        self.assertEqual(migrated.reveal_for_existing_job("a" * 32, 1), FAKE_KEY_A)
        with sqlite3.connect(legacy_path) as db:
            self.assertEqual(db.execute("SELECT ciphertext FROM credentials").fetchone()[0], ciphertext)
            self.assertIn("role", [row[1] for row in db.execute("PRAGMA table_info(credentials)")])

    def test_dispatch_guard_serializes_separate_users_of_credential_path(self):
        started = threading.Event()
        acquired = threading.Event()

        def contender():
            started.set()
            with credential_dispatch_guard(self.store.path):
                acquired.set()

        with credential_dispatch_guard(self.store.path):
            worker = threading.Thread(target=contender, daemon=True)
            worker.start()
            self.assertTrue(started.wait(1))
            self.assertFalse(acquired.wait(0.05))
        worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertTrue(acquired.is_set())
        self.assertEqual((self.store.path.parent / "credential-dispatch.lock").stat().st_mode & 0o777, 0o600)


class AdminHttpTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = self.root / "summary_private" / "credentials.sqlite3"
        self.master = Fernet.generate_key().decode("ascii")
        self.password = "synthetic-admin-password-for-tests"
        self.verifier = make_admin_password_hash(self.password)
        self.patch_db = patch.object(pipeline, "SUMMARY_CREDENTIAL_DB", self.db)
        self.patch_db.start()
        self.patch_state = patch.object(pipeline, "STATE", self.root)
        self.patch_state.start()
        self.origin = "http://127.0.0.1:8765"
        self.patch_env = patch.dict(os.environ, {
            "TRANSCRI_SUMMARY_MASTER_KEY": self.master,
            "TRANSCRI_SUMMARY_PYTHON": sys.executable,
            "TRANSCRI_SUMMARY_ADMIN_PASSWORD_HASH": self.verifier,
            "TRANSCRI_SUMMARY_ADMIN_ORIGIN": self.origin,
        })
        self.patch_env.start()

    def tearDown(self):
        self.patch_env.stop()
        self.patch_db.stop()
        self.patch_state.stop()
        self.temp.cleanup()
        pipeline.SUMMARY_ADMIN_FAILURES.clear()

    def request(self, method, path, body=None, auth=True, origin=True, host=None):
        handler = object.__new__(pipeline.DashboardHandler)
        handler.command = method
        handler.path = path
        handler.request_version = "HTTP/1.1"
        handler.requestline = method + " " + path + " HTTP/1.1"
        handler.client_address = ("127.0.0.1", 12345)
        handler.wfile = BytesIO()
        handler.headers = Message()
        headers = {"Host": host or "127.0.0.1:8765"}
        if auth:
            basic = base64.b64encode(("admin:" + self.password).encode()).decode()
            headers["Authorization"] = "Basic " + basic
        if body is not None:
            headers["Content-Type"] = "application/json"
            headers["X-Requested-With"] = "TranscriSummaryzator-Admin"
            if origin:
                headers["Origin"] = self.origin
            body = json.dumps(body)
            headers["Content-Length"] = str(len(body.encode("utf-8")))
        for name, value in headers.items():
            handler.headers[name] = value
        handler.rfile = BytesIO(body.encode("utf-8") if body is not None else b"")
        if method == "GET":
            handler.do_GET()
        else:
            handler.do_POST()
        response_headers, payload = handler.wfile.getvalue().split(b"\r\n\r\n", 1)
        lines = response_headers.decode("latin1").split("\r\n")
        status = int(lines[0].split()[1])
        metadata = dict(line.split(": ", 1) for line in lines[1:] if ": " in line)
        return status, payload, metadata

    def test_admin_auth_csrf_host_and_masking(self):
        status, _, _ = self.request("GET", "/api/summary/credentials", auth=False)
        self.assertEqual(status, 401)
        status, _, _ = self.request("POST", "/api/summary/credentials/add", {"label": "Первый", "key": FAKE_KEY_A}, origin=False)
        self.assertEqual(status, 403)
        status, _, _ = self.request("POST", "/api/summary/credentials/add", {"label": "Первый", "key": FAKE_KEY_A}, host="attacker.invalid")
        self.assertEqual(status, 403)
        status, raw, headers = self.request("POST", "/api/summary/credentials/add", {"label": "Первый", "key": FAKE_KEY_A})
        self.assertEqual(status, 201)
        self.assertNotIn(FAKE_KEY_A.encode(), raw)
        self.assertEqual(headers["Cache-Control"], "no-store, private")
        status, raw, _ = self.request("GET", "/api/summary/credentials")
        self.assertEqual(status, 200)
        self.assertNotIn(FAKE_KEY_A.encode(), raw)
        self.assertEqual(len(json.loads(raw)["keys"]), 1)
        status, page, _ = self.request("GET", "/summary-settings")
        self.assertEqual(status, 200)
        self.assertIn("OpenRouter".encode(), page)
        status, script, _ = self.request("GET", "/summary-settings.js")
        self.assertEqual(status, 200)
        self.assertIn(b"TranscriSummaryzator-Admin", script)

    def test_fail_closed_when_admin_not_configured(self):
        with patch.dict(os.environ, {"TRANSCRI_SUMMARY_ADMIN_PASSWORD_HASH": ""}):
            status, _, _ = self.request("GET", "/api/summary/credentials")
        self.assertEqual(status, 503)

    def test_replace_delete_use_persistent_job_count(self):
        status, raw, _ = self.request("POST", "/api/summary/credentials/add", {"label": "Первый", "key": FAKE_KEY_A})
        self.assertEqual(status, 201)
        identifier = json.loads(raw)["result"]["id"]
        status, raw, _ = self.request("POST", "/api/summary/credentials/replace", {"id": identifier, "key": FAKE_KEY_B})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw)["result"]["version"], 2)
        status, raw, _ = self.request("POST", "/api/summary/credentials/delete", {"id": identifier})
        self.assertEqual(status, 200)
        self.assertFalse(json.loads(raw)["result"]["revoked_upstream"])

    def test_judge_admin_role_and_attestation_protected(self):
        body = {"role": "judge", "label": "Gemini judge", "key": FAKE_GOOGLE_KEY}
        status, _, _ = self.request("POST", "/api/summary/credentials/add", body, auth=False)
        self.assertEqual(status, 401)
        status, _, _ = self.request("POST", "/api/summary/credentials/add", body, origin=False)
        self.assertEqual(status, 403)
        status, raw, _ = self.request("POST", "/api/summary/credentials/add", body)
        self.assertEqual(status, 201)
        self.assertNotIn(FAKE_GOOGLE_KEY.encode(), raw)
        identifier = json.loads(raw)["result"]["id"]
        policy = {"id": identifier, "paid_tier_confirmed": True, "project_scope": "paid-project-1"}
        status, _, _ = self.request("POST", "/api/summary/credentials/judge-policy", policy, origin=False)
        self.assertEqual(status, 403)
        status, raw, _ = self.request("POST", "/api/summary/credentials/judge-policy", policy)
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(raw)["result"]["paid_tier_confirmed"])
        status, raw, _ = self.request("GET", "/api/summary/credentials")
        self.assertEqual(status, 200)
        self.assertNotIn(FAKE_GOOGLE_KEY.encode(), raw)
        self.assertEqual(json.loads(raw)["keys"][0]["role"], "judge")
        status, page, _ = self.request("GET", "/summary-settings")
        self.assertEqual(status, 200)
        self.assertIn("Google Gemini".encode(), page)


if __name__ == "__main__":
    unittest.main()
