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


class FakeKeyResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, _):
        return json.dumps({"data": {"workspace_id": "0df9e665-d932-5740-b2c7-b52af166bc11", "limit": 1.0,
                                    "limit_remaining": 0.8, "limit_reset": "weekly",
                                    "usage_weekly": 0.2}}).encode()


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


if __name__ == "__main__":
    unittest.main()
