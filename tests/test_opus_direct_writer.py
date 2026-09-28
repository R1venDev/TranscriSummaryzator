"""No-network direct Opus comparison through the real ledger and scheduler."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from summary.luna_v1.engine import poll_once
from summary.luna_v1.contract import SCHEMA
from summary.luna_v1.ledger import (Ledger, OPUS_DIRECT_KIND, OPUS_DIRECT_POLICY,
                                    OPUS_DIRECT_POLICY_V1)
from summary.luna_v1.source import load_source
from summary.opus_v1.batch import BatchError, MODEL, Reply, canonical_request
from summary.opus_v1.direct_writer import authorize, submit
from tests.test_opus_engine import _SavedKeys, _source


def _document():
    return {
        "schema_version": "luna_summary_v1",
        "meeting": {"topic": "Проверка данных", "project": None},
        "main": [{"text": "Участники предложили проверить один вариант после получения данных.",
                  "source_ids": ["U00001", "U00003"]}],
        "timecodes": [{"topic": "Варианты проверки", "start_id": "U00001",
                       "end_id": "U00003"}],
        "tasks": [{"title": "Проверить X или Y",
                   "description": "После получения исходных данных проверить один выбранный вариант X или Y.",
                   "discussion_status": "proposed", "assignee": None, "due": None,
                   "priority": None, "recipient": None,
                   "source_ids": ["U00001", "U00003"],
                   "field_sources": {"action": ["U00001"], "assignee": [], "due": [],
                                     "priority": [], "recipient": [],
                                     "discussion_status": ["U00001"]}}],
        "questions": [{"text": "Когда получим исходные данные?", "source_ids": ["U00002"]}],
        "technical": [], "ideas": [], "verification": [],
        "chapters": [{"topic": "Проверка данных", "start_id": "U00001",
                      "end_id": "U00005", "summary": "Обсудили альтернативы и ожидание данных.",
                      "source_ids": ["U00001", "U00002", "U00003", "U00005"],
                      "details": []}],
    }


class _Client:
    posts = 0
    submissions = {}
    mode = "valid"

    def __init__(self, token):
        assert token == "synthetic-openrouter-judge"

    def submit(self, custom_id, request_body, *, allow_prompt_json=False):
        assert allow_prompt_json is True
        canonical_request(request_body, allow_prompt_json=True)
        type(self).posts += 1
        batch_id = f"batch_direct_{type(self).posts:03d}"
        type(self).submissions[batch_id] = (custom_id, request_body)
        if type(self).mode == "unknown":
            raise BatchError(None, "transport_unknown")
        return Reply(202, {"id": batch_id, "status": "validating",
                           "model": MODEL, "endpoint": "/v1/chat/completions"})

    def get(self, batch_id):
        custom_id, request = type(self).submissions[batch_id]
        document = _document() if type(self).mode != "invalid" else {"wrong": True}
        message = ({"content": ("{invalid" if type(self).mode == "invalid_json"
                                else json.dumps(document, ensure_ascii=False))}
                   if type(self).mode != "invalid_message" else ["malformed"])
        body = {"model": MODEL, "choices": [{"finish_reason": "stop",
                                              "message": message}]}
        return Reply(200, {"id": batch_id, "status": "completed", "model": MODEL,
                           "endpoint": "/v1/chat/completions",
                           "usage": {"cost": 0.025, "prompt_tokens": 500,
                                     "completion_tokens": 1000},
                           "results": [{"custom_id": custom_id, "error": None,
                                        "response": {"status_code": 200, "body": body}}]})

    def delete(self, batch_id):
        return Reply(200, {"id": batch_id, "deleted": True})


def _route(_client, request, *, max_output_tokens, allow_prompt_json=False):
    assert max_output_tokens == 24_000
    assert allow_prompt_json is True
    assert "response_format" not in request
    canonical_request(request, allow_prompt_json=True)
    return SimpleNamespace(
        workspace_id="judge-workspace", reserve_microusd=lambda **kw: 95_000,
        prompt_usd_per_token="0.000002", completion_usd_per_token="0.00001",
        request_usd="0")


class DirectOpusWriterTests(unittest.TestCase):
    def setUp(self):
        _Client.posts = 0
        _Client.submissions = {}
        _Client.mode = "valid"

    def _paths(self, root: Path):
        source_dir = root / "meeting"
        transcript = _source(source_dir)
        return transcript, root / "isolated-comparison", root / "private"

    def _authorize(self, transcript, output, private):
        return authorize(transcript_path=transcript, output_dir=output,
                         private_root=private, run_id="opus-direct-v2-test-001",
                         authorization_ref="user-20260928-opus-direct-v2-trial-1p6",
                         store_factory=lambda _root: _SavedKeys())

    def _submit(self, transcript, output, private):
        return submit(transcript_path=transcript, output_dir=output,
                      private_root=private, run_id="opus-direct-v2-test-001",
                      client_factory=_Client, store_factory=lambda _root: _SavedKeys(),
                      route_factory=_route)

    def test_historical_v1_cap_remains_readable_without_reusing_authorization(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(Path(tmp))
            ledger.db.execute("""INSERT INTO weekly_spending_authorizations
                (semantic_key,source_sha256,output_dir,quality_policy_version,
                 authorization_ref,cap_microusd,authorized_at,root_job_id)
                VALUES (?,?,?,?,?,?,?,?)""",
                ("a" * 64, "b" * 64, "/old/isolated", OPUS_DIRECT_POLICY_V1,
                 "historical-v1", 1_600_000, 1.0, "old-root"))
            self.assertEqual(ledger.weekly_cap_microusd("old-root"), 1_600_000)
            ledger.close()

    def test_exact_authorization_one_post_full_source_terminal_and_no_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            transcript, output, private = self._paths(Path(tmp))
            blocked = self._submit(transcript, output, private)
            self.assertEqual(blocked["reason"], "direct_opus_authorization_missing")
            self.assertEqual(_Client.posts, 0)
            auth = self._authorize(transcript, output, private)
            self.assertTrue(auth["created"])
            self.assertEqual(auth["weekly_cap_microusd"], 1_600_000)
            self.assertFalse(self._authorize(transcript, output, private)["created"])
            started = self._submit(transcript, output, private)
            self.assertEqual(started["status"], "submitted")
            again = self._submit(transcript, output, private)
            self.assertEqual(again["job_id"], started["job_id"])
            self.assertEqual(_Client.posts, 1)
            submitted = next(iter(_Client.submissions.values()))[1]
            self.assertEqual(OPUS_DIRECT_POLICY, "claude_opus_5_5_direct_writer_v2_prompt_json")
            self.assertNotIn("response_format", submitted)
            with self.assertRaisesRegex(ValueError, "structured_output_required"):
                canonical_request(submitted)
            system = submitted["messages"][0]["content"]
            schema_text = system.split("<output_schema_json>\n", 1)[1].split(
                "\n</output_schema_json>", 1)[0]
            self.assertEqual(json.loads(schema_text), SCHEMA)
            self.assertEqual(json.loads(submitted["messages"][1]["content"]),
                             json.loads(load_source(transcript)[0]))
            self.assertNotIn("DRAFT_DOCUMENT", submitted["messages"][1]["content"])
            ledger = Ledger(private)
            job = ledger.get(started["job_id"])
            self.assertEqual(job["kind"], OPUS_DIRECT_KIND)
            self.assertEqual(ledger.consumer_rows(job["semantic_key"]), [])
            ledger.db.execute("UPDATE jobs SET next_poll_at=0 WHERE id=?", (job["id"],))
            ledger.close()
            with patch("summary.luna_v1.engine._credential_store", return_value=_SavedKeys()):
                outcome = poll_once(private_root=private, opus_client_factory=_Client)
            self.assertTrue(any(item["status"] == "direct_complete" for item in outcome))
            ledger = Ledger(private)
            done = ledger.get(job["id"])
            ledger.close()
            artifacts = Path(done["artifact_dir"])
            self.assertEqual(done["status"], "direct_complete")
            self.assertEqual(done["billed_microusd"], 25_000)
            self.assertTrue((artifacts / "native_response.json").is_file())
            self.assertTrue((artifacts / "summary.md").is_file())
            self.assertTrue((artifacts / "batch_terminal.json").is_file())
            self.assertTrue((artifacts / "batch_delete.json").is_file())
            self.assertFalse((output / "summary_current.json").exists())
            self.assertFalse((output / "tasks.sqlite3").exists())
            self.assertEqual(_Client.posts, 1)

    def test_unknown_post_is_never_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            transcript, output, private = self._paths(Path(tmp))
            self._authorize(transcript, output, private)
            _Client.mode = "unknown"
            first = self._submit(transcript, output, private)
            self.assertEqual(first["status"], "submission_unknown")
            second = self._submit(transcript, output, private)
            self.assertEqual(second["status"], "submission_unknown")
            self.assertEqual(_Client.posts, 1)
            ledger = Ledger(private)
            job = ledger.get(first["job_id"])
            ledger.close()
            self.assertEqual(job["reserved_microusd"], 95_000)
            self.assertIsNone(job["billed_microusd"])
            self.assertFalse((output / "summary_current.json").exists())

    def test_crash_after_reservation_reseals_without_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            transcript, output, private = self._paths(Path(tmp))
            self._authorize(transcript, output, private)
            with patch("summary.opus_v1.direct_writer._pin", side_effect=OSError("synthetic crash")):
                with self.assertRaisesRegex(OSError, "synthetic crash"):
                    self._submit(transcript, output, private)
            self.assertEqual(_Client.posts, 0)
            ledger = Ledger(private)
            row = ledger.db.execute("SELECT * FROM jobs WHERE kind=?", (OPUS_DIRECT_KIND,)).fetchone()
            self.assertEqual(row["status"], "reserved")
            ledger.close()
            result = self._submit(transcript, output, private)
            self.assertEqual(result["status"], "submitted")
            self.assertEqual(result["job_id"], row["id"])
            self.assertEqual(_Client.posts, 1)

    def test_invalid_schema_retains_raw_without_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            transcript, output, private = self._paths(Path(tmp))
            self._authorize(transcript, output, private)
            _Client.mode = "invalid"
            started = self._submit(transcript, output, private)
            ledger = Ledger(private)
            ledger.db.execute("UPDATE jobs SET next_poll_at=0 WHERE id=?", (started["job_id"],))
            ledger.close()
            with patch("summary.luna_v1.engine._credential_store", return_value=_SavedKeys()):
                outcomes = poll_once(private_root=private, opus_client_factory=_Client)
            self.assertTrue(any(item["status"] == "failed_validation" for item in outcomes))
            ledger = Ledger(private)
            job = ledger.get(started["job_id"])
            ledger.close()
            self.assertEqual(job["status"], "failed_validation")
            self.assertTrue((Path(job["artifact_dir"]) / "native_response.json").is_file())
            self.assertFalse((output / "summary_current.json").exists())

    def test_invalid_json_retains_exact_raw_without_retry_or_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            transcript, output, private = self._paths(Path(tmp))
            self._authorize(transcript, output, private)
            _Client.mode = "invalid_json"
            started = self._submit(transcript, output, private)
            ledger = Ledger(private)
            ledger.db.execute("UPDATE jobs SET next_poll_at=0 WHERE id=?", (started["job_id"],))
            ledger.close()
            with patch("summary.luna_v1.engine._credential_store", return_value=_SavedKeys()):
                outcomes = poll_once(private_root=private, opus_client_factory=_Client)
            self.assertTrue(any(item["status"] == "failed_validation" for item in outcomes))
            ledger = Ledger(private)
            job = ledger.get(started["job_id"])
            ledger.close()
            artifacts = Path(job["artifact_dir"])
            self.assertEqual(job["status"], "failed_validation")
            self.assertEqual(json.loads((artifacts / "native_response.json").read_text())["text"],
                             "{invalid")
            self.assertFalse((artifacts / "candidate_document.json").exists())
            self.assertFalse((output / "summary_current.json").exists())
            self.assertEqual(self._submit(transcript, output, private)["status"],
                             "failed_validation")
            self.assertEqual(_Client.posts, 1)

    def test_malformed_message_is_failed_validation_without_scheduler_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            transcript, output, private = self._paths(Path(tmp))
            self._authorize(transcript, output, private)
            _Client.mode = "invalid_message"
            started = self._submit(transcript, output, private)
            ledger = Ledger(private)
            ledger.db.execute("UPDATE jobs SET next_poll_at=0 WHERE id=?", (started["job_id"],))
            ledger.close()
            with patch("summary.luna_v1.engine._credential_store", return_value=_SavedKeys()):
                outcomes = poll_once(private_root=private, opus_client_factory=_Client)
            self.assertTrue(any(item["status"] == "failed_validation" for item in outcomes))
            ledger = Ledger(private)
            job = ledger.get(started["job_id"])
            ledger.close()
            self.assertEqual(job["status"], "failed_validation")
            self.assertTrue((Path(job["artifact_dir"]) / "batch_terminal.json").is_file())
            self.assertFalse((output / "summary_current.json").exists())


if __name__ == "__main__":
    unittest.main()
