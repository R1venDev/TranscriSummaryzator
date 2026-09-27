"""Offline production-entry checks for the partitioned Opus Batch workflow."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlsplit

from summary.luna_v1.engine import (
    OPUS_PARTITIONED_QUALITY_POLICY_VERSION, poll_once, submit,
)
from summary.luna_v1.ledger import Ledger
from summary.luna_v1.source import load_source
from summary.opus_v1 import (BatchClient as OpusBatchClient, MODEL as OPUS_MODEL,
                             OPUS_SEGMENT_SCHEMA_ID)
from tests.test_luna_engine import FakeClient as WriterClient
from tests.test_opus_engine import (_HttpResponse, _OpusHttp, _SavedKeys,
                                    _arm, _source)


class _SegmentHttp:
    """OpenRouter transport stub with one response per durable Batch stage."""

    def __init__(self, *, fail_segment: str | None = None):
        self.fail_segment = fail_segment
        self.calls: list[tuple[str, str, dict | None]] = []
        self.submissions: dict[str, dict] = {}

    @property
    def posts(self) -> list[dict]:
        return [payload for method, path, payload in self.calls
                if method == "POST" and path == "/api/v1/batches"]

    @staticmethod
    def _report(payload: dict) -> dict:
        rows = {row["id"]: row for row in payload["TRANSCRIPT_SOURCE"]["utterances"]}
        anchors = payload["RISK_ANCHORS"]
        topic = payload["DRAFT_DOCUMENT"]["meeting"]["topic"]
        coverage = []
        for window in payload["SOURCE_WINDOWS"]:
            start, end = window["start_id"], window["end_id"]
            coverage.append({
                "window_id": window["window_id"],
                "start_id": start, "end_id": end,
                "salient": "Синтетический пример проверки источника",
                "source_quote": rows[start]["text"][:100],
                "draft_coverage": "covered",
                "draft_evidence": [{"section": "meeting", "index": 0,
                                    "quote": topic[: min(40, len(topic))]}],
                "finding_indices": [],
                "reviewed_anchor_ids": [
                    anchor["anchor_id"] for anchor in anchors
                    if start <= anchor["source_id"] <= end
                ],
            })
        return {"schema_version": OPUS_SEGMENT_SCHEMA_ID,
                "segment_id": payload["SEGMENT_ID"],
                "coverage": coverage, "findings": [], "patches": []}

    def open(self, request, timeout):
        assert timeout == 45
        assert request.get_header("Authorization") == "Bearer synthetic-openrouter-judge"
        method, path = request.get_method(), urlsplit(request.full_url).path
        body = json.loads(request.data) if request.data else None
        self.calls.append((method, path, body))
        if method == "GET" and not path.startswith("/api/v1/batches/"):
            return _HttpResponse(200, _OpusHttp._route(path))
        if method == "POST" and path == "/api/v1/batches":
            batch_id = f"batch_opus_segment_{len(self.submissions) + 1:03d}"
            self.submissions[batch_id] = body["requests"][0]
            return _HttpResponse(202, {"id": batch_id, "status": "validating",
                                       "endpoint": "/v1/chat/completions", "model": OPUS_MODEL})
        if path.startswith("/api/v1/batches/"):
            batch_id = path.rsplit("/", 1)[1]
            assert batch_id in self.submissions
            if method == "DELETE":
                return _HttpResponse(204, {})
            assert method == "GET"
            item = self.submissions[batch_id]
            chat = item["body"]
            payload = json.loads(chat["messages"][1]["content"])
            if payload["SEGMENT_ID"] == self.fail_segment:
                return _HttpResponse(200, {
                    "id": batch_id, "status": "failed", "model": OPUS_MODEL,
                    "endpoint": "/v1/chat/completions", "usage": {"cost": 0.001},
                    "request_counts": {"total": 1, "completed": 0, "failed": 1},
                })
            response = {"model": OPUS_MODEL, "choices": [{
                "finish_reason": "stop", "message": {
                    "role": "assistant", "content": json.dumps(self._report(payload),
                                                       ensure_ascii=False),
                },
            }]}
            return _HttpResponse(200, {
                "id": batch_id, "status": "completed", "model": OPUS_MODEL,
                "endpoint": "/v1/chat/completions", "usage": {"cost": 0.001},
                "request_counts": {"total": 1, "completed": 1, "failed": 0},
                "results": [{"custom_id": item["custom_id"], "error": None,
                             "response": {"status_code": 200, "body": response}}],
            })
        raise AssertionError(f"unexpected route: {method} {path}")


class OpusPartitionedEngineTests(unittest.TestCase):
    def setUp(self):
        WriterClient.submit_calls = 0
        WriterClient.submissions = {}
        WriterClient.report_factory = None

    @staticmethod
    def _start(output: Path, private: Path) -> dict:
        writer_route = SimpleNamespace(
            reserve_microusd=lambda payload, **kwargs: 20_000,
            workspace_id="writer-workspace",
            prompt_usd_per_token="0.00000005",
            completion_usd_per_token="0.00000025",
            cache_write_usd_per_token="0.0000000625", request_usd="0",
        )
        with patch("summary.luna_v1.engine._credential_store", return_value=_SavedKeys()), \
             patch("summary.luna_v1.engine.verify_batch_route",
                   return_value=writer_route):
            return submit(transcript_path=output / "transcript.json",
                          output_dir=output, private_root=private,
                          client_factory=WriterClient)

    @staticmethod
    def _poll(private: Path, http: _SegmentHttp) -> list[dict]:
        _arm(private)
        with patch("summary.luna_v1.engine._credential_store", return_value=_SavedKeys()):
            return poll_once(private_root=private, client_factory=WriterClient,
                             opus_client_factory=lambda token: OpusBatchClient(
                                 token, opener=http))

    @staticmethod
    def _published(output: Path) -> tuple[Path, dict]:
        pointer = json.loads((output / "summary_current.json").read_text(encoding="utf-8"))
        path = output / "summary_generations" / pointer["generation_id"]
        return path, json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))

    def test_three_scoped_batches_read_same_full_draft_and_reuse_after_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            output, private = Path(directory) / "meeting", Path(directory) / "private"
            _source(output)
            started = self._start(output, private)
            self.assertEqual(started["status"], "submitted")
            http = _SegmentHttp()
            for _ in range(7):
                outcomes = self._poll(private, http)
                if any(row["status"] == "accepted" for row in outcomes):
                    break
            else:
                self.fail("partitioned review did not complete")
            self.assertEqual(len(http.posts), 3)
            ledger = Ledger(private)
            root = ledger.get(started["job_id"])
            self.assertEqual(root["status"], "accepted")
            self.assertIsNone(ledger.stage(root["id"], "audit"))
            stages = [ledger.stage(root["id"], f"segment_{number}")
                      for number in (1, 2, 3)]
            self.assertEqual([stage["status"] for stage in stages],
                             ["stage_complete"] * 3)
            draft = json.loads((Path(root["artifact_dir"]) / "draft_document.json").read_text())
            plan = json.loads((Path(root["artifact_dir"]) / "opus_segment_plan.json").read_text())
            self.assertEqual(plan["version"], "opus_three_primary_parts_overlap8_v1")
            source_ids = list(load_source(output / "transcript.json")[1]["by_id"])
            primary_ids = []
            for number, post in enumerate(http.posts, 1):
                chat = post["requests"][0]["body"]
                payload = json.loads(chat["messages"][1]["content"])
                self.assertEqual(payload["DRAFT_DOCUMENT"], draft)
                self.assertEqual(payload["SEGMENT_ID"], f"S{number:02d}")
                self.assertEqual(chat["response_format"]["json_schema"]["name"],
                                 OPUS_SEGMENT_SCHEMA_ID)
                self.assertEqual(chat["max_completion_tokens"], 6_000)
                segment = plan["segments"][number - 1]
                self.assertEqual(payload["TRANSCRIPT_SOURCE"]["utterances"],
                                 segment["context_before"] +
                                 segment["primary_utterances"] +
                                 segment["context_after"])
                primary_ids.extend(row["id"] for row in segment["primary_utterances"])
            self.assertEqual(primary_ids, source_ids)
            _generation, run = self._published(output)
            self.assertEqual(run["quality_review"]["status"],
                             "opus_segment_reviewed_unverified")
            self.assertFalse(run["quality_review"]["verification_performed"])
            self.assertEqual(run["quality_review"]["segment_job_ids"],
                             [stage["id"] for stage in stages])
            ledger.close()
            self._poll(private, http)
            self.assertEqual(len(http.posts), 3)
            self.assertEqual(self._start(output, private)["status"], "accepted_cache_hit")
            self.assertEqual(WriterClient.submit_calls, 1)

    def test_failed_second_segment_publishes_unchanged_draft_with_incomplete_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            output, private = Path(directory) / "meeting", Path(directory) / "private"
            _source(output)
            started = self._start(output, private)
            http = _SegmentHttp(fail_segment="S02")
            for _ in range(6):
                outcomes = self._poll(private, http)
                if any(row["status"] == "accepted" for row in outcomes):
                    break
            else:
                self.fail("failed stage did not publish incomplete result")
            self.assertEqual(len(http.posts), 2)
            ledger = Ledger(private)
            root = ledger.get(started["job_id"])
            self.assertEqual(ledger.stage(root["id"], "segment_1")["status"],
                             "stage_complete")
            self.assertEqual(ledger.stage(root["id"], "segment_2")["status"], "remote_failed")
            self.assertIsNone(ledger.stage(root["id"], "segment_3"))
            artifacts = Path(root["artifact_dir"])
            self.assertTrue((artifacts / "opus_partition_incomplete.json").exists())
            draft = json.loads((artifacts / "draft_document.json").read_text())
            candidate = json.loads((artifacts / "candidate_document.json").read_text())
            self.assertEqual(candidate, draft)
            _generation, run = self._published(output)
            self.assertEqual(run["quality_review"]["status"], "coverage_incomplete")
            self.assertEqual(run["quality_review"]["segment_count"], 3)
            ledger.close()


if __name__ == "__main__":
    unittest.main()
