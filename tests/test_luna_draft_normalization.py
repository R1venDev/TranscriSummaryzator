"""Synthetic checks for source-metadata normalization before the quality stage."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from summary.luna_v1 import load_source, validate_document
from summary.luna_v1.batch import Reply
from summary.luna_v1.engine import (INVENTORY_V2_QUALITY_POLICY_VERSION,
                                    _advance_quality, _normalize_null_optional_field_sources,
                                    poll_once, submit)
from summary.luna_v1.ledger import Ledger
from tests.test_gemini_engine import FakeGemini, TwoRoleStore
from tests.test_luna_contract import _document
from tests.test_luna_engine import FakeClient, arm_poll


def _source(path: Path) -> dict:
    path.write_text(json.dumps({
        "source": "01.01.2030 — Учебная встреча.mkv", "duration_seconds": 12,
        "speakers": {"p1": "А", "p2": "Б"},
        "utterances": [
            {"start": 0.2, "end": 5.0, "speaker": "p1", "text": "Предлагаю проверить X или Y."},
            {"start": 5.2, "end": 11.0, "speaker": "p2", "text": "Предлагаю записать результат."},
        ],
    }, ensure_ascii=False), encoding="utf-8")
    return load_source(path)[1]


class OrphanWriter(FakeClient):
    def get(self, batch_id):
        reply = super().get(batch_id)
        body = copy.deepcopy(reply.body)
        message = body["results"][0]["response"]["body"]["choices"][0]["message"]
        document = json.loads(message["content"])
        document["tasks"][0]["field_sources"]["priority"] = ["U00002"]
        message["content"] = json.dumps(document, ensure_ascii=False)
        return Reply(reply.status_code, body)


class DraftNormalizationTests(unittest.TestCase):
    def test_only_known_orphans_of_null_optional_fields_are_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            index = _source(Path(directory) / "transcript.json")
            draft = _document()
            task = draft["tasks"][0]
            task["field_sources"]["priority"] = ["U00002"]
            original = copy.deepcopy(draft)
            normalized, removals = _normalize_null_optional_field_sources(draft, index)
            self.assertEqual(draft, original)
            self.assertEqual(removals, [{"task_index": 0, "field": "priority",
                                         "removed_source_ids": ["U00002"]}])
            self.assertEqual(normalized["tasks"][0]["field_sources"]["priority"], [])
            validate_document(normalized, index)

            for field, value, evidence in (
                ("priority", "urgent", "U00002"),
                ("priority", None, "U99999"),
                ("action", None, "U00002"),
            ):
                with self.subTest(field=field, evidence=evidence):
                    invalid = _document()
                    if field == "action":
                        invalid["tasks"][0]["field_sources"]["action"] = [evidence]
                    else:
                        invalid["tasks"][0][field] = value
                        invalid["tasks"][0]["field_sources"][field] = [evidence]
                    unchanged, changes = _normalize_null_optional_field_sources(invalid, index)
                    self.assertEqual(unchanged, invalid)
                    self.assertEqual(changes, [])
                    with self.assertRaises(ValueError):
                        validate_document(unchanged, index)

    def test_writer_raw_is_preserved_and_audit_sees_normalized_draft(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "meeting"
            output.mkdir()
            _source(output / "transcript.json")
            private = root / "private"
            writer_route = SimpleNamespace(
                reserve_microusd=lambda payload, **kwargs: 20_000,
                workspace_id="writer-workspace", prompt_usd_per_token="0.00000005",
                completion_usd_per_token="0.00000025",
                cache_write_usd_per_token="0.0000000625", request_usd="0")
            judge_route = SimpleNamespace(reserve_microusd=lambda: 25_000,
                                          input_tokens=1500, context_bound_tokens=2100,
                                          workspace_id="judge-workspace")
            OrphanWriter.submit_calls = 0
            OrphanWriter.submissions = {}
            FakeGemini.submit_calls = 0
            FakeGemini.submissions = {}
            FakeGemini.mode = "no_patch"
            with patch("summary.luna_v1.engine._credential_store", return_value=TwoRoleStore()), \
                 patch("summary.luna_v1.engine.QUALITY_POLICY_VERSION", INVENTORY_V2_QUALITY_POLICY_VERSION), \
                 patch("summary.luna_v1.engine.verify_batch_route", return_value=writer_route), \
                 patch("summary.luna_v1.engine.verify_gemini_batch_route", return_value=judge_route):
                started = submit(transcript_path=output / "transcript.json", output_dir=output,
                                 private_root=private, client_factory=OrphanWriter)
                arm_poll(private)
                first = poll_once(private_root=private, client_factory=OrphanWriter,
                                  gemini_client_factory=FakeGemini)
                self.assertIn("draft_ready", {event["status"] for event in first})
                ledger = Ledger(private)
                root_job = ledger.get(started["job_id"])
                artifacts = Path(root_job["artifact_dir"])
                raw = json.loads((artifacts / "native_response.json").read_text())
                draft = json.loads((artifacts / "draft_document.json").read_text())
                sidecar = json.loads((artifacts / "draft_normalization.json").read_text())
                self.assertEqual(json.loads(raw["text"])["tasks"][0]["field_sources"]["priority"],
                                 ["U00002"])
                self.assertEqual(draft["tasks"][0]["field_sources"]["priority"], [])
                self.assertEqual(sidecar["removals"], [{"task_index": 0, "field": "priority",
                                                        "removed_source_ids": ["U00002"]}])
                self.assertEqual(json.loads((artifacts / "draft_validation.json").read_text())["status"],
                                 "normalized")
                audit = ledger.stage(started["job_id"], "audit")
                self.assertEqual(audit["status"], "submitted")
                ledger.close()
                arm_poll(private)
                second = poll_once(private_root=private, client_factory=OrphanWriter,
                                   gemini_client_factory=FakeGemini)
                self.assertIn("accepted", {event["status"] for event in second})
            self.assertEqual(OrphanWriter.submit_calls, 1)
            self.assertEqual(FakeGemini.submit_calls, 1)
            audit_request = next(iter(FakeGemini.submissions.values()))[1]
            audit_input = json.loads(audit_request["messages"][1]["content"])
            self.assertEqual(audit_input["DRAFT_DOCUMENT"]["tasks"][0]["field_sources"]["priority"], [])

    def test_zero_patch_audit_reports_preexisting_invalid_draft(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "meeting"
            output.mkdir()
            index = _source(output / "transcript.json")
            draft = _document()
            draft["tasks"][0]["field_sources"]["priority"] = ["U00002"]
            artifacts = root / "artifacts"
            artifacts.mkdir()
            (artifacts / "draft_document.json").write_text(json.dumps(draft))
            report_path = root / "report.json"
            report_path.write_text(json.dumps({"patches": []}))
            job = {"id": "synthetic", "artifact_dir": str(artifacts),
                   "output_dir": str(output), "source_sha256": index["source_sha256"]}

            class StageLedger:
                def stage(self, job_id, kind):
                    return {"status": "stage_complete", "accepted_document_path": str(report_path)}

            with patch("summary.luna_v1.engine.apply_audit", side_effect=ValueError("invalid draft")), \
                 patch("summary.luna_v1.engine._finalize_quality", return_value={"status": "captured"}) as finalize:
                result = _advance_quality(StageLedger(), job, FakeClient)
            self.assertEqual(result["status"], "captured")
            self.assertEqual(finalize.call_args.kwargs["reason"], "invalid_draft_unrepaired")
            failure = json.loads((artifacts / "audit_apply_error.json").read_text())
            self.assertEqual(failure["failure_origin"], "invalid_draft_unrepaired")


if __name__ == "__main__":
    unittest.main()
