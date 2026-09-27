"""Production-entry continuation of a saved writer; transport and source are synthetic.

The first quality response is deliberately unusable, as a Batch result can be
after consuming tokens. A later quality-only run must preserve that receipt and
the original writer output while sharing the original logical budget.
"""
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from summary.gemini_v1.batch import Reply as GeminiReply
from summary.luna_v1.batch import Reply as LunaReply
from summary.luna_v1.engine import (SEGMENT_REVIEW_QUALITY_POLICY_VERSION,
                                     poll_once, resume_saved_draft, submit)
from summary.luna_v1.ledger import Ledger
from tests.test_gemini_engine import TwoRoleStore
from tests.test_gemini_segment_engine import SegmentGemini
from tests.test_luna_engine import FakeClient, arm_poll


class RecipientCitationWriter(FakeClient):
    """A saved historical draft with a known citation omitted from task navigation."""

    def get(self, batch_id):
        reply = super().get(batch_id)
        body = copy.deepcopy(reply.body)
        message = body["results"][0]["response"]["body"]["choices"][0]["message"]
        document = json.loads(message["content"])
        document["tasks"][0]["recipient"] = "Б"
        document["tasks"][0]["field_sources"]["recipient"] = ["U00002"]
        message["content"] = json.dumps(document, ensure_ascii=False)
        return LunaReply(reply.status_code, body)


class TruncatedThenValidGemini(SegmentGemini):
    failed_remote_id = None

    def submit(self, custom_id, request_body):
        reply = super().submit(custom_id, request_body)
        if self.__class__.failed_remote_id is None:
            self.__class__.failed_remote_id = reply.body["id"]
        return reply

    def get(self, identifier):
        reply = super().get(identifier)
        if identifier != self.__class__.failed_remote_id:
            return reply
        body = copy.deepcopy(reply.body)
        choice = body["results"][0]["response"]["body"]["choices"][0]
        choice["finish_reason"] = "length"
        choice["message"]["content"] = '{"schema_version":'
        return GeminiReply(reply.status_code, body)


class RecipientCitationChapterWriter(RecipientCitationWriter):
    """Add a chapter detail whose tense needs a source check after reuse."""

    def get(self, batch_id):
        reply = super().get(batch_id)
        body = copy.deepcopy(reply.body)
        message = body["results"][0]["response"]["body"]["choices"][0]["message"]
        document = json.loads(message["content"])
        document["chapters"][0]["details"] = [{
            "text": "Результат уже записан в журнал.", "source_ids": ["U00002"],
        }]
        message["content"] = json.dumps(document, ensure_ascii=False)
        return LunaReply(reply.status_code, body)


class MisassignedCoverageGemini(SegmentGemini):
    """Complete v3 report with one wrong coverage link, or two if ambiguous."""

    ambiguous = False

    def get(self, identifier):
        reply = super().get(identifier)
        custom_id, request = self.__class__.submissions[identifier]
        payload = json.loads(request["messages"][1]["content"])
        if payload["MODE"] != "segment_review" or payload["SOURCE_SEGMENT"]["segment_id"] != "S01":
            return reply
        primary = payload["SOURCE_SEGMENT"]["primary_utterances"]
        windows = payload["SOURCE_SEGMENT"]["coverage_windows"]
        assert len(primary) == len(windows) == 3
        body = copy.deepcopy(reply.body)
        message = body["results"][0]["response"]["body"]["choices"][0]["message"]
        report = json.loads(message["content"])
        assert report["schema_version"] == "gemini_segment_review_v3"
        report["items"] = [
            {"k": "action", "c": "Предложено проверить X или Y, один вариант.",
             "s": [primary[0]["id"]], "m": "proposed", "a": None, "r": None,
             "if": None, "or": ["X", "Y"], "fix": [], "v": "represented",
             "t": ["tasks:0"], "f": []},
            {"k": "action", "c": "Б обещал записать результат после проверки.",
             "s": [primary[1]["id"]], "m": "committed", "a": "Б", "r": None,
             "if": "после проверки", "or": [], "fix": [], "v": "partial",
             "t": ["chapters:0"], "f": [0]},
            {"k": "context", "c": "Проверять будут один вариант.",
             "s": [primary[2]["id"]], "m": "observation", "a": None, "r": None,
             "if": None, "or": [], "fix": [], "v": "represented",
             "t": ["tasks:0"], "f": []},
        ]
        report["coverage"] = [{
            "window_id": window["window_id"], "start_id": window["start_id"],
            "end_id": window["end_id"], "assessment": "material_items",
            "item_indices": [index],
        } for index, window in enumerate(windows)]
        report["coverage"][1]["item_indices"].insert(0, 0)
        if self.__class__.ambiguous:
            report["coverage"][2]["item_indices"].insert(0, 0)
        report["draft_assessments"] = [{
            "unit_id": "chapters:0:detail:0", "status": "unsupported",
            "source_ids": [primary[1]["id"]], "item_indices": [1],
            "finding_indices": [0],
        }]
        report["findings"] = [{
            "severity": "major", "kind": "modality",
            "description": "Реплика обещает записать результат; деталь ошибочно сообщает, что он уже записан.",
            "source_ids": [primary[1]["id"]],
            "affected": [{"section": "chapters", "index": 0}],
            "status": "unresolved", "patch_indices": [],
        }]
        report["patches"] = []
        message["content"] = json.dumps(report, ensure_ascii=False)
        return GeminiReply(reply.status_code, body)


def _source(path: Path) -> None:
    path.write_text(json.dumps({
        "source": "01.01.2030 — Учебная встреча.mkv", "duration_seconds": 12,
        "speakers": {"p1": "А", "p2": "Б"},
        "utterances": [
            {"start": 0.2, "end": 5.0, "speaker": "p1",
             "text": "Предлагаю проверить X или Y, не оба."},
            {"start": 5.2, "end": 11.0, "speaker": "p2",
             "text": "Передайте мне результат проверки; я запишу его в журнал."},
        ],
    }, ensure_ascii=False), encoding="utf-8")


def _source_six(path: Path) -> None:
    """Three primary utterances per segment, hence three coverage windows."""
    path.write_text(json.dumps({
        "source": "01.01.2030 — Учебная встреча.mkv", "duration_seconds": 32,
        "speakers": {"p1": "А", "p2": "Б"},
        "utterances": [
            {"start": 0.2, "end": 4.0, "speaker": "p1",
             "text": "Предлагаю проверить X или Y, один вариант, не оба."},
            {"start": 4.2, "end": 8.0, "speaker": "p2",
             "text": "Передайте мне результат проверки; я запишу его в журнал."},
            {"start": 8.2, "end": 12.0, "speaker": "p1",
             "text": "Согласен проверить только один вариант."},
            {"start": 12.2, "end": 17.0, "speaker": "p1",
             "text": "Ещё предлагаю записать результат проверки в журнал."},
            {"start": 17.2, "end": 23.0, "speaker": "p2",
             "text": "Срок следующего обсуждения определим позднее."},
            {"start": 23.2, "end": 31.0, "speaker": "p1",
             "text": "После записи результата вернёмся к вариантам."},
        ],
    }, ensure_ascii=False), encoding="utf-8")


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SavedWriterContinuationTests(unittest.TestCase):
    def _saved_report_with_bad_coverage(self, root: Path) -> tuple[Path, Path, dict, dict, dict]:
        """A failed writer, then an accepted continuation with an invalid report."""
        output = root / "meeting"
        output.mkdir()
        _source_six(output / "transcript.json")
        private = root / "private"
        RecipientCitationChapterWriter.submit_calls = 0
        RecipientCitationChapterWriter.submissions = {}
        MisassignedCoverageGemini.submit_calls = 0
        MisassignedCoverageGemini.submissions = {}
        started = submit(transcript_path=output / "transcript.json",
                         output_dir=output, private_root=private,
                         quality_policy_version=SEGMENT_REVIEW_QUALITY_POLICY_VERSION,
                         client_factory=RecipientCitationChapterWriter)
        self.assertEqual(started["status"], "submitted")
        with patch("summary.luna_v1.engine._normalize_known_field_source_membership",
                   side_effect=lambda draft, index: (draft, [])):
            arm_poll(private)
            first = poll_once(private_root=private,
                client_factory=RecipientCitationChapterWriter,
                gemini_client_factory=MisassignedCoverageGemini)
            self.assertIn("draft_ready", {event["status"] for event in first})
        # Reproduce a historical failed report saved before the narrowly
        # scoped form normalizer existed. The new continuation uses real code.
        with patch("summary.luna_v1.engine.normalize_v3_segment_review_report",
                   side_effect=lambda report, *args: (report, [])):
            for _ in range(4):
                arm_poll(private)
                poll_once(private_root=private,
                    client_factory=RecipientCitationChapterWriter,
                    gemini_client_factory=MisassignedCoverageGemini)
                ledger = Ledger(private)
                first_child = ledger.stage(started["job_id"], "segment_1")
                ledger.close()
                if first_child is not None and first_child["status"] == "failed_validation":
                    break
        ledger = Ledger(private)
        prior = ledger.get(started["job_id"])
        child = ledger.stage(prior["id"], "segment_1")
        self.assertEqual(prior["status"], "failed_validation")
        self.assertEqual(child["status"], "failed_validation")
        self.assertEqual(prior["dispatches"], 1)
        self.assertEqual(child["dispatches"], 1)
        self.assertIn("coverage", child["error_code"])
        self.assertEqual(MisassignedCoverageGemini.submit_calls, 1)
        ledger.close()
        first_continuation = resume_saved_draft(prior_job_id=prior["id"],
            private_root=private, gemini_client_factory=MisassignedCoverageGemini)
        self.assertEqual(first_continuation["status"], "quality_pending")
        with patch("summary.luna_v1.engine.normalize_v3_segment_review_report",
                   side_effect=lambda report, *args: (report, [])):
            for _ in range(4):
                arm_poll(private)
                poll_once(private_root=private,
                    client_factory=RecipientCitationChapterWriter,
                    gemini_client_factory=MisassignedCoverageGemini)
                ledger = Ledger(private)
                previous = ledger.get(first_continuation["job_id"])
                ledger.close()
                if previous["status"] == "accepted":
                    break
        self.assertEqual(previous["status"], "accepted")
        ledger = Ledger(private)
        reusable_child = ledger.stage(previous["id"], "segment_1")
        self.assertIsNotNone(reusable_child)
        self.assertEqual(reusable_child["status"], "failed_validation")
        self.assertIn("coverage", reusable_child["error_code"])
        ledger.close()
        self.assertEqual(MisassignedCoverageGemini.submit_calls, 2)
        self.assertTrue((output / "summary_current.json").is_file())
        return output, private, dict(prior), dict(previous), dict(reusable_child)

    def test_complete_failed_segment_is_reused_without_new_segment_one_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer_route = SimpleNamespace(
                reserve_microusd=lambda payload, **kwargs: 20_000,
                workspace_id="writer-workspace", prompt_usd_per_token="0.00000005",
                completion_usd_per_token="0.00000025",
                cache_write_usd_per_token="0.0000000625", request_usd="0")
            judge_route = SimpleNamespace(
                reserve_microusd=lambda: 15_000, input_tokens=1500,
                context_bound_tokens=2100, workspace_id="judge-workspace")
            MisassignedCoverageGemini.ambiguous = False
            with patch("summary.luna_v1.engine._credential_store", return_value=TwoRoleStore()), \
                 patch("summary.luna_v1.engine.verify_batch_route", return_value=writer_route), \
                 patch("summary.luna_v1.engine.verify_gemini_batch_route", return_value=judge_route):
                output, private, prior, previous, old_stage = self._saved_report_with_bad_coverage(root)
                old_dir = Path(old_stage["artifact_dir"])
                prior_dir = Path(prior["artifact_dir"])
                sealed = {str(path): _file_sha(path) for path in (
                    prior_dir / "manifest.json", prior_dir / "draft_document.json",
                    prior_dir / "native_response.json", prior_dir / "batch_terminal.json",
                    old_dir / "manifest.json", old_dir / "request.json",
                    old_dir / "native_response.json", old_dir / "batch_terminal.json",
                )}
                old_rows = {row["id"]: (row["status"], row["remote_id"],
                    row["billed_microusd"], row["dispatches"], row["error_code"])
                    for row in (prior, previous, old_stage)}
                # The real degraded generation from the earlier continuation
                # remains readable during a quality-only resume.
                pointer = output / "summary_current.json"
                accepted = output / "summary_generations" / json.loads(
                    pointer.read_text())["generation_id"]
                pointer_sha = _file_sha(pointer)
                accepted_sha = _file_sha(accepted / "summary.md")

                resumed = resume_saved_draft(prior_job_id=prior["id"],
                    private_root=private, reused_segment_stage_id=old_stage["id"],
                    gemini_client_factory=MisassignedCoverageGemini)
                self.assertEqual(resumed["status"], "quality_pending")
                self.assertEqual(resumed["new_writer_generations"], 0)
                same = resume_saved_draft(prior_job_id=prior["id"],
                    private_root=private, reused_segment_stage_id=old_stage["id"],
                    gemini_client_factory=MisassignedCoverageGemini)
                self.assertEqual(same["job_id"], resumed["job_id"])
                self.assertEqual(RecipientCitationChapterWriter.submit_calls, 1)
                self.assertEqual(MisassignedCoverageGemini.submit_calls, 2)

                ledger = Ledger(private)
                continuation = ledger.get(resumed["job_id"])
                self.assertEqual(continuation["status"], "quality_pending")
                self.assertEqual(continuation["dispatches"], 0)
                self.assertIsNone(continuation["remote_id"])
                self.assertIsNone(ledger.stage(continuation["id"], "segment_1"))
                self.assertIsNone(ledger.stage(continuation["id"], "segment_2"))
                self.assertEqual(continuation["billing_group_id"], prior["billing_group_id"])
                group_dispatches = ledger.db.execute(
                    "SELECT SUM(dispatches) FROM jobs WHERE billing_group_id=?",
                    (prior["billing_group_id"],)).fetchone()[0]
                self.assertEqual(group_dispatches, 3)
                manifest = json.loads((Path(continuation["artifact_dir"]) /
                    "manifest.json").read_text())
                self.assertEqual(manifest["reused_segment_1"]["source_stage_job_id"],
                                 old_stage["id"])
                reused = json.loads((Path(continuation["artifact_dir"]) /
                    "reused_segment_1_report.json").read_text())
                provenance = json.loads((Path(continuation["artifact_dir"]) /
                    "reused_segment_1_provenance.json").read_text())
                self.assertTrue(any(warning["code"] == "coverage_pointer_relocated"
                                    for warning in provenance["coverage_warnings"]))
                self.assertEqual(reused["coverage"][1]["item_indices"], [1])
                self.assertEqual(reused["draft_assessments"][0]["unit_id"],
                                 "chapters:0:detail:0")
                self.assertEqual(reused["draft_assessments"][0]["status"], "unsupported")
                rows = ledger.db.execute(
                    "SELECT id,status,remote_id,billed_microusd,dispatches,error_code "
                    "FROM jobs WHERE id IN (?,?,?)",
                    (prior["id"], previous["id"], old_stage["id"])).fetchall()
                self.assertEqual({row["id"]: (row["status"], row["remote_id"],
                    row["billed_microusd"], row["dispatches"], row["error_code"])
                    for row in rows}, old_rows)
                ledger.close()
                self.assertEqual({path: _file_sha(Path(path)) for path in sealed}, sealed)
                self.assertEqual(_file_sha(pointer), pointer_sha)

                arm_poll(private)
                poll_once(private_root=private,
                    client_factory=RecipientCitationChapterWriter,
                    gemini_client_factory=MisassignedCoverageGemini)
                self.assertEqual(MisassignedCoverageGemini.submit_calls, 3)
                ledger = Ledger(private)
                self.assertIsNone(ledger.stage(continuation["id"], "segment_1"))
                self.assertIsNotNone(ledger.stage(continuation["id"], "segment_2"))
                self.assertEqual(ledger.stage(continuation["id"], "segment_2")["dispatches"], 1)
                ledger.close()
                self.assertEqual(_file_sha(pointer), pointer_sha)
                self.assertEqual(_file_sha(accepted / "summary.md"), accepted_sha)
                self.assertEqual({path: _file_sha(Path(path)) for path in sealed}, sealed)

    def test_reuse_rejects_changed_source_and_ambiguous_coverage(self):
        writer_route = SimpleNamespace(
            reserve_microusd=lambda payload, **kwargs: 20_000,
            workspace_id="writer-workspace", prompt_usd_per_token="0.00000005",
            completion_usd_per_token="0.00000025",
            cache_write_usd_per_token="0.0000000625", request_usd="0")
        judge_route = SimpleNamespace(
            reserve_microusd=lambda: 15_000, input_tokens=1500,
            context_bound_tokens=2100, workspace_id="judge-workspace")
        for reason, ambiguous in (("source", False), ("ambiguous", True),
                                  ("route", False), ("model", False)):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as directory:
                MisassignedCoverageGemini.ambiguous = ambiguous
                with patch("summary.luna_v1.engine._credential_store", return_value=TwoRoleStore()), \
                     patch("summary.luna_v1.engine.verify_batch_route", return_value=writer_route), \
                     patch("summary.luna_v1.engine.verify_gemini_batch_route", return_value=judge_route):
                    output, private, prior, previous, old_stage = self._saved_report_with_bad_coverage(Path(directory))
                    if reason == "source":
                        source = output / "transcript.json"
                        changed = json.loads(source.read_text())
                        changed["utterances"][0]["text"] += " Изменён источник."
                        source.write_text(json.dumps(changed, ensure_ascii=False), encoding="utf-8")
                    elif reason == "route":
                        path = Path(old_stage["artifact_dir"]) / "manifest.json"
                        changed = json.loads(path.read_text())
                        changed["provider_only"] = ["other"]
                        path.write_text(json.dumps(changed), encoding="utf-8")
                    elif reason == "model":
                        path = Path(old_stage["artifact_dir"]) / "submit_response.json"
                        changed = json.loads(path.read_text())
                        changed["model"] = "other-model"
                        path.write_text(json.dumps(changed), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        resume_saved_draft(prior_job_id=prior["id"],
                            private_root=private, reused_segment_stage_id=old_stage["id"],
                            gemini_client_factory=MisassignedCoverageGemini)
                    ledger = Ledger(private)
                    self.assertEqual(ledger.db.execute(
                        "SELECT COUNT(*) FROM jobs WHERE writer_parent_id=?",
                        (prior["id"],)).fetchone()[0], 1)
                    self.assertEqual(ledger.get(old_stage["id"])["status"], "failed_validation")
                    ledger.close()
                    self.assertEqual(RecipientCitationChapterWriter.submit_calls, 1)
                    self.assertEqual(MisassignedCoverageGemini.submit_calls, 2)

    def test_failed_writer_is_reused_once_and_published_after_sealed_resume(self):
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
            judge_route = SimpleNamespace(
                reserve_microusd=lambda: 15_000, input_tokens=1500,
                context_bound_tokens=2100, workspace_id="judge-workspace")
            RecipientCitationWriter.submit_calls = 0
            RecipientCitationWriter.submissions = {}
            TruncatedThenValidGemini.submit_calls = 0
            TruncatedThenValidGemini.submissions = {}
            TruncatedThenValidGemini.failed_remote_id = None

            with patch("summary.luna_v1.engine._credential_store", return_value=TwoRoleStore()), \
                 patch("summary.luna_v1.engine.verify_batch_route", return_value=writer_route), \
                 patch("summary.luna_v1.engine.verify_gemini_batch_route", return_value=judge_route):
                started = submit(transcript_path=output / "transcript.json",
                                 output_dir=output, private_root=private,
                                 client_factory=RecipientCitationWriter,
                                 quality_policy_version=SEGMENT_REVIEW_QUALITY_POLICY_VERSION)
                self.assertEqual(started["status"], "submitted")
                # Reproduce the earlier code path, before the deterministic
                # citation membership repair was added. The new code is then
                # used normally for the saved-draft continuation below.
                with patch("summary.luna_v1.engine._normalize_known_field_source_membership",
                           side_effect=lambda draft, index: (draft, [])):
                    arm_poll(private)
                    first = poll_once(private_root=private,
                        client_factory=RecipientCitationWriter,
                        gemini_client_factory=TruncatedThenValidGemini)
                    self.assertIn("draft_ready", {event["status"] for event in first})
                    self.assertEqual(TruncatedThenValidGemini.submit_calls, 1)
                    arm_poll(private)
                    second = poll_once(private_root=private,
                        client_factory=RecipientCitationWriter,
                        gemini_client_factory=TruncatedThenValidGemini)
                    self.assertIn("failed_validation", {event["status"] for event in second})

                ledger = Ledger(private)
                prior = ledger.get(started["job_id"])
                first_child = ledger.stage(prior["id"], "segment_1")
                self.assertEqual(prior["status"], "failed_validation")
                self.assertEqual(first_child["status"], "failed_validation")
                self.assertEqual(prior["dispatches"], 1)
                self.assertEqual(first_child["dispatches"], 1)
                self.assertIsNotNone(prior["billed_microusd"])
                self.assertIsNotNone(first_child["billed_microusd"])
                prior_dir = Path(prior["artifact_dir"])
                child_dir = Path(first_child["artifact_dir"])
                sealed_before = {str(path): _file_sha(path) for path in (
                    prior_dir / "manifest.json", prior_dir / "request.json",
                    prior_dir / "submit_response.json", prior_dir / "batch_terminal.json",
                    prior_dir / "native_response.json", prior_dir / "draft_document.json",
                    child_dir / "request.json", child_dir / "batch_terminal.json",
                )}
                old_rows = {row["id"]: (row["status"], row["remote_id"],
                    row["billed_microusd"], row["dispatches"], row["error_code"])
                    for row in (prior, first_child)}
                ledger.close()
                self.assertFalse((output / "summary_current.json").exists())

                # A crash after pinning the continuation, before opening it
                # to the scheduler, must not create an incomplete active job.
                with patch.object(Ledger, "mark_continuation_ready",
                                  side_effect=RuntimeError("synthetic setup crash")):
                    with self.assertRaisesRegex(RuntimeError, "synthetic setup crash"):
                        resume_saved_draft(prior_job_id=prior["id"],
                            private_root=private,
                            gemini_client_factory=TruncatedThenValidGemini)
                ledger = Ledger(private)
                continuation = ledger.db.execute(
                    "SELECT * FROM jobs WHERE writer_parent_id=?", (prior["id"],)).fetchone()
                self.assertIsNotNone(continuation)
                self.assertEqual(continuation["status"], "preparing")
                continuation_id = continuation["id"]
                ledger.close()
                idle = poll_once(private_root=private,
                    client_factory=RecipientCitationWriter,
                    gemini_client_factory=TruncatedThenValidGemini)
                self.assertFalse(any(event.get("job_id") == continuation_id for event in idle))
                self.assertEqual(TruncatedThenValidGemini.submit_calls, 1)

                resumed = resume_saved_draft(prior_job_id=prior["id"],
                    private_root=private,
                    gemini_client_factory=TruncatedThenValidGemini)
                self.assertEqual(resumed["status"], "quality_pending")
                self.assertEqual(resumed["job_id"], continuation_id)
                self.assertEqual(resumed["new_writer_generations"], 0)
                same = resume_saved_draft(prior_job_id=prior["id"],
                    private_root=private,
                    gemini_client_factory=TruncatedThenValidGemini)
                self.assertEqual(same["job_id"], continuation_id)
                self.assertEqual(RecipientCitationWriter.submit_calls, 1)

                outcomes = []
                for _ in range(8):
                    arm_poll(private)
                    outcomes.extend(poll_once(private_root=private,
                        client_factory=RecipientCitationWriter,
                        gemini_client_factory=TruncatedThenValidGemini))
                    if any(event["status"] == "accepted" and
                           event["job_id"] == continuation_id for event in outcomes):
                        break
                self.assertTrue(any(event["status"] == "accepted" and
                    event["job_id"] == continuation_id for event in outcomes), outcomes)
                completed_again = resume_saved_draft(prior_job_id=prior["id"],
                    private_root=private,
                    gemini_client_factory=TruncatedThenValidGemini)
                self.assertEqual(completed_again["status"], "accepted")
                self.assertEqual(completed_again["job_id"], continuation_id)
                self.assertEqual(completed_again["new_writer_generations"], 0)

            self.assertEqual(RecipientCitationWriter.submit_calls, 1)
            ledger = Ledger(private)
            accepted = ledger.get(continuation_id)
            self.assertEqual(accepted["status"], "accepted")
            self.assertEqual(accepted["writer_parent_id"], prior["id"])
            self.assertEqual(accepted["billing_group_id"], prior["billing_group_id"])
            self.assertEqual(accepted["dispatches"], 0)
            self.assertIsNone(accepted["remote_id"])
            rows = ledger.db.execute(
                "SELECT id,status,remote_id,billed_microusd,dispatches,error_code "
                "FROM jobs WHERE id IN (?,?)", (prior["id"], first_child["id"])).fetchall()
            self.assertEqual({row["id"]: (row["status"], row["remote_id"],
                row["billed_microusd"], row["dispatches"], row["error_code"])
                for row in rows}, old_rows)
            group = ledger.db.execute(
                "SELECT COUNT(*),SUM(dispatches),SUM(COALESCE(billed_microusd,reserved_microusd)) "
                "FROM jobs WHERE billing_group_id=?", (prior["billing_group_id"],)).fetchone()
            self.assertLessEqual(group[1], 6)
            self.assertLessEqual(group[2], 100_000)
            self.assertEqual(group[1], 5)  # Luna + failed review + two reviews + verify
            ledger.close()
            self.assertEqual({path: _file_sha(Path(path)) for path in sealed_before}, sealed_before)
            pointer = json.loads((output / "summary_current.json").read_text())
            generation = output / "summary_generations" / pointer["generation_id"]
            published = json.loads((generation / "model_document.json").read_text())
            self.assertEqual(published["tasks"][0]["source_ids"], ["U00001", "U00002"])
            self.assertEqual(published["tasks"][0]["recipient"], "Б")
            provenance = json.loads((Path(accepted["artifact_dir"]) /
                "continuation_provenance.json").read_text())
            self.assertEqual(provenance["writer_parent_job_id"], prior["id"])
            self.assertEqual(provenance["writer_remote_batch_id"], prior["remote_id"])
            self.assertEqual(provenance["already_counted_dispatches"], 2)


if __name__ == "__main__":
    unittest.main()
