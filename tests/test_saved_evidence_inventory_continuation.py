"""A paid malformed v2 inventory is reused without repeating its Batch calls."""
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
from summary.gemini_v1.inventory_contract import plan_inventory_segments
from summary.luna_v1.engine import (
    EVIDENCE_RECONCILE_QUALITY_POLICY_VERSION,
    _relocate_inventory_coverage_pointers, poll_once,
    resume_saved_evidence_inventory, submit,
)
from summary.luna_v1.ledger import Ledger
from summary.luna_v1.source import load_source
from tests.test_gemini_engine import TwoRoleStore
from tests.test_gemini_evidence_engine import EvidenceGemini
from tests.test_luna_engine import FakeClient, arm_poll


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MisplacedEvidenceGemini(EvidenceGemini):
    """Place four otherwise valid S03 references in its adjacent window."""

    def get(self, name):
        reply = super().get(name)
        _, request = self.__class__.submissions[name]
        payload = json.loads(request["messages"][1]["content"])
        if "SOURCE_SEGMENT" not in payload or payload["SOURCE_SEGMENT"]["segment_id"] != "S03":
            return reply
        body = copy.deepcopy(reply.body)
        response = body["results"][0]["response"]["body"]
        report = json.loads(response["choices"][0]["message"]["content"])
        assert len(report["coverage"]) == 2 and len(report["items"]) == 2
        first, second = report["items"]
        report["items"] = [copy.deepcopy(first) for _ in range(3)] + [
            copy.deepcopy(second) for _ in range(3)]
        report["coverage"][0]["item_indices"] = [0, 4, 5]
        report["coverage"][1]["item_indices"] = [3, 1, 2]
        response["choices"][0]["message"]["content"] = json.dumps(report, ensure_ascii=False)
        return GeminiReply(reply.status_code, body)


class SavedEvidenceInventoryContinuationTests(unittest.TestCase):
    def setUp(self):
        # These legacy continuation scenarios require the original logging-OFF policy.
        logging_policy = patch("summary.luna_v1.engine.OPUS_WORKSPACE_IO_LOGGING_ENABLED", False)
        logging_policy.start()
        self.addCleanup(logging_policy.stop)

    def _make_degraded(self, root: Path):
        output = root / "meeting"
        output.mkdir()
        source = output / "transcript.json"
        filler = "Служебное пояснение без новой работы. " * 650
        source.write_text(json.dumps({
            "source": "01.01.2030 — Синтетика.mkv", "duration_seconds": 82,
            "speakers": {"p1": "А"}, "utterances": [
                {"start": start, "end": start + 5, "speaker": "p1", "text": line + filler}
                for start, line in zip((0, 35, 70, 75), (
                    "Предлагаю проверить X или Y, не оба. ",
                    "Ещё предлагаю записать результат в журнал. ",
                    "Обсудили проверку данных. ",
                    "Повторили контекст проверки данных. ",
                ))],
        }, ensure_ascii=False), encoding="utf-8")
        private = root / "private"
        writer_route = SimpleNamespace(
            reserve_microusd=lambda payload, **kwargs: 20_000,
            workspace_id="writer-workspace", prompt_usd_per_token="0.00000005",
            completion_usd_per_token="0.00000025",
            cache_write_usd_per_token="0.0000000625", request_usd="0")
        judge_route = SimpleNamespace(
            reserve_microusd=lambda: 15_000,
            input_tokens=1500, context_bound_tokens=2100,
            workspace_id="judge-workspace", key_limit_remaining_usd=None)
        FakeClient.submit_calls = 0
        FakeClient.submissions = {}
        FakeClient.report_factory = None
        MisplacedEvidenceGemini.submit_calls = 0
        MisplacedEvidenceGemini.submissions = {}
        MisplacedEvidenceGemini.active = set()
        MisplacedEvidenceGemini.stage_order = []
        MisplacedEvidenceGemini.bad_quote_segment = None
        patches = (
            patch("summary.luna_v1.engine._credential_store", return_value=TwoRoleStore()),
            patch("summary.luna_v1.engine.verify_batch_route", return_value=writer_route),
            patch("summary.luna_v1.engine.verify_gemini_batch_route", return_value=judge_route),
        )
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        started = submit(transcript_path=source, output_dir=output, private_root=private,
                         client_factory=FakeClient,
                         quality_policy_version=EVIDENCE_RECONCILE_QUALITY_POLICY_VERSION)
        self.assertEqual(started["status"], "submitted")
        for _ in range(8):
            arm_poll(private)
            poll_once(private_root=private, client_factory=FakeClient,
                      gemini_client_factory=MisplacedEvidenceGemini)
            ledger = Ledger(private)
            prior = ledger.get(started["job_id"])
            ledger.close()
            if prior["status"] == "accepted":
                break
        self.assertEqual(prior["status"], "accepted")
        self.assertEqual(FakeClient.submit_calls + MisplacedEvidenceGemini.submit_calls, 4)
        self.assertEqual(MisplacedEvidenceGemini.stage_order, ["S01", "S02", "S03"])
        ledger = Ledger(private)
        old_stages = [ledger.stage(prior["id"], f"inventory_{number}")
                      for number in (1, 2, 3)]
        ledger.close()
        self.assertEqual([stage["status"] for stage in old_stages],
                         ["stage_complete", "stage_complete", "failed_validation"])
        pointer = output / "summary_current.json"
        generation = output / "summary_generations" / json.loads(pointer.read_text())["generation_id"]
        self.assertEqual(json.loads((generation / "run_manifest.json").read_text())[
            "quality_review"]["status"], "inventory_unavailable")
        return output, private, prior, old_stages, pointer, generation

    def test_four_paid_calls_then_zero_call_recovery_then_two_reconciliations(self):
        with tempfile.TemporaryDirectory() as directory:
            output, private, prior, old_stages, pointer, generation = self._make_degraded(Path(directory))
            old_files = [Path(prior["artifact_dir"]) / "manifest.json",
                         Path(prior["artifact_dir"]) / "draft_document.json"]
            for stage in old_stages:
                old_files.extend(Path(stage["artifact_dir"]) / name for name in (
                    "manifest.json", "request.json", "target.json", "native_response.json",
                    "batch_terminal.json"))
            old_hashes = {_file: _digest(_file) for _file in old_files}
            old_pointer_hash = _digest(pointer)
            old_generation_hash = _digest(generation / "summary.md")
            resumed = resume_saved_evidence_inventory(
                prior_job_id=prior["id"], private_root=private,
                gemini_client_factory=MisplacedEvidenceGemini)
            self.assertEqual(resumed["status"], "quality_pending")
            self.assertEqual(resumed["parser_correction_warning_count"], 4)
            self.assertEqual(FakeClient.submit_calls + MisplacedEvidenceGemini.submit_calls, 4)
            again = resume_saved_evidence_inventory(
                prior_job_id=prior["id"], private_root=private,
                gemini_client_factory=MisplacedEvidenceGemini)
            self.assertEqual(again["job_id"], resumed["job_id"])
            ledger = Ledger(private)
            continuation = ledger.get(resumed["job_id"])
            self.assertEqual(continuation["billing_group_id"], prior["billing_group_id"])
            self.assertEqual(continuation["dispatches"], 0)
            self.assertIsNone(continuation["remote_id"])
            self.assertTrue(all(ledger.stage(continuation["id"], f"inventory_{i}") is None
                                for i in (1, 2, 3)))
            ledger.close()
            self.assertEqual({_file: _digest(_file) for _file in old_files}, old_hashes)
            self.assertEqual(_digest(pointer), old_pointer_hash)
            self.assertEqual(_digest(generation / "summary.md"), old_generation_hash)
            for _ in range(7):
                arm_poll(private)
                poll_once(private_root=private, client_factory=FakeClient,
                          gemini_client_factory=MisplacedEvidenceGemini)
                ledger = Ledger(private)
                continuation = ledger.get(resumed["job_id"])
                ledger.close()
                if continuation["status"] == "accepted":
                    break
            self.assertEqual(continuation["status"], "accepted")
            self.assertEqual(FakeClient.submit_calls + MisplacedEvidenceGemini.submit_calls, 6)
            self.assertEqual(MisplacedEvidenceGemini.stage_order[-2:],
                             ["reconcile_1", "reconcile_2"])
            ledger = Ledger(private)
            self.assertEqual(sum(row[0] for row in ledger.db.execute(
                "SELECT dispatches FROM jobs WHERE billing_group_id=?",
                (prior["billing_group_id"],))), 6)
            self.assertEqual(ledger.get(prior["id"])["status"], "accepted")
            self.assertEqual(ledger.stage(prior["id"], "inventory_3")["status"],
                             "failed_validation")
            first = ledger.stage(continuation["id"], "reconcile_1")
            second = ledger.stage(continuation["id"], "reconcile_2")
            first_revised = json.loads((Path(continuation["artifact_dir"]) /
                                        "reconcile_1_revised_document.json").read_text())
            second_target = json.loads((Path(second["artifact_dir"]) / "target.json").read_text())
            self.assertEqual(second_target["draft"], first_revised)
            self.assertEqual(first["status"], "stage_complete")
            self.assertEqual(second["status"], "stage_complete")
            ledger.close()
            self.assertEqual({_file: _digest(_file) for _file in old_files}, old_hashes)
            self.assertEqual(_digest(generation / "summary.md"), old_generation_hash)
            current = output / "summary_generations" / json.loads(pointer.read_text())["generation_id"]
            review = json.loads((current / "run_manifest.json").read_text())["quality_review"]
            self.assertEqual(review["reused_inventory_stage_ids"], [s["id"] for s in old_stages])
            self.assertEqual(review["inventory_job_ids"], [s["id"] for s in old_stages])
            self.assertEqual(review["parser_correction_warning_count"], 4)
            self.assertGreaterEqual(review["coverage_warning_count"], 4)
            self.assertEqual(review["inventory_correction_source"],
                             "saved_terminal_native_response")
            self.assertNotEqual(review["status"], "model_reconciled_checked")
            terminal_repeat = resume_saved_evidence_inventory(
                prior_job_id=prior["id"], private_root=private,
                gemini_client_factory=MisplacedEvidenceGemini)
            self.assertEqual(terminal_repeat["status"], "accepted")
            self.assertEqual(terminal_repeat["job_id"], continuation["id"])
            arm_poll(private)
            poll_once(private_root=private, client_factory=FakeClient,
                      gemini_client_factory=MisplacedEvidenceGemini)
            self.assertEqual(FakeClient.submit_calls + MisplacedEvidenceGemini.submit_calls, 6)

    def test_initial_pointer_change_blocks_new_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            _, private, prior, _, pointer, _ = self._make_degraded(Path(directory))
            saved = json.loads(pointer.read_text())
            saved["generation_id"] = "20990101-000000-000000000000"
            pointer.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, "current_pointer_changed"):
                resume_saved_evidence_inventory(
                    prior_job_id=prior["id"], private_root=private,
                    gemini_client_factory=MisplacedEvidenceGemini)
            self.assertEqual(FakeClient.submit_calls + MisplacedEvidenceGemini.submit_calls, 4)

    def test_native_mismatch_is_blocked_before_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            _, private, prior, stages, _, _ = self._make_degraded(Path(directory))
            native = Path(stages[2]["artifact_dir"]) / "native_response.json"
            payload = json.loads(native.read_text())
            payload["text"] += " "
            native.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "native_stop_or_usage_differs"):
                resume_saved_evidence_inventory(prior_job_id=prior["id"], private_root=private,
                                                gemini_client_factory=MisplacedEvidenceGemini)
            self.assertEqual(MisplacedEvidenceGemini.submit_calls, 3)

    def test_target_mismatch_is_blocked_before_continuation(self):
        with tempfile.TemporaryDirectory() as directory:
            _, private, prior, stages, _, _ = self._make_degraded(Path(directory))
            target = Path(stages[2]["artifact_dir"]) / "target.json"
            payload = json.loads(target.read_text())
            payload["segment_id"] = "S99"
            target.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "request_or_route_changed"):
                resume_saved_evidence_inventory(prior_job_id=prior["id"], private_root=private,
                                                gemini_client_factory=MisplacedEvidenceGemini)
            self.assertEqual(MisplacedEvidenceGemini.submit_calls, 3)

    def test_missing_reuse_manifest_cannot_dispatch_new_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            _, private, prior, _, _, _ = self._make_degraded(Path(directory))
            resumed = resume_saved_evidence_inventory(
                prior_job_id=prior["id"], private_root=private,
                gemini_client_factory=MisplacedEvidenceGemini)
            ledger = Ledger(private)
            continuation = ledger.get(resumed["job_id"])
            ledger.close()
            manifest_path = Path(continuation["artifact_dir"]) / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            del manifest["reused_inventories"]
            manifest_path.write_text(json.dumps(manifest))
            arm_poll(private)
            outcomes = poll_once(private_root=private, client_factory=FakeClient,
                                 gemini_client_factory=MisplacedEvidenceGemini)
            self.assertTrue(any(row["status"] == "failed_quality_integrity" for row in outcomes))
            self.assertEqual(FakeClient.submit_calls + MisplacedEvidenceGemini.submit_calls, 4)

    def test_ambiguous_destination_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "transcript.json"
            source.write_text(json.dumps({
                "source": "Synthetic", "duration_seconds": 12,
                "speakers": {"p1": "А"},
                "utterances": [{"start": i, "end": i + 0.8, "speaker": "p1",
                                "text": f"Факт {i}"} for i in range(6)],
            }, ensure_ascii=False))
            source_text, _, _ = load_source(source)
            segment = plan_inventory_segments(source_text, count=2,
                                              windows_per_segment=3)[1]
            self.assertEqual(len(segment["coverage_windows"]), 3)
            primary = segment["primary_utterances"]
            item = {"kind": "technical", "claim": "Факт", "source_ids": [
                primary[0]["id"], primary[2]["id"]], "speaker": primary[0]["speaker"],
                "actor": None, "recipient": None, "action": None,
                "modality": "explanation", "condition": None,
                "alternatives": [], "correction_of": [], "uncertainty": None,
                "source_quote": primary[0]["text"]}
            report = {"schema_version": "gemini_source_inventory_v2",
                      "segment_id": segment["segment_id"], "items": [item],
                      "coverage": [{"window_id": row["window_id"],
                                    "start_id": row["start_id"], "end_id": row["end_id"],
                                    "assessment": "uncertain", "item_indices": [0] if i == 1 else []}
                                   for i, row in enumerate(segment["coverage_windows"])]}
            with self.assertRaisesRegex(ValueError, "destination_ambiguous"):
                _relocate_inventory_coverage_pointers(report, segment)


if __name__ == "__main__":
    unittest.main()
