"""Offline two-request diagnostic through the real ledger and scheduler."""
from __future__ import annotations

import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from summary.gemini_v1.batch import BatchError, MODEL, Reply
from summary.gemini_v1.diagnostic import inspect_run, submit_pair
from summary.gemini_v1.reconcile_contract import draft_units
from summary.luna_v1.engine import poll_once
from summary.luna_v1.ledger import Ledger


SCHEMA_ROOT = Path(__file__).resolve().parents[1] / "summary" / "gemini_v1"


def _payload() -> dict:
    draft = {
        "schema_version": "luna_summary_v1",
        "meeting": {"topic": "Проверка", "project": None},
        "main": [{"text": "Тестовое действие обсуждено.", "source_ids": ["U00001"]}],
        "timecodes": [], "tasks": [], "questions": [], "technical": [],
        "ideas": [], "verification": [], "chapters": [],
    }
    units = draft_units(draft)
    return {
        "SOURCE_EXCERPTS": [
            {"id": "U00001", "speaker": "А", "text": "Обсудим тестовое действие."},
            {"id": "U00002", "speaker": "Б", "text": "Отдельно проверим ответ."},
        ],
        "INDEPENDENT_SOURCE_INVENTORY": {"items": [
            {"item_id": "C01", "source_ids": ["U00001"]},
            {"item_id": "C02", "source_ids": ["U00002"]},
        ]},
        "DRAFT_DOCUMENT": draft,
        "DRAFT_UNITS": [next(row for row in units if row["unit_id"] == "main:0")],
        "DRAFT_REFERENCE_UNIT_IDS": [row["unit_id"] for row in units],
        "SOURCE_WINDOWS_TO_RECHECK": [{"window_id": "W01", "start_id": "U00001",
                                       "end_id": "U00002"}],
        "TASK": "Синтетическая диагностика.",
    }


def _request(system: str, content: str, slot: str) -> dict:
    name, file = (("gemini_inventory_reconcile_v2", "output_schema_reconcile_v2.json")
                  if slot == "baseline" else
                  ("gemini_diagnostic_compact_v1", "output_schema_diagnostic_compact_v1.json"))
    return {
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": content}],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": name, "strict": True,
            "schema": json.loads((SCHEMA_ROOT / file).read_text())}},
        "max_completion_tokens": 5000, "reasoning": {"effort": "low"},
        "tool_choice": "none", "plugins": [], "modalities": ["text"],
    }


class _Store:
    def __init__(self, path):
        self.path = path

    def dispatch_candidates(self, role):
        assert role == "judge"
        return [{"id": "judge-test", "version": 1, "workspace_id": "ws-test"}]

    def reveal_for_dispatch(self, identifier, version, *, role):
        assert (identifier, version, role) == ("judge-test", 1, "judge")
        return "synthetic-secret"

    def reveal_for_existing_job(self, identifier, version):
        assert (identifier, version) == ("judge-test", 1)
        return "synthetic-secret"


class _Batch:
    posts = []
    fail_first = False
    finish_reason = "stop"
    mutator = None

    def __init__(self, token):
        assert token == "synthetic-secret"

    def submit(self, custom_id, body):
        if self.fail_first and not self.posts:
            self.posts.append((custom_id, body, None))
            raise BatchError(None, "transport_unknown")
        remote = "batch_synthetic_diagnostic_" + str(len(self.posts) + 1)
        self.posts.append((custom_id, body, remote))
        return Reply(202, {"id": remote, "model": MODEL,
                           "endpoint": "/v1/chat/completions", "status": "validating"})

    def get(self, remote):
        custom_id, request, _ = next(row for row in self.posts if row[2] == remote)
        payload = json.loads(request["messages"][1]["content"])
        quote = payload["DRAFT_DOCUMENT"]["main"][0]["text"]
        if custom_id.startswith("diagnostic_baseline-"):
            result = {
                "schema_version": "gemini_inventory_reconcile_v2",
                "source_window_assessments": [{
                    "window_id": row["window_id"], "status": "material_in_inventory",
                    "source_ids": [row["start_id"]], "finding_indices": [],
                } for row in payload["SOURCE_WINDOWS_TO_RECHECK"]],
                "inventory_assessments": [{
                    "item_id": item["item_id"], "status": "represented",
                    "draft_targets": [{"section": "main", "index": 0}],
                    "draft_evidence": [{"unit_id": "main:0", "quote": quote}],
                    "finding_indices": [],
                } for item in payload["INDEPENDENT_SOURCE_INVENTORY"]["items"]],
                "draft_assessments": [{
                    "unit_id": unit["unit_id"], "status": "supported",
                    "source_ids": ["U00001"], "inventory_ids": ["C01"],
                    "finding_indices": [],
                } for unit in payload["DRAFT_UNITS"]],
                "findings": [], "patches": [],
            }
        else:
            result = {
                "schema_version": "gemini_diagnostic_compact_v1",
                "item_assessments": [{
                    "item_id": item["item_id"], "source_status": "supported",
                    "draft_status": "represented", "source_ids": item["source_ids"],
                    "draft_evidence": [{"unit_id": "main:0", "quote": quote}],
                    "missing_facets": [], "task_modality": "technical_or_question",
                    "reason": "Синтетическая проверка.", "proposed_edit_target": None,
                    "proposed_edit_text": None,
                } for item in payload["INDEPENDENT_SOURCE_INVENTORY"]["items"]],
            }
        if type(self).mutator is not None:
            result = type(self).mutator(result, custom_id)
        return Reply(200, {"id": remote, "model": MODEL,
                           "endpoint": "/v1/chat/completions", "status": "completed",
                           "request_counts": {"total": 1, "completed": 1, "failed": 0},
                           "usage": {"cost": 0.001, "prompt_tokens": 100,
                                     "completion_tokens": 40, "total_tokens": 140,
                                     "is_byok": False},
                           "results": [{"custom_id": custom_id, "error": None,
                                        "response": {"status_code": 200, "body": {
                                            "model": MODEL, "choices": [{
                                                "finish_reason": self.finish_reason,
                                                "message": {"role": "assistant",
                                                            "content": json.dumps(result)}}]}}}]})

    def delete(self, remote):
        return Reply(204, {})


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        _Batch.posts = []
        _Batch.fail_first = False
        _Batch.finish_reason = "stop"
        _Batch.mutator = None
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.private = self.root / "private"
        ledger = Ledger(self.private)
        ledger.close()
        (self.private / "credentials.sqlite3").touch()
        self.plan = self.root / "plan.json"
        self.output = self.root / "diagnostic"
        self._write_plan(_payload())

    def _write_plan(self, payload):
        content = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        (self.root / "baseline.json").write_text(json.dumps(_request("Старый способ.", content, "baseline")))
        (self.root / "compact.json").write_text(json.dumps(_request("Новый короткий способ.", content, "compact")))
        self.plan.write_text(json.dumps({
            "run_id": "diagnostic-synthetic-001", "source_sha256": "a" * 64,
            "draft_sha256": "b" * 64, "cases": ["C01", "C02"],
            "requests": {"baseline": "baseline.json", "compact": "compact.json"},
        }))

    def _patches(self, reserve=30_000):
        return [patch("summary.gemini_v1.diagnostic.CredentialStore", _Store),
                patch("summary.gemini_v1.diagnostic.credential_dispatch_guard",
                      lambda _path: contextlib.nullcontext()),
                patch("summary.gemini_v1.diagnostic.verify_batch_route",
                      lambda _client, _body, *, max_output_tokens: SimpleNamespace(
                          workspace_id="ws-test", key_limit_remaining_usd=None,
                          reserve_microusd=lambda: reserve)),
                patch("summary.luna_v1.engine._credential_store", _Store)]

    def _submit(self, reserve=30_000):
        with contextlib.ExitStack() as stack:
            for item in self._patches(reserve):
                stack.enter_context(item)
            return submit_pair(plan_path=self.plan, private_root=self.private,
                               output_dir=self.output, client_factory=_Batch)

    def _finish_with_mutator(self, mutator):
        self._submit()
        _Batch.mutator = mutator
        ledger = Ledger(self.private)
        ledger.db.execute("UPDATE jobs SET next_poll_at=0 WHERE status='submitted'")
        ledger.close()
        with patch("summary.luna_v1.engine._credential_store", _Store):
            result = poll_once(private_root=self.private, gemini_client_factory=_Batch)
        ledger = Ledger(self.private)
        jobs = {row["kind"]: dict(row) for row in ledger.db.execute("SELECT * FROM jobs")}
        ledger.close()
        return result, jobs

    def test_two_jobs_share_budget_finish_without_publication_and_do_not_repeat(self):
        outcome = self._submit()
        self.assertEqual([row["status"] for row in outcome["slots"]],
                         ["submitted", "submitted"])
        self.assertEqual(len(_Batch.posts), 2)
        ledger = Ledger(self.private)
        jobs = ledger.db.execute("SELECT kind,status,dispatches,billing_group_id "
                                 "FROM jobs ORDER BY kind").fetchall()
        self.assertEqual([row["kind"] for row in jobs],
                         ["diagnostic_baseline", "diagnostic_compact"])
        self.assertEqual({row["billing_group_id"] for row in jobs},
                         {"diagnostic-diagnostic-synthetic-001"})
        ledger.db.execute("UPDATE jobs SET next_poll_at=0 WHERE status='submitted'")
        ledger.close()
        with patch("summary.luna_v1.engine._credential_store", _Store):
            states = poll_once(private_root=self.private,
                               gemini_client_factory=_Batch)
        self.assertEqual([row["status"] for row in states],
                         ["diagnostic_complete", "diagnostic_complete"])
        snapshot = inspect_run(private_root=self.private, output_dir=self.output,
                               run_id="diagnostic-synthetic-001")
        self.assertEqual([row["status"] for row in snapshot["stages"]],
                         ["diagnostic_complete", "diagnostic_complete"])
        self.assertEqual(snapshot["publication"], "forbidden")
        ledger = Ledger(self.private)
        jobs = ledger.db.execute("SELECT status,billed_microusd,artifact_dir "
                                 "FROM jobs ORDER BY kind").fetchall()
        self.assertEqual([row["status"] for row in jobs],
                         ["diagnostic_complete", "diagnostic_complete"])
        self.assertEqual([row["billed_microusd"] for row in jobs], [1000, 1000])
        for row in jobs:
            path = Path(row["artifact_dir"])
            self.assertTrue((path / "batch_terminal.json").is_file())
            self.assertTrue((path / "native_response.json").is_file())
            self.assertTrue((path / "diagnostic_report.json").is_file())
        ledger.close()
        self.assertFalse((self.output / "current").exists())
        self.assertEqual([row["status"] for row in self._submit()["slots"]],
                         ["diagnostic_complete", "diagnostic_complete"])
        self.assertEqual(len(_Batch.posts), 2)

    def test_mismatched_source_or_cost_blocks_before_post(self):
        changed = _request("Новый короткий способ.", "Другой текст.", "compact")
        (self.root / "compact.json").write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "different source"):
            self._submit()
        self.assertEqual(_Batch.posts, [])
        self._write_plan(_payload())
        outcome = self._submit(reserve=60_000)
        self.assertEqual(outcome["reason"], "diagnostic_group_budget_exceeded")
        self.assertEqual(_Batch.posts, [])
        ledger = Ledger(self.private)
        self.assertEqual(ledger.db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 0)
        ledger.close()

    def test_unknown_post_is_counted_and_never_repeated(self):
        _Batch.fail_first = True
        outcome = self._submit()
        self.assertEqual([row["status"] for row in outcome["slots"]],
                         ["submission_unknown", "submitted"])
        self.assertEqual(len(_Batch.posts), 2)
        self._submit()
        self.assertEqual(len(_Batch.posts), 2)

    def test_existing_weekly_spend_blocks_pair_before_any_post(self):
        ledger = Ledger(self.private)
        for number in range(10):
            decision = ledger.reserve(
                semantic_key=f"{number:064x}", source_sha256="c" * 64,
                output_dir=self.root / f"prior-{number}", credential_id="writer-test",
                credential_version=1, workspace_id="ws-test",
                max_cost_microusd=98_000)
            self.assertEqual(decision.kind, "new")
        ledger.close()
        outcome = self._submit()
        self.assertEqual((outcome["status"], outcome["reason"]),
                         ("blocked", "weekly_budget_exceeded"))
        self.assertEqual(_Batch.posts, [])

    def test_length_response_is_saved_but_not_accepted(self):
        self._submit()
        _Batch.finish_reason = "length"
        ledger = Ledger(self.private)
        ledger.db.execute("UPDATE jobs SET next_poll_at=0 WHERE status='submitted'")
        ledger.close()
        with patch("summary.luna_v1.engine._credential_store", _Store):
            states = poll_once(private_root=self.private, gemini_client_factory=_Batch)
        self.assertEqual([row["status"] for row in states],
                         ["failed_validation", "failed_validation"])
        ledger = Ledger(self.private)
        for row in ledger.db.execute("SELECT artifact_dir FROM jobs"):
            path = Path(row["artifact_dir"])
            self.assertTrue((path / "native_response.json").is_file())
            self.assertFalse((path / "diagnostic_report.json").exists())
        ledger.close()

    def test_wrong_version_or_missing_case_is_invalid_with_raw_preserved(self):
        def bad(report, custom_id):
            if custom_id.startswith("diagnostic_baseline-"):
                report["schema_version"] = "other"
            else:
                report["item_assessments"].pop()
            return report

        result, jobs = self._finish_with_mutator(bad)
        self.assertEqual([row["status"] for row in result],
                         ["failed_validation", "failed_validation"])
        for job in jobs.values():
            artifacts = Path(job["artifact_dir"])
            self.assertTrue((artifacts / "batch_terminal.json").is_file())
            self.assertTrue((artifacts / "native_response.json").is_file())
            self.assertFalse((artifacts / "diagnostic_report.json").exists())

    def test_forged_draft_quote_is_invalid_in_both_forms(self):
        def bad(report, custom_id):
            field = ("inventory_assessments" if custom_id.startswith("diagnostic_baseline-")
                     else "item_assessments")
            report[field][0]["draft_evidence"][0]["quote"] = "Такой фразы в сохранённом черновике нет"
            return report

        result, _ = self._finish_with_mutator(bad)
        self.assertEqual([row["status"] for row in result],
                         ["failed_validation", "failed_validation"])

    def test_source_id_outside_frozen_request_is_invalid(self):
        def bad(report, custom_id):
            if custom_id.startswith("diagnostic_baseline-"):
                report["source_window_assessments"][0]["source_ids"] = ["U99999"]
            else:
                report["item_assessments"][0]["source_ids"] = ["U99999"]
            return report

        result, _ = self._finish_with_mutator(bad)
        self.assertEqual([row["status"] for row in result],
                         ["failed_validation", "failed_validation"])


if __name__ == "__main__":
    unittest.main()
