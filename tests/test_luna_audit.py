"""Audit/repair contract tests use only artificial utterances and no API."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from summary.luna_v1 import SCHEMA_ID, load_source, validate_document
from summary.luna_v1.audit import (
    AUDIT_PROMPT_PATH, AUDIT_SCHEMA, AUDIT_SCHEMA_ID, apply_audit,
    build_audit_input, source_windows, validate_audit,
)


def _task(title: str, description: str, source_id: str) -> dict:
    return {
        "title": title, "description": description, "discussion_status": "proposed",
        "assignee": None, "due": None, "priority": None, "recipient": None,
        "source_ids": [source_id],
        "field_sources": {"action": [source_id], "assignee": [], "due": [],
                          "priority": [], "recipient": [], "discussion_status": [source_id]},
    }


def _document() -> dict:
    return {
        "schema_version": SCHEMA_ID,
        "meeting": {"topic": "Учебная проверка вариантов", "project": None},
        "main": [{"text": "Предложено проверить X.", "source_ids": ["U00001"]}],
        "timecodes": [{"topic": "Варианты", "start_id": "U00001", "end_id": "U00003"}],
        "tasks": [_task("Проверить X", "Предложено проверить один вариант X.", "U00001")],
        "questions": [], "technical": [], "ideas": [], "verification": [],
        "chapters": [{"topic": "Обсуждение проверки", "start_id": "U00001", "end_id": "U00003",
                      "summary": "Участники обсудили два действия и открытый вопрос.",
                      "source_ids": ["U00001", "U00002", "U00003"], "details": []}],
    }


def _finding(*, status: str, source_id: str, patches: list[int], description: str,
             kind: str = "omission", affected: list[dict] | None = None) -> dict:
    return {"severity": "major", "kind": kind, "description": description,
            "source_ids": [source_id], "affected": affected or [],
            "status": status, "patch_indices": patches}


class LunaAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        transcript_path = Path(self.tmp.name) / "transcript.json"
        transcript_path.write_text(json.dumps({
            "source": "01.01.2030 — Учебный диалог.mkv",
            "duration_seconds": 25,
            "speakers": {"p1": "А", "p2": "Б"},
            "utterances": [
                {"start": 0, "end": 6, "speaker": "p1", "text": "Предлагаю проверить X."},
                {"start": 7, "end": 13, "speaker": "p2", "text": "Нужно также отдельно проверить Y."},
                {"start": 14, "end": 24, "speaker": "p1", "text": "Пока не ясно, успеем ли сегодня."},
            ],
        }, ensure_ascii=False), encoding="utf-8")
        self.source_text, self.index, _ = load_source(transcript_path)
        self.draft = _document()
        validate_document(self.draft, self.index)

    def _report(self) -> dict:
        new_task = _task("Проверить Y", "Отдельно проверить вариант Y в продолжение обсуждения.", "U00002")
        new_main = {"text": "Предложено проверить X и отдельно Y.", "source_ids": ["U00001", "U00002"]}
        return {
            "schema_version": AUDIT_SCHEMA_ID,
            "coverage": [
                {"window_id": "W01", "start_id": "U00001", "end_id": "U00001",
                 "salient": "Предложение проверить X.", "draft_coverage": "covered", "finding_indices": []},
                {"window_id": "W02", "start_id": "U00002", "end_id": "U00002",
                 "salient": "Отдельная работа по Y.", "draft_coverage": "missing", "finding_indices": [0]},
                {"window_id": "W03", "start_id": "U00003", "end_id": "U00003",
                 "salient": "Срок остаётся неизвестным.", "draft_coverage": "covered", "finding_indices": [1]},
            ],
            "findings": [
                _finding(status="repaired", source_id="U00002", patches=[0, 1],
                         description="Отдельная задача Y отсутствует в черновике."),
                _finding(status="unresolved", source_id="U00003", patches=[],
                         description="Неясно, установлен ли срок на сегодня."),
            ],
            "patches": [
                {"section": "tasks", "operation": "insert", "index": 1,
                 "item_json": json.dumps(new_task, ensure_ascii=False)},
                {"section": "main", "operation": "replace", "index": 0,
                 "item_json": json.dumps(new_main, ensure_ascii=False)},
            ],
        }

    def test_versioned_schema_and_prompt_cover_two_modes(self):
        self.assertEqual(AUDIT_SCHEMA["properties"]["schema_version"]["enum"], [AUDIT_SCHEMA_ID])
        for definition in AUDIT_SCHEMA["properties"].values():
            if definition.get("type") == "array" and "properties" in definition["items"]:
                item = definition["items"]
                self.assertFalse(item["additionalProperties"])
                self.assertEqual(set(item["properties"]), set(item["required"]))
        prompt = AUDIT_PROMPT_PATH.read_text(encoding="utf-8")
        self.assertIn("MODE=audit", prompt)
        self.assertIn("MODE=verify", prompt)
        self.assertIn("всей исходной стенограмме", prompt)
        self.assertEqual(len(source_windows(self.index)), 3)

    def test_canonical_input_keeps_full_source_and_draft_separate(self):
        first = build_audit_input(self.source_text, self.draft)
        second = build_audit_input(self.source_text, copy.deepcopy(self.draft))
        self.assertEqual(first, second)
        payload = json.loads(first)
        self.assertEqual(payload["MODE"], "audit")
        self.assertEqual(len(payload["TRANSCRIPT_SOURCE"]["utterances"]), 3)
        self.assertEqual(payload["SOURCE_WINDOWS"], source_windows(self.index))
        self.assertEqual(payload["DRAFT_DOCUMENT"], self.draft)
        self.assertEqual(payload["PRIOR_FINDINGS"], [])
        verify = json.loads(build_audit_input(self.source_text, self.draft, mode="verify",
                                               prior_findings=self._report()["findings"]))
        self.assertEqual(verify["MODE"], "verify")
        self.assertEqual(len(verify["PRIOR_FINDINGS"]), 2)

    def test_applies_complete_section_items_without_mutating_draft(self):
        original = copy.deepcopy(self.draft)
        report = self._report()
        self.assertIs(validate_audit(report, self.draft, self.index), report)
        revised, unresolved = apply_audit(self.draft, report, self.index)
        self.assertEqual(self.draft, original)
        self.assertEqual(len(revised["tasks"]), 2)
        self.assertEqual(revised["tasks"][1]["title"], "Проверить Y")
        self.assertEqual(revised["main"][0]["source_ids"], ["U00001", "U00002"])
        self.assertEqual(len(unresolved), 1)
        self.assertIn("Неясно, установлен ли срок", revised["verification"][0]["text"])
        self.assertEqual(revised["verification"][0]["source_ids"], ["U00003"])
        validate_document(revised, self.index)

    def test_verify_mode_can_annotate_remaining_issue_without_patch(self):
        report = {"schema_version": AUDIT_SCHEMA_ID,
                  "coverage": [
                      {"window_id": f"W{number:02d}", "start_id": f"U{number:05d}",
                       "end_id": f"U{number:05d}", "salient": "Краткая тема окна.",
                       "draft_coverage": "covered", "finding_indices": [0] if number == 3 else []}
                      for number in (1, 2, 3)],
                  "findings": [_finding(status="unresolved", source_id="U00003", patches=[],
                                        description="Поздний ответ не разрешает срок.")],
                  "patches": []}
        revised, unresolved = apply_audit(self.draft, report, self.index, mode="verify")
        self.assertEqual(len(unresolved), 1)
        self.assertEqual(len(revised["verification"]), 1)

    def test_uncertain_optional_field_needs_no_corrective_patch_when_draft_is_cautious(self):
        report = self._report()
        report["findings"][1]["kind"] = "time"
        self.assertIs(validate_audit(report, self.draft, self.index), report)
        revised, unresolved = apply_audit(self.draft, report, self.index)
        self.assertIsNone(revised["tasks"][0]["due"])
        self.assertEqual(len(unresolved), 1)

    def test_verify_corrects_confident_error_and_keeps_uncertainty_visible(self):
        draft = copy.deepcopy(self.draft)
        draft["main"][0] = {"text": "Всё нужно закончить сегодня.", "source_ids": ["U00003"]}
        replacement = {"text": "Участники не определили, успеют ли сегодня.", "source_ids": ["U00003"]}
        report = {
            "schema_version": AUDIT_SCHEMA_ID,
            "coverage": [
                {"window_id": f"W{number:02d}", "start_id": f"U{number:05d}",
                 "end_id": f"U{number:05d}", "salient": "Краткая тема окна.",
                 "draft_coverage": "covered", "finding_indices": [0] if number == 3 else []}
                for number in (1, 2, 3)],
            "findings": [_finding(status="unresolved", source_id="U00003", patches=[0],
                                  kind="time", description="Срок на сегодня не подтверждён.",
                                  affected=[{"section": "main", "index": 0}])],
            "patches": [{"section": "main", "operation": "replace", "index": 0,
                         "item_json": json.dumps(replacement, ensure_ascii=False)}],
        }
        revised, unresolved = apply_audit(draft, report, self.index, mode="verify")
        self.assertEqual(revised["main"][0], replacement)
        self.assertEqual(len(unresolved), 1)
        self.assertIn("Срок на сегодня не подтверждён", revised["verification"][0]["text"])
        self.assertEqual(draft["main"][0]["text"], "Всё нужно закончить сегодня.")
        report["findings"][0]["patch_indices"] = []
        with self.assertRaisesRegex(ValueError, "corrective patch"):
            validate_audit(report, draft, self.index, mode="verify")
        report["findings"][0]["patch_indices"] = [0]
        report["patches"][0]["item_json"] = json.dumps(draft["main"][0], ensure_ascii=False)
        with self.assertRaisesRegex(ValueError, "no-op"):
            validate_audit(report, draft, self.index, mode="verify")

    def test_conflicting_claim_requires_patch_for_every_affected_element(self):
        report = self._report()
        report["findings"][0]["kind"] = "unsupported_claim"
        report["findings"][0]["affected"] = [
            {"section": "main", "index": 0}, {"section": "chapters", "index": 0}]
        with self.assertRaisesRegex(ValueError, "every confident affected claim"):
            validate_audit(report, self.draft, self.index)

    def test_rejects_unlinked_or_invalid_patches(self):
        report = self._report()
        report["findings"][0]["patch_indices"] = [0]
        with self.assertRaisesRegex(ValueError, "every patch"):
            validate_audit(report, self.draft, self.index)
        report = self._report()
        report["patches"][0]["index"] = 3
        with self.assertRaisesRegex(ValueError, "index exceeds"):
            validate_audit(report, self.draft, self.index)
        report = self._report()
        report["patches"][0]["item_json"] = '{"title":"incomplete"}'
        with self.assertRaises(ValueError):
            apply_audit(self.draft, report, self.index)

    def test_rejects_unknown_source_and_duplicate_original_target(self):
        report = self._report()
        report["findings"][0]["source_ids"] = ["U99999"]
        with self.assertRaisesRegex(ValueError, "unknown source ID"):
            validate_audit(report, self.draft, self.index)
        report = self._report()
        report["patches"].append(copy.deepcopy(report["patches"][0]))
        report["findings"][0]["patch_indices"].append(2)
        with self.assertRaisesRegex(ValueError, "duplicate target"):
            validate_audit(report, self.draft, self.index)

    def test_requires_exact_whole_source_coverage(self):
        report = self._report()
        report["coverage"].pop()
        with self.assertRaisesRegex(ValueError, "one coverage row"):
            validate_audit(report, self.draft, self.index)
        report = self._report()
        report["coverage"][1]["end_id"] = "U00003"
        with self.assertRaisesRegex(ValueError, "source window identity"):
            validate_audit(report, self.draft, self.index)
        report = self._report()
        report["coverage"][1]["finding_indices"] = []
        with self.assertRaisesRegex(ValueError, "omission finding"):
            validate_audit(report, self.draft, self.index)

    def test_can_repair_parseable_structurally_invalid_draft(self):
        broken = copy.deepcopy(self.draft)
        broken["tasks"][0].pop("field_sources")
        broken.pop("ideas")
        with self.assertRaises(ValueError):
            validate_document(broken, self.index)
        report = self._report()
        report["coverage"][0]["finding_indices"] = [0]
        report["findings"][0]["source_ids"].append("U00001")
        report["patches"] = [{"section": "tasks", "operation": "replace", "index": 0,
                              "item_json": json.dumps(self.draft["tasks"][0], ensure_ascii=False)}]
        report["findings"][0]["patch_indices"] = [0]
        report["coverage"][1]["draft_coverage"] = "covered"
        report["coverage"][1]["finding_indices"] = []
        revised, _ = apply_audit(broken, report, self.index)
        self.assertEqual(revised["ideas"], [])
        self.assertEqual(revised["tasks"][0], self.draft["tasks"][0])


if __name__ == "__main__":
    unittest.main()
