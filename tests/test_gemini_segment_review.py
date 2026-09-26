"""Offline contract tests for one-pass source-segment Gemini review."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from summary.gemini_v1.inventory_contract import (
    merge_inventory_reports,
    plan_inventory_segments,
    validate_inventory_plan,
)
from summary.gemini_v1.segment_review_contract import (
    SEGMENT_REVIEW_PROMPT_PATH,
    SEGMENT_REVIEW_SCHEMA,
    SEGMENT_REVIEW_SCHEMA_ID,
    build_segment_review_input,
    draft_units_for_segment,
    merge_segment_review_reports,
    segment_inventory_view,
    segment_review_warnings,
    validate_segment_review_report,
)
from summary.luna_v1 import SCHEMA_ID, load_source, validate_document
from summary.luna_v1.audit import apply_audit


def _task(assignee: str | None, source_ids: list[str]) -> dict:
    return {
        "title": "Подготовить таблицу",
        "description": "Подготовить таблицу для проверки трёх вариантов.",
        "discussion_status": "committed",
        "assignee": assignee, "due": None, "priority": None, "recipient": None,
        "source_ids": source_ids,
        "field_sources": {
            "action": source_ids, "assignee": source_ids if assignee else [],
            "due": [], "priority": [], "recipient": [],
            "discussion_status": source_ids,
        },
    }


def _item(source_id: str, claim: str, speaker: str, actor: str | None,
          *, kind: str = "action", modality: str = "committed") -> dict:
    return {
        "kind": kind, "claim": claim, "source_ids": [source_id],
        "speaker": speaker, "actor": actor, "recipient": None,
        "action": claim if kind == "action" else None,
        "modality": modality, "condition": None, "alternatives": [],
        "correction_of": [], "uncertainty": None,
    }


class SegmentReviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "transcript.json"
        path.write_text(json.dumps({
            "source": "Учебная встреча.mkv", "duration_seconds": 40,
            "speakers": {"p1": "Алиса", "p2": "Борис"},
            "utterances": [
                {"start": 0, "end": 4, "speaker": "p1",
                 "text": "Борис, подготовь таблицу для трёх вариантов."},
                {"start": 5, "end": 8, "speaker": "p2",
                 "text": "Да, я подготовлю таблицу."},
                {"start": 9, "end": 13, "speaker": "p1",
                 "text": "Для отчёта нужен месячный объём, не недельный."},
                {"start": 14, "end": 19, "speaker": "p2",
                 "text": "Поздняя поправка: проверяем пять вариантов, а не три."},
                {"start": 20, "end": 24, "speaker": "p1",
                 "text": "Срок исходных данных пока не известен."},
                {"start": 25, "end": 29, "speaker": "p2",
                 "text": "Прежнюю проверку двух вариантов мы уже завершили."},
            ],
        }, ensure_ascii=False), encoding="utf-8")
        self.source_text, self.source_index, _ = load_source(path)
        self.segments = plan_inventory_segments(self.source_text, count=2, overlap=2,
                                                windows_per_segment=2)
        validate_inventory_plan(self.segments, self.source_index)
        self.draft = {
            "schema_version": SCHEMA_ID,
            "meeting": {"topic": "Таблица и варианты", "project": None},
            "main": [{"text": "Обсудили таблицу и варианты.",
                      "source_ids": ["U00001", "U00004"]}],
            "timecodes": [],
            "tasks": [_task("Алиса", ["U00001", "U00002"])],
            "questions": [{"text": "Когда будут данные?", "source_ids": ["U00005"]}],
            "technical": [{"text": "Проверяют три варианта.", "source_ids": ["U00004"]}],
            "ideas": [], "verification": [], "chapters": [],
        }
        validate_document(self.draft, self.source_index)

    def _report(self, segment, *, item_source_id=None, finding=None, patch=None):
        primary = segment["primary_utterances"]
        source_id = item_source_id or primary[0]["id"]
        source_row = self.source_index["by_id"][source_id]
        item = _item(source_id, "Подготовить таблицу", source_row["speaker"], "Борис")
        coverage = []
        for window in segment["coverage_windows"]:
            members = [row["id"] for row in primary]
            member_ids = set(members[members.index(window["start_id"]):
                                     members.index(window["end_id"]) + 1])
            indices = [0] if source_id in member_ids else []
            coverage.append({
                "window_id": window["window_id"], "start_id": window["start_id"],
                "end_id": window["end_id"],
                "assessment": "material_items" if indices else "no_material_items",
                "item_indices": indices,
            })
        findings = [finding] if finding else []
        patches = [patch] if patch else []
        units = draft_units_for_segment(self.draft, segment, self.source_index)
        draft_rows = []
        for unit in units:
            related = unit["section"] == "tasks" and finding is not None
            draft_rows.append({
                "unit_id": unit["unit_id"],
                "status": "partial" if related else "supported",
                "source_ids": [source_id], "item_indices": [0] if related else [],
                "finding_indices": [0] if related else [],
            })
        return {
            "schema_version": SEGMENT_REVIEW_SCHEMA_ID,
            "segment_id": segment["segment_id"], "coverage": coverage, "items": [item],
            "item_assessments": [{
                "item_index": 0, "status": "partial" if finding else "represented",
                "draft_targets": [{"section": "tasks", "index": 0}],
                "finding_indices": [0] if finding else [],
            }],
            "draft_assessments": draft_rows,
            "findings": findings, "patches": patches,
        }

    def test_payload_has_exact_segment_and_full_draft(self):
        segment = self.segments[0]
        payload = json.loads(build_segment_review_input(
            self.source_text, segment, self.draft, self.source_index))
        self.assertEqual(payload["SOURCE_SEGMENT"], segment)
        self.assertEqual(payload["DRAFT_DOCUMENT"], self.draft)
        self.assertNotIn(self.segments[1]["primary_utterances"][-1],
                         payload["SOURCE_SEGMENT"]["primary_utterances"])
        self.assertEqual(payload["MODE"], "segment_review")
        self.assertEqual(list(payload)[-1], "TASK")
        prompt = SEGMENT_REVIEW_PROMPT_PATH.read_text(encoding="utf-8")
        self.assertIn("Сначала исходник", prompt)
        self.assertIn("полным черновиком", prompt)
        self.assertEqual(SEGMENT_REVIEW_SCHEMA["properties"]["items"]["type"], "array")

    def test_report_inventory_adapter_and_bookkeeping(self):
        reports = [self._report(segment) for segment in self.segments]
        for report, segment in zip(reports, self.segments):
            self.assertIs(validate_segment_review_report(
                report, segment, self.draft, self.source_index), report)
        merged = merge_inventory_reports(
            [segment_inventory_view(report) for report in reports],
            self.segments, self.source_index,
        )
        self.assertEqual(merged["primary_utterance_count"], 6)
        self.assertEqual([item["item_id"] for item in merged["items"]],
                         ["S01-I001", "S02-I001"])
        bad = copy.deepcopy(reports[0])
        bad["draft_assessments"].pop()
        with self.assertRaisesRegex(ValueError, "omitted an item or draft assessment"):
            validate_segment_review_report(bad, self.segments[0],
                                           self.draft, self.source_index)

    def test_source_fingerprint_context_and_primary_support(self):
        segment = copy.deepcopy(self.segments[1])
        segment["context_before"][0]["text"] = "Подменённая реплика"
        with self.assertRaisesRegex(ValueError, "span or adjacent overlap differs"):
            build_segment_review_input(self.source_text, segment,
                                       self.draft, self.source_index)
        report = self._report(self.segments[1])
        report["items"][0]["source_ids"] = [self.segments[1]["context_before"][0]["id"]]
        with self.assertRaisesRegex(ValueError, "context-only"):
            validate_segment_review_report(report, self.segments[1],
                                           self.draft, self.source_index)

    def test_safe_patch_merge_and_conflict_visibility(self):
        first = self.segments[0]
        second = self.segments[1]
        corrected = _task("Борис", ["U00001", "U00002"])
        finding = {
            "severity": "major", "kind": "role", "description": "Таблицу делает Борис.",
            "source_ids": [first["primary_utterances"][0]["id"]],
            "affected": [{"section": "tasks", "index": 0}],
            "status": "repaired", "patch_indices": [0],
        }
        patch = {"section": "tasks", "operation": "replace", "index": 0,
                 "item_json": json.dumps(corrected, ensure_ascii=False)}
        one = self._report(first, finding=finding, patch=patch)
        two = self._report(second)
        combined, warnings = merge_segment_review_reports(
            [one, two], self.segments, self.draft, self.source_index)
        self.assertEqual(warnings, [])
        revised, unresolved = apply_audit(self.draft, combined, self.source_index)
        self.assertEqual(unresolved, [])
        self.assertEqual(revised["tasks"][0]["assignee"], "Борис")
        partial, partial_warnings = merge_segment_review_reports(
            [one], self.segments, self.draft, self.source_index, allow_partial=True)
        partial_revised, partial_unresolved = apply_audit(
            self.draft, partial, self.source_index)
        self.assertEqual(partial_warnings, [])
        self.assertEqual(partial_unresolved, [])
        self.assertEqual(partial_revised["tasks"][0]["assignee"], "Борис")
        with self.assertRaisesRegex(ValueError, "one report per segment"):
            merge_segment_review_reports(
                [one], self.segments, self.draft, self.source_index)
        self.assertEqual(segment_review_warnings(one, first,
                                                 self.draft, self.source_index), [])

        internally_conflicting = copy.deepcopy(one)
        internally_conflicting["findings"][0]["status"] = "unresolved"
        with self.assertRaisesRegex(ValueError, "unresolved finding cannot apply patches"):
            validate_segment_review_report(
                internally_conflicting, first, self.draft, self.source_index)

        conflicting = copy.deepcopy(two)
        conflicting["findings"] = [{
            "severity": "major", "kind": "role", "description": "Роль требует уточнения.",
            "source_ids": [second["primary_utterances"][0]["id"]],
            "affected": [{"section": "tasks", "index": 0}],
            "status": "repaired", "patch_indices": [0],
        }]
        other = _task(None, ["U00001", "U00002"])
        conflicting["patches"] = [{"section": "tasks", "operation": "replace", "index": 0,
                                   "item_json": json.dumps(other, ensure_ascii=False)}]
        conflicting["item_assessments"][0]["finding_indices"] = [0]
        conflicting["item_assessments"][0]["status"] = "uncertain"
        # The other segment's task assessment is on its primary source ID.
        combined, warnings = merge_segment_review_reports(
            [one, conflicting], self.segments, self.draft, self.source_index)
        self.assertIn("cross_segment_patch_conflict", {w["code"] for w in warnings})
        self.assertEqual(combined["patches"], [])
        self.assertEqual([f["status"] for f in combined["findings"]],
                         ["unresolved", "unresolved"])
        revised, unresolved = apply_audit(self.draft, combined, self.source_index)
        self.assertEqual(revised["tasks"][0]["assignee"], "Алиса")
        self.assertEqual(len(unresolved), 2)


if __name__ == "__main__":
    unittest.main()
