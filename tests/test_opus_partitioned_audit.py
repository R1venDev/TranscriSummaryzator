"""Synthetic offline checks for Opus review of the whole source in parts.

These cases are deliberately unrelated to the private meeting. They test
source accounting and report grounding; a valid report still does not prove
that a model noticed every material fact.
"""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from summary.gemini_v1.inventory_contract import validate_inventory_plan
from summary.luna_v1.audit import apply_audit
from summary.luna_v1.source import load_source
from summary.opus_v1.contract import (
    OPUS_SEGMENT_SCHEMA_ID,
    merge_opus_segment_reports,
    validate_opus_segment_report,
)
from summary.opus_v1.payload import (
    build_opus_segment_audit_input,
    plan_opus_audit_segments,
)


def _source(count: int) -> tuple[str, dict]:
    """A varied 411-utterance-like conversation with boundary risk cases."""
    utterances = []
    for number in range(1, count + 1):
        text = f"Обсуждаем пункт {number}."
        if number == 2:
            text = "Если данные придут завтра, проверим зоны или дисбалансы, не оба варианта."
        elif number == 3:
            text = "Предлагаю это проверить, решение ещё не принято."
        elif number == count - 1:
            text = "Точка появится после пробоя M15."
        elif number == count:
            text = "Поправка: точка появится при пересечении нарисованной линии."
        utterances.append({
            "start": number * 1.2,
            "end": number * 1.2 + 0.7,
            "speaker": "a" if number % 2 else "b",
            "text": text,
        })
    transcript = {
        "source": "Синтетический пример.mkv",
        "duration_seconds": count * 1.2 + 2,
        "speakers": {"a": "Алиса", "b": "Борис"},
        "utterances": utterances,
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "transcript.json"
        path.write_text(json.dumps(transcript, ensure_ascii=False), encoding="utf-8")
        source_text, index, _sha = load_source(path)
    return source_text, index


def _draft() -> dict:
    return {
        "schema_version": "luna_summary_v1",
        "meeting": {"topic": "Учебная встреча", "project": None},
        "main": [{"text": "Проверят зоны при поступлении данных.",
                  "source_ids": ["U00002"]}],
        "timecodes": [],
        "tasks": [],
        "questions": [],
        "technical": [{"text": "Точка появится после пробоя M15.",
                       "source_ids": ["U00410"]}],
        "ideas": [],
        "verification": [],
        "chapters": [],
    }


def _empty_report(segment: dict, source_text: str, draft: dict) -> dict:
    """A shape-only report; tests below replace material rows explicitly."""
    payload = json.loads(build_opus_segment_audit_input(source_text, draft, segment))
    primary_ids = [row["id"] for row in segment["primary_utterances"]]
    positions = {source_id: number for number, source_id in enumerate(primary_ids)}

    def anchors(window: dict) -> list[str]:
        start, end = positions[window["start_id"]], positions[window["end_id"]]
        members = set(primary_ids[start:end + 1])
        return [item["anchor_id"] for item in payload["RISK_ANCHORS"]
                if item["source_id"] in members]

    return {
        "schema_version": OPUS_SEGMENT_SCHEMA_ID,
        "segment_id": segment["segment_id"],
        "coverage": [{
            "window_id": window["window_id"],
            "start_id": window["start_id"],
            "end_id": window["end_id"],
            "salient": "Несущественный организационный обмен",
            "source_quote": next(row["text"] for row in segment["primary_utterances"]
                                 if row["id"] == window["start_id"]),
            "draft_coverage": "no_material_item",
            "draft_evidence": [],
            "reviewed_anchor_ids": anchors(window),
            "finding_indices": [],
        } for window in segment["coverage_windows"]],
        "findings": [],
        "patches": [],
    }


class OpusPartitionedAuditTests(unittest.TestCase):
    def test_short_sources_still_cover_every_utterance_once(self):
        for count in (1, 2, 3, 4):
            with self.subTest(count=count):
                source_text, index = _source(count)
                segments = plan_opus_audit_segments(source_text)
                self.assertEqual(len(segments), min(3, count))
                self.assertIs(validate_inventory_plan(segments, index), segments)
                self.assertEqual(
                    [row["id"] for segment in segments
                     for row in segment["primary_utterances"]],
                    list(index["by_id"]),
                )

    def test_all_411_utterances_are_primary_once_with_bounded_real_overlap(self):
        source_text, index = _source(411)
        segments = plan_opus_audit_segments(source_text)
        self.assertEqual(len(segments), 3)
        self.assertEqual([item["segment_id"] for item in segments],
                         ["S01", "S02", "S03"])
        self.assertIs(validate_inventory_plan(segments, index), segments)
        source_ids = list(index["by_id"])
        primary_ids = [item["id"] for segment in segments
                       for item in segment["primary_utterances"]]
        self.assertEqual(primary_ids, source_ids)
        self.assertEqual(len(set(primary_ids)), 411)
        self.assertEqual(primary_ids[-1], "U00411")
        for segment in segments:
            self.assertLessEqual(len(segment["context_before"]), 12)
            self.assertLessEqual(len(segment["context_after"]), 12)
            self.assertTrue(segment["coverage_windows"])

    def test_plan_rejects_a_missing_primary_and_forged_context(self):
        source_text, index = _source(31)
        segments = plan_opus_audit_segments(source_text)
        missing = copy.deepcopy(segments)
        missing[1]["primary_utterances"].pop(0)
        with self.assertRaisesRegex(ValueError, "complete source exactly once"):
            validate_inventory_plan(missing, index)
        forged = copy.deepcopy(segments)
        forged[1]["context_before"][-1]["text"] = "Подменённый текст"
        with self.assertRaisesRegex(ValueError, "canonical source"):
            validate_inventory_plan(forged, index)

    def test_each_payload_contains_scoped_source_and_entire_draft(self):
        source_text, index = _source(411)
        draft = _draft()
        segments = plan_opus_audit_segments(source_text)
        primary_seen = []
        for segment in segments:
            payload = json.loads(build_opus_segment_audit_input(
                source_text, draft, segment))
            self.assertEqual(payload["DRAFT_DOCUMENT"], draft)
            self.assertEqual(payload["SEGMENT_ID"], segment["segment_id"])
            self.assertEqual(payload["SOURCE_WINDOWS"],
                             segment["coverage_windows"])
            source_rows = payload["TRANSCRIPT_SOURCE"]["utterances"]
            expected_rows = (segment["context_before"] + segment["primary_utterances"]
                             + segment["context_after"])
            self.assertEqual(source_rows, expected_rows)
            for row in expected_rows:
                self.assertEqual(index["by_id"][row["id"]], row)
            self.assertLess(len(source_rows), 411)
            self.assertIn("RISK_ANCHORS", payload)
            primary_seen.extend(item["id"] for item in segment["primary_utterances"])
        self.assertEqual(primary_seen, list(index["by_id"]))

    def test_risk_hints_include_condition_or_and_late_correction(self):
        source_text, _index = _source(411)
        segments = plan_opus_audit_segments(source_text)
        draft = _draft()
        first = json.loads(build_opus_segment_audit_input(source_text, draft, segments[0]))
        last = json.loads(build_opus_segment_audit_input(source_text, draft, segments[-1]))
        self.assertIn("U00002", json.dumps(first["RISK_ANCHORS"], ensure_ascii=False))
        self.assertIn("U00411", json.dumps(last["RISK_ANCHORS"], ensure_ascii=False))
        # Anchors are hints, not an expected answer or proof of coverage.
        self.assertNotIn("RISK_ANCHORS", draft)

    def test_covered_window_requires_a_real_current_draft_quote(self):
        source_text, index = _source(411)
        segment = plan_opus_audit_segments(source_text)[0]
        draft = _draft()
        draft["main"][0]["text"] = (
            "Если данные придут завтра, предложено проверить зоны или "
            "дисбалансы, не оба варианта. Решение ещё не принято."
        )
        report = _empty_report(segment, source_text, draft)
        row = report["coverage"][0]
        row["salient"] = "Условный выбор одного из двух тестов"
        row["source_quote"] = "Если данные придут завтра"
        row["draft_coverage"] = "covered"
        row["draft_evidence"] = [{
            "section": "main", "index": 0,
            "quote": "проверить зоны или дисбалансы, не оба варианта",
        }]
        self.assertIs(validate_opus_segment_report(report, segment, draft, index), report)

        missing = copy.deepcopy(report)
        missing["coverage"][0]["draft_evidence"] = []
        with self.assertRaises(ValueError):
            validate_opus_segment_report(missing, segment, draft, index)

        invented = copy.deepcopy(report)
        invented["coverage"][0]["draft_evidence"][0]["quote"] = (
            "проверить зоны и дисбалансы обязательно"
        )
        with self.assertRaises(ValueError):
            validate_opus_segment_report(invented, segment, draft, index)

        wrong_index = copy.deepcopy(report)
        wrong_index["coverage"][0]["draft_evidence"][0]["index"] = 1
        with self.assertRaises(ValueError):
            validate_opus_segment_report(wrong_index, segment, draft, index)

    def test_every_risk_anchor_is_acknowledged_once_in_its_primary_window(self):
        source_text, index = _source(411)
        segment = plan_opus_audit_segments(source_text)[0]
        draft = _draft()
        report = _empty_report(segment, source_text, draft)
        self.assertGreater(len(report["coverage"][0]["reviewed_anchor_ids"]), 1)
        self.assertTrue(report["coverage"][1]["reviewed_anchor_ids"])
        self.assertIs(validate_opus_segment_report(report, segment, draft, index), report)

        missing = copy.deepcopy(report)
        missing["coverage"][0]["reviewed_anchor_ids"].pop(0)
        with self.assertRaises(ValueError):
            validate_opus_segment_report(missing, segment, draft, index)

        repeated = copy.deepcopy(report)
        repeated["coverage"][0]["reviewed_anchor_ids"].append(
            repeated["coverage"][0]["reviewed_anchor_ids"][0])
        with self.assertRaises(ValueError):
            validate_opus_segment_report(repeated, segment, draft, index)

        wrong_window = copy.deepcopy(report)
        displaced = wrong_window["coverage"][0]["reviewed_anchor_ids"].pop(0)
        wrong_window["coverage"][1]["reviewed_anchor_ids"].append(displaced)
        with self.assertRaises(ValueError):
            validate_opus_segment_report(wrong_window, segment, draft, index)

        fabricated = copy.deepcopy(report)
        fabricated["coverage"][0]["reviewed_anchor_ids"].append("R999")
        with self.assertRaises(ValueError):
            validate_opus_segment_report(fabricated, segment, draft, index)

    def test_condition_alternative_and_late_correction_findings_are_grounded(self):
        source_text, index = _source(411)
        segments = plan_opus_audit_segments(source_text)
        draft = _draft()
        first = _empty_report(segments[0], source_text, draft)
        first["coverage"][0].update({
            "salient": "Условие и выбор теста",
            "source_quote": "Если данные придут завтра",
            "draft_coverage": "partial",
            "draft_evidence": [{"section": "main", "index": 0,
                                "quote": "Проверят зоны при поступлении данных"}],
            "finding_indices": [0],
        })
        corrected_main = {
            "text": "Если данные придут завтра, предложено проверить зоны или "
                    "дисбалансы, не оба варианта. Решение ещё не принято.",
            "source_ids": ["U00002", "U00003"],
        }
        first["findings"] = [{
            "severity": "major", "kind": "alternative",
            "description": "Черновик утратил условие, альтернативу и статус предложения.",
            "evidence_quote": "зоны или дисбалансы, не оба варианта",
            "source_ids": ["U00002", "U00003"],
            "affected": [{"section": "main", "index": 0}],
            "status": "repaired", "patch_indices": [0],
        }]
        first["patches"] = [{
            "section": "main", "operation": "replace", "index": 0,
            "item_json": json.dumps(corrected_main, ensure_ascii=False),
        }]
        self.assertIs(validate_opus_segment_report(
            first, segments[0], draft, index), first)

        last = _empty_report(segments[-1], source_text, draft)
        last["coverage"][-1].update({
            "salient": "Поздняя адресная поправка к триггеру",
            "source_quote": "Поправка: точка появится",
            "draft_coverage": "partial",
            "draft_evidence": [{"section": "technical", "index": 0,
                                "quote": "после пробоя M15"}],
            "finding_indices": [0],
        })
        last["findings"] = [{
            "severity": "major", "kind": "late_correction",
            "description": "Поздняя реплика меняет момент появления точки.",
            "evidence_quote": "при пересечении нарисованной линии",
            "source_ids": ["U00410", "U00411"],
            "affected": [{"section": "technical", "index": 0}],
            "status": "repaired", "patch_indices": [0],
        }]
        last["patches"] = [{
            "section": "technical", "operation": "replace", "index": 0,
            "item_json": json.dumps({
                "text": "Точка появится при пересечении нарисованной линии.",
                "source_ids": ["U00410", "U00411"],
            }, ensure_ascii=False),
        }]
        self.assertIs(validate_opus_segment_report(
            last, segments[-1], draft, index), last)

        forged = copy.deepcopy(last)
        forged["findings"][0]["evidence_quote"] = "Точка появится ровно в 12:00"
        with self.assertRaises(ValueError):
            validate_opus_segment_report(forged, segments[-1], draft, index)

    def test_conflicting_edits_to_one_draft_item_are_not_silently_applied(self):
        source_text, index = _source(411)
        segments = plan_opus_audit_segments(source_text)
        draft = _draft()
        reports = [_empty_report(segment, source_text, draft) for segment in segments]
        alternatives = (
            "Если данные придут завтра, предложено проверить зоны или дисбалансы.",
            "Нужно проверить только зоны после получения данных.",
        )
        for number in (0, 1):
            segment = segments[number]
            report = reports[number]
            source_row = segment["primary_utterances"][1 if number == 0 else 0]
            report["coverage"][0].update({
                "salient": "Спорная правка того же пункта черновика",
                "source_quote": source_row["text"],
                "draft_coverage": "partial",
                "draft_evidence": [{"section": "main", "index": 0,
                                    "quote": "Проверят зоны при поступлении данных"}],
                "finding_indices": [0],
            })
            report["findings"] = [{
                "severity": "major", "kind": "alternative",
                "description": "Предложена несовместимая редакция пункта.",
                "evidence_quote": source_row["text"],
                "source_ids": [source_row["id"]],
                "affected": [{"section": "main", "index": 0}],
                "status": "repaired", "patch_indices": [0],
            }]
            report["patches"] = [{
                "section": "main", "operation": "replace", "index": 0,
                "item_json": json.dumps({"text": alternatives[number],
                                         "source_ids": [source_row["id"]]},
                                        ensure_ascii=False),
            }]
            validate_opus_segment_report(report, segment, draft, index)
        combined, warnings = merge_opus_segment_reports(
            reports, segments, draft, index)
        self.assertIn("cross_segment_patch_conflict", {row["code"] for row in warnings})
        self.assertFalse(any(patch["section"] == "main" and patch["index"] == 0
                             for patch in combined["patches"]))
        self.assertEqual([finding["status"] for finding in combined["findings"]],
                         ["unresolved", "unresolved"])
        with self.assertRaises(ValueError):
            merge_opus_segment_reports(reports[:-1], segments, draft, index)

    def test_cross_scope_replacement_is_held_but_local_insert_survives(self):
        source_text, index = _source(411)
        segments = plan_opus_audit_segments(source_text)
        draft = _draft()
        # This broad item also relies on a distant source. A reviewer of S01
        # has not seen that source and cannot safely rewrite the entire item.
        draft["main"][0]["source_ids"] = ["U00002", "U00410"]
        reports = [_empty_report(segment, source_text, draft) for segment in segments]
        first = reports[0]
        first["coverage"][0].update({
            "salient": "Условная альтернатива и открытый статус",
            "source_quote": "Если данные придут завтра",
            "draft_coverage": "partial",
            "draft_evidence": [{"section": "main", "index": 0,
                                "quote": "Проверят зоны при поступлении данных"}],
            "finding_indices": [0, 1],
        })
        first["findings"] = [
            {
                "severity": "major", "kind": "alternative",
                "description": "Изменить условную альтернативу в широком пункте.",
                "evidence_quote": "зоны или дисбалансы",
                "source_ids": ["U00002"],
                "affected": [{"section": "main", "index": 0}],
                "status": "repaired", "patch_indices": [0],
            },
            {
                "severity": "minor", "kind": "omission",
                "description": "Отразить открытый статус отдельным вопросом.",
                "evidence_quote": "решение ещё не принято",
                "source_ids": ["U00003"], "affected": [],
                "status": "repaired", "patch_indices": [1],
            },
        ]
        first["patches"] = [
            {"section": "main", "operation": "replace", "index": 0,
             "item_json": json.dumps({
                 "text": "Если данные придут завтра, предложено проверить зоны или дисбалансы.",
                 "source_ids": ["U00002", "U00410"],
             }, ensure_ascii=False)},
            {"section": "questions", "operation": "insert", "index": 0,
             "item_json": json.dumps({
                 "text": "Решено ли проводить проверку?",
                 "source_ids": ["U00003"],
             }, ensure_ascii=False)},
        ]
        validate_opus_segment_report(first, segments[0], draft, index)
        combined, warnings = merge_opus_segment_reports(
            reports, segments, draft, index)
        self.assertIn("cross_scope_patch_requires_review",
                      {warning["code"] for warning in warnings})
        self.assertEqual([(patch["section"], patch["operation"])
                          for patch in combined["patches"]],
                         [("questions", "insert")])
        self.assertEqual([finding["status"] for finding in combined["findings"]],
                         ["unresolved", "repaired"])
        revised, unresolved = apply_audit(draft, combined, index)
        self.assertEqual(revised["main"][0], draft["main"][0])
        self.assertEqual(revised["questions"][0]["text"], "Решено ли проводить проверку?")
        self.assertEqual(len(unresolved), 1)

        removal_reports = copy.deepcopy(reports)
        removal_reports[0]["patches"][0].update({
            "operation": "remove", "item_json": None,
        })
        validate_opus_segment_report(removal_reports[0], segments[0], draft, index)
        removal_combined, removal_warnings = merge_opus_segment_reports(
            removal_reports, segments, draft, index)
        self.assertIn("cross_scope_patch_requires_review",
                      {warning["code"] for warning in removal_warnings})
        removed_candidate, _ = apply_audit(draft, removal_combined, index)
        self.assertEqual(removed_candidate["main"][0], draft["main"][0])
        self.assertEqual(removed_candidate["questions"][0]["text"],
                         "Решено ли проводить проверку?")


if __name__ == "__main__":
    unittest.main()
