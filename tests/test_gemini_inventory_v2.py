"""Free, synthetic regression tests for evidence-backed source inventory v2."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from summary.gemini_v1.inventory_contract import (
    INVENTORY_SCHEMA_ID, plan_inventory_segments,
)
from summary.gemini_v1.inventory_v2 import (
    INVENTORY_PROMPT_PATH_V2, INVENTORY_SCHEMA_ID_V2, INVENTORY_SCHEMA_V2,
    MERGED_INVENTORY_SCHEMA_ID_V2, build_inventory_input_v2,
    inventory_risk_warnings_v2, merge_inventory_reports_v2,
    validate_inventory_report_v2,
)
from summary.luna_v1 import load_source


class SourceInventoryV2Tests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        source = Path(temporary.name) / "transcript.json"
        lines = [
            ("p1", "Борис, подготовь таблицу."),
            ("p2", "В расчёте 80 000 свечей."),
            ("p1", "Проверим с 17:00 до 21:00."),
            ("p2", "Не выполняем без согласия."),
            ("p1", "Если задержка останется, попробуем тест зоны."),
            ("p2", "CSV или XLSX — пока выбор открыт."),
            ("p1", "Нет, подожди: прежний срок отменён."),
            ("p2", "Я передам отчёт Борису."),
        ]
        source.write_text(json.dumps({
            "source": "Учебная встреча.mkv", "duration_seconds": 40,
            "speakers": {"p1": "Алиса", "p2": "Борис"},
            "utterances": [
                {"start": float(i * 5), "end": float(i * 5 + 4),
                 "speaker": speaker, "text": line}
                for i, (speaker, line) in enumerate(lines)
            ],
        }, ensure_ascii=False), encoding="utf-8")
        self.source_text, self.source_index, _ = load_source(source)
        self.segments = plan_inventory_segments(
            self.source_text, count=2, overlap=2, windows_per_segment=2)

    def _report(self, segment: dict, *, with_item: bool) -> dict:
        primary = segment["primary_utterances"]
        first = primary[0]
        item = {
            "kind": "action", "claim": "Подготовить таблицу.",
            "source_ids": [first["id"]], "speaker": first["speaker"],
            "actor": None, "recipient": None, "action": "Подготовить таблицу",
            "modality": "proposed", "condition": None,
            "alternatives": [], "correction_of": [], "uncertainty": None,
            "source_quote": first["text"],
        }
        coverage = []
        for window in segment["coverage_windows"]:
            has_first = (window["start_id"] <= first["id"] <= window["end_id"])
            indices = [0] if with_item and has_first else []
            coverage.append({
                "window_id": window["window_id"], "start_id": window["start_id"],
                "end_id": window["end_id"], "assessment": "material_items" if indices else "uncertain",
                "item_indices": indices,
            })
        return {
            "schema_version": INVENTORY_SCHEMA_ID_V2,
            "segment_id": segment["segment_id"],
            "coverage": coverage, "items": [item] if with_item else [],
        }

    def test_payload_is_source_only_and_schema_requires_exact_quote(self):
        payload = json.loads(build_inventory_input_v2(self.segments[0]))
        self.assertEqual(list(payload), ["SOURCE_SEGMENT", "TASK"])
        self.assertNotIn("DRAFT_DOCUMENT", json.dumps(payload))
        self.assertIn("source_quote", payload["TASK"])
        self.assertIn("source_quote", INVENTORY_SCHEMA_V2["properties"]["items"]["items"]["required"])
        self.assertEqual(INVENTORY_SCHEMA_V2["properties"]["schema_version"]["enum"],
                         [INVENTORY_SCHEMA_ID_V2])
        self.assertIn("Черновик конспекта тебе не показан",
                      INVENTORY_PROMPT_PATH_V2.read_text(encoding="utf-8"))

    def test_quote_mismatch_is_visible_uncertainty_and_missing_field_is_hard(self):
        segment = self.segments[0]
        report = self._report(segment, with_item=True)
        self.assertIs(validate_inventory_report_v2(report, segment), report)

        wrong = copy.deepcopy(report)
        wrong["items"][0]["source_quote"] = "Подготовь таблицу"  # wrong case
        self.assertIs(validate_inventory_report_v2(wrong, segment), wrong)
        self.assertEqual([row["code"] for row in inventory_risk_warnings_v2(wrong, segment)
                          if row["risk_kind"] == "quote"], ["source_quote_mismatch"])

        other = copy.deepcopy(report)
        other["items"][0]["source_quote"] = segment["primary_utterances"][1]["text"]
        self.assertIs(validate_inventory_report_v2(other, segment), other)
        merged = merge_inventory_reports_v2([other, self._report(self.segments[1], with_item=False)],
                                             self.segments, self.source_index)
        self.assertEqual(merged["items"][0]["source_quote"],
                         segment["primary_utterances"][1]["text"])
        self.assertTrue(any(row["code"] == "source_quote_mismatch"
                            for row in merged["risk_warnings"]))

        no_quote = copy.deepcopy(report)
        del no_quote["items"][0]["source_quote"]
        with self.assertRaisesRegex(ValueError, "source_quote"):
            validate_inventory_report_v2(no_quote, segment)

        bad_ids = copy.deepcopy(report)
        bad_ids["items"][0]["source_ids"] = ["U99999"]
        with self.assertRaisesRegex(ValueError, "source ID"):
            validate_inventory_report_v2(bad_ids, segment)

    def test_uncited_risk_anchors_are_visible_uncertainty_not_model_facts(self):
        reports = [self._report(self.segments[0], with_item=True),
                   self._report(self.segments[1], with_item=False)]
        merged = merge_inventory_reports_v2(reports, self.segments, self.source_index)
        self.assertEqual(merged["schema_version"], MERGED_INVENTORY_SCHEMA_ID_V2)
        self.assertEqual(merged["items"][0]["source_quote"],
                         self.segments[0]["primary_utterances"][0]["text"])
        self.assertEqual(merged["primary_utterance_count"], 8)
        warnings = merged["risk_warnings"]
        self.assertTrue(warnings)
        self.assertTrue(all(row["code"] == "source_risk_anchor_uncited" and
                            row["status"] == "uncertain" and len(row["source_ids"]) == 1
                            for row in warnings))
        self.assertTrue({"number", "time", "negation", "condition", "alternative",
                         "correction", "actor"}.issubset({row["risk_kind"] for row in warnings}))
        self.assertTrue(any(row["source_ids"] == ["U00002"] and "80 000" in row["anchors"]
                            for row in warnings))
        self.assertTrue(any(row["source_ids"] == ["U00005"] and row["risk_kind"] == "condition"
                            for row in warnings))
        self.assertTrue(any(row["source_ids"] == ["U00007"] and row["risk_kind"] == "correction"
                            for row in warnings))

        # A citation can be semantically wrong; lexical warnings deliberately
        # disappear for that U and reconciliation must still check its claim.
        first = copy.deepcopy(reports[0])
        first["items"][0]["claim"] = "Таблица уже передана."
        self.assertIs(validate_inventory_report_v2(first, self.segments[0]), first)
        self.assertFalse(any(row["source_ids"] == ["U00001"] for row in
                             inventory_risk_warnings_v2(first, self.segments[0])))

    def test_v1_report_version_is_not_implicitly_upgraded(self):
        report = self._report(self.segments[0], with_item=True)
        report["schema_version"] = INVENTORY_SCHEMA_ID
        with self.assertRaisesRegex(ValueError, "version differs"):
            validate_inventory_report_v2(report, self.segments[0])


if __name__ == "__main__":
    unittest.main()
