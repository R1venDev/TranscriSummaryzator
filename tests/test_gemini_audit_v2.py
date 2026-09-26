"""V2 Gemini source inventory uses only artificial dialogue and no API."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from summary.gemini_v1.audit_v2 import (
    GEMINI_AUDIT_PROMPT_PATH_V2,
    GEMINI_AUDIT_SCHEMA_ID_V2,
    GEMINI_AUDIT_SCHEMA_V2,
    apply_gemini_audit_v2,
    coverage_warnings_gemini_v2,
    legacy_report_v1,
    validate_gemini_audit_v2,
)
from summary.luna_v1 import SCHEMA_ID, load_source, validate_document
from summary.luna_v1.audit import AUDIT_SCHEMA_ID, validate_audit


def _task(title: str, description: str, source_id: str, assignee: str | None) -> dict:
    return {
        "title": title, "description": description, "discussion_status": "proposed",
        "assignee": assignee, "due": None, "priority": None, "recipient": None,
        "source_ids": [source_id],
        "field_sources": {
            "action": [source_id], "assignee": [source_id] if assignee else [],
            "due": [], "priority": [], "recipient": [],
            "discussion_status": [source_id],
        },
    }


def _finding(kind: str, source_id: str, affected: list[dict], patch_index: int) -> dict:
    return {
        "severity": "major", "kind": kind, "description": f"Синтетическая ошибка: {kind}.",
        "source_ids": [source_id], "affected": affected,
        "status": "repaired", "patch_indices": [patch_index],
    }


def _material(kind: str, claim: str, source_id: str, targets: list[dict],
              findings: list[int]) -> dict:
    return {
        "kind": kind, "claim": claim, "source_ids": [source_id],
        "draft_targets": targets, "finding_indices": findings,
    }


class GeminiAuditV2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        transcript = Path(self.tmp.name) / "transcript.json"
        transcript.write_text(json.dumps({
            "source": "Учебный разговор.mkv", "duration_seconds": 24,
            "speakers": {"p1": "Алиса", "p2": "Борис"},
            "utterances": [
                {"start": 0, "end": 5, "speaker": "p1",
                 "text": "Борис подготовит таблицу; я только передам критерии."},
                {"start": 6, "end": 11, "speaker": "p2",
                 "text": "Предлагаю передать Алисе выгрузку за месяц, с шагом 15 минут или час."},
                {"start": 12, "end": 17, "speaker": "p1",
                 "text": "Поправка: проверяем три варианта, а не пять."},
                {"start": 18, "end": 23, "speaker": "p2",
                 "text": "Остаётся вопрос, когда получим исходные данные."},
            ],
        }, ensure_ascii=False), encoding="utf-8")
        _, self.index, _ = load_source(transcript)
        self.draft = {
            "schema_version": SCHEMA_ID,
            "meeting": {"topic": "Учебная проверка", "project": None},
            "main": [{"text": "Обсудили таблицу и варианты.", "source_ids": ["U00001", "U00003"]}],
            "timecodes": [{"topic": "Проверка", "start_id": "U00001", "end_id": "U00004"}],
            "tasks": [_task("Подготовить таблицу", "Подготовить таблицу для проверки.",
                            "U00001", "Алиса")],
            "questions": [{"text": "Когда поступят исходные данные?", "source_ids": ["U00004"]}],
            "technical": [{"text": "Нужно проверить пять вариантов.", "source_ids": ["U00003"]}],
            "ideas": [], "verification": [],
            "chapters": [{"topic": "Проверка", "start_id": "U00001", "end_id": "U00004",
                          "summary": "Обсудили таблицу, данные и варианты.",
                          "source_ids": ["U00001", "U00002", "U00003", "U00004"],
                          "details": []}],
        }
        validate_document(self.draft, self.index)

    def _report(self):
        correct_actor = _task("Подготовить таблицу", "Подготовить таблицу для проверки.",
                              "U00001", "Борис")
        handoff = _task("Передать выгрузку", "Передать Алисе выгрузку за месяц с шагом 15 минут или час.",
                        "U00002", None)
        correction = {"text": "Поправка: проверить три варианта, не пять.",
                      "source_ids": ["U00003"]}
        return {
            "schema_version": GEMINI_AUDIT_SCHEMA_ID_V2,
            "coverage": [
                {"window_id": "W01", "start_id": "U00001", "end_id": "U00001",
                 "salient": "Исполнитель таблицы", "draft_coverage": "covered",
                 "finding_indices": [0], "material_items": [
                     _material("action", "Борис подготовит таблицу.", "U00001",
                               [{"section": "tasks", "index": 0}], [0])]},
                {"window_id": "W02", "start_id": "U00002", "end_id": "U00002",
                 "salient": "Передача данных", "draft_coverage": "missing",
                 "finding_indices": [1], "material_items": [
                     _material("action", "Предложена передача месячной выгрузки Алисе с альтернативным шагом.",
                               "U00002", [], [1])]},
                {"window_id": "W03", "start_id": "U00003", "end_id": "U00003",
                 "salient": "Поздняя поправка", "draft_coverage": "covered",
                 "finding_indices": [2], "material_items": [
                     _material("correction", "Поправка меняет пять вариантов на три.", "U00003",
                               [{"section": "technical", "index": 0}], [2])]},
                {"window_id": "W04", "start_id": "U00004", "end_id": "U00004",
                 "salient": "Открытый вопрос", "draft_coverage": "covered",
                 "finding_indices": [], "material_items": [
                     _material("question", "Срок получения исходных данных открыт.", "U00004",
                               [{"section": "questions", "index": 0}], [])]},
            ],
            "findings": [
                _finding("role", "U00001", [{"section": "tasks", "index": 0}], 0),
                _finding("omission", "U00002", [], 1),
                _finding("late_correction", "U00003", [{"section": "technical", "index": 0}], 2),
            ],
            "patches": [
                {"section": "tasks", "operation": "replace", "index": 0,
                 "item_json": json.dumps(correct_actor, ensure_ascii=False)},
                {"section": "tasks", "operation": "insert", "index": 1,
                 "item_json": json.dumps(handoff, ensure_ascii=False)},
                {"section": "technical", "operation": "replace", "index": 0,
                 "item_json": json.dumps(correction, ensure_ascii=False)},
            ],
        }

    def test_versioned_contract_tracks_actor_omission_and_late_correction(self):
        original = copy.deepcopy(self.draft)
        report = self._report()
        self.assertIs(validate_gemini_audit_v2(report, self.draft, self.index), report)
        self.assertEqual(coverage_warnings_gemini_v2(report, self.index, self.draft), [])
        projected = legacy_report_v1(report)
        self.assertEqual(projected["schema_version"], AUDIT_SCHEMA_ID)
        self.assertNotIn("material_items", projected["coverage"][0])
        self.assertIs(validate_audit(projected, self.draft, self.index), projected)
        revised, unresolved = apply_gemini_audit_v2(self.draft, report, self.index)
        self.assertEqual(original, self.draft)
        self.assertEqual(unresolved, [])
        self.assertEqual(revised["tasks"][0]["assignee"], "Борис")
        self.assertEqual(len(revised["tasks"]), 2)
        self.assertIn("15 минут или час", revised["tasks"][1]["description"])
        self.assertIn("три варианта", revised["technical"][0]["text"])
        validate_document(revised, self.index)

    def test_all_covered_but_empty_atomic_inventory_downgrades_quality(self):
        report = self._report()
        report["coverage"][0]["material_items"] = []
        self.assertIs(validate_gemini_audit_v2(report, self.draft, self.index), report)
        warnings = coverage_warnings_gemini_v2(report, self.index, self.draft)
        self.assertIn("material_inventory_empty", {item["code"] for item in warnings})
        revised, _ = apply_gemini_audit_v2(self.draft, report, self.index)
        self.assertEqual(revised["tasks"][0]["assignee"], "Борис")

    def test_distant_window_anchor_and_missing_action_mapping_warn(self):
        report = self._report()
        report["coverage"][0]["material_items"][0]["source_ids"] = ["U00003"]
        self.assertIs(validate_gemini_audit_v2(report, self.draft, self.index), report)
        warnings = coverage_warnings_gemini_v2(report, self.index, self.draft)
        self.assertIn("material_source_outside_window", {item["code"] for item in warnings})

        report = self._report()
        report["coverage"][3]["material_items"][0]["kind"] = "action"
        self.assertIs(validate_gemini_audit_v2(report, self.draft, self.index), report)
        warnings = coverage_warnings_gemini_v2(report, self.index, self.draft)
        self.assertIn("action_missing_task_or_finding", {item["code"] for item in warnings})

    def test_window_finding_links_and_patches_remain_accounted_for(self):
        report = self._report()
        report["coverage"][1]["material_items"][0]["finding_indices"] = []
        self.assertIs(validate_gemini_audit_v2(report, self.draft, self.index), report)
        warnings = coverage_warnings_gemini_v2(report, self.index, self.draft)
        self.assertTrue({"material_unmapped", "material_finding_links_mismatch",
                         "material_findings_unlinked"}.issubset({item["code"] for item in warnings}))

        report = self._report()
        report["findings"][0]["patch_indices"] = []
        with self.assertRaisesRegex(ValueError, "corrective patch|repaired finding needs a patch"):
            validate_gemini_audit_v2(report, self.draft, self.index)

        report = self._report()
        report["coverage"][2]["finding_indices"] = []
        report["coverage"][2]["material_items"][0]["finding_indices"] = []
        self.assertIs(validate_gemini_audit_v2(report, self.draft, self.index), report)
        warnings = coverage_warnings_gemini_v2(report, self.index, self.draft)
        self.assertIn("material_findings_unlinked", {item["code"] for item in warnings})

    def test_salvageable_role_patch_survives_partial_label(self):
        report = self._report()
        report["coverage"][0]["draft_coverage"] = "partial"
        self.assertIs(validate_gemini_audit_v2(report, self.draft, self.index), report)
        warnings = coverage_warnings_gemini_v2(report, self.index, self.draft)
        self.assertIn("coverage_without_omission", {item["code"] for item in warnings})
        revised, _ = apply_gemini_audit_v2(self.draft, report, self.index)
        self.assertEqual(revised["tasks"][0]["assignee"], "Борис")

    def test_salvageable_unsupported_claim_patch_survives_no_material_label(self):
        report = self._report()
        report["findings"][2]["kind"] = "unsupported_claim"
        report["coverage"][2]["draft_coverage"] = "no_material_item"
        self.assertIs(validate_gemini_audit_v2(report, self.draft, self.index), report)
        warnings = coverage_warnings_gemini_v2(report, self.index, self.draft)
        self.assertIn("no_material_item_conflict", {item["code"] for item in warnings})
        revised, _ = apply_gemini_audit_v2(self.draft, report, self.index)
        self.assertIn("три варианта", revised["technical"][0]["text"])

    def test_missing_or_malformed_inventory_warns_without_discarding_patch(self):
        report = self._report()
        report["coverage"][0].pop("material_items")
        report["coverage"][1]["material_items"] = "not an array"
        self.assertIs(validate_gemini_audit_v2(report, self.draft, self.index), report)
        warnings = coverage_warnings_gemini_v2(report, self.index, self.draft)
        self.assertTrue({"material_inventory_missing", "material_inventory_not_array"}.issubset(
            {item["code"] for item in warnings}))
        revised, _ = apply_gemini_audit_v2(self.draft, report, self.index)
        self.assertEqual(len(revised["tasks"]), 2)

    def test_whole_dialogue_claimed_no_material_is_not_false_clean(self):
        report = self._report()
        report["findings"] = []
        report["patches"] = []
        for row in report["coverage"]:
            row["draft_coverage"] = "no_material_item"
            row["finding_indices"] = []
            row["material_items"] = []
        self.assertIs(validate_gemini_audit_v2(report, self.draft, self.index), report)
        warnings = coverage_warnings_gemini_v2(report, self.index, self.draft)
        self.assertIn("all_windows_no_material", {item["code"] for item in warnings})

    def test_schema_and_prompt_are_pinned_and_do_not_relabel_v1(self):
        self.assertEqual(GEMINI_AUDIT_SCHEMA_V2["properties"]["schema_version"]["enum"],
                         [GEMINI_AUDIT_SCHEMA_ID_V2])
        coverage = GEMINI_AUDIT_SCHEMA_V2["properties"]["coverage"]["items"]
        self.assertIn("material_items", coverage["required"])
        item = coverage["properties"]["material_items"]["items"]
        self.assertEqual(set(item["properties"]), set(item["required"]))
        prompt = GEMINI_AUDIT_PROMPT_PATH_V2.read_text(encoding="utf-8")
        for phrase in ("MODE=audit", "MODE=verify", "material_items", "позднюю поправку"):
            self.assertIn(phrase, prompt)


if __name__ == "__main__":
    unittest.main()
