"""Bidirectional reconciliation checks with artificial dialogue and no API."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from summary.gemini_v1.reconcile_contract import (
    RECONCILE_PROMPT_PATH,
    RECONCILE_SCHEMA,
    RECONCILE_SCHEMA_ID,
    apply_reconciliation_report,
    build_reconcile_input,
    draft_units,
    reconciliation_warnings,
    validate_reconciliation_report,
)
from summary.luna_v1 import SCHEMA_ID, load_source, validate_document


def _task(title: str, description: str, source_ids: list[str],
          assignee: str | None, *, status: str = "proposed") -> dict:
    return {
        "title": title, "description": description, "discussion_status": status,
        "assignee": assignee, "due": None, "priority": None, "recipient": None,
        "source_ids": source_ids,
        "field_sources": {
            "action": list(source_ids), "assignee": list(source_ids) if assignee else [],
            "due": [], "priority": [], "recipient": [],
            "discussion_status": list(source_ids),
        },
    }


class GeminiReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        transcript = Path(self.tmp.name) / "transcript.json"
        lines = [
            ("p1", "Борис, подготовь таблицу. Я только передам критерии."),
            ("p2", "Да, я подготовлю таблицу."),
            ("p1", "Предлагаю передать Алисе выгрузку за месяц: CSV или Excel."),
            ("p2", "Уточнение: проверяем три варианта, а не пять."),
            ("p1", "Когда придут исходные данные, пока неясно."),
            ("p2", "Для графика нужна выборка не менее 80 строк."),
            ("p1", "Ранее я отправила тестовый файл; это уже сделано."),
            ("p2", "Отдельно обсуждали интерфейс, будущей задачи нет."),
        ]
        transcript.write_text(json.dumps({
            "source": "Учебный диалог.mkv", "duration_seconds": 60,
            "speakers": {"p1": "Алиса", "p2": "Борис"},
            "utterances": [
                {"start": number * 7.0, "end": number * 7.0 + 5.0,
                 "speaker": speaker, "text": line}
                for number, (speaker, line) in enumerate(lines)
            ],
        }, ensure_ascii=False), encoding="utf-8")
        self.source_text, self.source_index, self.source_sha = load_source(transcript)
        self.draft = {
            "schema_version": SCHEMA_ID,
            "meeting": {"topic": "Проверка таблицы и вариантов", "project": None},
            "main": [{"text": "Обсудили таблицу и варианты.", "source_ids": ["U00001", "U00004"]}],
            "timecodes": [{"topic": "Таблица", "start_id": "U00001", "end_id": "U00004"}],
            "tasks": [_task("Подготовить таблицу", "Подготовить таблицу для проверки.",
                            ["U00001", "U00002"], "Алиса", status="committed")],
            "questions": [{"text": "Когда придут исходные данные?", "source_ids": ["U00005"]}],
            "technical": [{"text": "Проверяют пять вариантов.", "source_ids": ["U00004"]}],
            "ideas": [], "verification": [],
            "chapters": [{"topic": "Таблица и варианты", "start_id": "U00001",
                          "end_id": "U00005", "summary": "Обсудили таблицу, выгрузку и варианты.",
                          "source_ids": ["U00001", "U00003", "U00004", "U00005"],
                          "details": [{"text": "Исходные данные пока ждут.", "source_ids": ["U00005"]}]}],
        }
        validate_document(self.draft, self.source_index)
        self.inventory = {
            "schema_version": "gemini_source_inventory_merged_v1",
            "source_sha256": self.source_sha,
            "source_fingerprint": hashlib.sha256(self.source_text.encode("utf-8")).hexdigest(),
            "primary_utterance_count": 8, "segment_count": 2,
            "coverage": [],
            "items": [
                {"item_id": "S01-I001", "segment_id": "S01", "kind": "action",
                 "claim": "Борис обещал подготовить таблицу.", "source_ids": ["U00001", "U00002"],
                 "speaker": "Борис", "actor": "Борис", "recipient": None,
                 "action": "Подготовить таблицу", "modality": "committed", "condition": None,
                 "alternatives": [], "correction_of": [], "uncertainty": None},
                {"item_id": "S01-I002", "segment_id": "S01", "kind": "action",
                 "claim": "Предложена передача месячной выгрузки в CSV или Excel.",
                 "source_ids": ["U00003"], "speaker": "Алиса", "actor": None,
                 "recipient": "Алиса", "action": "Передать месячную выгрузку",
                 "modality": "proposed", "condition": None,
                 "alternatives": ["CSV", "Excel"], "correction_of": [], "uncertainty": None},
                {"item_id": "S02-I001", "segment_id": "S02", "kind": "correction",
                 "claim": "Три варианта заменяют пять.", "source_ids": ["U00004"],
                 "speaker": "Борис", "actor": None, "recipient": None, "action": None,
                 "modality": "observation", "condition": None,
                 "alternatives": [], "correction_of": [], "uncertainty": None},
            ],
        }

    def _report(self):
        correct_task = _task("Подготовить таблицу", "Подготовить таблицу для проверки.",
                             ["U00001", "U00002"], "Борис", status="committed")
        offered_task = _task("Передать месячную выгрузку",
                             "Передать Алисе месячную выгрузку в CSV или Excel.",
                             ["U00003"], None)
        replacement = {"text": "После поправки проверяют три варианта, а не пять.",
                       "source_ids": ["U00004"]}
        assessments = []
        for unit in draft_units(self.draft):
            source_ids = (["U00001"] if unit["unit_id"] == "meeting:0" else
                          ["U00001", "U00002"] if unit["section"] == "tasks" else
                          ["U00004"] if unit["section"] == "technical" else
                          ["U00005"] if unit["section"] in {"questions", "chapters"} else
                          ["U00001"])
            status = ("partial" if unit["unit_id"] == "tasks:0:relations" else
                      "unsupported" if unit["section"] == "technical" else "supported")
            findings = ([0] if unit["unit_id"] == "tasks:0:relations" else
                        [2] if unit["section"] == "technical" else [])
            assessments.append({
                "unit_id": unit["unit_id"], "status": status,
                "source_ids": source_ids,
                "inventory_ids": ["S01-I001"] if unit["section"] == "tasks" else [],
                "finding_indices": findings,
            })
        return {
            "schema_version": RECONCILE_SCHEMA_ID,
            "source_window_assessments": [],
            "inventory_assessments": [
                {"item_id": "S01-I001", "status": "partial",
                 "draft_targets": [{"section": "tasks", "index": 0}], "finding_indices": [0]},
                {"item_id": "S01-I002", "status": "missing",
                 "draft_targets": [], "finding_indices": [1]},
                {"item_id": "S02-I001", "status": "contradicted",
                 "draft_targets": [{"section": "technical", "index": 0}], "finding_indices": [2]},
            ],
            "draft_assessments": assessments,
            "findings": [
                {"severity": "major", "kind": "role", "description": "Таблицу обещал Борис, не Алиса.",
                 "source_ids": ["U00001", "U00002"],
                 "affected": [{"section": "tasks", "index": 0}],
                 "status": "repaired", "patch_indices": [0]},
                {"severity": "major", "kind": "omission",
                 "description": "Пропущено конкретное предложение передать месячную выгрузку.",
                 "source_ids": ["U00003"], "affected": [],
                 "status": "repaired", "patch_indices": [1]},
                {"severity": "major", "kind": "late_correction",
                 "description": "Поздняя поправка: три варианта вместо пяти.",
                 "source_ids": ["U00004"],
                 "affected": [{"section": "technical", "index": 0}],
                 "status": "repaired", "patch_indices": [2]},
            ],
            "patches": [
                {"section": "tasks", "operation": "replace", "index": 0,
                 "item_json": json.dumps(correct_task, ensure_ascii=False)},
                {"section": "tasks", "operation": "insert", "index": 1,
                 "item_json": json.dumps(offered_task, ensure_ascii=False)},
                {"section": "technical", "operation": "replace", "index": 0,
                 "item_json": json.dumps(replacement, ensure_ascii=False)},
            ],
        }

    def test_bounded_excerpts_and_bidirectional_input(self):
        payload = json.loads(build_reconcile_input(self.source_text, self.draft, self.inventory))
        self.assertEqual(payload["MODE"], "reconcile")
        self.assertEqual(payload["DRAFT_DOCUMENT"], self.draft)
        self.assertEqual(len(payload["INDEPENDENT_SOURCE_INVENTORY"]["items"]), 3)
        included = [row["id"] for row in payload["SOURCE_EXCERPTS"]]
        self.assertEqual(included, [f"U{number:05d}" for number in range(1, 7)])
        self.assertNotIn("U00008", included)
        self.assertEqual(payload["EXCERPT_SCOPE"]["total_utterance_count"], 8)
        self.assertEqual(len(payload["DRAFT_UNITS"]), len(draft_units(self.draft)))
        self.assertEqual(list(payload)[-1], "TASK")

    def test_apply_actor_omission_and_correction_without_changing_sealed_draft(self):
        report = self._report()
        before = copy.deepcopy(self.draft)
        self.assertIs(validate_reconciliation_report(
            report, self.draft, self.inventory, self.source_index), report)
        self.assertEqual(reconciliation_warnings(report, self.draft, self.inventory,
                                                 self.source_index), [])
        revised, unresolved = apply_reconciliation_report(
            self.draft, report, self.inventory, self.source_index)
        self.assertEqual(self.draft, before)
        self.assertEqual(unresolved, [])
        self.assertEqual(revised["tasks"][0]["assignee"], "Борис")
        self.assertEqual(revised["tasks"][1]["assignee"], None)
        self.assertEqual(revised["tasks"][1]["discussion_status"], "proposed")
        self.assertIn("CSV или Excel", revised["tasks"][1]["description"])
        self.assertIn("три варианта", revised["technical"][0]["text"])
        validate_document(revised, self.source_index)

    def test_wrong_source_and_unsafe_patch_are_rejected(self):
        inventory = copy.deepcopy(self.inventory)
        inventory["source_fingerprint"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            build_reconcile_input(self.source_text, self.draft, inventory)
        inventory = copy.deepcopy(self.inventory)
        inventory["source_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "raw source SHA"):
            validate_reconciliation_report(self._report(), self.draft, inventory,
                                           self.source_index)
        report = self._report()
        report["patches"][0]["item_json"] = json.dumps(self.draft["tasks"][0],
                                                        ensure_ascii=False)
        with self.assertRaisesRegex(ValueError, "no-op"):
            validate_reconciliation_report(report, self.draft, self.inventory,
                                           self.source_index)
        report = self._report()
        report["findings"][0]["source_ids"] = ["U99999"]
        with self.assertRaisesRegex(ValueError, "unknown source ID"):
            validate_reconciliation_report(report, self.draft, self.inventory,
                                           self.source_index)
        draft = copy.deepcopy(self.draft)
        draft["chapters"][0]["details"][0]["source_ids"] = ["U99999"]
        with self.assertRaisesRegex(ValueError, "unknown draft source ID|invalid draft source IDs"):
            build_reconcile_input(self.source_text, draft, self.inventory)

    def test_missing_assessment_warns_but_does_not_discard_valid_patch(self):
        report = self._report()
        report["inventory_assessments"].pop(1)
        report["draft_assessments"].pop()
        self.assertIs(validate_reconciliation_report(report, self.draft,
                                                    self.inventory, self.source_index), report)
        codes = {row["code"] for row in reconciliation_warnings(
            report, self.draft, self.inventory, self.source_index)}
        self.assertIn("inventory_item_unassessed", codes)
        self.assertIn("draft_unit_unassessed", codes)
        revised, _ = apply_reconciliation_report(self.draft, report, self.inventory,
                                                  self.source_index)
        self.assertEqual(revised["tasks"][0]["assignee"], "Борис")

    def test_internal_conflicting_coverage_is_visible(self):
        report = self._report()
        report["inventory_assessments"][0]["status"] = "represented"
        warnings = reconciliation_warnings(report, self.draft, self.inventory,
                                           self.source_index)
        self.assertIn("represented_target_not_supported", {row["code"] for row in warnings})
        self.assertIs(validate_reconciliation_report(
            report, self.draft, self.inventory, self.source_index), report)

    def test_unresolved_role_with_affected_task_remains_visible_without_patch(self):
        report = self._report()
        report["findings"][0]["status"] = "unresolved"
        report["findings"][0]["patch_indices"] = []
        report["patches"].pop(0)
        report["findings"][1]["patch_indices"] = [0]
        report["findings"][2]["patch_indices"] = [1]
        self.assertIs(validate_reconciliation_report(
            report, self.draft, self.inventory, self.source_index), report)
        revised, unresolved = apply_reconciliation_report(
            self.draft, report, self.inventory, self.source_index)
        self.assertEqual(len(unresolved), 1)
        self.assertEqual(revised["tasks"][0]["assignee"], "Алиса")
        self.assertTrue(any("Таблицу обещал Борис" in row["text"]
                            for row in revised["verification"]))
        report["findings"][0]["status"] = "repaired"
        with self.assertRaisesRegex(ValueError, "repaired finding needs a patch"):
            validate_reconciliation_report(report, self.draft, self.inventory,
                                           self.source_index)

    def test_focused_verify_sends_only_related_inventory_and_units(self):
        prior = [self._report()["findings"][0]]
        payload = json.loads(build_reconcile_input(
            self.source_text, self.draft, self.inventory,
            mode="verify", prior_findings=prior))
        self.assertEqual(payload["MODE"], "verify")
        self.assertEqual([item["item_id"] for item in
                          payload["INDEPENDENT_SOURCE_INVENTORY"]["items"]],
                         ["S01-I001"])
        self.assertIn("tasks:0:relations", [unit["unit_id"] for unit in payload["DRAFT_UNITS"]])
        self.assertNotIn("U00008", [row["id"] for row in payload["SOURCE_EXCERPTS"]])
        report = self._report()
        report["inventory_assessments"] = report["inventory_assessments"][:1]
        report["draft_assessments"] = [row for row in report["draft_assessments"]
                                       if row["unit_id"].startswith("tasks:0")]
        self.assertNotIn("inventory_item_unassessed", {row["code"] for row in
            reconciliation_warnings(report, self.draft, self.inventory,
                                    self.source_index, mode="verify")})

    def test_uncertain_and_no_material_windows_are_sent_whole_and_accounted(self):
        inventory = copy.deepcopy(self.inventory)
        inventory["coverage"] = [
            {"segment_id": "S01", "window_id": "S01-W01", "start_id": "U00001",
             "end_id": "U00004", "assessment": "uncertain", "item_ids": []},
            {"segment_id": "S02", "window_id": "S02-W01", "start_id": "U00005",
             "end_id": "U00008", "assessment": "no_material_items", "item_ids": []},
        ]
        payload = json.loads(build_reconcile_input(self.source_text, self.draft, inventory))
        self.assertEqual(len(payload["SOURCE_WINDOWS_TO_RECHECK"]), 2)
        self.assertEqual([row["id"] for row in payload["SOURCE_EXCERPTS"]],
                         [f"U{number:05d}" for number in range(1, 9)])
        report = self._report()
        report["source_window_assessments"] = [
            {"window_id": "S01-W01", "status": "material_in_inventory",
             "source_ids": ["U00001", "U00002"], "finding_indices": []},
            {"window_id": "S02-W01", "status": "confirmed_no_material",
             "source_ids": ["U00007", "U00008"], "finding_indices": []},
        ]
        warnings = reconciliation_warnings(report, self.draft, inventory,
                                           self.source_index)
        self.assertIn("source_inventory_uncertain_window", {row["code"] for row in warnings})
        self.assertNotIn("no_material_window_unverified", {row["code"] for row in warnings})
        report["source_window_assessments"].pop()
        warnings = reconciliation_warnings(report, self.draft, inventory,
                                           self.source_index)
        self.assertIn("no_material_window_unverified", {row["code"] for row in warnings})

        inventory["items"] = []
        inventory["coverage"][0]["assessment"] = "no_material_items"
        warnings = reconciliation_warnings(report, self.draft, inventory,
                                           self.source_index)
        self.assertIn("all_inventory_empty", {row["code"] for row in warnings})

    def test_prompt_schema_are_versioned_and_generic(self):
        self.assertEqual(RECONCILE_SCHEMA["properties"]["schema_version"]["enum"],
                         [RECONCILE_SCHEMA_ID])
        self.assertEqual(set(RECONCILE_SCHEMA["required"]), set(RECONCILE_SCHEMA["properties"]))
        prompt = RECONCILE_PROMPT_PATH.read_text(encoding="utf-8")
        for marker in ("MODE=reconcile", "MODE=verify", "supported_uninventoried",
                       "tasks:*:relations", "не установленная истина"):
            self.assertIn(marker, prompt)
        self.assertNotIn("U003", prompt)


if __name__ == "__main__":
    unittest.main()
