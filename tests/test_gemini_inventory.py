"""Source-only inventory uses artificial dialogue and never sends an API call."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from summary.gemini_v1.inventory_contract import (
    INVENTORY_PROMPT_PATH,
    INVENTORY_SCHEMA,
    INVENTORY_SCHEMA_ID,
    MERGED_INVENTORY_SCHEMA_ID,
    build_inventory_input,
    merge_inventory_reports,
    plan_inventory_segments,
    validate_inventory_plan,
    validate_inventory_report,
)
from summary.luna_v1 import load_source


class GeminiSourceInventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        transcript = Path(self.tmp.name) / "transcript.json"
        lines = [
            ("p1", "Борис, подготовь таблицу по данным за месяц."),
            ("p2", "Я подготовлю таблицу к пятнице."),
            ("p1", "Можем проверить CSV или Excel; это один выбор формата."),
            ("p2", "Ранее я отправил тестовый файл, он уже готов."),
            ("p1", "Отдельно соберу замечания по формулам."),
            ("p2", "Только если данные поступят сегодня, успеем проверить график."),
            ("p1", "Нет, подожди: к понедельнику, а не к пятнице."),
            ("p2", "Принял поправку к сроку таблицы."),
            ("p1", "Для графика нужна выборка минимум 80 строк."),
            ("p2", "Когда получим эту выборку?"),
            ("p1", "Пока не знаю точную дату передачи."),
            ("p2", "Проверка графика требует тех же данных."),
        ]
        utterances = []
        for number, (speaker, line) in enumerate(lines):
            start = float(number * 6)
            # A plausible pause near the middle tests boundary hygiene; the
            # overlap is still necessary because the correction can cross it.
            if number >= 6:
                start += 20.0
            utterances.append({"start": start, "end": start + 4.0,
                               "speaker": speaker, "text": line})
        transcript.write_text(json.dumps({
            "source": "Учебная встреча.mkv", "duration_seconds": 90,
            "speakers": {"p1": "Алиса", "p2": "Борис"},
            "utterances": utterances,
        }, ensure_ascii=False), encoding="utf-8")
        self.source_text, self.source_index, self.source_sha = load_source(transcript)
        self.segments = plan_inventory_segments(
            self.source_text, count=2, overlap=2, windows_per_segment=3
        )

    @staticmethod
    def _action(source_id: str, speaker: str = "Алиса") -> dict:
        return {
            "kind": "action", "claim": "Алиса соберёт замечания по формулам.",
            "source_ids": [source_id], "speaker": speaker, "actor": "Алиса",
            "recipient": None, "action": "Собрать замечания по формулам",
            "modality": "committed", "condition": None, "alternatives": [],
            "correction_of": [], "uncertainty": None,
        }

    def _report(self, segment: dict, *, item: dict | None = None) -> dict:
        primary_id = segment["primary_utterances"][0]["id"]
        item = item or self._action(primary_id,
                                    speaker=segment["primary_utterances"][0]["speaker"])
        coverage = []
        for number, window in enumerate(segment["coverage_windows"]):
            coverage.append({
                "window_id": window["window_id"],
                "start_id": window["start_id"], "end_id": window["end_id"],
                # The artificial report does not assert semantic absence for
                # the other windows; coordinate validation is the test scope.
                "assessment": "material_items" if number == 0 else "uncertain",
                "item_indices": [0] if number == 0 else [],
            })
        return {
            "schema_version": INVENTORY_SCHEMA_ID,
            "segment_id": segment["segment_id"], "coverage": coverage,
            "items": [item],
        }

    def test_segments_partition_every_source_id_once_and_mark_overlap(self):
        self.assertEqual(len(self.segments), 2)
        primary_ids = [row["id"] for segment in self.segments
                       for row in segment["primary_utterances"]]
        self.assertEqual(primary_ids, list(self.source_index["by_id"]))
        self.assertEqual(len(primary_ids), len(set(primary_ids)))
        self.assertEqual(self.segments[0]["context_before"], [])
        self.assertEqual(self.segments[-1]["context_after"], [])
        self.assertEqual(
            [row["id"] for row in self.segments[0]["context_after"]],
            [row["id"] for row in self.segments[1]["primary_utterances"][:2]],
        )
        self.assertEqual(
            [row["id"] for row in self.segments[1]["context_before"]],
            [row["id"] for row in self.segments[0]["primary_utterances"][-2:]],
        )
        self.assertEqual(
            self.segments[0]["source_fingerprint"],
            hashlib.sha256(self.source_text.encode("utf-8")).hexdigest(),
        )
        self.assertIs(validate_inventory_plan(self.segments, self.source_index),
                      self.segments)
        for segment in self.segments:
            windows = segment["coverage_windows"]
            covered_ids = []
            ordered = [row["id"] for row in segment["primary_utterances"]]
            positions = {source_id: i for i, source_id in enumerate(ordered)}
            for window in windows:
                covered_ids.extend(ordered[positions[window["start_id"]]:
                                           positions[window["end_id"]] + 1])
            self.assertEqual(covered_ids, ordered)

    def test_input_is_source_only_and_keeps_host_task_last(self):
        payload = json.loads(build_inventory_input(self.segments[0]))
        self.assertEqual(list(payload), ["SOURCE_SEGMENT", "TASK"])
        self.assertNotIn("DRAFT_DOCUMENT", json.dumps(payload))
        self.assertNotIn("PRIOR_FINDINGS", json.dumps(payload))
        self.assertEqual(payload["SOURCE_SEGMENT"]["segment_id"], "S01")
        self.assertTrue(payload["SOURCE_SEGMENT"]["primary_utterances"])
        self.assertIn("CONTEXT_BEFORE/AFTER", payload["TASK"])
        self.assertEqual(INVENTORY_SCHEMA["properties"]["schema_version"]["enum"],
                         [INVENTORY_SCHEMA_ID])
        prompt = INVENTORY_PROMPT_PATH.read_text(encoding="utf-8")
        self.assertIn("Черновик конспекта тебе не показан", prompt)
        self.assertIn("уже выполненное", prompt)

    def test_reports_merge_to_host_assigned_ids_with_exact_source_identity(self):
        reports = [self._report(segment) for segment in self.segments]
        merged = merge_inventory_reports(reports, self.segments, self.source_index)
        self.assertEqual(merged["schema_version"], MERGED_INVENTORY_SCHEMA_ID)
        self.assertEqual(merged["source_sha256"], self.source_sha)
        self.assertEqual(merged["primary_utterance_count"], 12)
        self.assertEqual([item["item_id"] for item in merged["items"]],
                         ["S01-I001", "S02-I001"])
        self.assertEqual(merged["coverage"][0]["item_ids"], ["S01-I001"])

    def test_citation_in_context_only_is_rejected(self):
        second = self.segments[1]
        context_id = second["context_before"][0]["id"]
        report = self._report(second)
        report["items"][0]["source_ids"] = [context_id]
        with self.assertRaisesRegex(ValueError, "context-only"):
            validate_inventory_report(report, second)

    def test_unknown_duplicate_and_wrong_window_links_are_rejected(self):
        first = self.segments[0]
        report = self._report(first)
        report["items"][0]["source_ids"] = ["U99999"]
        with self.assertRaisesRegex(ValueError, "source ID"):
            validate_inventory_report(report, first)
        report = self._report(first)
        source_id = report["items"][0]["source_ids"][0]
        report["items"][0]["source_ids"] = [source_id, source_id]
        with self.assertRaisesRegex(ValueError, "source ID"):
            validate_inventory_report(report, first)
        report = self._report(first)
        report["coverage"][0]["item_indices"] = []
        report["coverage"][0]["assessment"] = "uncertain"
        report["coverage"][1]["item_indices"] = [0]
        report["coverage"][1]["assessment"] = "material_items"
        with self.assertRaisesRegex(ValueError, "no source in this window"):
            validate_inventory_report(report, first)

    def test_missing_window_or_broken_global_partition_is_rejected(self):
        first = self.segments[0]
        report = self._report(first)
        report["coverage"].pop()
        with self.assertRaisesRegex(ValueError, "every window"):
            validate_inventory_report(report, first)

        segments = copy.deepcopy(self.segments)
        segments[1]["primary_utterances"].pop()
        reports = [self._report(segment) for segment in segments]
        with self.assertRaisesRegex(ValueError, "complete source exactly once"):
            merge_inventory_reports(reports, segments, self.source_index)

        segments = copy.deepcopy(self.segments)
        segments[0]["primary_utterances"][0]["text"] = "Строка после подмены"
        with self.assertRaisesRegex(ValueError, "differs from canonical source"):
            validate_inventory_plan(segments, self.source_index)

        segments = copy.deepcopy(self.segments)
        # A genuine but distant utterance is still invalid contextual overlap.
        segments[1]["context_before"][0] = self.segments[0]["primary_utterances"][0]
        with self.assertRaisesRegex(ValueError, "not adjacent"):
            validate_inventory_plan(segments, self.source_index)

    def test_validator_does_not_claim_semantic_truth(self):
        report = self._report(self.segments[0])
        # A wrong actor/claim can have valid source coordinates. Semantic
        # scrutiny belongs to a separate source-grounded reconciliation.
        report["items"][0]["claim"] = "Алиса уже передала тысячу файлов."
        report["items"][0]["actor"] = "Борис"
        self.assertIs(validate_inventory_report(report, self.segments[0]), report)

    def test_three_segments_and_one_utterance_are_supported(self):
        segments = plan_inventory_segments(self.source_text, count=3, overlap=4)
        self.assertEqual(len(segments), 3)
        self.assertEqual(
            [row["id"] for segment in segments for row in segment["primary_utterances"]],
            list(self.source_index["by_id"]),
        )
        source = json.loads(self.source_text)
        source["utterances"] = source["utterances"][:1]
        one = plan_inventory_segments(json.dumps(source), count=2)
        self.assertEqual(len(one), 1)
        self.assertEqual(one[0]["coverage_windows"][0]["start_id"], "U00001")


if __name__ == "__main__":
    unittest.main()
