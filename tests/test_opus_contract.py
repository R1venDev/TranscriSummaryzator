"""Synthetic, offline Opus source-audit and patch contract tests."""

from __future__ import annotations

import copy
import json
import unittest

from summary.opus_v1 import (
    OPUS_AUDIT_PROMPT_PATH, OPUS_AUDIT_SCHEMA, OPUS_AUDIT_SCHEMA_ID,
    OUTPUT_CAP_AUDIT, OUTPUT_CAP_VERIFY, REASONING_EFFORT,
    apply_opus_audit, build_opus_audit_input, build_opus_audit_request,
    coverage_warnings_opus, validate_opus_audit,
)
from summary.luna_v1.audit import source_windows


def _fixtures():
    utterances = [
        {"id": "U00001", "speaker": "Алиса", "start_ms": 0, "end_ms": 1000,
         "text": "Борис подготовит таблицу."},
        {"id": "U00002", "speaker": "Борис", "start_ms": 1000, "end_ms": 2000,
         "text": "Когда получим исходные данные?"},
    ]
    source = {"source_kind": "TRANSCRIPT_SOURCE", "utterances": utterances}
    index = {"by_id": {row["id"]: row for row in utterances}}
    draft = {
        "schema_version": "luna_summary_v1",
        "meeting": {"topic": "Учебная встреча", "project": None},
        "main": [{"text": "Борис подготовит таблицу.", "source_ids": ["U00001"]}],
        "timecodes": [], "tasks": [], "questions": [], "technical": [],
        "ideas": [], "verification": [], "chapters": [],
    }
    return json.dumps(source, ensure_ascii=False), index, draft


def _report(index):
    windows = source_windows(index)
    question = {"text": "Когда получим исходные данные?", "source_ids": ["U00002"]}
    return {
        "schema_version": OPUS_AUDIT_SCHEMA_ID,
        "coverage": [
            {"window_id": windows[0]["window_id"], "start_id": "U00001", "end_id": "U00001",
             "salient": "Подготовка таблицы", "source_quote": "Борис подготовит таблицу",
             "draft_coverage": "covered", "finding_indices": []},
            {"window_id": windows[1]["window_id"], "start_id": "U00002", "end_id": "U00002",
             "salient": "Открытый вопрос", "source_quote": "Когда получим исходные данные?",
             "draft_coverage": "missing", "finding_indices": [0]},
        ],
        "findings": [{
            "severity": "major", "kind": "omission", "description": "Потерян открытый вопрос о сроке данных.",
            "evidence_quote": "Когда получим исходные данные?",
            "source_ids": ["U00002"], "affected": [], "status": "repaired", "patch_indices": [0],
        }],
        "patches": [{"section": "questions", "operation": "insert", "index": 0,
                     "item_json": json.dumps(question, ensure_ascii=False)}],
    }


class OpusContractTests(unittest.TestCase):
    def test_complete_source_draft_and_adaptive_chat_request(self):
        source, _index, draft = _fixtures()
        payload = json.loads(build_opus_audit_input(source, draft))
        self.assertEqual(list(payload), ["TRANSCRIPT_SOURCE", "SOURCE_WINDOWS",
                                         "DRAFT_DOCUMENT", "PRIOR_FINDINGS", "MODE", "TASK"])
        self.assertEqual(payload["TRANSCRIPT_SOURCE"], json.loads(source))
        self.assertEqual(len(payload["SOURCE_WINDOWS"]), 2)
        request = build_opus_audit_request(source, draft)
        self.assertEqual(request["max_completion_tokens"], OUTPUT_CAP_AUDIT)
        self.assertEqual(request["reasoning"], {"effort": REASONING_EFFORT})
        self.assertEqual(REASONING_EFFORT, "medium")
        self.assertEqual(request["response_format"]["json_schema"]["name"], OPUS_AUDIT_SCHEMA_ID)
        self.assertEqual(request["response_format"]["json_schema"]["schema"], OPUS_AUDIT_SCHEMA)
        self.assertEqual([message["role"] for message in request["messages"]], ["system", "user"])
        self.assertFalse({"temperature", "tools", "tool_choice", "cache_control", "plugins"} & set(request))
        self.assertIn("evidence_quote", OPUS_AUDIT_PROMPT_PATH.read_text(encoding="utf-8"))

    def test_verify_scopes_source_but_includes_full_revised_draft(self):
        source, _index, draft = _fixtures()
        long_source = json.loads(source)
        long_source["utterances"] = [
            {"id": f"U{number:05d}", "text": f"Реплика {number}.",
             "speaker": "Алиса", "start_ms": number * 1000, "end_ms": number * 1000 + 500}
            for number in range(1, 11)
        ]
        draft["main"] = [
            {"text": "Затронутый пункт", "source_ids": ["U00002"]},
            {"text": "Посторонний пункт", "source_ids": ["U00005"]},
        ]
        draft["questions"] = [{"text": "Вопрос", "source_ids": ["U00009"]}]
        draft["chapters"] = [{"topic": "Большой раздел", "start_id": "U00001",
                               "end_id": "U00010", "summary": "Связанный обзор",
                               "source_ids": ["U00001", "U00010"], "details": []}]
        prior = [{"description": "Проверь реплики отдельно", "source_ids": ["U00002", "U00009"],
                  "affected": [{"section": "questions", "index": 0}]}]
        verify = json.loads(build_opus_audit_input(
            json.dumps(long_source, ensure_ascii=False), draft,
            mode="verify", prior_findings=prior,
        ))
        self.assertEqual([row["id"] for row in verify["TRANSCRIPT_SOURCE"]["utterances"]],
                         ["U00001", "U00002", "U00003", "U00008", "U00009", "U00010"])
        self.assertEqual([(row["window_id"], row["start_id"], row["end_id"])
                          for row in verify["SOURCE_WINDOWS"]],
                         [("V01", "U00001", "U00003"), ("V02", "U00008", "U00010")])
        self.assertNotIn("DRAFT_TARGETS", verify)
        self.assertEqual(verify["DRAFT_DOCUMENT"], draft)
        self.assertEqual(verify["DRAFT_DOCUMENT"]["main"][1]["text"], "Посторонний пункт")
        self.assertEqual(verify["PRIOR_FINDINGS"], prior)
        self.assertEqual(build_opus_audit_request(
            json.dumps(long_source, ensure_ascii=False), draft, mode="verify",
            prior_findings=prior)["max_completion_tokens"], OUTPUT_CAP_VERIFY)
        with self.assertRaisesRegex(ValueError, "verify_requires_prior_findings"):
            build_opus_audit_input(source, draft, mode="verify")

    def test_quote_grounding_and_targeted_patch(self):
        _source, index, draft = _fixtures()
        original = copy.deepcopy(draft)
        report = _report(index)
        self.assertIs(validate_opus_audit(report, draft, index), report)
        self.assertEqual(coverage_warnings_opus(report, index), [])
        revised, unresolved = apply_opus_audit(draft, report, index)
        self.assertEqual(unresolved, [])
        self.assertEqual(draft, original)
        self.assertEqual(revised["questions"], [{"text": "Когда получим исходные данные?",
                                                "source_ids": ["U00002"]}])
        self.assertEqual(revised["main"], draft["main"])

    def test_verify_patch_validator_accepts_focused_coverage_only(self):
        _source, index, draft = _fixtures()
        report = _report(index)
        report["coverage"] = [{
            "window_id": "V01", "start_id": "U00001", "end_id": "U00002",
            "salient": "Вопрос о данных", "source_quote": "Когда получим исходные данные?",
            "draft_coverage": "missing", "finding_indices": [0],
        }]
        self.assertIs(validate_opus_audit(
            report, draft, index, mode="verify",
            expected_windows=[{"window_id": "V01", "start_id": "U00001", "end_id": "U00002"}],
        ), report)
        with self.assertRaisesRegex(ValueError, "coverage_scope_mismatch"):
            validate_opus_audit(
                report, draft, index, mode="verify",
                expected_windows=[{"window_id": "V01", "start_id": "U00002", "end_id": "U00002"}],
            )
        skipped = copy.deepcopy(report)
        skipped["coverage"] = [{"window_id": "V01", "start_id": "U00001",
                                "end_id": "U00001", "salient": "Подготовка таблицы",
                                "source_quote": "Борис подготовит таблицу",
                                "draft_coverage": "covered", "finding_indices": []}]
        with self.assertRaisesRegex(ValueError, "coverage_scope_mismatch"):
            validate_opus_audit(
                skipped, draft, index, mode="verify", expected_windows=[
                    {"window_id": "V01", "start_id": "U00001", "end_id": "U00001"},
                    {"window_id": "V02", "start_id": "U00002", "end_id": "U00002"},
                ],
            )
        self.assertEqual(coverage_warnings_opus(report, index), [])
        revised, _ = apply_opus_audit(draft, report, index, mode="verify")
        self.assertEqual(revised["questions"][0]["text"], "Когда получим исходные данные?")

    def test_verify_uses_current_revised_index_after_prior_insert(self):
        source, index, draft = _fixtures()
        revised = copy.deepcopy(draft)
        revised["main"] = [
            {"text": "Новая вводная после первого аудита.", "source_ids": ["U00001"]},
            {"text": "Данные уже получены.", "source_ids": ["U00002"]},
        ]
        prior = [{"description": "Старый индекс относится к черновику до вставки.",
                  "source_ids": ["U00002"], "affected": [{"section": "main", "index": 0}]}]
        verify_input = json.loads(build_opus_audit_input(
            source, revised, mode="verify", prior_findings=prior))
        self.assertEqual(verify_input["DRAFT_DOCUMENT"], revised)
        self.assertEqual(verify_input["PRIOR_FINDINGS"][0]["affected"][0]["index"], 0)
        corrected = {"text": "Срок получения данных остаётся открытым.",
                     "source_ids": ["U00002"]}
        report = {
            "schema_version": OPUS_AUDIT_SCHEMA_ID,
            "coverage": [{"window_id": "V01", "start_id": "U00001", "end_id": "U00002",
                          "salient": "Открытый срок данных",
                          "source_quote": "Когда получим исходные данные?",
                          "draft_coverage": "partial", "finding_indices": [0]}],
            "findings": [{"severity": "major", "kind": "unsupported_claim",
                          "description": "Черновик утверждает, что данные получены, хотя это вопрос.",
                          "evidence_quote": "Когда получим исходные данные?",
                          "source_ids": ["U00002"],
                          "affected": [{"section": "main", "index": 1}],
                          "status": "repaired", "patch_indices": [0]}],
            "patches": [{"section": "main", "operation": "replace", "index": 1,
                         "item_json": json.dumps(corrected, ensure_ascii=False)}],
        }
        updated, _ = apply_opus_audit(
            revised, report, index, mode="verify",
            expected_windows=verify_input["SOURCE_WINDOWS"],
            verify_source_ids={item["id"] for item in verify_input["TRANSCRIPT_SOURCE"]["utterances"]},
        )
        self.assertEqual(updated["main"][0], revised["main"][0])
        self.assertEqual(updated["main"][1], corrected)

    def test_verify_finding_cannot_cite_untransmitted_source_id(self):
        source, index, draft = _fixtures()
        extended = json.loads(source)
        extended["utterances"].extend([
            {"id": "U00003", "speaker": "Алиса", "start_ms": 2000,
             "end_ms": 3000, "text": "Промежуточная реплика."},
            {"id": "U00004", "speaker": "Борис", "start_ms": 3000,
             "end_ms": 4000, "text": "Четвёртая реплика вне проверяемого контекста."},
        ])
        for item in extended["utterances"][2:]:
            index["by_id"][item["id"]] = item
        verify_input = json.loads(build_opus_audit_input(
            json.dumps(extended, ensure_ascii=False), draft, mode="verify",
            prior_findings=[{"source_ids": ["U00001"]}],
        ))
        scope = {item["id"] for item in verify_input["TRANSCRIPT_SOURCE"]["utterances"]}
        self.assertEqual(scope, {"U00001", "U00002"})
        report = _report(index)
        report["coverage"] = [{"window_id": "V01", "start_id": "U00001",
                               "end_id": "U00002", "salient": "Таблица и данные",
                               "source_quote": "Борис подготовит таблицу",
                               "draft_coverage": "covered", "finding_indices": []}]
        report["findings"][0]["source_ids"] = ["U00004"]
        report["findings"][0]["evidence_quote"] = "Четвёртая реплика вне проверяемого контекста."
        with self.assertRaisesRegex(ValueError, "source_outside_verify_scope"):
            validate_opus_audit(
                report, draft, index, mode="verify",
                expected_windows=verify_input["SOURCE_WINDOWS"],
                verify_source_ids=scope,
            )

    def test_fabricated_finding_quote_fails_but_bad_coverage_quote_warns(self):
        _source, index, draft = _fixtures()
        report = _report(index)
        report["findings"][0]["evidence_quote"] = "Выдуманное решение"
        with self.assertRaisesRegex(ValueError, "evidence_quote_not_in_source"):
            validate_opus_audit(report, draft, index)
        report = _report(index)
        report["coverage"][0]["source_quote"] = "Выдуманная фраза"
        self.assertIs(validate_opus_audit(report, draft, index), report)
        self.assertIn("coverage_quote_not_in_window",
                      {warning["code"] for warning in coverage_warnings_opus(report, index)})


if __name__ == "__main__":
    unittest.main()
