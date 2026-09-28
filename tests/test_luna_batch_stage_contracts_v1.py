"""Offline tests for the versioned Batch sidecar wire contracts."""

from __future__ import annotations

import json
import unittest

from pydantic import ValidationError

from summary.luna_v1 import SCHEMA
from summary.luna_v1.batch_stage_contracts_v1 import (
    InventoryReport,
    PublicationReviewSidecar,
    REVIEW_SIDECAR_SCHEMA_PATH,
    developer_message_for_stage,
    parse_stage_report,
    prompt_sha256_for_stage,
    response_format_for_stage,
    review_sidecar_schema,
    schema_for_stage,
    schema_path_for_stage,
)


STAGES = ("writer", "extract", "audit", "global", "repair", "verify")


def _walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


class BatchStageContractTests(unittest.TestCase):
    def test_writer_schema_is_the_existing_contract(self):
        writer = schema_for_stage("writer")
        self.assertEqual(writer, SCHEMA)
        self.assertIsNot(writer, SCHEMA)
        self.assertEqual(writer["required"], [
            "schema_version", "meeting", "main", "timecodes", "tasks",
            "questions", "technical", "ideas", "verification", "chapters",
        ])
        self.assertEqual(response_format_for_stage("writer")["json_schema"]["name"], "luna_summary_v1")

    def test_generated_sidecar_schemas_are_strict_and_versioned(self):
        for stage in STAGES[1:]:
            with self.subTest(stage=stage):
                schema = schema_for_stage(stage)
                saved = json.loads(schema_path_for_stage(stage).read_text(encoding="utf-8"))
                self.assertEqual(schema, saved)
                self.assertLess(len(json.dumps(schema)), 12_000)
                self.assertEqual(response_format_for_stage(stage)["json_schema"]["schema"], schema)
                for node in _walk(schema):
                    self.assertNotIn("default", node)
                    if node.get("type") == "object":
                        self.assertIs(node.get("additionalProperties"), False)
                        self.assertEqual(set(node.get("required", [])), set(node.get("properties", {})))
                self.assertEqual(schema["properties"]["schema_version"]["enum"], [
                    response_format_for_stage(stage)["json_schema"]["name"]
                ])
        evidence = schema_for_stage("extract")["$defs"]["Evidence"]
        self.assertEqual(evidence["properties"]["u_id"]["type"], "string")
        self.assertNotIn("enum", evidence["properties"]["u_id"])

    def test_prompt_is_common_plus_exactly_one_stage(self):
        common = (schema_path_for_stage("extract").parent.parent / "prompts_batch_v1" / "00_common.md").read_text(encoding="utf-8").strip()
        for stage in STAGES:
            with self.subTest(stage=stage):
                message = developer_message_for_stage(stage)
                self.assertTrue(message.startswith(common + "\n\n"))
                self.assertEqual(len(prompt_sha256_for_stage(stage)), 64)
                self.assertNotIn("Короткие синтетические примеры", message)
        self.assertIn("Стадия S1W", developer_message_for_stage("writer"))
        self.assertNotIn("Стадия S1W", developer_message_for_stage("audit"))
        self.assertIn("Стадия S6", developer_message_for_stage("verify"))

    def test_reports_require_nullable_fields_and_forbid_unknown_keys(self):
        inventory = {
            "schema_version": "luna_inventory_v1", "complete": True,
            "evidence": [{"evidence_id": "E1", "u_id": "U00001", "quote": "Проверим X"}],
            "units": [{"unit_id": "unit-1", "kind": "action", "text": "Предложено проверить X.",
                       "evidence_ids": ["E1"], "facets": [{"facet_id": "F1", "axis": "action",
                       "value": "проверить X", "evidence_ids": ["E1"]}]}],
            "source_accounting": [{"u_id": "U00001", "disposition": "content",
                                   "unit_ids": ["unit-1"], "note": None}],
            "open_links": [], "unprocessed_ids": [],
        }
        report = parse_stage_report("extract", inventory)
        self.assertIsInstance(report, InventoryReport)
        self.assertIsNone(report.source_accounting[0].note)
        missing = json.loads(json.dumps(inventory))
        del missing["source_accounting"][0]["note"]
        with self.assertRaises(ValidationError):
            parse_stage_report("extract", missing)
        unknown = dict(inventory, provider="other")
        with self.assertRaises(ValidationError):
            parse_stage_report("extract", unknown)
        with self.assertRaisesRegex(ValueError, "writer uses"):
            parse_stage_report("writer", {})

    def test_each_report_shape_parses_without_semantic_authority(self):
        reports = {
            "audit": {"schema_version": "luna_audit_v1", "complete": False,
                      "evidence": [], "source_checks": [], "document_checks": [],
                      "additional_units": [], "findings": [], "context_requests": [],
                      "unprocessed_ids": ["surface-1"]},
            "global": {"schema_version": "luna_global_v1", "complete": False,
                       "evidence": [], "resolutions": [{"request_id": "request-1",
                       "status": "unresolved", "conclusion": None, "evidence_ids": [],
                       "affected_surface_ids": ["surface-1"]}], "link_checks": [],
                       "additional_units": [], "findings": [],
                       "affected_surfaces": ["surface-1"], "unprocessed_ids": []},
            "repair": {"schema_version": "luna_patch_plan_v1", "complete": True,
                       "bundles": [{"bundle_id": "bundle-1", "finding_ids": ["finding-1"],
                       "evidence_ids": ["E1"], "affected_surface_ids": ["surface-1"],
                       "operations": [{"kind": "replace_field", "target_id": "surface-1",
                       "field_key": "text", "value_json": '"Уточнённый текст"',
                       "position_after_id": None, "temp_id": None, "lineage_ids": []}],
                       "preservation_notes": "Сохранить условие.", "dependency_bundle_ids": []}],
                       "unresolved": [], "unprocessed_finding_ids": []},
            "verify": {"schema_version": "luna_verification_v1", "complete": True,
                       "bundle_checks": [{"bundle_id": "bundle-1", "verdict": "unresolved",
                       "evidence": [], "preserved_surface_ids": [], "regressions": [],
                       "note": "Связь остаётся неясной."}],
                       "new_findings": [], "unprocessed_bundle_ids": []},
        }
        for stage, payload in reports.items():
            with self.subTest(stage=stage):
                parsed = parse_stage_report(stage, payload)
                self.assertEqual(parsed.model_dump(), payload)
        # A model cannot supply an app-owned expected-before digest.
        forged = json.loads(json.dumps(reports["repair"]))
        forged["bundles"][0]["operations"][0]["expected_before_hash"] = "forged"
        with self.assertRaises(ValidationError):
            parse_stage_report("repair", forged)

    def test_publication_sidecar_is_local_and_strict(self):
        schema = review_sidecar_schema()
        self.assertEqual(schema, json.loads(REVIEW_SIDECAR_SCHEMA_PATH.read_text(encoding="utf-8")))
        for node in _walk(schema):
            if node.get("type") == "object":
                self.assertIs(node.get("additionalProperties"), False)
                self.assertEqual(set(node.get("required", [])), set(node.get("properties", {})))
        payload = {
            "schema_version": "luna_review_sidecar_v1", "generation_id": "generation-1",
            "source_revision": "sha-1", "execution_status": "terminal",
            "review_status": "review_incomplete", "source_scope_ids": ["U00001"],
            "checked_ids": [], "unreviewed_ids": ["surface-1"],
            "accepted_bundle_ids": [], "rejected_bundle_ids": [],
            "unresolved_bundle_ids": [], "source_ambiguities": [],
            "model_disagreements": [], "technical_failures": ["audit unavailable"],
            "billed_cost_microusd": None, "held_cost_microusd": 1000,
            "unknown_bill_ids": ["batch-1"], "artifact_hashes": [],
            "generation_lineage": [],
        }
        self.assertEqual(PublicationReviewSidecar.model_validate(payload).model_dump(), payload)
        with self.assertRaises(ValidationError):
            PublicationReviewSidecar.model_validate(dict(payload, approval="model"))


if __name__ == "__main__":
    unittest.main()
