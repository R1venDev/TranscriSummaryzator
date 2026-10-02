"""Budget capacity and independent grounded findings on production helpers."""
from dataclasses import replace
from decimal import Decimal
from copy import deepcopy
import json
import unittest

from summary.luna_v1.capacity import allocate_outputs
from summary.luna_v1.route import Route, RouteBlocked
from summary.luna_v1.review_salvage import salvage_review
from summary.luna_v1.source_first_core import (
    accepted_patch_document, build_surfaces, plan_packets, review_evidence,
    validate_patch_plan, stage_patch_candidate, validate_verification,
)
from summary.luna_v1.source_first_runtime import _repair_payload
from tests import test_luna_source_first_core as fixtures


class CapacityTests(unittest.TestCase):
    def route(self):
        return Route("openai/gpt-6-luna", "openai", "scope", 1_000_000, 128_000,
            Decimal("0.0000001"), Decimal("0.000000375"), Decimal("0.000000125"), Decimal(0), None)

    def test_capacity_changes_with_money_and_remaining_responsibility(self):
        route = self.route()
        body = {"messages": [{"role": "user", "content": "source" * 1000}]}
        def allocate(money, future):
            return allocate_outputs(route, [body], available_microusd=money,
                future_items=future, future_input_tokens=20_000)
        small, basis = allocate(30_000, 3)
        large, _ = allocate(120_000, 3)
        final, _ = allocate(120_000, 0)
        self.assertLess(small[0]["max_tokens"], large[0]["max_tokens"])
        self.assertLess(large[0]["max_tokens"], final[0]["max_tokens"])
        self.assertGreater(large[0]["max_tokens"], 25_000)
        self.assertLessEqual(sum(basis["reserved_microusd"]) + basis["future_input_hold_microusd"], 30_000)
        self.assertNotIn("max_tokens", body)

    def test_confirmed_endpoint_and_remaining_context_are_hard_boundaries(self):
        route = replace(self.route(), max_completion_tokens=7200)
        result, _ = allocate_outputs(route, [{}], available_microusd=200_000,
                                    future_items=0, future_input_tokens=0)
        self.assertEqual(result[0]["max_tokens"], 7200)
        route = replace(route, max_context_tokens=3000)
        result, _ = allocate_outputs(route, [{}], available_microusd=200_000,
                                    future_items=0, future_input_tokens=0)
        self.assertLess(result[0]["max_tokens"], 3000)
        with self.assertRaises(RouteBlocked):
            allocate_outputs(self.route(), [{"source": "x" * 10000}],
                available_microusd=1, future_items=0, future_input_tokens=0)


class GroundedReviewTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.SourceFirstCoreTests()
        fixture.setUp()
        self.addCleanup(fixture.temp.cleanup)
        self.snapshot = fixture.snapshot
        self.packet = plan_packets(self.snapshot)[0]
        self.document = fixtures._document()
        self.registry = build_surfaces(self.document, self.snapshot.index)
        self.target = next(k for k, v in self.registry.items() if v["field_key"] == "main.0.text")
        self.evidence = {"E1": {"evidence_id": "E1", "u_id": "U00001",
                                "quote": "Предлагаю проверить сигнал."}}
        self.finding = {"finding_id": "F1", "kind": "unsupported_claim", "severity": "material",
            "affected_surface_ids": [self.target], "affected_unit_ids": [], "evidence_ids": ["E1"],
            "problem": "Предложение представлено как решение", "required_preservation": "Статус предложения",
            "proposed_resolution": None}

    def audit(self):
        return {"schema_version": "luna_audit_v1", "complete": True, "evidence": [],
            "source_checks": [], "document_checks": [], "additional_units": [],
            "findings": [deepcopy(self.finding)], "context_requests": [], "unprocessed_ids": []}

    def salvage(self, raw):
        return salvage_review(raw, stage="audit", snapshot=self.snapshot, registry=self.registry,
            known_evidence=self.evidence, known_units=set(), packet=self.packet,
            known_context_targets=set(self.registry))

    def test_input_evidence_resolves_but_collision_and_forgery_do_not(self):
        self.assertIn("E1", review_evidence([], self.evidence, self.snapshot))
        with self.assertRaisesRegex(ValueError, "collision"):
            review_evidence([{"evidence_id": "E1", "u_id": "U00002",
                "quote": "Или взять другую запись."}], self.evidence, self.snapshot)
        local = review_evidence([{"evidence_id": "E1", "u_id": "U00001",
            "quote": "проверить сигнал"}], self.evidence, self.snapshot)["E1"]
        self.assertEqual(local["quote"], "проверить сигнал")
        self.assertEqual(local["reference_resolution"], "report_local_span_same_u_v1")
        raw = self.audit()
        raw["evidence"] = [{"evidence_id": "E1", "u_id": "U00001", "quote": "Обещаю сделать"}]
        checked = self.salvage(raw)
        self.assertEqual(checked["findings"], [])
        self.assertFalse(checked["complete"])

    def test_bad_neighbor_does_not_destroy_valid_finding_or_claim_complete(self):
        raw = self.audit()
        raw["findings"].append({**self.finding, "finding_id": "F2", "evidence_ids": ["missing"]})
        checked = self.salvage(raw)
        self.assertEqual([row["finding_id"] for row in checked["findings"]], ["P001:audit:F1"])
        self.assertEqual(checked["source_checks"], {})
        self.assertFalse(checked["complete"])
        self.assertEqual(raw["findings"][0]["finding_id"], "F1")
        self.assertIn("finding", [row["kind"] for row in checked["discarded"]])

    def test_global_unknown_link_does_not_hide_finding_and_missing_is_unreviewed(self):
        raw = {"schema_version": "luna_global_v1", "complete": True, "evidence": [],
            "resolutions": [], "additional_units": [], "findings": [self.finding],
            "link_checks": [{"link_id": "unit-accident", "unit_ids": [], "relation": "unresolved",
                             "evidence_ids": ["E1"], "conclusion": "Нет контекста"}],
            "affected_surfaces": [self.target], "unprocessed_ids": []}
        checked = salvage_review(raw, stage="global", snapshot=self.snapshot,
            registry=self.registry, known_evidence=self.evidence, known_units=set(),
            known_links={"link-1"}, known_contexts=set())
        self.assertEqual(len(checked["findings"]), 1)
        self.assertEqual(checked["link_checks"], [])
        self.assertFalse(checked["complete"])
        self.assertIn({"kind": "link_checks", "id": "link-1", "reason": "missing_verdict"}, checked["discarded"])

    def test_salvaged_finding_is_repaired_verified_and_applied_without_mutating_d0(self):
        checked = self.salvage(self.audit())
        findings = {row["finding_id"]: row for row in checked["findings"]}
        evidence = checked["evidence"]
        payload = _repair_payload(self.snapshot, {"document": self.document, "registry": self.registry},
            findings, None, evidence)
        self.assertEqual(payload["SOURCE_EVIDENCE"], evidence)
        value = "Предложено проверить сигнал."
        raw = {"schema_version": "luna_patch_plan_v1", "complete": True, "bundles": [{
            "bundle_id": "B1", "finding_ids": list(findings), "evidence_ids": list(evidence),
            "affected_surface_ids": [self.target], "operations": [{"kind": "replace_field",
            "target_id": self.target, "field_key": "main.0.text", "value_json": json.dumps(value),
            "position_after_id": None, "temp_id": None, "lineage_ids": []}],
            "preservation_notes": "Сохранить предложение", "dependency_bundle_ids": []}],
            "unresolved": [], "unprocessed_finding_ids": []}
        plan = validate_patch_plan(raw, findings=findings, registry=self.registry, known_evidence_ids=set(evidence))
        candidate, before = stage_patch_candidate(self.document, plan, self.registry, self.snapshot.index, evidence)
        self.assertIn("expected_before_hash", before["B1"][0])
        verification = {"schema_version": "luna_verification_v1", "complete": True,
            "bundle_checks": [{"bundle_id": "B1", "verdict": "accept", "evidence": list(self.evidence.values()),
                "preserved_surface_ids": [self.target], "regressions": [], "note": "Предложение сохранено"}],
            "new_findings": [], "unprocessed_bundle_ids": []}
        result = validate_verification(verification, plan=plan, snapshot=self.snapshot, registry=self.registry)
        published = accepted_patch_document(self.document, plan, result, self.registry, self.snapshot.index, evidence)
        self.assertEqual(published["main"][0]["text"], value)
        self.assertEqual(published, candidate)
        self.assertEqual(self.document["main"][0]["text"], "Обсудили проверку сигнала.")
        self.assertEqual(published["tasks"], self.document["tasks"])

    def test_incomplete_verification_applies_only_independent_explicit_acceptance(self):
        other = next(k for k, v in self.registry.items() if v["field_key"] == "tasks.0.description")
        plan = {"bundles": {
            "B1": {"dependency_bundle_ids": [], "affected_surface_ids": [self.target]},
            "B2": {"dependency_bundle_ids": [], "affected_surface_ids": [other]},
            "B3": {"dependency_bundle_ids": ["B2"], "affected_surface_ids": []}}}
        raw = {"schema_version": "luna_verification_v1", "complete": False,
            "bundle_checks": [{"bundle_id": identity, "verdict": "accept", "evidence": [
                {**self.evidence["E1"], "evidence_id": identity}],
                "preserved_surface_ids": [], "regressions": [], "note": "Проверено"}
                for identity in ("B1", "B2")],
            "new_findings": [{**self.finding, "evidence_ids": ["B2"], "affected_surface_ids": [other]}],
            "unprocessed_bundle_ids": ["B3"]}
        checked = validate_verification(raw, plan=plan, snapshot=self.snapshot, registry=self.registry)
        self.assertEqual(checked["accepted"], {"B1"})
        self.assertFalse(checked["complete"])
        self.assertEqual(checked["missing_bundle_ids"], ["B3"])
