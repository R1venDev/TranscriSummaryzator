"""Offline checks for source-first coordinates and patch boundaries."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from summary.luna_v1.source_first_core import (
    accepted_patch_document, build_surfaces, load_snapshot, normalize_inventories, partition_surfaces,
    plan_dimensions, plan_packets, safe_mark_document, stage_patch_candidate, validate_evidence,
    validate_audit, validate_global, validate_inventory, validate_patch_plan, validate_verification,
    remap_mark_targets, salvage_inventory,
)


def _document() -> dict:
    return {"schema_version": "luna_summary_v1",
            "meeting": {"topic": "Тестирование", "project": None},
            "main": [{"text": "Обсудили проверку сигнала.", "source_ids": ["U00001"]}],
            "timecodes": [{"topic": "Проверка", "start_id": "U00001", "end_id": "U00003"}],
            "tasks": [{"title": "Проверить сигнал", "description": "Проверить сигнал на тестовой записи.",
                       "discussion_status": "proposed", "assignee": None, "due": None,
                       "priority": None, "recipient": None, "source_ids": ["U00001"],
                       "field_sources": {"action": ["U00001"], "assignee": [], "due": [],
                                         "priority": [], "recipient": [],
                                         "discussion_status": ["U00001"]}}],
            "questions": [], "technical": [], "ideas": [], "verification": [],
            "chapters": [{"topic": "Тестирование", "start_id": "U00001", "end_id": "U00003",
                          "summary": "Обсудили условия теста.", "source_ids": ["U00001", "U00003"],
                          "details": [{"text": "Сигнал проверят на записи.",
                                       "source_ids": ["U00001"]}]}]}


class SourceFirstCoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        path = Path(self.temp.name) / "transcript.json"
        path.write_text(json.dumps({
            "source": "01.01.2026 test.mkv", "duration_seconds": 18,
            "speakers": {"s1": "Аня"},
            "utterances": [
                {"start": 0, "end": 4, "speaker": "s1", "text": "Предлагаю проверить сигнал."},
                {"start": 5, "end": 8, "speaker": "s1", "text": "Или взять другую запись."},
                {"start": 9, "end": 12, "speaker": None, "text": "Условие ещё не согласовано."},
            ],
        }, ensure_ascii=False), encoding="utf-8")
        self.snapshot = load_snapshot(path)

    def test_partition_covers_exact_source_with_unknown_speaker(self):
        packets = plan_packets(self.snapshot, target_chars=220, overlap=1)
        self.assertEqual([u for packet in packets for u in packet["core_ids"]],
                         list(self.snapshot.ids))
        self.assertEqual(self.snapshot.records[-1]["speaker_label"], "Участник не определён")
        self.assertEqual(plan_dimensions(packets)["status"], "ready")
        self.assertEqual(plan_dimensions(tuple({} for _ in range(4)))["planned_items"], 12)
        self.assertEqual(plan_dimensions(tuple({} for _ in range(4)))["maximum_items"], 12)
        self.assertEqual(plan_dimensions(tuple({} for _ in range(5)))["status"],
                         "dimension_budget_blocked")

    def test_evidence_uses_exact_raw_substring(self):
        self.assertEqual(validate_evidence([{"evidence_id": "E1", "u_id": "U00001",
                                             "quote": "проверить сигнал"}], self.snapshot)["E1"]["source_offset"], 10)
        with self.assertRaisesRegex(ValueError, "evidence_quote"):
            validate_evidence([{"evidence_id": "E1", "u_id": "U00001",
                                "quote": "проверить  сигнал"}], self.snapshot)

    def test_evidence_normalizes_only_unique_case_and_terminal_period(self):
        report = validate_evidence([{"evidence_id": "E1", "u_id": "U00001",
                                     "quote": "ПРЕДЛАГАЮ."}], self.snapshot)["E1"]
        self.assertEqual(report["quote"], "Предлагаю")
        self.assertEqual(report["source_offset"], 0)
        self.assertEqual(report["normalization"], "casefold_terminal_period_v1")
        self.assertEqual(len(report["native_quote_sha256"]), 64)
        case_only = validate_evidence([{"evidence_id": "E2", "u_id": "U00001",
                                        "quote": "ПРЕДЛАГАЮ ПРОВЕРИТЬ СИГНАЛ."}],
                                       self.snapshot)["E2"]
        self.assertEqual(case_only["quote"], "Предлагаю проверить сигнал.")
        self.assertEqual(case_only["normalization"], "casefold_source_span_v1")
        with self.assertRaisesRegex(ValueError, "evidence_quote"):
            validate_evidence([{"evidence_id": "E1", "u_id": "U00001",
                                "quote": "Предлагаю не проверить сигнал."}], self.snapshot)
        with self.assertRaisesRegex(ValueError, "evidence_quote"):
            validate_evidence([{"evidence_id": "E1", "u_id": "U00001",
                                "quote": "Предлагаю проверить сигнал 2."}], self.snapshot)
        ambiguous = replace(self.snapshot, records=tuple(
            {**row, "text": "Тест тест."} if row["id"] == "U00001" else row
            for row in self.snapshot.records))
        with self.assertRaisesRegex(ValueError, "evidence_quote"):
            validate_evidence([{"evidence_id": "E1", "u_id": "U00001",
                                "quote": "ТЕСТ"}], ambiguous)

    def test_negative_audit_verdict_requires_matching_finding_or_context(self):
        document = _document()
        packet = plan_packets(self.snapshot, target_chars=9999)[0]
        surfaces = list(build_surfaces(document, self.snapshot.index).values())
        inventory = {"units": {"P001:A1": {"unit_id": "P001:A1"}},
                     "facets": {"P001:A1:F1": {"unit_id": "P001:A1"}}}
        report = {"schema_version": "luna_audit_v1", "complete": True,
            "evidence": [{"evidence_id": "E1", "u_id": "U00001",
                          "quote": "Предлагаю проверить сигнал."}],
            "source_checks": [{"unit_id": "P001:A1", "inventory_verdict": "source_supported",
                "coverage": "partial", "facet_checks": [{"facet_id": "P001:A1:F1",
                    "verdict": "missing", "document_evidence": [], "note": None}],
                "finding_ids": []}],
            "document_checks": [{"surface_id": row["surface_id"],
                "claims": ([{"claim_text": str(row["text_or_scalar"]),
                            "verdict": "supported", "evidence_ids": ["E1"], "note": None}]
                           if row["text_or_scalar"] not in (None, "") else []),
                "finding_ids": []} for row in surfaces],
            "additional_units": [], "findings": [], "context_requests": [],
            "unprocessed_ids": []}
        registry = build_surfaces(document, self.snapshot.index)
        with self.assertRaisesRegex(ValueError, "negative_source_verdict_unaccounted"):
            validate_audit(report, packet=packet, packet_inventory=inventory,
                           surfaces=surfaces, full_registry=registry, snapshot=self.snapshot)
        report["source_checks"][0]["coverage"] = "full"
        report["source_checks"][0]["facet_checks"][0]["verdict"] = "preserved"
        report["source_checks"][0]["facet_checks"][0]["document_evidence"] = [
            {"surface_id": surfaces[2]["surface_id"],
             "quote": str(surfaces[2]["text_or_scalar"])}]
        report["document_checks"][0]["claims"][0]["verdict"] = "unsupported"
        with self.assertRaisesRegex(ValueError, "negative_document_verdict_unaccounted"):
            validate_audit(report, packet=packet, packet_inventory=inventory,
                           surfaces=surfaces, full_registry=registry, snapshot=self.snapshot)

    def test_unresolved_mark_follows_original_claim_after_insert(self):
        document = _document()
        document["main"].append({"text": "Условие осталось открытым.",
                                 "source_ids": ["U00003"]})
        old = build_surfaces(document, self.snapshot.index)
        first = next(key for key, row in old.items() if row["field_key"] == "main.0.text")
        second = next(key for key, row in old.items() if row["field_key"] == "main.1.text")
        plan = {"bundles": {"B1": {"evidence_ids": ["E1"],
            "operations": [{"kind": "add_section_item", "target_id": None,
                "field_key": "main", "value_json": json.dumps({"text": "Новая проверка.",
                    "source_ids": ["U00001"]}, ensure_ascii=False),
                "position_after_id": first}]}}}
        candidate, _ = stage_patch_candidate(document, plan, old, self.snapshot.index)
        finding = {"finding_id": "F2", "problem": "Неясно условие",
                   "affected_surface_ids": [second], "evidence_ids": ["E2"]}
        new = build_surfaces(candidate, self.snapshot.index)
        mapped = remap_mark_targets([finding], plan=plan, accepted={"B1"},
                                    old_registry=old, new_registry=new)
        self.assertEqual(new[mapped[0]["affected_surface_ids"][0]]["field_key"],
                         "main.2.text")
        marked, _ = safe_mark_document(candidate, mapped, new, self.snapshot,
            {"E2": {"u_id": "U00003"}})
        self.assertEqual(marked["main"][1]["text"], "Новая проверка.")
        self.assertTrue(marked["main"][2]["text"].startswith("Требует проверки"))

    def test_replace_uses_original_target_even_when_insert_precedes_it(self):
        document = _document()
        document["main"].append({"text": "Условие осталось открытым.",
                                 "source_ids": ["U00003"]})
        registry = build_surfaces(document, self.snapshot.index)
        first = next(key for key, row in registry.items() if row["field_key"] == "main.0.text")
        second = next(key for key, row in registry.items() if row["field_key"] == "main.1.text")
        plan = {"bundles": {
            "B-insert": {"evidence_ids": ["E1"], "operations": [{
                "kind": "add_section_item", "field_key": "main", "position_after_id": first,
                "value_json": json.dumps({"text": "Вставка.", "source_ids": ["U00001"]})}]},
            "B-replace": {"evidence_ids": ["E2"], "operations": [{
                "kind": "replace_field", "target_id": second,
                "value_json": json.dumps("Условие изменено.")}]},
        }}
        candidate, _ = stage_patch_candidate(document, plan, registry, self.snapshot.index)
        self.assertEqual([row["text"] for row in candidate["main"]], [
            "Обсудили проверку сигнала.", "Вставка.", "Условие изменено."])

    def test_optional_task_repair_derives_source_refs_from_bundle_evidence(self):
        document = _document()
        old = build_surfaces(document, self.snapshot.index)
        target = next(key for key, row in old.items()
                      if row["field_key"] == "tasks.0.assignee")
        plan = {"bundles": {"B1": {"evidence_ids": ["E2"],
            "operations": [{"kind": "update_task", "target_id": target,
                "field_key": "tasks.0.assignee", "value_json": json.dumps("Аня")} ]}}}
        evidence = {"E2": {"u_id": "U00002"}}
        candidate, _ = stage_patch_candidate(document, plan, old,
                                             self.snapshot.index, evidence)
        self.assertEqual(candidate["tasks"][0]["assignee"], "Аня")
        self.assertEqual(candidate["tasks"][0]["field_sources"]["assignee"], ["U00002"])
        self.assertIn("U00002", candidate["tasks"][0]["source_ids"])
        published = accepted_patch_document(document, plan, {"accepted": {"B1"}},
                                            old, self.snapshot.index, evidence)
        self.assertEqual(published, candidate)

    def test_repaired_task_content_and_status_keep_late_source_provenance(self):
        document = _document()
        old = build_surfaces(document, self.snapshot.index)
        evidence = {"E2": {"u_id": "U00002"}}
        cases = (
            ("title", "Проверить сигнал или взять другую запись", "action"),
            ("description", "Проверить сигнал на тестовой или другой записи.", "action"),
            ("discussion_status", "unknown", "discussion_status"),
        )
        for field_name, replacement, provenance_field in cases:
            with self.subTest(field=field_name):
                target = next(identity for identity, row in old.items()
                              if row["field_key"] == "tasks.0." + field_name)
                plan = {"bundles": {"B1": {"evidence_ids": ["E2"],
                    "operations": [{"kind": "update_task", "target_id": target,
                        "field_key": "tasks.0." + field_name,
                        "value_json": json.dumps(replacement, ensure_ascii=False)}]}}}
                candidate, _ = stage_patch_candidate(document, plan, old,
                                                     self.snapshot.index, evidence)
                task = candidate["tasks"][0]
                self.assertEqual(task[field_name], replacement)
                self.assertEqual(task["source_ids"], ["U00001", "U00002"])
                self.assertEqual(task["field_sources"][provenance_field],
                                 ["U00001", "U00002"])
                self.assertEqual(document["tasks"][0]["source_ids"], ["U00001"])
                published = accepted_patch_document(document, plan,
                    {"accepted": {"B1"}}, old, self.snapshot.index, evidence)
                self.assertEqual(published, candidate)

    def test_repaired_task_content_without_known_evidence_is_rejected(self):
        document = _document()
        old = build_surfaces(document, self.snapshot.index)
        target = next(identity for identity, row in old.items()
                      if row["field_key"] == "tasks.0.description")
        plan = {"bundles": {"B1": {"evidence_ids": ["E2"],
            "operations": [{"kind": "update_task", "target_id": target,
                "field_key": "tasks.0.description",
                "value_json": json.dumps("Проверить другую запись.")}]}}}
        with self.assertRaisesRegex(ValueError, "task_repair_without_evidence"):
            stage_patch_candidate(document, plan, old, self.snapshot.index)
        with self.assertRaisesRegex(ValueError, "task_repair_evidence_unknown"):
            stage_patch_candidate(document, plan, old, self.snapshot.index, {})
        self.assertEqual(document["tasks"][0]["source_ids"], ["U00001"])

    def test_inventory_requires_every_core_id_and_unit_evidence(self):
        packet = plan_packets(self.snapshot, target_chars=9999)[0]
        report = {"schema_version": "luna_inventory_v1", "complete": True,
                  "evidence": [{"evidence_id": "E1", "u_id": "U00001",
                                "quote": "Предлагаю проверить сигнал."}],
                  "units": [{"unit_id": "A1", "kind": "action", "text": "Предложена проверка сигнала.",
                             "evidence_ids": ["E1"], "facets": [{"facet_id": "F1", "axis": "status",
                             "value": "предложение", "evidence_ids": ["E1"]}]}],
                  "source_accounting": [
                      {"u_id": "U00001", "disposition": "content", "unit_ids": ["A1"], "note": None},
                      {"u_id": "U00002", "disposition": "context_only", "unit_ids": [], "note": None},
                      {"u_id": "U00003", "disposition": "uncertain", "unit_ids": [], "note": None},
                  ], "open_links": [], "unprocessed_ids": []}
        checked = validate_inventory(report, packet, self.snapshot)
        self.assertIn("P001:A1", checked["units"])
        self.assertFalse(checked["complete"])
        self.assertEqual(checked["source_accounting"]["U00003"]["disposition"],
                         "uncertain")
        self.assertEqual(set(normalize_inventories([checked], (packet,), self.snapshot)["source_accounting"]),
                         set(self.snapshot.ids))
        del report["source_accounting"][-1]
        with self.assertRaisesRegex(ValueError, "source_accounting_coverage"):
            validate_inventory(report, packet, self.snapshot)

    def test_inventory_salvage_keeps_valid_units_and_marks_bad_accounting_uncertain(self):
        packet = plan_packets(self.snapshot, target_chars=9999)[0]
        report = {"schema_version": "luna_inventory_v1", "complete": True,
            "evidence": [{"evidence_id": "E1", "u_id": "U00001",
                          "quote": "Предлагаю проверить сигнал."}],
            "units": [{"unit_id": "A1", "kind": "action", "text": "Проверить сигнал",
                       "evidence_ids": ["E1"], "facets": [{"facet_id": "F1",
                       "axis": "action", "value": "проверка", "evidence_ids": ["E1"]}]}],
            "source_accounting": [
                {"u_id": "U00001", "disposition": "content", "unit_ids": ["A1"], "note": None},
                {"u_id": "U00002", "disposition": "content", "unit_ids": ["A1"], "note": None},
                {"u_id": "U00003", "disposition": "context_only", "unit_ids": [], "note": None}],
            "open_links": [], "unprocessed_ids": []}
        with self.assertRaisesRegex(ValueError, "content_unit_lacks_core_evidence"):
            validate_inventory(report, packet, self.snapshot)
        checked = salvage_inventory(report, packet, self.snapshot)
        self.assertFalse(checked["complete"])
        self.assertEqual(set(checked["units"]), {"P001:A1"})
        self.assertEqual(checked["source_accounting"]["U00002"]["disposition"], "uncertain")
        self.assertEqual(checked["uncertain_core_ids"], ["U00002"])
        self.assertEqual(checked["native_report"], report)

    def test_inventory_salvage_discards_unit_with_missing_evidence_and_fails_on_collisions(self):
        packet = plan_packets(self.snapshot, target_chars=9999)[0]
        report = {"schema_version": "luna_inventory_v1", "complete": True,
            "evidence": [{"evidence_id": "E1", "u_id": "U00001",
                          "quote": "Предлагаю проверить сигнал."}],
            "units": [
                {"unit_id": "A1", "kind": "action", "text": "Проверить сигнал",
                 "evidence_ids": ["E1"], "facets": []},
                {"unit_id": "A2", "kind": "action", "text": "Взять другую запись",
                 "evidence_ids": ["E2"], "facets": []}],
            "source_accounting": [
                {"u_id": "U00001", "disposition": "content", "unit_ids": ["A1"], "note": None},
                {"u_id": "U00002", "disposition": "content", "unit_ids": ["A2"], "note": None},
                {"u_id": "U00003", "disposition": "context_only", "unit_ids": [], "note": None}],
            "open_links": [], "unprocessed_ids": []}
        checked = salvage_inventory(report, packet, self.snapshot)
        self.assertEqual(set(checked["units"]), {"P001:A1"})
        self.assertFalse(checked["complete"])
        self.assertEqual(checked["source_accounting"]["U00002"]["disposition"], "uncertain")
        self.assertIn({"kind": "unit", "id": "A2", "reason": "unit_evidence_invalid"},
                      checked["discarded"])
        report["evidence"].append(report["evidence"][0].copy())
        with self.assertRaisesRegex(ValueError, "evidence_identity_invalid"):
            salvage_inventory(report, packet, self.snapshot)

    def test_needs_review_survives_exact_snapshot_and_source_packet(self):
        path = self.snapshot.transcript_path
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["utterances"][1]["uncertainty"] = {"needs_review": True}
        path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        snapshot = load_snapshot(path)
        self.assertTrue(snapshot.records[1]["needs_review"])
        self.assertNotIn("needs_review", snapshot.records[0])
        packet = plan_packets(snapshot)[0]
        self.assertTrue(packet["records"][1]["needs_review"])

    def test_all_surfaces_are_partitioned_and_patch_target_is_allowlisted(self):
        document = _document()
        registry = build_surfaces(document, self.snapshot.index)
        self.assertIn("tasks.0.assignee", [row["field_key"] for row in registry.values()])
        packets = plan_packets(self.snapshot, target_chars=220)
        bins = partition_surfaces(registry, packets)
        self.assertEqual({row["surface_id"] for group in bins for row in group}, set(registry))
        target = next(identity for identity, row in registry.items()
                      if row["field_key"] == "main.0.text")
        finding = {"F1": {"finding_id": "F1"}}
        raw = {"schema_version": "luna_patch_plan_v1", "complete": True,
               "bundles": [{"bundle_id": "B1", "finding_ids": ["F1"],
                            "evidence_ids": ["E1"], "affected_surface_ids": [target],
                            "operations": [{"kind": "replace_field", "target_id": target,
                                            "field_key": "main.0.text",
                                            "value_json": json.dumps("Обсудили проверку сигнала и её условие."),
                                            "position_after_id": None, "temp_id": None,
                                            "lineage_ids": []}],
                            "preservation_notes": "Сохранить предложение.",
                            "dependency_bundle_ids": []}],
               "unresolved": [], "unprocessed_finding_ids": []}
        plan = validate_patch_plan(raw, findings=finding, registry=registry,
                                   known_evidence_ids={"E1"})
        candidate, hashes = stage_patch_candidate(document, plan, registry, self.snapshot.index)
        self.assertIn("условие", candidate["main"][0]["text"])
        self.assertEqual(document["main"][0]["text"], "Обсудили проверку сигнала.")
        self.assertEqual(len(hashes["B1"][0]["expected_before_hash"]), 64)
        raw["bundles"][0]["operations"][0]["field_key"] = "tasks.0.assignee"
        with self.assertRaisesRegex(ValueError, "patch_target_unauthorized"):
            validate_patch_plan(raw, findings=finding, registry=registry,
                                known_evidence_ids={"E1"})

    def test_known_contested_actor_is_marked_without_mutating_draft(self):
        document = _document()
        document["tasks"][0]["assignee"] = "Аня"
        document["tasks"][0]["field_sources"]["assignee"] = ["U00001"]
        registry = build_surfaces(document, self.snapshot.index)
        target = next(identity for identity, row in registry.items()
                      if row["field_key"] == "tasks.0.assignee")
        finding = {"finding_id": "F1", "problem": "Исполнитель не согласован",
                   "affected_surface_ids": [target], "evidence_ids": ["E1"]}
        marked, marks = safe_mark_document(document, [finding], registry, self.snapshot,
                                           {"E1": {"u_id": "U00001"}})
        self.assertIsNone(marked["tasks"][0]["assignee"])
        self.assertEqual(document["tasks"][0]["assignee"], "Аня")
        self.assertEqual(marks[0]["source_ids"], ["U00001"])
        self.assertIn("Исполнитель не согласован", marked["verification"][-1]["text"])

    def test_global_can_cite_newly_found_source_unit(self):
        raw = {"schema_version": "luna_global_v1", "complete": True,
               "evidence": [{"evidence_id": "E1", "u_id": "U00001",
                             "quote": "Предлагаю проверить сигнал."}],
               "resolutions": [], "link_checks": [],
               "additional_units": [{"unit_id": "A1", "kind": "action",
                                     "text": "Проверить сигнал", "evidence_ids": ["E1"],
                                     "facets": []}],
               "findings": [{"finding_id": "F1", "kind": "omission",
                             "severity": "material", "affected_surface_ids": [],
                             "affected_unit_ids": ["A1"], "evidence_ids": ["E1"],
                             "problem": "Пропущено действие", "required_preservation": "Сохранить предложение",
                             "proposed_resolution": None}],
               "affected_surfaces": [], "unprocessed_ids": []}
        checked = validate_global(raw, snapshot=self.snapshot,
            expected_context_ids=set(), full_registry={}, expected_link_ids=set(),
            expected_unit_ids=set())
        self.assertEqual(checked["findings"][0]["affected_unit_ids"], ["global:extra:A1"])

    def test_dependency_diamond_keeps_atomic_verdict(self):
        plan = {"bundles": {
            "B1": {"dependency_bundle_ids": []},
            "B2": {"dependency_bundle_ids": ["B1"]},
            "B3": {"dependency_bundle_ids": ["B1"]},
            "B4": {"dependency_bundle_ids": ["B2", "B3"]},
        }}
        def check(bundle_id, verdict):
            return {"bundle_id": bundle_id, "verdict": verdict,
                    "evidence": [{"evidence_id": bundle_id, "u_id": "U00001",
                                  "quote": "Предлагаю проверить сигнал."}],
                    "preserved_surface_ids": [], "regressions": [], "note": "Источник"}
        raw = {"schema_version": "luna_verification_v1", "complete": True,
               "bundle_checks": [check(name, "accept") for name in plan["bundles"]],
               "new_findings": [], "unprocessed_bundle_ids": []}
        accepted = validate_verification(raw, plan=plan, snapshot=self.snapshot,
                                         registry={})["accepted"]
        self.assertEqual(accepted, set(plan["bundles"]))
        raw["bundle_checks"][2]["verdict"] = "reject"
        rejected_group = validate_verification(raw, plan=plan, snapshot=self.snapshot,
                                               registry={})["accepted"]
        self.assertEqual(rejected_group, set())

    def test_verification_new_finding_has_validated_source_and_surface(self):
        registry = build_surfaces(_document(), self.snapshot.index)
        surface = next(iter(registry))
        plan = {"bundles": {"B1": {"dependency_bundle_ids": []}}}
        raw = {"schema_version": "luna_verification_v1", "complete": True,
               "bundle_checks": [{
                   "bundle_id": "B1", "verdict": "accept",
                   "evidence": [{"evidence_id": "E1", "u_id": "U00001",
                                 "quote": "Предлагаю проверить сигнал."}],
                   "preserved_surface_ids": [], "regressions": [], "note": "Источник",
               }],
               "new_findings": [{
                   "finding_id": "F1", "kind": "unsupported_claim",
                   "severity": "material", "affected_surface_ids": [surface],
                   "affected_unit_ids": [], "evidence_ids": ["E1"],
                   "problem": "Новое противоречие", "required_preservation": "Сохранить условие",
                   "proposed_resolution": None,
               }], "unprocessed_bundle_ids": []}
        checked = validate_verification(raw, plan=plan, snapshot=self.snapshot,
                                         registry=registry)
        self.assertEqual(checked["accepted"], set())
        self.assertEqual(checked["new_findings"][0]["finding_id"], "verify:F1")
        self.assertEqual(checked["new_findings"][0]["evidence_ids"], ["verify:E1"])
        raw["new_findings"][0]["evidence_ids"] = ["forged"]
        with self.assertRaisesRegex(ValueError, "verification_new_finding_invalid"):
            validate_verification(raw, plan=plan, snapshot=self.snapshot,
                                  registry=registry)


if __name__ == "__main__":
    unittest.main()
