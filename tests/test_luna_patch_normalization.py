"""Offline recovery of repair envelopes without relaxing source/target guards."""
from copy import deepcopy
import json
import unittest

from summary.luna_v1.patch_normalization import _merge_text, normalize_patch_report
from summary.luna_v1.source_first_core import build_surfaces, stage_patch_candidate
from tests import test_luna_source_first_core as core_tests
from tests import test_luna_source_first_runtime as runtime_tests
from summary.luna_v1.ledger import write_private_json
import hashlib
from unittest.mock import patch

_document = core_tests._document


class PatchNormalizationTests(unittest.TestCase):
    setUp = core_tests.SourceFirstCoreTests.setUp

    def _bundle(self, identity, field="main.0.text", value="Проверка предложена."):
        registry = build_surfaces(_document(), self.snapshot.index)
        target = next(k for k, s in registry.items() if s["field_key"] == field)
        return {"bundle_id": identity, "finding_ids": ["F" + identity],
            "evidence_ids": ["E1"], "affected_surface_ids": [target],
            "operations": [{"kind": "replace_field", "target_id": target,
                "field_key": field, "value_json": json.dumps(value, ensure_ascii=False),
                "position_after_id": None, "temp_id": None, "lineage_ids": []}],
            "preservation_notes": "Сохранить статус предложения.", "dependency_bundle_ids": []}

    def _normalize(self, bundles, *, unresolved=None, document=None):
        raw = {"schema_version": "luna_patch_plan_v1", "complete": True,
            "bundles": bundles, "unresolved": unresolved or [], "unprocessed_finding_ids": []}
        findings = {"F" + b["bundle_id"]: {} for b in bundles}
        if unresolved:
            findings["F-unresolved"] = {}
        document = document or _document()
        plan, provenance = normalize_patch_report(raw, document=document,
            registry=build_surfaces(document, self.snapshot.index), findings=findings,
            evidence={"E1": {"u_id": "U00001"}}, source_index=self.snapshot.index)
        return plan, provenance, raw

    def test_exact_id_explanation_normalization_keeps_native(self):
        plan, provenance, raw = self._normalize([self._bundle("1")],
            unresolved=["F-unresolved — Неясная связь."])
        self.assertEqual(plan["unresolved"], ["F-unresolved"])
        self.assertEqual(raw["unresolved"], ["F-unresolved — Неясная связь."])
        self.assertEqual(provenance["transforms"][0]["explanation"], "Неясная связь.")

    def test_unknown_or_partial_id_is_not_guessed(self):
        with self.assertRaisesRegex(ValueError, "patch_finding_unknown"):
            self._normalize([self._bundle("1")], unresolved=["F-unresolved-other — Объяснение"])

    def test_entity_plus_exact_field_resolves_to_app_surface(self):
        bundle = self._bundle("1")
        target = bundle["operations"][0]["target_id"]
        bundle["operations"][0]["target_id"] = "main.0"
        plan, provenance, _ = self._normalize([bundle])
        self.assertEqual(plan["bundles"]["1"]["operations"][0]["target_id"], target)
        self.assertEqual(provenance["transforms"][0]["kind"], "entity_field_to_surface")

    def test_mismatched_entity_is_quarantined(self):
        bad, good = self._bundle("1"), self._bundle("2", "tasks.0.description", "Проверка по предложению.")
        bad["operations"][0]["target_id"] = "main.999"
        plan, provenance, _ = self._normalize([bad, good])
        self.assertEqual(set(plan["bundles"]), {"2"})
        self.assertEqual(plan["unprocessed_finding_ids"], ["F1"])
        self.assertFalse(plan["complete"])
        self.assertTrue(provenance["discarded"])

    def test_whole_task_expands_changed_fields_and_retains_action_provenance(self):
        document = _document()
        task = deepcopy(document["tasks"][0]); task["description"] = "Проверка только предложена."
        task["due"] = "После согласования"
        task["field_sources"]["due"] = ["U00001"]
        bundle = self._bundle("1", "tasks.0.description")
        bundle["operations"][0].update(kind="update_task", target_id="tasks.0", field_key=None,
            value_json=json.dumps(task, ensure_ascii=False))
        plan, provenance, raw = self._normalize([bundle])
        operations = plan["bundles"]["1"]["operations"]
        self.assertEqual({op["field_key"] for op in operations}, {"tasks.0.description", "tasks.0.due"})
        candidate, _ = stage_patch_candidate(document, plan,
            build_surfaces(document, self.snapshot.index), self.snapshot.index, {"E1": {"u_id": "U00001"}})
        self.assertIsNone(candidate["tasks"][0]["assignee"])
        self.assertEqual(candidate["tasks"][0]["field_sources"]["action"], ["U00001"])
        self.assertEqual(raw["bundles"][0]["operations"][0]["field_key"], None)

    def test_whole_task_cannot_inject_uuid_or_unknown_source(self):
        for injected in ("uuid", "source"):
            with self.subTest(injected=injected):
                task = deepcopy(_document()["tasks"][0])
                if injected == "uuid": task["id"] = "invented"
                else: task["source_ids"].append("U99999")
                bundle = self._bundle("1", "tasks.0.description")
                bundle["operations"][0].update(kind="update_task", target_id="tasks.0", field_key=None,
                    value_json=json.dumps(task))
                plan, provenance, _ = self._normalize([bundle])
                self.assertFalse(plan["bundles"])
                self.assertTrue(provenance["discarded"])

    def test_disjoint_and_identical_text_edits_merge(self):
        base = "Начало один середина два конец."
        self.assertEqual(_merge_text(base, ["Начало первый середина два конец.",
            "Начало один середина второй конец."]), "Начало первый середина второй конец.")
        self.assertEqual(_merge_text(base, [base, base]), base)
        with self.assertRaisesRegex(ValueError, "patch_text_merge_conflict"):
            _merge_text(base, ["Начало три середина два конец.", "Начало четыре середина два конец."])

    def test_shared_target_components_merge_with_lineage(self):
        doc = _document(); doc["main"][0]["text"] = "Начало один середина два конец."
        a = self._bundle("1", value="Начало первый середина два конец.")
        b = self._bundle("2", value="Начало один середина второй конец.")
        plan, provenance, _ = self._normalize([a, b], document=doc)
        self.assertEqual(len(plan["bundles"]), 1)
        identity = next(iter(plan["bundles"]))
        self.assertEqual(provenance["bundle_lineage"][identity], ["1", "2"])
        self.assertEqual(plan["bundles"][identity]["finding_ids"], ["F1", "F2"])

    def test_conflict_does_not_drop_unrelated_fix(self):
        a, b = self._bundle("1", value="Вариант А."), self._bundle("2", value="Вариант Б.")
        good = self._bundle("3", "tasks.0.description", "Проверка предложена.")
        plan, provenance, _ = self._normalize([a, b, good])
        self.assertEqual(set(plan["bundles"]), {"3"})
        self.assertEqual(set(plan["unprocessed_finding_ids"]), {"F1", "F2"})

    def test_dependency_group_is_atomic_and_cycles_remain_invalid(self):
        a, b = self._bundle("1"), self._bundle("2", "tasks.0.description", "Проверка предложена.")
        a["dependency_bundle_ids"] = ["2"]
        plan, _, _ = self._normalize([a, b])
        self.assertEqual(len(plan["bundles"]), 1)
        b["dependency_bundle_ids"] = ["1"]
        plan, provenance, _ = self._normalize([a, b])
        self.assertFalse(plan["bundles"])
        self.assertEqual(provenance["discarded"][0]["reason"], "patch_dependency_cycle")

    def test_duplicate_finding_accounting_never_becomes_supported(self):
        a, b = self._bundle("1"), self._bundle("2")
        b["finding_ids"] = ["F1"]
        with self.assertRaisesRegex(ValueError, "patch_findings_not_accounted"):
            self._normalize([a, b])

    def test_shared_insertion_anchor_is_quarantined_without_losing_other_fix(self):
        a, b = self._bundle("1"), self._bundle("2")
        for bundle in (a, b):
            op = bundle["operations"][0]
            op.update(kind="add_section_item", target_id=None, field_key="main",
                position_after_id=bundle["affected_surface_ids"][0],
                value_json=json.dumps({"text": "Дополнение.", "source_ids": ["U00001"]}))
        plan, provenance, _ = self._normalize([a, b,
            self._bundle("3", "tasks.0.description", "Проверка предложена.")])
        self.assertEqual(set(plan["bundles"]), {"3"})
        self.assertEqual(set(plan["unprocessed_finding_ids"]), {"F1", "F2"})
        self.assertEqual(provenance["discarded"][0]["reason"], "reused_patch_insertion_anchor")


class SavedRepairLedgerTests(unittest.TestCase):
    setUp = runtime_tests.SourceFirstRuntimeTests.setUp
    _reserved = runtime_tests.SourceFirstRuntimeTests._reserved

    def _completed_repair(self):
        attempt_id, _, _, _ = self._reserved()
        terminal = self.root / "repair-terminal.json"
        terminal.write_text('{"status":"completed"}')
        terminal_sha = hashlib.sha256(terminal.read_bytes()).hexdigest()
        self.ledger.db.execute("UPDATE batch_attempts SET stage='repair',status='completed',"
            "billed_microusd=500,terminal_path=?,terminal_sha256=? WHERE id=?",
            (str(terminal), terminal_sha, attempt_id))
        accepted = self.root / "model_document.json"
        accepted.write_text('{"published":"draft with cautions"}')
        accepted_sha = hashlib.sha256(accepted.read_bytes()).hexdigest()
        self.ledger.finish_batch_workflow(self.workflow["id"], status="accepted",
                                         result_path=accepted, result_sha256=accepted_sha)
        receipt = self.root / "repair-recovery.json"
        write_private_json(receipt, {"workflow_id": self.workflow["id"],
            "source_sha256": self.workflow["source_sha256"],
            "repair_terminal_sha256": terminal_sha,
            "accepted_document_path": str(accepted), "accepted_document_sha256": accepted_sha})
        return receipt, hashlib.sha256(receipt.read_bytes()).hexdigest(), accepted

    def test_continuation_retains_bills_cap_and_old_document_is_idempotent(self):
        receipt, sha, accepted = self._completed_repair()
        before = accepted.read_bytes()
        self.assertTrue(self.ledger.resume_saved_repair(self.workflow["id"],
            receipt_path=receipt, receipt_sha256=sha))
        self.assertFalse(self.ledger.resume_saved_repair(self.workflow["id"],
            receipt_path=receipt, receipt_sha256=sha))
        self.assertEqual(self.ledger.get_batch_workflow(self.workflow["id"])["status"], "active")
        self.assertEqual(self.ledger.get_batch_workflow(self.workflow["id"])["planned_reserve_microusd"], 10000)
        self.assertEqual(self.ledger.list_batch_attempts(self.workflow["id"])[0]["billed_microusd"], 500)
        self.assertEqual(accepted.read_bytes(), before)
        self.assertEqual(self.ledger._rolling_spent_microusd(__import__('time').time()), 10000)

    def test_changed_published_document_blocks_continuation(self):
        receipt, sha, accepted = self._completed_repair(); accepted.write_text('changed')
        with self.assertRaisesRegex(ValueError, "repair_recovery_not_eligible"):
            self.ledger.resume_saved_repair(self.workflow["id"], receipt_path=receipt, receipt_sha256=sha)
        self.assertEqual(self.ledger.get_batch_workflow(self.workflow["id"])["status"], "accepted")

    def test_verification_already_exists_cannot_be_sent_again(self):
        receipt, sha, _ = self._completed_repair()
        self.ledger.db.execute("UPDATE batch_attempts SET stage='verify' WHERE workflow_id=?",
                               (self.workflow["id"],))
        with self.assertRaisesRegex(ValueError, "repair_recovery_not_eligible"):
            self.ledger.resume_saved_repair(self.workflow["id"], receipt_path=receipt, receipt_sha256=sha)

    def test_shared_weekly_cap_blocks_reopening(self):
        receipt, sha, _ = self._completed_repair()
        with patch('summary.luna_v1.ledger.WEEK_CAP_MICROUSD', 9999):
            with self.assertRaisesRegex(ValueError, "repair_recovery_weekly_budget_exceeded"):
                self.ledger.resume_saved_repair(self.workflow["id"], receipt_path=receipt, receipt_sha256=sha)
        self.assertEqual(self.ledger.get_batch_workflow(self.workflow["id"])["status"], "accepted")

    def test_old_bills_do_not_underreserve_reopened_weekly_plan(self):
        receipt, sha, _ = self._completed_repair()
        self.ledger.db.execute("UPDATE batch_attempts SET created_at=? WHERE workflow_id=?",
            (__import__('time').time() - 8 * 86400, self.workflow["id"]))
        self.assertEqual(self.ledger._rolling_spent_microusd(__import__('time').time()), 0)
        with patch('summary.luna_v1.ledger.WEEK_CAP_MICROUSD', 9999):
            with self.assertRaisesRegex(ValueError, "repair_recovery_weekly_budget_exceeded"):
                self.ledger.resume_saved_repair(self.workflow["id"], receipt_path=receipt, receipt_sha256=sha)
        self.assertEqual(self.ledger.get_batch_workflow(self.workflow["id"])["status"], "accepted")
        self.assertIsNone(self.ledger.get_batch_workflow(self.workflow["id"])["repair_recovery_sha256"])


if __name__ == "__main__":
    unittest.main()
