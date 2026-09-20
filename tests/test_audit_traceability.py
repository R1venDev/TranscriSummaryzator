"""Executable traceability for the 2026-09-17 deep audit.

This does not pretend that a finite suite proves correctness for every future
conversation.  It makes the narrower guarantee reviewable: every reported
TS-001..TS-068 issue and every publication stage has at least one executable
regression/invariant test in this repository.
"""
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


def refs(file_name, *test_names):
    return tuple(f"tests/{file_name}::{name}" for name in test_names)


SEMANTIC_STATE = refs(
    "test_v26_deep_audit.py",
    "test_retyping_keeps_reported_work_instead_of_deleting_it",
    "test_past_attempt_and_intent_are_preserved_as_states",
    "test_internal_action_identifier_keeps_human_task_and_counterparty_acceptance",
)
ACTION_FRAMES = refs(
    "test_quality_schema.py",
    "test_two_source_actions_survive_one_opaque_model_frame",
    "test_nearby_speaker_cannot_replace_audited_task_owner",
)
RELATIONS = refs(
    "test_v26_deep_audit.py",
    "test_intervening_question_captures_yes_instead_of_proposal",
    "test_semantic_equivalence_preserves_polarity_numbers_conditions_and_actor",
)
QUESTIONS = refs(
    "test_latest_audit.py",
    "test_typed_slots_close_direct_answers",
    "test_unverified_candidate_answer_is_not_published_as_known",
)
PUBLIC_AST = refs(
    "test_reaudit_run13_v25.py",
    "test_unknown_evidence_and_unrendered_card_are_rejected",
    "test_contextual_sections_do_not_invent_relations_from_proximity",
    "test_chapter_label_never_hides_mid_sentence_truncation",
)
LINEAGE = refs(
    "test_summary_worker.py",
    "test_candidate_lineage_prefers_exact_action_then_safe_rekey",
    "test_rejected_revision_is_terminal_despite_origin_rewrite",
)
NUMBERS = refs(
    "test_quality_schema.py",
    "test_quantity_without_valid_evidence_is_rejected",
    "test_signed_index_quantity_is_not_a_timeframe",
)
FINAL_GATES = refs(
    "test_latest_audit.py",
    "test_utility_gate_rejects_editorially_bad_public_surface",
    "test_publication_gate_rejects_surface_regressions",
)
PLANNING = refs(
    "test_v21_architecture.py",
    "test_planner_builds_view_specific_subgraphs_and_contracts",
    "test_actual_output_verifier_preserves_modality_condition_and_negation",
)
OPERATIONS = refs(
    "test_v23_operations.py",
    "test_same_stage_key_restores_artifact_across_jobs",
    "test_per_item_decisions_go_to_trace_and_summary_has_operations",
)
DELIVERY = refs(
    "test_reaudit_run13_v25.py",
    "test_incomplete_generation_manifest_is_rejected",
    "test_generation_pointer_is_visible_to_status_polling",
) + refs(
    "test_v26_deep_audit.py",
    "test_standalone_time_is_plain_and_absolute_application_link_is_supported",
)


AUDIT_ISSUE_TESTS = {
    # Each entry names the narrowest executable regression for the audit item;
    # this is intentionally not a blanket range-to-suite mapping.
    "TS-001": refs("test_integrity.py", "test_final_auditor_retypes_supported_content_instead_of_deleting_it"),
    "TS-002": refs("test_quality_schema.py", "test_source_past_attempt_overrides_model_in_progress_label"),
    "TS-003": refs("test_v26_deep_audit.py", "test_source_intent_to_try_an_extra_timeframe_is_retained"),
    "TS-004": refs("test_v26_deep_audit.py", "test_past_attempt_and_intent_are_preserved_as_states"),
    "TS-005": refs("test_v26_deep_audit.py", "test_second_person_marker_and_imbalance_experiment_remain_separate_tasks"),
    "TS-006": refs("test_v26_deep_audit.py", "test_second_person_marker_and_imbalance_experiment_remain_separate_tasks"),
    "TS-007": refs("test_v26_deep_audit.py", "test_multiple_actions_keep_independent_roles_and_lineage"),
    "TS-008": refs("test_summary_worker.py", "test_bare_maxim_is_marked_ambiguous"),
    "TS-009": refs("test_v26_deep_audit.py", "test_intervening_question_captures_yes_instead_of_proposal"),
    "TS-010": refs("test_v26_deep_audit.py", "test_proposal_cannot_supply_its_own_acceptance_evidence"),
    "TS-011": refs("test_latest_audit.py", "test_typed_slots_close_direct_answers"),
    "TS-012": refs("test_v26_deep_audit.py", "test_structured_answer_cannot_override_opposite_source_text"),
    "TS-013": refs("test_v26_deep_audit.py", "test_unverified_candidate_answer_is_not_reopened_as_raw_question"),
    "TS-014": refs("test_reaudit_run13_v25.py", "test_question_projection_does_not_overwrite_answer_claim_in_outcome"),
    "TS-015": refs("test_v26_deep_audit.py", "test_late_technical_pass_recovers_full_path_correction_and_objection"),
    "TS-016": refs("test_quality_schema.py", "test_explicit_correction_supersedes_low_overlap_claim"),
    "TS-017": refs("test_latest_audit.py", "test_internal_claim_label_falls_back_to_safe_dialogue_evidence"),
    "TS-018": refs("test_reaudit_run13_v25.py", "test_contextual_sections_do_not_invent_relations_from_proximity"),
    "TS-019": refs("test_v24_generation.py", "test_equivalent_outcome_field_represents_chronology_item"),
    "TS-020": refs("test_v26_deep_audit.py", "test_resource_is_not_labeled_as_work_result"),
    "TS-021": refs("test_summary_worker.py", "test_rejected_revision_is_terminal_despite_origin_rewrite"),
    "TS-022": refs("test_v26_deep_audit.py", "test_semantic_equivalence_preserves_polarity_numbers_conditions_and_actor"),
    "TS-023": refs("test_v26_deep_audit.py", "test_semantic_equivalence_preserves_polarity_numbers_conditions_and_actor"),
    "TS-024": refs("test_summary_worker.py", "test_evidence_containment_does_not_remove_additional_action"),
    "TS-025": refs("test_v24_generation.py", "test_quarantined_claim_always_has_public_sentence_plan"),
    "TS-026": refs("test_v26_deep_audit.py", "test_same_timestamp_fact_cannot_steal_compound_proposal_acceptance"),
    "TS-027": refs("test_summary_worker.py", "test_candidate_lineage_prefers_exact_action_then_safe_rekey"),
    "TS-028": refs("test_reaudit_run13_v25.py", "test_semantic_revision_changes_generation_id"),
    "TS-029": refs("test_v24_generation.py", "test_merged_task_cites_all_canonical_source_claims"),
    "TS-030": refs("test_quality_schema.py", "test_two_source_actions_survive_one_opaque_model_frame"),
    "TS-031": refs("test_reaudit_run13_v25.py", "test_unknown_evidence_and_unrendered_card_are_rejected"),
    "TS-032": refs("test_reaudit_run13_v25.py", "test_number_time_and_profile_normalization"),
    "TS-033": refs("test_reaudit_run13_v25.py", "test_number_time_and_profile_normalization"),
    "TS-034": refs("test_v23_operations.py", "test_conflicting_end_of_meeting_times_leave_exact_time_open"),
    "TS-035": refs("test_summary_worker.py", "test_conflicting_percent_estimates_are_not_reconciled_by_writer"),
    "TS-036": refs("test_summary_worker.py", "test_malformed_asr_term_is_kept_out_of_public_summary"),
    "TS-037": refs("test_summary_worker.py", "test_overlap_uncertainty_blocks_main_content"),
    "TS-038": refs("test_summary_worker.py", "test_semantic_registry_splits_incomplete_response"),
    "TS-039": refs("test_latest_audit.py", "test_utility_gate_rejects_editorially_bad_public_surface"),
    "TS-040": refs("test_summary_worker.py", "test_final_document_reconciliation_does_not_hide_bad_public_item"),
    "TS-041": refs("test_v24_generation.py", "test_final_document_verifier_uses_renderer_normalization"),
    "TS-042": refs("test_summary_worker.py", "test_structure_gate_requires_detailed_topic_coverage"),
    "TS-043": refs("test_v21_architecture.py", "test_planner_builds_view_specific_subgraphs_and_contracts"),
    "TS-044": refs("test_v26_deep_audit.py", "test_chapter_count_is_adaptive_below_configured_ceiling"),
    "TS-045": refs("test_v26_deep_audit.py", "test_title_repair_uses_supported_themes_not_raw_tasks"),
    "TS-046": refs("test_reaudit_run13_v25.py", "test_chapter_uses_episode_end"),
    "TS-047": refs("test_v26_deep_audit.py", "test_final_audit_inventory_includes_chronology_items"),
    "TS-048": refs("test_summary_worker.py", "test_navigation_audit_uses_evidence_start_and_has_no_noise"),
    "TS-049": refs("test_v26_deep_audit.py", "test_multi_episode_document_title_uses_two_supported_claims"),
    "TS-050": refs("test_summary_worker.py", "test_overview_uses_real_chapter_evidence"),
    "TS-051": refs("test_v26_deep_audit.py", "test_rules_have_a_dedicated_protected_reader_view"),
    "TS-052": refs("test_reaudit_run13_v25.py", "test_goal_only_claim_is_not_labeled_as_experiment"),
    "TS-053": refs("test_v26_deep_audit.py", "test_late_path_recovery_is_not_tied_to_specific_timeframes"),
    "TS-054": refs("test_v23_operations.py", "test_schedule_is_not_hardcoded_and_preserves_or_alternatives"),
    "TS-055": refs("test_quality_schema.py", "test_uncertain_task_is_not_eligible_for_automatic_creation"),
    "TS-056": refs("test_v26_deep_audit.py", "test_capability_idea_is_not_promoted_to_current_task"),
    "TS-057": refs("test_v26_deep_audit.py", "test_topic_entities_are_not_empty_when_topic_is_known"),
    "TS-058": refs("test_v26_deep_audit.py", "test_task_surface_rules_generalize_to_unrelated_entities_and_artifacts"),
    "TS-059": refs("test_summary_worker.py", "test_final_document_audit_bounds_model_batches"),
    "TS-060": refs("test_v23_operations.py", "test_per_item_decisions_go_to_trace_and_summary_has_operations"),
    "TS-061": refs("test_integrity.py", "test_length_limited_ollama_stream_rejected"),
    "TS-062": refs("test_latest_audit.py", "test_safe_diagnostics_keep_counts_and_hashes"),
    "TS-063": refs("test_v26_deep_audit.py", "test_force_policy_separates_view_rebuild_from_fresh_model_calls"),
    "TS-064": refs("test_v26_deep_audit.py", "test_release_manifest_ignores_operating_system_sidecars"),
    "TS-065": refs("test_v26_deep_audit.py", "test_exported_schema_versions_match_registry"),
    "TS-066": refs("test_v26_deep_audit.py", "test_standalone_time_is_plain_and_absolute_application_link_is_supported"),
    "TS-067": refs("test_v26_deep_audit.py", "test_meeting_date_prefers_explicit_timestamp_and_rejects_partial_dates"),
    "TS-068": refs("test_latest_audit.py", "test_visual_mixed_script_typos_are_normalized_without_translation"),
}


STAGE_TESTS = {
    "asr_boundaries": refs("test_asr_boundaries.py", "test_duplicate_in_actual_overlap_removed"),
    "evidence_repair": refs("test_evidence_repair.py", "test_repair_compares_units_entities_modality_and_direction"),
    "fact_policy": refs("test_integrity.py", "test_final_auditor_retypes_supported_content_instead_of_deleting_it"),
    "semantic_normalization": ACTION_FRAMES,
    "meeting_graph": RELATIONS,
    "canonical_reducers": SEMANTIC_STATE + QUESTIONS,
    "planner": PLANNING,
    "public_items": refs("test_v24_generation.py", "test_quarantined_claim_always_has_public_sentence_plan"),
    "document_ast": PUBLIC_AST,
    "final_semantic_audit": refs("test_summary_worker.py", "test_final_document_audit_bounds_model_batches"),
    "render_verification": refs("test_v24_generation.py", "test_final_document_mutations_are_rejected"),
    "runtime_quality_gates": FINAL_GATES,
    "candidate_disposition": LINEAGE,
    "atomic_generation": DELIVERY,
    "replay_and_diagnostics": OPERATIONS,
}


class AuditTraceabilityTests(unittest.TestCase):
    def test_every_audit_issue_has_executable_regression_coverage(self):
        expected = {f"TS-{index:03d}" for index in range(1, 69)}
        self.assertEqual(set(AUDIT_ISSUE_TESTS), expected)
        self._assert_references_exist(AUDIT_ISSUE_TESTS)

    def test_every_pipeline_stage_has_executable_contract_coverage(self):
        self.assertGreaterEqual(len(STAGE_TESTS), 15)
        self._assert_references_exist(STAGE_TESTS)

    def _assert_references_exist(self, mapping):
        for key, references in mapping.items():
            self.assertTrue(references, key)
            for reference in references:
                path_text, test_name = reference.split("::", 1)
                path = ROOT / path_text
                self.assertTrue(path.is_file(), reference)
                source = path.read_text(encoding="utf-8")
                self.assertRegex(source, rf"(?m)^\s*def\s+{re.escape(test_name)}\s*\(", reference)


if __name__ == "__main__":
    unittest.main()
