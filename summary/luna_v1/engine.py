"""Summary-only Luna Batch dispatch, recovery and local publication.

The watcher calls submit once and a low-frequency scheduler calls poll_once.
Neither path touches audio, ASR, diarization, Ollama, or Plane.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_UP
from email.utils import parsedate_to_datetime
from pathlib import Path

from scripts.summary_credentials import CredentialError, CredentialStore, credential_dispatch_guard

from . import PROMPT_PATH, SCHEMA, SCHEMA_ID, load_source, validate_document
from .audit import (AUDIT_PROMPT_PATH, AUDIT_SCHEMA, AUDIT_SCHEMA_ID,
                    apply_audit, build_audit_input, coverage_warnings,
                    validate_audit)
from .batch import BatchClient, BatchError, MODEL, TERMINAL, extract_one_completed, valid_batch_id
from summary.gemini_v1 import GEMINI_AUDIT_PROMPT_PATH, build_gemini_audit_input
from summary.gemini_v1.audit_v2 import (GEMINI_AUDIT_PROMPT_PATH_V2,
                                       GEMINI_AUDIT_SCHEMA_V2,
                                       GEMINI_AUDIT_SCHEMA_ID_V2,
                                       coverage_warnings_gemini_v2,
                                       legacy_report_v1,
                                       validate_gemini_audit_v2)
from summary.gemini_v1.inventory_contract import (
    INVENTORY_PROMPT_PATH, INVENTORY_SCHEMA, INVENTORY_SCHEMA_ID,
    build_inventory_input, merge_inventory_reports, plan_inventory_segments,
    validate_inventory_plan,
    validate_inventory_report,
)
from summary.gemini_v1.inventory_v2 import (
    INVENTORY_PROMPT_PATH_V2, INVENTORY_SCHEMA_V2, INVENTORY_SCHEMA_ID_V2,
    build_inventory_input_v2, inventory_risk_warnings_v2,
    merge_inventory_reports_v2, validate_inventory_report_v2,
)
from summary.gemini_v1.reconcile_contract import (
    RECONCILE_PROMPT_PATH, RECONCILE_SCHEMA, RECONCILE_SCHEMA_ID,
    build_reconcile_input, validate_reconciliation_report,
    apply_reconciliation_report, reconciliation_warnings,
)
from summary.gemini_v1.reconcile_v2 import (
    RECONCILE_PROMPT_PATH_V2, RECONCILE_SCHEMA_V2, RECONCILE_SCHEMA_ID_V2,
    build_reconcile_input_v2, validate_reconciliation_report_v2,
    apply_reconciliation_report_v2, reconciliation_warnings_v2,
    partition_reconcile_targets, build_reconcile_target_input_v2,
    validate_reconciliation_target_report_v2, reconciliation_target_warnings_v2,
)
from summary.gemini_v1.segment_review_contract import (
    SEGMENT_REVIEW_PROMPT_PATH, SEGMENT_REVIEW_SCHEMA,
    SEGMENT_REVIEW_SCHEMA_ID, SEGMENT_REVIEW_PROMPT_PATH_V2,
    SEGMENT_REVIEW_SCHEMA_V2, SEGMENT_REVIEW_SCHEMA_ID_V2,
    SEGMENT_REVIEW_PROMPT_PATH_V3, SEGMENT_REVIEW_SCHEMA_V3,
    SEGMENT_REVIEW_SCHEMA_ID_V3,
    build_segment_review_input,
    normalize_v3_segment_review_report,
    segment_form_warnings, validate_segment_review_report, segment_review_warnings,
    merge_segment_review_reports, segment_inventory_view,
)
from summary.gemini_v1.batch import BatchClient as GeminiBatchClient
from summary.gemini_v1.batch import BatchError as GeminiBatchError
from summary.gemini_v1.batch import BATCH_MODEL_IDS as GEMINI_BATCH_MODEL_IDS
from summary.gemini_v1.batch import extract_one_completed as extract_one_gemini_completed
from summary.gemini_v1.route import (RouteBlocked as GeminiRouteBlocked,
                                     verify_batch_route as verify_gemini_batch_route)
from summary.opus_v1 import (OPUS_AUDIT_PROMPT_PATH, OPUS_AUDIT_SCHEMA,
                             OPUS_AUDIT_SCHEMA_ID, apply_opus_audit,
                             build_opus_audit_request, coverage_warnings_opus,
                             validate_opus_audit,
                             OPUS_SEGMENT_OUTPUT_CAP, OPUS_SEGMENT_PROMPT_PATH,
                             OPUS_SEGMENT_SCHEMA, OPUS_SEGMENT_SCHEMA_ID,
                             OPUS_SEGMENT_OUTPUT_CAP_V3, OPUS_SEGMENT_EFFORT_V3,
                             OPUS_SEGMENT_PROMPT_PATH_V3, OPUS_SEGMENT_SCHEMA_V3,
                             OPUS_SEGMENT_SCHEMA_ID_V3,
                             build_opus_segment_audit_request,
                             merge_opus_segment_reports,
                             plan_opus_audit_segments, validate_opus_segment_report)
from summary.opus_v1.batch import (BatchClient as OpusBatchClient,
                                    BatchError as OpusBatchError,
                                    BATCH_MODEL_IDS as OPUS_BATCH_MODEL_IDS,
                                    MODEL as OPUS_MODEL,
                                    PROVIDER as OPUS_BATCH_PROVIDER,
                                    PRIVACY_MODE as OPUS_PRIVACY_MODE,
                                    WORKSPACE_IO_LOGGING_ENABLED as OPUS_WORKSPACE_IO_LOGGING_ENABLED,
                                    OUTPUT_CAP_AUDIT as OPUS_AUDIT_OUTPUT_CAP,
                                    OUTPUT_CAP_VERIFY as OPUS_VERIFY_OUTPUT_CAP,
                                    extract_one_completed as extract_one_opus_completed,
                                    valid_batch_id as valid_opus_batch_id)
from summary.opus_v1.route import (RouteBlocked as OpusRouteBlocked,
                                    verify_batch_route as verify_opus_batch_route)
from .ledger import (DIAGNOSTIC_KINDS, JOB_CAP_MICROUSD, MAX_DISPATCHES_PER_JOB, WEEK_CAP_MICROUSD, Ledger,
                     usd_micros, write_private_json)
from .publication import _verify_staged_target, publish_document
from .route import MAX_COMPLETION_TOKENS, RouteBlocked, verify_batch_route
from .tasks import RevisionConflict, TaskStore


PRIVACY_MODE = "batch_gateway_retention_up_to_30d_provider_zdr_off_user_authorized"
OPUS_WRITER_PRIVACY_MODE = (
    "batch_gateway_retention_up_to_30d_provider_zdr_off_"
    "workspace_input_output_logging_on_min_3mo_or_longer_user_authorized"
)
REASONING_EFFORT = "medium"
LEGACY_QUALITY_POLICY_VERSION = "luna_auto_audit_v3"
PREVIOUS_GEMINI_QUALITY_POLICY_VERSION = "gemini_openrouter_judge_repair_v2"
INVENTORY_V2_QUALITY_POLICY_VERSION = "gemini_openrouter_judge_repair_v3_source_inventory"
INVENTORY_RECONCILE_QUALITY_POLICY_VERSION = "gemini_openrouter_source_inventory_reconcile_v1"
EVIDENCE_RECONCILE_QUALITY_POLICY_VERSION = "gemini_openrouter_source_inventory_reconcile_v2"
SEGMENT_REVIEW_QUALITY_POLICY_VERSION_V1 = "gemini_openrouter_segment_review_v1"
SEGMENT_REVIEW_QUALITY_POLICY_VERSION_V2 = "gemini_openrouter_segment_review_v2_compact"
SEGMENT_REVIEW_QUALITY_POLICY_VERSION = "gemini_openrouter_segment_review_v3_packed"
SEGMENT_REVIEW_POLICIES = frozenset({SEGMENT_REVIEW_QUALITY_POLICY_VERSION_V1,
                                      SEGMENT_REVIEW_QUALITY_POLICY_VERSION_V2,
                                      SEGMENT_REVIEW_QUALITY_POLICY_VERSION})
OPUS_QUALITY_POLICY_VERSION = "claude_opus_5_5_full_source_audit_v1"
OPUS_PARTITIONED_QUALITY_POLICY_VERSION_V2 = "claude_opus_5_5_partitioned_audit_v2"
OPUS_PARTITIONED_QUALITY_POLICY_VERSION = "claude_opus_5_5_partitioned_audit_v3"
OPUS_PARTITIONED_POLICIES = frozenset({OPUS_PARTITIONED_QUALITY_POLICY_VERSION_V2,
                                       OPUS_PARTITIONED_QUALITY_POLICY_VERSION})
OPUS_QUALITY_POLICIES = frozenset({OPUS_QUALITY_POLICY_VERSION,
                                    *OPUS_PARTITIONED_POLICIES})
OPUS_QUALITY_PROVIDER = "openrouter_claude_opus"
OPUS_PARTITION_VERSION = "opus_three_primary_parts_overlap8_v1"
QUALITY_POLICY_VERSION = OPUS_PARTITIONED_QUALITY_POLICY_VERSION
# Retained for all already-submitted Gemini jobs and their saved manifests.
QUALITY_PROVIDER = "openrouter_gemini"
GEMINI_MODEL = "google/gemini-3.7-flash:batch"
GEMINI_PRIVACY_MODE = "openrouter_batch_30d_google_vertex_user_authorized"


def _opus_partition_profile(policy: str) -> tuple[Path, dict, str, int, str, str]:
    if policy == OPUS_PARTITIONED_QUALITY_POLICY_VERSION_V2:
        return (OPUS_SEGMENT_PROMPT_PATH, OPUS_SEGMENT_SCHEMA,
                OPUS_SEGMENT_SCHEMA_ID, OPUS_SEGMENT_OUTPUT_CAP, "medium", "v2")
    if policy == OPUS_PARTITIONED_QUALITY_POLICY_VERSION:
        return (OPUS_SEGMENT_PROMPT_PATH_V3, OPUS_SEGMENT_SCHEMA_V3,
                OPUS_SEGMENT_SCHEMA_ID_V3, OPUS_SEGMENT_OUTPUT_CAP_V3,
                OPUS_SEGMENT_EFFORT_V3, "v3")
    raise ValueError("unknown Opus partition policy")


QUALITY_CREDENTIAL_WAIT_SECONDS = 30 * 60
QUALITY_BATCH_WAIT_SECONDS = 26 * 60 * 60
INVENTORY_SEGMENTER_VERSION = "source_inventory_segments_v1"
INVENTORY_OUTPUT_CAP = 5_000
RECONCILE_OUTPUT_CAP = 7_000
FOCUSED_VERIFY_OUTPUT_CAP = 3_000
EVIDENCE_INVENTORY_OUTPUT_CAP = 10_000
EVIDENCE_RECONCILE_OUTPUT_CAP = 14_000
EVIDENCE_VERIFY_OUTPUT_CAP = 5_000
EVIDENCE_THIRD_SEGMENT_AFTER_CHARS = 60_000
EVIDENCE_RECONCILE_PARTITION_VERSION = "source_window_halves_v1"
SEGMENT_REVIEW_OUTPUT_CAP_V1 = 10_000
SEGMENT_REVIEW_OUTPUT_CAP_V2 = 6_500
# The v2 full-source first segment reached its 6,500-token cap mid-JSON.
# v3 removes duplicated rows; this bounded headroom protects complete output.
SEGMENT_REVIEW_OUTPUT_CAP = 7_500
SEGMENT_REVIEW_THIRD_SEGMENT_AFTER_CHARS = 60_000
INVENTORY_THIRD_SEGMENT_AFTER_CHARS = 120_000
INVENTORY_CONTEXT_OVERLAP = 4
INVENTORY_WINDOWS_PER_SEGMENT = 4
SAVED_SEGMENT_REUSE_VERSION = "saved_segment_form_reuse_v1"
SAVED_INVENTORY_REUSE_VERSION = "saved_inventory_coverage_reuse_v1"


def _json_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _pin_stage_files(artifacts: Path, manifest: dict, request_body: dict,
                     target: dict | None = None) -> None:
    """Seal replay inputs before the manifest that permits a paid POST.

    A crash between individual atomic writes leaves a reserved stage. On its
    next deterministic attempt, only byte-equivalent sidecars are reused.
    """
    files = [(artifacts / "request.json", request_body)]
    if target is not None:
        files.append((artifacts / "target.json", target))
    files.append((artifacts / "manifest.json", manifest))
    for path, value in files:
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if _sha(_json_bytes(saved)) != _sha(_json_bytes(value)):
                raise ValueError("saved_quality_sidecar_changed")
        else:
            write_private_json(path, value)


def _pin_private_json(path: Path, value: dict) -> None:
    """Create a replay input once; a repeated setup must have identical bytes."""
    if path.exists():
        if _sha(_json_bytes(json.loads(path.read_text(encoding="utf-8")))) != _sha(_json_bytes(value)):
            raise ValueError("saved_continuation_sidecar_changed")
    else:
        write_private_json(path, value)


def _normalize_null_optional_field_sources(document: dict, source_index: dict) -> tuple[dict, list[dict]]:
    """Drop only known, out-of-task evidence for an unset optional task field.

    This changes source metadata, never task content or a populated field. An
    unknown source ID is left in place so normal validation still rejects it.
    """
    tasks = document.get("tasks")
    if not isinstance(tasks, list):
        return document, []
    normalized = deepcopy(document)
    removals = []
    known = source_index.get("by_id", {})
    for number, task in enumerate(normalized["tasks"]):
        if not isinstance(task, dict):
            continue
        source_ids = task.get("source_ids")
        field_sources = task.get("field_sources")
        if (not isinstance(source_ids, list)
                or not all(isinstance(source_id, str) for source_id in source_ids)
                or not isinstance(field_sources, dict)):
            continue
        task_sources = set(source_ids)
        for field in ("assignee", "due", "priority", "recipient"):
            references = field_sources.get(field)
            if task.get(field) is not None or not isinstance(references, list):
                continue
            if not all(isinstance(source_id, str) for source_id in references):
                continue
            removed = [source_id for source_id in references
                       if source_id not in task_sources and source_id in known]
            if removed:
                field_sources[field] = [source_id for source_id in references
                                        if source_id in task_sources or source_id not in known]
                removals.append({"task_index": number, "field": field,
                                 "removed_source_ids": removed})
    return (normalized, removals) if removals else (document, [])


def _normalize_known_field_source_membership(document: dict, source_index: dict) -> tuple[dict, list[dict]]:
    """Include a field's existing, known citations in its task navigation span.

    This changes only the outer source list. It never creates evidence, changes
    a field value, or claims that a cited utterance entails the field. Unknown
    IDs are left for ordinary validation to reject.
    """
    tasks = document.get("tasks")
    known = source_index.get("by_id", {})
    if not isinstance(tasks, list) or not isinstance(known, dict):
        return document, []
    normalized = deepcopy(document)
    additions = []
    order = {source_id: number for number, source_id in enumerate(known)}
    for number, task in enumerate(normalized["tasks"]):
        if not isinstance(task, dict):
            continue
        source_ids = task.get("source_ids")
        field_sources = task.get("field_sources")
        if (not isinstance(source_ids, list)
                or not all(isinstance(source_id, str) for source_id in source_ids)
                or not isinstance(field_sources, dict)):
            continue
        cited = []
        for references in field_sources.values():
            if not isinstance(references, list) or not all(isinstance(value, str) for value in references):
                cited = []
                break
            cited.extend(references)
        if not cited or any(source_id not in known for source_id in [*source_ids, *cited]):
            continue
        missing = sorted(set(cited) - set(source_ids), key=order.__getitem__)
        if missing:
            task["source_ids"] = sorted(set(source_ids) | set(missing), key=order.__getitem__)
            additions.append({"task_index": number, "added_source_ids": missing})
    return (normalized, additions) if additions else (document, [])


def _quality_route(manifest: dict) -> str:
    """A submitted writer keeps the quality backend pinned in its manifest."""
    policy = manifest.get("quality_policy_version")
    if policy == LEGACY_QUALITY_POLICY_VERSION and not manifest.get("quality_provider"):
        return "luna"
    if (policy in OPUS_QUALITY_POLICIES
            and manifest.get("quality_provider") == OPUS_QUALITY_PROVIDER
            and manifest.get("audit_model") == OPUS_MODEL):
        return "opus"
    if (policy in {PREVIOUS_GEMINI_QUALITY_POLICY_VERSION,
                   INVENTORY_V2_QUALITY_POLICY_VERSION,
                   INVENTORY_RECONCILE_QUALITY_POLICY_VERSION,
                   EVIDENCE_RECONCILE_QUALITY_POLICY_VERSION,
                   *SEGMENT_REVIEW_POLICIES}
            and manifest.get("quality_provider") == QUALITY_PROVIDER):
        return "gemini"
    raise ValueError("unknown_quality_policy")


def _quality_contract(provider: str, policy: str | None = None) -> tuple[Path, dict, str]:
    if provider == "luna":
        return AUDIT_PROMPT_PATH, AUDIT_SCHEMA, AUDIT_SCHEMA_ID
    if provider == "opus" and policy == OPUS_QUALITY_POLICY_VERSION:
        return OPUS_AUDIT_PROMPT_PATH, OPUS_AUDIT_SCHEMA, OPUS_AUDIT_SCHEMA_ID
    if provider == "opus" and policy in OPUS_PARTITIONED_POLICIES:
        prompt, schema, schema_id, _cap, _effort, _profile = _opus_partition_profile(policy)
        return prompt, schema, schema_id
    if provider == "gemini":
        if policy == PREVIOUS_GEMINI_QUALITY_POLICY_VERSION:
            return GEMINI_AUDIT_PROMPT_PATH, AUDIT_SCHEMA, AUDIT_SCHEMA_ID
        if policy == INVENTORY_V2_QUALITY_POLICY_VERSION:
            return GEMINI_AUDIT_PROMPT_PATH_V2, GEMINI_AUDIT_SCHEMA_V2, GEMINI_AUDIT_SCHEMA_ID_V2
        if policy == SEGMENT_REVIEW_QUALITY_POLICY_VERSION_V1:
            return SEGMENT_REVIEW_PROMPT_PATH, SEGMENT_REVIEW_SCHEMA, SEGMENT_REVIEW_SCHEMA_ID
        if policy == SEGMENT_REVIEW_QUALITY_POLICY_VERSION_V2:
            return SEGMENT_REVIEW_PROMPT_PATH_V2, SEGMENT_REVIEW_SCHEMA_V2, SEGMENT_REVIEW_SCHEMA_ID_V2
        if policy == SEGMENT_REVIEW_QUALITY_POLICY_VERSION or policy is None:
            return SEGMENT_REVIEW_PROMPT_PATH_V3, SEGMENT_REVIEW_SCHEMA_V3, SEGMENT_REVIEW_SCHEMA_ID_V3
        if policy == INVENTORY_RECONCILE_QUALITY_POLICY_VERSION:
            return RECONCILE_PROMPT_PATH, RECONCILE_SCHEMA, RECONCILE_SCHEMA_ID
        if policy == EVIDENCE_RECONCILE_QUALITY_POLICY_VERSION:
            return RECONCILE_PROMPT_PATH_V2, RECONCILE_SCHEMA_V2, RECONCILE_SCHEMA_ID_V2
    raise ValueError("unknown_quality_provider")


def _quality_stage_contract(provider: str, policy: str | None, kind: str) -> tuple[Path, dict, str]:
    if (provider == "gemini" and policy == INVENTORY_RECONCILE_QUALITY_POLICY_VERSION
            and kind.startswith("inventory_")):
        return INVENTORY_PROMPT_PATH, INVENTORY_SCHEMA, INVENTORY_SCHEMA_ID
    if (provider == "gemini" and policy == EVIDENCE_RECONCILE_QUALITY_POLICY_VERSION
            and kind.startswith("inventory_")):
        return INVENTORY_PROMPT_PATH_V2, INVENTORY_SCHEMA_V2, INVENTORY_SCHEMA_ID_V2
    if (provider == "gemini" and policy in SEGMENT_REVIEW_POLICIES
            and kind == "verify"):
        return RECONCILE_PROMPT_PATH, RECONCILE_SCHEMA, RECONCILE_SCHEMA_ID
    return _quality_contract(provider, policy)


def _inventory_contract_for_policy(policy: str) -> tuple[Path, dict]:
    return ((INVENTORY_PROMPT_PATH_V2, INVENTORY_SCHEMA_V2)
            if policy == EVIDENCE_RECONCILE_QUALITY_POLICY_VERSION
            else (INVENTORY_PROMPT_PATH, INVENTORY_SCHEMA))


def _reconcile_contract_for_policy(policy: str) -> tuple[Path, dict]:
    return ((RECONCILE_PROMPT_PATH_V2, RECONCILE_SCHEMA_V2)
            if policy == EVIDENCE_RECONCILE_QUALITY_POLICY_VERSION
            else (RECONCILE_PROMPT_PATH, RECONCILE_SCHEMA))


def _inventory_profile(policy: str) -> tuple[int, int, int, int]:
    if policy == EVIDENCE_RECONCILE_QUALITY_POLICY_VERSION:
        return (EVIDENCE_THIRD_SEGMENT_AFTER_CHARS,
                EVIDENCE_INVENTORY_OUTPUT_CAP,
                EVIDENCE_RECONCILE_OUTPUT_CAP, EVIDENCE_VERIFY_OUTPUT_CAP)
    return (INVENTORY_THIRD_SEGMENT_AFTER_CHARS, INVENTORY_OUTPUT_CAP,
            RECONCILE_OUTPUT_CAP, FOCUSED_VERIFY_OUTPUT_CAP)


def _quality_prompt_path(provider: str, policy: str | None = None) -> Path:
    return _quality_contract(provider, policy)[0]


def _segment_review_schema_id(policy: str) -> str:
    return {
        SEGMENT_REVIEW_QUALITY_POLICY_VERSION_V1: SEGMENT_REVIEW_SCHEMA_ID,
        SEGMENT_REVIEW_QUALITY_POLICY_VERSION_V2: SEGMENT_REVIEW_SCHEMA_ID_V2,
        SEGMENT_REVIEW_QUALITY_POLICY_VERSION: SEGMENT_REVIEW_SCHEMA_ID_V3,
    }[policy]


def _segment_review_output_cap(policy: str) -> int:
    return {
        SEGMENT_REVIEW_QUALITY_POLICY_VERSION_V1: SEGMENT_REVIEW_OUTPUT_CAP_V1,
        SEGMENT_REVIEW_QUALITY_POLICY_VERSION_V2: SEGMENT_REVIEW_OUTPUT_CAP_V2,
        SEGMENT_REVIEW_QUALITY_POLICY_VERSION: SEGMENT_REVIEW_OUTPUT_CAP,
    }[policy]


def _segment_review_third_after(policy: str) -> int:
    return (INVENTORY_THIRD_SEGMENT_AFTER_CHARS
            if policy == SEGMENT_REVIEW_QUALITY_POLICY_VERSION_V1
            else SEGMENT_REVIEW_THIRD_SEGMENT_AFTER_CHARS)


def _request_body(source_text: str) -> dict:
    instruction = PROMPT_PATH.read_text(encoding="utf-8")
    return {
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": source_text},
        ],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": SCHEMA_ID, "strict": True, "schema": SCHEMA,
        }},
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "reasoning": {"effort": REASONING_EFFORT},
    }


def _semantic_identity(source_sha256: str, source_text: str, workspace_scope: str,
                       prompt_sha256: str, schema_sha256: str, force_nonce: str | None,
                       judge_scope: str = "judge-unavailable",
                       quality_policy_version: str | None = None) -> str:
    policy = quality_policy_version or QUALITY_POLICY_VERSION
    if policy == OPUS_QUALITY_POLICY_VERSION:
        return _sha(_json_bytes({
            "source_sha256": source_sha256,
            "projected_source_sha256": _sha(source_text.encode("utf-8")),
            "workspace_scope": workspace_scope,
            "judge_scope": judge_scope,
            "prompt_sha256": prompt_sha256,
            "schema_sha256": schema_sha256,
            "model": MODEL,
            "privacy_mode": OPUS_WRITER_PRIVACY_MODE,
            "workspace_io_logging_enabled": OPUS_WORKSPACE_IO_LOGGING_ENABLED,
            "reasoning_effort": REASONING_EFFORT,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "quality_policy_version": policy,
            "quality_provider": OPUS_QUALITY_PROVIDER,
            "audit_model": OPUS_MODEL,
            "audit_privacy_mode": OPUS_PRIVACY_MODE,
            "audit_workspace_io_logging_enabled": OPUS_WORKSPACE_IO_LOGGING_ENABLED,
            "audit_prompt_sha256": _sha(OPUS_AUDIT_PROMPT_PATH.read_bytes()),
            "audit_schema_sha256": _sha(_json_bytes(OPUS_AUDIT_SCHEMA)),
            "audit_output_cap": OPUS_AUDIT_OUTPUT_CAP,
            "verify_output_cap": OPUS_VERIFY_OUTPUT_CAP,
            "force_nonce": force_nonce,
        }))
    if policy in OPUS_PARTITIONED_POLICIES:
        audit_prompt, audit_schema, _id, output_cap, effort, _profile = _opus_partition_profile(policy)
        return _sha(_json_bytes({
            "source_sha256": source_sha256,
            "projected_source_sha256": _sha(source_text.encode("utf-8")),
            "workspace_scope": workspace_scope,
            "judge_scope": judge_scope,
            "prompt_sha256": prompt_sha256,
            "schema_sha256": schema_sha256,
            "model": MODEL,
            "privacy_mode": OPUS_WRITER_PRIVACY_MODE,
            "workspace_io_logging_enabled": OPUS_WORKSPACE_IO_LOGGING_ENABLED,
            "reasoning_effort": REASONING_EFFORT,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "quality_policy_version": policy,
            "quality_provider": OPUS_QUALITY_PROVIDER,
            "audit_model": OPUS_MODEL,
            "audit_privacy_mode": OPUS_PRIVACY_MODE,
            "audit_workspace_io_logging_enabled": OPUS_WORKSPACE_IO_LOGGING_ENABLED,
            "audit_prompt_sha256": _sha(audit_prompt.read_bytes()),
            "audit_schema_sha256": _sha(_json_bytes(audit_schema)),
            "segment_output_cap": output_cap,
            "partition_version": OPUS_PARTITION_VERSION,
            **({"segment_reasoning_effort": effort}
               if policy == OPUS_PARTITIONED_QUALITY_POLICY_VERSION else {}),
            "force_nonce": force_nonce,
        }))
    audit_prompt_path, audit_schema, _ = _quality_contract("gemini", policy)
    inventory_prompt_path, inventory_schema = _inventory_contract_for_policy(policy)
    verify_prompt_path, verify_schema = _reconcile_contract_for_policy(policy)
    inventory_third_after, inventory_cap, reconcile_cap, verify_cap = _inventory_profile(policy)
    material = {
        "source_sha256": source_sha256,
        "projected_source_sha256": _sha(source_text.encode("utf-8")),
        "workspace_scope": workspace_scope,
        "judge_scope": judge_scope,
        "prompt_sha256": prompt_sha256,
        "schema_sha256": schema_sha256,
        "model": MODEL,
        "privacy_mode": PRIVACY_MODE,
        "reasoning_effort": REASONING_EFFORT,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "quality_policy_version": policy,
        "quality_provider": QUALITY_PROVIDER,
        "audit_model": GEMINI_MODEL,
        "audit_privacy_mode": GEMINI_PRIVACY_MODE,
        "audit_prompt_sha256": _sha(audit_prompt_path.read_bytes()),
        "audit_schema_sha256": _sha(_json_bytes(audit_schema)),
        "inventory_prompt_sha256": _sha(inventory_prompt_path.read_bytes()),
        "inventory_schema_sha256": _sha(_json_bytes(inventory_schema)),
        "inventory_segmenter_version": INVENTORY_SEGMENTER_VERSION,
        "inventory_third_segment_after_chars": inventory_third_after,
        "inventory_context_overlap": INVENTORY_CONTEXT_OVERLAP,
        "inventory_windows_per_segment": INVENTORY_WINDOWS_PER_SEGMENT,
        "inventory_output_cap": inventory_cap,
        "reconcile_output_cap": reconcile_cap,
        "focused_verify_output_cap": verify_cap,
        "segment_review_prompt_sha256": _sha(audit_prompt_path.read_bytes()),
        "segment_review_schema_sha256": _sha(_json_bytes(audit_schema)),
        "segment_review_output_cap": SEGMENT_REVIEW_OUTPUT_CAP,
        "segment_review_third_segment_after_chars": SEGMENT_REVIEW_THIRD_SEGMENT_AFTER_CHARS,
        "verify_prompt_sha256": _sha(verify_prompt_path.read_bytes()),
        "verify_schema_sha256": _sha(_json_bytes(verify_schema)),
        "force_nonce": force_nonce,
    }
    return _sha(_json_bytes(material))


def _credential_store(private_root: Path) -> CredentialStore:
    return CredentialStore(Path(private_root) / "credentials.sqlite3")


def _publish_verified_document(*, document: dict, job: dict, request_manifest: dict,
                               source_index: dict, transcript_path: Path,
                               output_dir: Path, private_root: Path) -> tuple[str, Path]:
    """Seal one model document, then commit matching task edits before pointer swap."""
    validate_document(document, source_index)
    task_store = TaskStore(Path(private_root) / "tasks.sqlite3")
    plan = task_store.preview_reconcile(source_index["source_sha256"], document["tasks"])
    generation_id = datetime.fromtimestamp(job["created_at"], timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + job["id"][:12]
    quality_file = Path(job["artifact_dir"]) / "quality_decision.json"
    quality_review = json.loads(quality_file.read_text(encoding="utf-8"))["quality_review"] if quality_file.exists() else None
    remote_batch_id = job["remote_id"]
    writer_parent_id = job.get("writer_parent_id")
    if writer_parent_id is not None:
        if request_manifest.get("writer_parent_job_id") != writer_parent_id:
            raise ValueError("continuation_writer_identity_changed")
        writer_ledger = Ledger(private_root)
        try:
            writer = writer_ledger.get(writer_parent_id)
        finally:
            writer_ledger.close()
        if (writer is None or writer["kind"] != "summary"
                or writer["source_sha256"] != job["source_sha256"]
                or writer["credential_id"] != job["credential_id"]
                or writer["remote_id"] != request_manifest.get("writer_remote_batch_id")
                or job["remote_id"] is not None):
            raise ValueError("continuation_writer_provenance_invalid")
        remote_batch_id = writer["remote_id"]
    return publish_document(
        document=document, source_index=source_index, transcript_path=transcript_path,
        output_dir=output_dir, semantic_key=job["semantic_key"],
        job_id=job["id"], remote_batch_id=remote_batch_id,
        credential_id=job["credential_id"],
        prompt_sha256=request_manifest["prompt_sha256"],
        schema_sha256=request_manifest["schema_sha256"],
        effective_tasks=plan.effective_tasks, generation_id=generation_id,
        quality_review=quality_review,
        before_pointer=lambda: task_store.commit_reconcile(plan),
    )


def _reuse_accepted(*, job: dict, output_dir: Path, transcript_path: Path,
                    source_index: dict, private_root: Path, ledger: Ledger) -> dict:
    """Publish a previously accepted model document for another local consumer."""
    artifacts = Path(job["artifact_dir"])
    request_manifest = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
    if request_manifest["source_sha256"] != source_index["source_sha256"]:
        raise ValueError("accepted_cache_source_mismatch")
    document = json.loads(Path(job["accepted_document_path"]).read_text(encoding="utf-8"))
    generation_id, _ = _publish_verified_document(
        document=document, job=job, request_manifest=request_manifest,
        source_index=source_index, transcript_path=transcript_path,
        output_dir=output_dir, private_root=private_root,
    )
    ledger.mark_consumer(job["semantic_key"], output_dir, generation_id=generation_id)
    return {"status": "accepted_cache_hit", "job_id": job["id"],
            "semantic_key": job["semantic_key"], "generation_id": generation_id,
            "new_generations": 0}


def submit(*, transcript_path: Path, output_dir: Path, private_root: Path,
           force_nonce: str | None = None, client_factory=BatchClient,
           quality_policy_version: str | None = None,
           dynamic_budget_authorization_ref: str | None = None) -> dict:
    """Persist intent and upper-bound reserve before one physical POST."""
    policy = quality_policy_version or QUALITY_POLICY_VERSION
    if policy not in {QUALITY_POLICY_VERSION, OPUS_QUALITY_POLICY_VERSION,
                      OPUS_PARTITIONED_QUALITY_POLICY_VERSION_V2,
                      SEGMENT_REVIEW_QUALITY_POLICY_VERSION,
                      EVIDENCE_RECONCILE_QUALITY_POLICY_VERSION}:
        raise ValueError("unsupported_new_summary_quality_policy")
    if dynamic_budget_authorization_ref is not None and policy != EVIDENCE_RECONCILE_QUALITY_POLICY_VERSION:
        raise ValueError("dynamic_review_budget_requires_evidence_policy")
    transcript_path = Path(transcript_path)
    output_dir = Path(output_dir)
    source_text, source_index, source_sha = load_source(transcript_path)
    prompt_sha = _sha(PROMPT_PATH.read_bytes())
    schema_sha = _sha(_json_bytes(SCHEMA))
    if policy in OPUS_QUALITY_POLICIES:
        audit_prompt_path, audit_schema, _ = _quality_contract("opus", policy)
        quality_metadata = {
            "quality_provider": OPUS_QUALITY_PROVIDER,
            "audit_model": OPUS_MODEL,
            "audit_privacy_mode": OPUS_PRIVACY_MODE,
            "audit_workspace_io_logging_enabled": OPUS_WORKSPACE_IO_LOGGING_ENABLED,
            "audit_prompt_sha256": _sha(audit_prompt_path.read_bytes()),
            "audit_schema_sha256": _sha(_json_bytes(audit_schema)),
            **({"audit_output_cap": OPUS_AUDIT_OUTPUT_CAP,
                "verify_output_cap": OPUS_VERIFY_OUTPUT_CAP}
               if policy == OPUS_QUALITY_POLICY_VERSION else
               {"segment_output_cap": _opus_partition_profile(policy)[3],
                "partition_version": OPUS_PARTITION_VERSION,
                **({"segment_reasoning_effort": OPUS_SEGMENT_EFFORT_V3}
                   if policy == OPUS_PARTITIONED_QUALITY_POLICY_VERSION else {})}),
        }
    else:
        audit_prompt_path, audit_schema, _ = _quality_contract("gemini", policy)
        inventory_prompt_path, inventory_schema = _inventory_contract_for_policy(policy)
        verify_prompt_path, verify_schema = _reconcile_contract_for_policy(policy)
        inventory_third_after, inventory_cap, reconcile_cap, verify_cap = _inventory_profile(policy)
        quality_metadata = {
            "quality_provider": QUALITY_PROVIDER,
            "audit_model": GEMINI_MODEL,
            "audit_privacy_mode": GEMINI_PRIVACY_MODE,
            "audit_prompt_sha256": _sha(audit_prompt_path.read_bytes()),
            "audit_schema_sha256": _sha(_json_bytes(audit_schema)),
            "inventory_prompt_sha256": _sha(inventory_prompt_path.read_bytes()),
            "inventory_schema_sha256": _sha(_json_bytes(inventory_schema)),
            "inventory_segmenter_version": INVENTORY_SEGMENTER_VERSION,
            "inventory_third_segment_after_chars": inventory_third_after,
            "inventory_context_overlap": INVENTORY_CONTEXT_OVERLAP,
            "inventory_windows_per_segment": INVENTORY_WINDOWS_PER_SEGMENT,
            "inventory_output_cap": inventory_cap,
            "reconcile_output_cap": reconcile_cap,
            "focused_verify_output_cap": verify_cap,
            "segment_review_prompt_sha256": _sha(audit_prompt_path.read_bytes()),
            "segment_review_schema_sha256": _sha(_json_bytes(audit_schema)),
            "segment_review_output_cap": SEGMENT_REVIEW_OUTPUT_CAP,
            "segment_review_third_segment_after_chars": SEGMENT_REVIEW_THIRD_SEGMENT_AFTER_CHARS,
            "verify_prompt_sha256": _sha(verify_prompt_path.read_bytes()),
            "verify_schema_sha256": _sha(_json_bytes(verify_schema)),
        }
    request_body = _request_body(source_text)
    serialized_request = _json_bytes(request_body)
    ledger = Ledger(private_root)
    guard = None
    try:
        store = _credential_store(private_root)
        guard_context = credential_dispatch_guard(getattr(store, "path", Path(private_root) / "credentials.sqlite3"))
        guard_context.__enter__()
        guard = guard_context
        candidates = store.dispatch_candidates(role="writer")
        if not candidates:
            return {"status": "credential_required", "source_sha256": source_sha}
        selected = candidates[0]  # ordered primary; no automatic account hopping
        scope = selected.get("workspace_id") or ("unverified-key-version:" + selected["id"] + ":" + str(selected["version"]))
        judge_candidates = store.dispatch_candidates(role="judge")
        judge_scope = (judge_candidates[0].get("workspace_id") or "judge-unavailable") if judge_candidates else "judge-unavailable"
        semantic_key = _semantic_identity(source_sha, source_text, scope, prompt_sha,
                                          schema_sha, force_nonce, judge_scope, policy)
        prior = ledger.attach_consumer(semantic_key, source_sha, output_dir)
        if prior:
            if prior["status"] == "accepted":
                return _reuse_accepted(job=prior, output_dir=output_dir,
                                       transcript_path=transcript_path,
                                       source_index=source_index, private_root=private_root,
                                       ledger=ledger)
            return {"status": prior["status"], "job_id": prior["id"], "semantic_key": semantic_key}
        if (policy not in OPUS_QUALITY_POLICIES
                and QUALITY_POLICY_VERSION in OPUS_QUALITY_POLICIES
                and OPUS_WORKSPACE_IO_LOGGING_ENABLED):
            return {"status": "privacy_policy_blocked",
                    "reason": "gemini_workspace_logging_policy_mismatch"}
        token = store.reveal_for_dispatch(selected["id"], selected["version"])
        client = client_factory(token)
        route = verify_batch_route(client)  # free GET /key and catalog checks
        if route.workspace_id != selected.get("workspace_id"):
            raise RouteBlocked("credential_workspace_changed_since_check")
        reserve_micros = route.reserve_microusd(serialized_request)
        decision = ledger.reserve(
            semantic_key=semantic_key, source_sha256=source_sha, output_dir=output_dir,
            credential_id=selected["id"], credential_version=selected["version"],
            workspace_id=selected.get("workspace_id"), max_cost_microusd=reserve_micros,
            dynamic_group_authorization_ref=dynamic_budget_authorization_ref,
        )
        if decision.kind != "new":
            return {"status": decision.kind, "reason": decision.reason, "job_id": decision.job_id}
        job = ledger.get(decision.job_id)
        artifacts = Path(job["artifact_dir"])
        write_private_json(artifacts / "manifest.json", {
            "job_id": job["id"], "semantic_key": semantic_key,
            "source_sha256": source_sha, "output_dir": str(output_dir),
            "credential_id": selected["id"], "credential_version": selected["version"],
            "workspace_id": selected.get("workspace_id"), "model": MODEL,
            "provider": "openai",
            "privacy_mode": (OPUS_WRITER_PRIVACY_MODE
                             if policy in OPUS_QUALITY_POLICIES else PRIVACY_MODE),
            **({"workspace_io_logging_enabled": OPUS_WORKSPACE_IO_LOGGING_ENABLED}
               if policy in OPUS_QUALITY_POLICIES else {}),
            "prompt_sha256": prompt_sha, "schema_sha256": schema_sha,
            "quality_policy_version": policy,
            "dynamic_budget_authorization_ref": dynamic_budget_authorization_ref,
            "judge_workspace_id": judge_scope,
            **quality_metadata,
            "request_sha256": _sha(serialized_request),
            "reserve_microusd": reserve_micros,
            "pricing": {
                "prompt_usd_per_token": str(route.prompt_usd_per_token),
                "completion_usd_per_token": str(route.completion_usd_per_token),
                "cache_write_usd_per_token": str(route.cache_write_usd_per_token),
                "request_usd": str(route.request_usd),
            },
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "reasoning_effort": REASONING_EFFORT,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        write_private_json(artifacts / "request.json", request_body)
        # An admin may rotate after the free preflight. The durable reserved
        # row now blocks replacement; re-read the exact version before POST.
        try:
            token = store.reveal_for_dispatch(selected["id"], selected["version"])
        except CredentialError:
            ledger.cancel_before_submit(job["id"], "credential_changed_before_post")
            return {"status": "cancelled_before_submit", "job_id": job["id"]}
        if not ledger.mark_submitting(job["id"]):
            return {"status": "submission_state_conflict", "job_id": job["id"]}
        client = client_factory(token)
        try:
            reply = client.submit(job["custom_id"], request_body)
            write_private_json(artifacts / "submit_response.json", reply.body)
            remote_id = reply.body.get("id")
            if reply.status_code != 202 or not valid_batch_id(remote_id):
                ledger.submission_result(job["id"], remote_id=None, error_code="unexpected_submit_response")
                return {"status": "submission_unknown", "job_id": job["id"]}
            ledger.submission_result(job["id"], remote_id=remote_id)
            return {"status": "submitted", "job_id": job["id"], "remote_id": remote_id,
                    "reserved_usd": reserve_micros / 1_000_000}
        except BatchError as exc:
            definite = exc.code in {400, 401, 402, 403, 404, 422, 429}
            ledger.submission_result(job["id"], remote_id=None, error_code=exc.reason,
                                     definite_rejection=definite)
            write_private_json(artifacts / "submit_error.json", {
                "code": exc.code, "reason": exc.reason, "retry_after": exc.retry_after,
                "definite_rejection": definite,
            })
            return {"status": "rejected_before_submit" if definite else "submission_unknown",
                    "job_id": job["id"], "error_code": exc.reason}
    except (CredentialError, RouteBlocked, BatchError) as exc:
        return {"status": "preflight_blocked", "reason": str(exc)}
    finally:
        try:
            if guard is not None:
                guard.__exit__(None, None, None)
        finally:
            ledger.close()


def _reusable_saved_segment_1(ledger: Ledger, *, stage_job_id: str,
                              writer_job_id: str, source_text: str,
                              source_index: dict, source_sha: str,
                              output_dir: str, draft: dict,
                              segment: dict) -> tuple[dict, dict, list[dict]]:
    """Reparse one completed failed item without changing its physical record.

    Only exact prior inputs and a complete stop response are reusable. The
    normalizer may repair coordinates/form, but cannot add source facts or
    turn sparse assessments into verified findings.
    """
    stage = ledger.get(stage_job_id)
    if (stage is None or stage["kind"] != "segment_1"
            or stage["status"] != "failed_validation"
            or stage["dispatches"] != 1 or stage["remote_id"] is None
            or stage["billed_microusd"] is None
            or stage["billing_group_id"] != writer_job_id
            or stage["source_sha256"] != source_sha
            or stage["output_dir"] != output_dir
            or not str(stage.get("error_code") or "").startswith("ValueError:inventory coverage")):
        raise ValueError("saved_segment_1_is_not_reusable_terminal_form_failure")
    previous_root = ledger.get(stage["root_job_id"])
    if (previous_root is None or previous_root["status"] != "accepted"
            or previous_root["writer_parent_id"] != writer_job_id
            or previous_root["billing_group_id"] != writer_job_id
            or previous_root["source_sha256"] != source_sha
            or previous_root["output_dir"] != output_dir):
        raise ValueError("saved_segment_1_parent_identity_mismatch")
    previous_decision = json.loads((Path(previous_root["artifact_dir"]) /
                                    "quality_decision.json").read_text(encoding="utf-8"))
    previous_manifest = json.loads((Path(previous_root["artifact_dir"]) /
                                    "manifest.json").read_text(encoding="utf-8"))
    quality = previous_decision.get("quality_review", {})
    if (quality.get("status") != "segment_review_unavailable"
            or quality.get("reason") != "segment_1:failed_validation"
            or quality.get("segment_job_ids") != [stage_job_id]):
        raise ValueError("saved_segment_1_was_not_the_publication_blocker")
    artifacts = Path(stage["artifact_dir"])
    names = ("manifest.json", "request.json", "target.json", "submit_response.json",
             "batch_terminal.json", "native_response.json")
    paths = {name: artifacts / name for name in names}
    if any(path.is_symlink() or not path.is_file() for path in paths.values()):
        raise ValueError("saved_segment_1_artifacts_unavailable")
    stage_manifest = json.loads(paths["manifest.json"].read_text(encoding="utf-8"))
    request = json.loads(paths["request.json"].read_text(encoding="utf-8"))
    target = json.loads(paths["target.json"].read_text(encoding="utf-8"))
    submission = json.loads(paths["submit_response.json"].read_text(encoding="utf-8"))
    terminal = json.loads(paths["batch_terminal.json"].read_text(encoding="utf-8"))
    native = json.loads(paths["native_response.json"].read_text(encoding="utf-8"))
    expected_target = {"segment": segment, "draft": draft}
    expected_request = _quality_request_body(
        source_text, expected_target, kind="segment_1", prior_findings=[],
        provider="gemini", policy=SEGMENT_REVIEW_QUALITY_POLICY_VERSION, source_index=source_index)
    if (target != expected_target or request != expected_request
            or stage_manifest.get("job_id") != stage_job_id
            or stage_manifest.get("root_job_id") != previous_root["id"]
            or stage_manifest.get("stage") != "segment_1"
            or stage_manifest.get("custom_id") != stage["custom_id"]
            or stage_manifest.get("source_sha256") != source_sha
            or stage_manifest.get("quality_policy_version") != SEGMENT_REVIEW_QUALITY_POLICY_VERSION
            or stage_manifest.get("prompt_sha256") != _sha(SEGMENT_REVIEW_PROMPT_PATH_V3.read_bytes())
            or stage_manifest.get("schema_sha256") != _sha(_json_bytes(SEGMENT_REVIEW_SCHEMA_V3))
            or stage_manifest.get("request_sha256") != _sha(_json_bytes(request))
            or stage_manifest.get("target_document_sha256") != _sha(_json_bytes(target))
            or stage_manifest.get("model") != GEMINI_MODEL
            or stage_manifest.get("provider") != QUALITY_PROVIDER
            or stage_manifest.get("privacy_mode") != GEMINI_PRIVACY_MODE
            or stage_manifest.get("provider_only") != ["google-vertex"]
            or stage_manifest.get("workspace_id") != stage["workspace_id"]
            or stage_manifest.get("workspace_id") != previous_manifest.get("judge_workspace_id")
            or stage_manifest.get("credential_id") != stage["credential_id"]
            or stage_manifest.get("credential_version") != stage["credential_version"]
            or stage_manifest.get("max_output_tokens") != SEGMENT_REVIEW_OUTPUT_CAP
            or submission.get("id") != stage["remote_id"]
            or terminal.get("id") != stage["remote_id"]
            or submission.get("model") not in GEMINI_BATCH_MODEL_IDS
            or terminal.get("model") != submission.get("model")):
        raise ValueError("saved_segment_1_request_or_route_changed")
    body, usage = extract_one_gemini_completed(
        terminal, stage["custom_id"], expected_batch_id=stage["remote_id"],
        manifest=stage_manifest, saved_request=request)
    choices = body.get("choices") if isinstance(body, dict) else None
    if (not isinstance(body, dict)
            or body.get("model") not in GEMINI_BATCH_MODEL_IDS | {GEMINI_MODEL.removesuffix(":batch")}
            or not isinstance(choices, list) or len(choices) != 1
            or not isinstance(choices[0], dict)
            or choices[0].get("finish_reason") != "stop"
            or not isinstance(choices[0].get("message"), dict)
            or choices[0]["message"].get("refusal")
            or choices[0]["message"].get("content") != native.get("text")
            or native.get("usage") != usage):
        raise ValueError("saved_segment_1_native_stop_or_usage_differs")
    report = json.loads(native["text"])
    normalized, changes = normalize_v3_segment_review_report(
        report, segment, draft, source_index)
    validate_segment_review_report(normalized, segment, draft, source_index)
    warnings = segment_review_warnings(normalized, segment, draft, source_index)
    warnings.extend(segment_form_warnings(changes))
    provenance = {
        "version": SAVED_SEGMENT_REUSE_VERSION,
        "source_stage_job_id": stage_job_id,
        "source_stage_root_job_id": previous_root["id"],
        "source_remote_batch_id": stage["remote_id"],
        "source_stage_billed_microusd": stage["billed_microusd"],
        "source_stage_dispatches": stage["dispatches"],
        "source_artifact_sha256": {name: _sha(path.read_bytes())
                                   for name, path in paths.items()},
        "normalized_report_sha256": _sha(_json_bytes(normalized)),
        "normalization_changes": changes,
        "coverage_warnings": warnings,
    }
    return normalized, provenance, warnings


def _relocate_inventory_coverage_pointers(report: dict, segment: dict) -> tuple[dict, list[dict]]:
    """Correct only unambiguous adjacent-window bookkeeping in a v2 report."""
    try:
        validate_inventory_report_v2(report, segment)
    except ValueError as exc:
        if "item has no source in this window" not in str(exc):
            raise ValueError("saved_inventory_failure_is_not_coverage_only") from exc
    else:
        return report, []
    normalized = deepcopy(report)
    primary = [row["id"] for row in segment["primary_utterances"]]
    positions = {source_id: number for number, source_id in enumerate(primary)}
    windows = segment["coverage_windows"]
    if (not isinstance(normalized, dict) or not isinstance(normalized.get("coverage"), list)
            or len(normalized["coverage"]) != len(windows)
            or not isinstance(normalized.get("items"), list)):
        raise ValueError("saved_inventory_coverage_shape_changed")
    members = [set(primary[positions[window["start_id"]]:
                           positions[window["end_id"]] + 1]) for window in windows]
    moves = []
    for source_number, row in enumerate(normalized["coverage"]):
        if (not isinstance(row, dict) or not isinstance(row.get("item_indices"), list)):
            raise ValueError("saved_inventory_coverage_shape_changed")
        for item_index in list(row["item_indices"]):
            if (type(item_index) is not int or not 0 <= item_index < len(normalized["items"])
                    or not isinstance(normalized["items"][item_index], dict)
                    or not isinstance(normalized["items"][item_index].get("source_ids"), list)):
                raise ValueError("saved_inventory_pointer_shape_changed")
            cited = set(normalized["items"][item_index]["source_ids"])
            if members[source_number] & cited:
                continue
            destinations = [number for number, sources in enumerate(members)
                            if sources & cited]
            if (len(destinations) != 1
                    or abs(destinations[0] - source_number) != 1
                    or item_index in normalized["coverage"][destinations[0]]["item_indices"]):
                raise ValueError("saved_inventory_pointer_destination_ambiguous")
            destination = destinations[0]
            row["item_indices"].remove(item_index)
            normalized["coverage"][destination]["item_indices"].append(item_index)
            moves.append({
                "code": "coverage_pointer_relocated", "segment_id": segment["segment_id"],
                "item_index": item_index,
                "from_window_id": windows[source_number]["window_id"],
                "to_window_id": windows[destination]["window_id"],
                "cited_primary_source_ids": sorted(members[destination] & cited),
                "status": "parser_correction_unverified",
            })
    if not moves:
        raise ValueError("saved_inventory_coverage_has_no_relocatable_pointers")
    validate_inventory_report_v2(normalized, segment)
    return normalized, moves


def _reusable_saved_inventory(ledger: Ledger, *, root: dict, stage: dict,
                              segment: dict, source_text: str,
                              source_index: dict) -> tuple[dict, dict]:
    """Reparse an exact terminal v2 inventory item without touching its row."""
    kind = f"inventory_{int(segment['segment_id'][1:])}"
    failed = stage["status"] == "failed_validation"
    if (stage["root_job_id"] != root["id"] or stage["kind"] != kind
            or stage["status"] not in {"stage_complete", "failed_validation"}
            or stage["dispatches"] != 1 or not stage["remote_id"]
            or stage["billed_microusd"] is None
            or stage["source_sha256"] != root["source_sha256"]
            or stage["output_dir"] != root["output_dir"]
            or stage["billing_group_id"] != root["billing_group_id"]
            or (failed and not str(stage.get("error_code") or "").startswith(
                "ValueError:inventory coverage["))):
        raise ValueError("saved_inventory_stage_is_not_reusable")
    root_manifest = json.loads((Path(root["artifact_dir"]) / "manifest.json").read_text(encoding="utf-8"))
    artifacts = Path(stage["artifact_dir"])
    names = ("manifest.json", "request.json", "target.json", "submit_response.json",
             "batch_terminal.json", "native_response.json")
    paths = {name: artifacts / name for name in names}
    if any(path.is_symlink() or not path.is_file() for path in paths.values()):
        raise ValueError("saved_inventory_artifacts_unavailable")
    manifest, request, target, submission, terminal, native = (
        json.loads(paths[name].read_text(encoding="utf-8")) for name in names)
    expected_request = _quality_request_body(
        source_text, segment, kind=kind, prior_findings=[], provider="gemini",
        policy=EVIDENCE_RECONCILE_QUALITY_POLICY_VERSION, source_index=source_index)
    if (target != segment or request != expected_request
            or stage["semantic_key"] != _quality_stage_key(
                root, kind, expected_request, "gemini", EVIDENCE_RECONCILE_QUALITY_POLICY_VERSION)
            or manifest.get("job_id") != stage["id"]
            or manifest.get("root_job_id") != root["id"]
            or manifest.get("stage") != kind
            or manifest.get("semantic_key") != stage["semantic_key"]
            or manifest.get("custom_id") != stage["custom_id"]
            or manifest.get("source_sha256") != root["source_sha256"]
            or manifest.get("quality_policy_version") != EVIDENCE_RECONCILE_QUALITY_POLICY_VERSION
            or manifest.get("prompt_sha256") != _sha(INVENTORY_PROMPT_PATH_V2.read_bytes())
            or manifest.get("schema_sha256") != _sha(_json_bytes(INVENTORY_SCHEMA_V2))
            or manifest.get("request_sha256") != _sha(_json_bytes(request))
            or manifest.get("target_document_sha256") != _sha(_json_bytes(target))
            or manifest.get("model") != GEMINI_MODEL
            or manifest.get("provider") != QUALITY_PROVIDER
            or manifest.get("privacy_mode") != GEMINI_PRIVACY_MODE
            or manifest.get("provider_only") != ["google-vertex"]
            or manifest.get("workspace_id") != stage["workspace_id"]
            or manifest.get("workspace_id") != root_manifest.get("judge_workspace_id")
            or manifest.get("credential_id") != stage["credential_id"]
            or manifest.get("credential_version") != stage["credential_version"]
            or manifest.get("reserve_microusd") != stage["reserved_microusd"]
            or manifest.get("max_output_tokens") != EVIDENCE_INVENTORY_OUTPUT_CAP
            or submission.get("id") != stage["remote_id"]
            or terminal.get("id") != stage["remote_id"]
            or submission.get("model") not in GEMINI_BATCH_MODEL_IDS
            or terminal.get("model") != submission.get("model")):
        raise ValueError("saved_inventory_request_or_route_changed")
    body, usage = extract_one_gemini_completed(
        terminal, stage["custom_id"], expected_batch_id=stage["remote_id"],
        manifest=manifest, saved_request=request)
    choices = body.get("choices") if isinstance(body, dict) else None
    if (not isinstance(body, dict)
            or body.get("model") not in GEMINI_BATCH_MODEL_IDS | {GEMINI_MODEL.removesuffix(":batch")}
            or not isinstance(choices, list) or len(choices) != 1
            or not isinstance(choices[0], dict)
            or choices[0].get("finish_reason") != "stop"
            or not isinstance(choices[0].get("message"), dict)
            or choices[0]["message"].get("refusal")
            or not isinstance(choices[0]["message"].get("content"), str)
            or choices[0]["message"]["content"] != native.get("text")
            or native.get("usage") != usage
            or _cost_micros(usage) != stage["billed_microusd"]):
        raise ValueError("saved_inventory_native_stop_or_usage_differs")
    raw = json.loads(native["text"])
    if failed:
        report, corrections = _relocate_inventory_coverage_pointers(raw, segment)
        if not corrections:
            raise ValueError("saved_inventory_failed_stage_has_no_correction")
    else:
        validate_inventory_report_v2(raw, segment)
        report, corrections = raw, []
        accepted_path = Path(stage["accepted_document_path"] or "")
        if (accepted_path.is_symlink() or not accepted_path.is_file()
                or json.loads(accepted_path.read_text(encoding="utf-8")) != report):
            raise ValueError("saved_inventory_accepted_report_differs")
    provenance = {
        "version": SAVED_INVENTORY_REUSE_VERSION,
        "correction_source": "saved_terminal_native_response",
        "source_stage_job_id": stage["id"],
        "source_root_job_id": root["id"],
        "source_remote_batch_id": stage["remote_id"],
        "source_stage_status": stage["status"],
        "source_stage_billed_microusd": stage["billed_microusd"],
        "source_stage_dispatches": stage["dispatches"],
        "source_artifact_sha256": {name: _sha(path.read_bytes())
                                   for name, path in paths.items()},
        "raw_report_sha256": _sha(_json_bytes(raw)),
        "normalized_report_sha256": _sha(_json_bytes(report)),
        "corrections": corrections,
        "risk_warnings": inventory_risk_warnings_v2(report, segment),
    }
    return report, provenance


def resume_saved_evidence_inventory(*, prior_job_id: str, private_root: Path,
                                    gemini_client_factory=GeminiBatchClient) -> dict:
    """Seal a zero-writer v2 continuation from a degraded accepted inventory.

    Route metadata GETs provide current bounds. No Batch POST occurs here;
    the scheduler alone may submit the two new reconciliation stages.
    """
    ledger = Ledger(private_root)
    guard = None
    try:
        prior = ledger.get(prior_job_id)
        if (prior is None or prior["kind"] != "summary" or prior["status"] != "accepted"
                or prior["dispatches"] != 1 or not prior["remote_id"]
                or prior["billed_microusd"] is None
                or prior["accepted_document_path"] is None or prior["generation_id"] is None):
            raise ValueError("saved_evidence_root_is_not_accepted_terminal")
        prior_artifacts = Path(prior["artifact_dir"])
        required = ("manifest.json", "request.json", "draft_document.json",
                    "submit_response.json", "batch_terminal.json", "native_response.json",
                    "inventory_plan.json", "quality_decision.json")
        paths = {name: prior_artifacts / name for name in required}
        if any(path.is_symlink() or not path.is_file() for path in paths.values()):
            raise ValueError("saved_evidence_root_artifacts_unavailable")
        manifest, request, draft, submission, terminal, native, frozen, decision = (
            json.loads(paths[name].read_text(encoding="utf-8")) for name in required)
        if (manifest.get("job_id") != prior["id"]
                or manifest.get("semantic_key") != prior["semantic_key"]
                or manifest.get("source_sha256") != prior["source_sha256"]
                or manifest.get("output_dir") != prior["output_dir"]
                or manifest.get("credential_id") != prior["credential_id"]
                or manifest.get("credential_version") != prior["credential_version"]
                or manifest.get("workspace_id") != prior["workspace_id"]
                or manifest.get("model") != MODEL
                or manifest.get("provider") != "openai"
                or manifest.get("privacy_mode") != PRIVACY_MODE
                or manifest.get("quality_policy_version") != EVIDENCE_RECONCILE_QUALITY_POLICY_VERSION
                or manifest.get("quality_provider") != QUALITY_PROVIDER
                or manifest.get("audit_model") != GEMINI_MODEL
                or manifest.get("audit_privacy_mode") != GEMINI_PRIVACY_MODE
                or manifest.get("prompt_sha256") != _sha(PROMPT_PATH.read_bytes())
                or manifest.get("schema_sha256") != _sha(_json_bytes(SCHEMA))
                or manifest.get("audit_prompt_sha256") != _sha(RECONCILE_PROMPT_PATH_V2.read_bytes())
                or manifest.get("audit_schema_sha256") != _sha(_json_bytes(RECONCILE_SCHEMA_V2))
                or manifest.get("verify_prompt_sha256") != _sha(RECONCILE_PROMPT_PATH_V2.read_bytes())
                or manifest.get("verify_schema_sha256") != _sha(_json_bytes(RECONCILE_SCHEMA_V2))
                or manifest.get("inventory_prompt_sha256") != _sha(INVENTORY_PROMPT_PATH_V2.read_bytes())
                or manifest.get("inventory_schema_sha256") != _sha(_json_bytes(INVENTORY_SCHEMA_V2))
                or manifest.get("inventory_segmenter_version") != INVENTORY_SEGMENTER_VERSION
                or manifest.get("inventory_third_segment_after_chars") != EVIDENCE_THIRD_SEGMENT_AFTER_CHARS
                or manifest.get("inventory_context_overlap") != INVENTORY_CONTEXT_OVERLAP
                or manifest.get("inventory_windows_per_segment") != INVENTORY_WINDOWS_PER_SEGMENT
                or manifest.get("inventory_output_cap") != EVIDENCE_INVENTORY_OUTPUT_CAP
                or manifest.get("reconcile_output_cap") != EVIDENCE_RECONCILE_OUTPUT_CAP
                or manifest.get("focused_verify_output_cap") != EVIDENCE_VERIFY_OUTPUT_CAP
                or manifest.get("request_sha256") != _sha(_json_bytes(request))
                or manifest.get("reserve_microusd") != prior["reserved_microusd"]
                or submission.get("id") != prior["remote_id"]
                or terminal.get("id") != prior["remote_id"]):
            raise ValueError("saved_evidence_root_manifest_or_route_changed")
        judge_scope = manifest.get("judge_workspace_id")
        if not isinstance(judge_scope, str) or not judge_scope or judge_scope == "judge-unavailable":
            raise ValueError("saved_evidence_judge_scope_unpinned")
        transcript_path = Path(prior["output_dir"]) / "transcript.json"
        if transcript_path.is_symlink() or not transcript_path.is_file():
            raise ValueError("saved_evidence_source_unavailable")
        source_text, source_index, source_sha = load_source(transcript_path)
        if (source_sha != prior["source_sha256"] or request != _request_body(source_text)):
            raise ValueError("saved_evidence_source_or_writer_request_changed")
        writer_body, writer_usage = extract_one_completed(terminal, prior["custom_id"])
        choices = writer_body.get("choices") if isinstance(writer_body, dict) else None
        writer_model = submission.get("model")
        if (not isinstance(choices, list) or len(choices) != 1
                or not isinstance(terminal.get("results"), list)
                or len(terminal["results"]) != 1
                or terminal.get("model") != writer_model
                or not isinstance(writer_model, str)
                or (writer_model not in {MODEL, "openai/gpt-6-luna"}
                    and not writer_model.startswith("openai/gpt-6-luna-"))
                or writer_body.get("model") not in {writer_model, MODEL, "openai/gpt-6-luna"}
                or _cost_micros(writer_usage) != prior["billed_microusd"]
                or not isinstance(choices[0], dict)
                or choices[0].get("finish_reason") != "stop"
                or not isinstance(choices[0].get("message"), dict)
                or choices[0]["message"].get("refusal")
                or choices[0]["message"].get("content") != native.get("text")
                or native.get("usage") != writer_usage):
            raise ValueError("saved_evidence_writer_native_or_usage_changed")
        raw_draft = json.loads(native["text"])
        normalized_draft, _ = _normalize_null_optional_field_sources(raw_draft, source_index)
        normalized_draft, _ = _normalize_known_field_source_membership(
            normalized_draft, source_index)
        if draft not in (raw_draft, normalized_draft):
            raise ValueError("saved_evidence_writer_draft_differs_from_native")
        validate_document(draft, source_index)
        count = 3 if len(source_text) > EVIDENCE_THIRD_SEGMENT_AFTER_CHARS else 2
        segments = plan_inventory_segments(
            source_text, count=count, overlap=INVENTORY_CONTEXT_OVERLAP,
            windows_per_segment=INVENTORY_WINDOWS_PER_SEGMENT)
        validate_inventory_plan(segments, source_index)
        expected_plan = {"version": INVENTORY_SEGMENTER_VERSION,
                         "source_sha256": source_sha, "segments": segments}
        if frozen != expected_plan:
            raise ValueError("saved_evidence_inventory_plan_changed")
        stages = [ledger.stage(prior["id"], f"inventory_{number}")
                  for number in range(1, len(segments) + 1)]
        if any(stage is None for stage in stages):
            raise ValueError("saved_evidence_inventory_stage_missing")
        failed = [stage for stage in stages if stage["status"] == "failed_validation"]
        review = decision.get("quality_review", {})
        if (len(failed) != 1 or review.get("status") != "inventory_unavailable"
                or review.get("reason") != f"{failed[0]['kind']}:failed_validation"
                or review.get("inventory_job_ids") != [stage["id"] for stage in stages]
                or any(stage["status"] not in {"stage_complete", "failed_validation"}
                       for stage in stages)):
            raise ValueError("saved_evidence_inventory_was_not_publication_blocker")
        accepted_path = Path(prior["accepted_document_path"])
        generation = (Path(prior["output_dir"]) / "summary_generations" /
                      prior["generation_id"])
        pointer_path = Path(prior["output_dir"]) / "summary_current.json"
        model_path = generation / "model_document.json"
        if (accepted_path != prior_artifacts / "candidate_document.json"
                or any(path.is_symlink() or not path.is_file()
                       for path in (accepted_path, pointer_path, model_path))
                or json.loads(accepted_path.read_text(encoding="utf-8")) != draft
                or json.loads(model_path.read_text(encoding="utf-8")) != draft):
            raise ValueError("saved_evidence_accepted_document_changed")
        sealed_generation = _verify_staged_target(
            generation, generation_id=prior["generation_id"], document=draft,
            source_sha=source_sha, semantic_key=prior["semantic_key"],
            job_id=prior["id"], remote_batch_id=prior["remote_id"],
            credential_id=prior["credential_id"],
            prompt_sha256=manifest["prompt_sha256"],
            schema_sha256=manifest["schema_sha256"], quality_review=review)
        reused = [_reusable_saved_inventory(
            ledger, root=prior, stage=stage, segment=segment,
            source_text=source_text, source_index=source_index)
            for stage, segment in zip(stages, segments)]
        reports = [report for report, _ in reused]
        inventory = merge_inventory_reports_v2(reports, segments, source_index)
        targets = partition_reconcile_targets(source_text, source_index, draft, inventory)
        if len(targets) != 2:
            raise ValueError("saved_evidence_reconcile_partition_invalid")
        if (sorted(source_id for target in targets
                   for source_id in target["scope"]["primary_source_ids"])
                != sorted(source_index["by_id"])
                or sorted(row["window_id"] for target in targets
                          for row in target["inventory"]["coverage"])
                != sorted(row["window_id"] for row in inventory["coverage"])
                or {item["item_id"] for target in targets
                    for item in target["inventory"]["items"]}
                != {item["item_id"] for item in inventory["items"]}):
            raise ValueError("saved_evidence_reconcile_partition_incomplete")
        requests = [_quality_request_body(
            source_text, target, kind=f"reconcile_{number}", prior_findings=[],
            provider="gemini", policy=EVIDENCE_RECONCILE_QUALITY_POLICY_VERSION,
            source_index=source_index)
            for number, target in enumerate(targets, 1)]
        identity = {
            "mode": SAVED_INVENTORY_REUSE_VERSION,
            "source_root_job_id": prior["id"],
            "source_sha256": source_sha,
            "writer_manifest_sha256": _sha(paths["manifest.json"].read_bytes()),
            "writer_draft_file_sha256": _sha(paths["draft_document.json"].read_bytes()),
            "normalized_draft_sha256": _sha(_json_bytes(draft)),
            "inventory_plan_sha256": _sha(_json_bytes(expected_plan)),
            "reused_inventories": [{
                "source_stage_job_id": stage["id"],
                "source_native_response_sha256": provenance["source_artifact_sha256"]["native_response.json"],
                "normalized_report_sha256": provenance["normalized_report_sha256"],
                "provenance_sha256": _sha(_json_bytes(provenance)),
            } for stage, (_, provenance) in zip(stages, reused)],
            "merged_inventory_sha256": _sha(_json_bytes(inventory)),
            "initial_reconcile_request_sha256": [_sha(_json_bytes(body)) for body in requests],
            "quality_policy_version": EVIDENCE_RECONCILE_QUALITY_POLICY_VERSION,
            "judge_workspace_id": judge_scope,
        }
        semantic_key = _sha(_json_bytes(identity))
        existing = ledger.by_semantic_key(semantic_key)
        if existing is not None and existing["status"] != "preparing":
            return {"status": existing["status"], "job_id": existing["id"],
                    "semantic_key": semantic_key, "new_writer_generations": 0}
        # The old generation must still be selected when sealing a new root.
        # A completed continuation has legitimately switched this pointer;
        # the verified semantic-key hit above remains idempotent after that.
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        if (pointer.get("generation_id") != prior["generation_id"]
                or pointer.get("verified_artifact_sha256") !=
                sealed_generation["artifact_sha256"]["summary.md"]):
            raise ValueError("saved_evidence_current_pointer_changed")
        store = _credential_store(private_root)
        guard_context = credential_dispatch_guard(
            getattr(store, "path", Path(private_root) / "credentials.sqlite3"))
        guard_context.__enter__()
        guard = guard_context
        selected = next((candidate for candidate in store.dispatch_candidates(role="judge")
                         if candidate.get("workspace_id") == judge_scope
                         and candidate["id"] == stages[0]["credential_id"]
                         and candidate["version"] == stages[0]["credential_version"]), None)
        if selected is None:
            return {"status": "credential_required", "reason": "pinned_judge_credential_unavailable"}
        if any(stage["credential_id"] != selected["id"]
               or stage["credential_version"] != selected["version"]
               or stage["workspace_id"] != judge_scope for stage in stages):
            raise ValueError("saved_evidence_stage_credential_or_workspace_changed")
        token = store.reveal_for_dispatch(selected["id"], selected["version"], role="judge")
        routes = [verify_gemini_batch_route(
            gemini_client_factory(token), body,
            max_output_tokens=body["max_completion_tokens"])
            for body in requests]
        if any(route.workspace_id != judge_scope for route in routes):
            raise ValueError("saved_evidence_judge_workspace_changed")
        reserves = [route.reserve_microusd() for route in routes]
        if (routes[0].key_limit_remaining_usd is not None
                and sum(reserves) > int(routes[0].key_limit_remaining_usd * 1_000_000)):
            return {"status": "budget_blocked", "reason": "judge_key_budget_insufficient"}
        group = prior["billing_group_id"]
        rows = ledger.db.execute("""SELECT status,dispatches,reserved_microusd,billed_microusd
            FROM jobs WHERE billing_group_id=?""", (group,)).fetchall()
        spent = sum((row["billed_microusd"] if row["billed_microusd"] is not None
                     else row["reserved_microusd"])
                    for row in rows if row["status"] not in
                    {"rejected_before_submit", "cancelled_before_submit"})
        dispatches = sum(row["dispatches"] for row in rows)
        group_cap = ledger.logical_group_cap_microusd(group)
        weekly_spent = ledger.db.execute(
            "SELECT COALESCE(SUM(CASE WHEN billed_microusd IS NULL THEN reserved_microusd "
            "ELSE billed_microusd END),0) FROM jobs WHERE created_at>=? AND status NOT IN "
            "('rejected_before_submit','cancelled_before_submit')",
            (time.time() - 7 * 24 * 60 * 60,)).fetchone()[0]
        if ((group_cap is not None and spent + sum(reserves) > group_cap)
                or dispatches + 2 > MAX_DISPATCHES_PER_JOB
                or weekly_spent + sum(reserves) > WEEK_CAP_MICROUSD):
            return {"status": "budget_blocked", "reason": "saved_logical_job_cap",
                    "already_counted_microusd": spent,
                    "reconcile_reserve_microusd": reserves,
                    "already_counted_dispatches": dispatches}
        decision_row = ledger.reserve(
            semantic_key=semantic_key, source_sha256=source_sha,
            output_dir=Path(prior["output_dir"]), credential_id=prior["credential_id"],
            credential_version=prior["credential_version"], workspace_id=prior["workspace_id"],
            max_cost_microusd=0, continuation_of_job_id=prior["id"],
            continuation_mode="accepted_inventory_unavailable")
        if decision_row.kind not in {"new", "pending"}:
            return {"status": decision_row.kind, "reason": decision_row.reason,
                    "job_id": decision_row.job_id}
        job = ledger.get(decision_row.job_id)
        if job["status"] != "preparing":
            return {"status": job["status"], "job_id": job["id"],
                    "new_writer_generations": 0}
        artifacts = Path(job["artifact_dir"])
        continuation_manifest = {**manifest,
            "job_id": job["id"], "semantic_key": semantic_key,
            "writer_parent_job_id": prior["id"],
            "writer_remote_batch_id": prior["remote_id"],
            "writer_manifest_sha256": identity["writer_manifest_sha256"],
            "writer_draft_file_sha256": identity["writer_draft_file_sha256"],
            "normalized_draft_sha256": identity["normalized_draft_sha256"],
            "continuation_mode": "accepted_inventory_unavailable",
            "source_inventory_root_job_id": prior["id"],
            "reused_inventories": identity["reused_inventories"],
            "merged_inventory_sha256": identity["merged_inventory_sha256"],
            "reserve_microusd": 0,
            "created_at": datetime.fromtimestamp(job["created_at"], timezone.utc).isoformat(),
        }
        _pin_private_json(artifacts / "draft_document.json", draft)
        _pin_private_json(artifacts / "inventory_plan.json", expected_plan)
        for number, (report, provenance) in enumerate(reused, 1):
            _pin_private_json(artifacts / f"reused_inventory_{number}_report.json", report)
            _pin_private_json(artifacts / f"reused_inventory_{number}_provenance.json", provenance)
        _pin_private_json(artifacts / "merged_inventory.json", inventory)
        _pin_private_json(artifacts / "reconcile_partition_plan.json", {
            "version": EVIDENCE_RECONCILE_PARTITION_VERSION,
            "source_sha256": source_sha,
            "parts": [{
                "primary_source_ids": target["scope"]["primary_source_ids"],
                "primary_window_ids": target["scope"]["primary_window_ids"],
                "inventory_item_ids": [item["item_id"] for item in target["inventory"]["items"]],
            } for target in targets],
        })
        _pin_private_json(artifacts / "manifest.json", continuation_manifest)
        _pin_private_json(artifacts / "continuation_provenance.json", {
            **identity,
            "correction_source": "saved_terminal_native_response",
            "parser_correction_warning_count": sum(len(provenance["corrections"])
                                                   for _, provenance in reused),
            "inventory_risk_warning_count": len(inventory["risk_warnings"]),
            "already_counted_microusd": spent,
            "already_counted_dispatches": dispatches,
            "reconcile_reserve_microusd": reserves,
            "reconcile_2_preflight_is_provisional": True,
        })
        ledger.mark_continuation_ready(job["id"])
        return {"status": "quality_pending", "job_id": job["id"],
                "semantic_key": semantic_key, "writer_parent_job_id": prior["id"],
                "new_writer_generations": 0,
                "reused_inventory_stage_ids": [stage["id"] for stage in stages],
                "parser_correction_warning_count": sum(len(provenance["corrections"])
                                                       for _, provenance in reused),
                "inventory_risk_warning_count": len(inventory["risk_warnings"]),
                "already_counted_microusd": spent,
                "already_counted_dispatches": dispatches,
                "reconcile_reserve_microusd": reserves,
                "reconcile_2_preflight_is_provisional": True}
    except (CredentialError, GeminiRouteBlocked, GeminiBatchError) as exc:
        return {"status": "preflight_blocked", "reason": str(exc)[:120]}
    finally:
        if guard is not None:
            guard.__exit__(None, None, None)
        ledger.close()


def resume_saved_draft(*, prior_job_id: str, private_root: Path,
                       reused_segment_stage_id: str | None = None,
                       gemini_client_factory=GeminiBatchClient) -> dict:
    """Start a new, bounded quality review of one saved failed Luna draft.

    The original writer and failed review keep their terminal rows and raw
    files. This entry never calls Luna or submits a Gemini request; the normal
    scheduler dispatches the pinned review after its manifest is sealed.
    """
    ledger = Ledger(private_root)
    guard = None
    try:
        prior = ledger.get(prior_job_id)
        if (prior is None or prior["kind"] != "summary"
                or prior["status"] != "failed_validation"
                or not isinstance(prior["remote_id"], str)
                or prior["billed_microusd"] is None):
            raise ValueError("saved_writer_is_not_terminal_failed_with_usage")
        prior_dir = Path(prior["artifact_dir"])
        original_manifest_bytes = (prior_dir / "manifest.json").read_bytes()
        original_manifest = json.loads(original_manifest_bytes)
        if (original_manifest.get("source_sha256") != prior["source_sha256"]
                or original_manifest.get("output_dir") != prior["output_dir"]
                or original_manifest.get("quality_provider") != QUALITY_PROVIDER
                or original_manifest.get("quality_policy_version") not in SEGMENT_REVIEW_POLICIES
                or original_manifest.get("model") != MODEL
                or original_manifest.get("credential_id") != prior["credential_id"]):
            raise ValueError("saved_writer_manifest_identity_mismatch")
        original_request = json.loads((prior_dir / "request.json").read_text(encoding="utf-8"))
        if _sha(_json_bytes(original_request)) != original_manifest.get("request_sha256"):
            raise ValueError("saved_writer_request_changed")
        submission = json.loads((prior_dir / "submit_response.json").read_text(encoding="utf-8"))
        terminal = json.loads((prior_dir / "batch_terminal.json").read_text(encoding="utf-8"))
        if submission.get("id") != prior["remote_id"] or terminal.get("id") != prior["remote_id"]:
            raise ValueError("saved_writer_remote_identity_mismatch")
        source_path = Path(prior["output_dir"]) / "transcript.json"
        source_text, source_index, source_sha = load_source(source_path)
        if source_sha != prior["source_sha256"]:
            raise ValueError("saved_writer_source_changed")
        original_draft_bytes = (prior_dir / "draft_document.json").read_bytes()
        original_draft = json.loads(original_draft_bytes)
        native = json.loads((prior_dir / "native_response.json").read_text(encoding="utf-8"))
        if (not isinstance(native.get("text"), str)
                or json.loads(native["text"]) != original_draft):
            raise ValueError("saved_writer_draft_differs_from_native_response")
        draft, removals = _normalize_null_optional_field_sources(original_draft, source_index)
        draft, additions = _normalize_known_field_source_membership(draft, source_index)
        validate_document(draft, source_index)
        prompt_path, schema, _ = _quality_contract("gemini", SEGMENT_REVIEW_QUALITY_POLICY_VERSION)
        judge_scope = original_manifest.get("judge_workspace_id")
        if not isinstance(judge_scope, str) or not judge_scope or judge_scope == "judge-unavailable":
            return {"status": "preflight_blocked", "reason": "judge_scope_not_pinned"}
        segments = plan_inventory_segments(
            source_text,
            count=3 if len(source_text) > SEGMENT_REVIEW_THIRD_SEGMENT_AFTER_CHARS else 2,
            overlap=INVENTORY_CONTEXT_OVERLAP,
            windows_per_segment=INVENTORY_WINDOWS_PER_SEGMENT,
        )
        validate_inventory_plan(segments, source_index)
        reused = None
        if reused_segment_stage_id is not None:
            if not isinstance(reused_segment_stage_id, str) or not reused_segment_stage_id:
                raise ValueError("invalid_saved_segment_stage_id")
            reused = _reusable_saved_segment_1(
                ledger, stage_job_id=reused_segment_stage_id,
                writer_job_id=prior_job_id, source_text=source_text,
                source_index=source_index, source_sha=source_sha,
                output_dir=prior["output_dir"], draft=draft, segment=segments[0])
        identity = {
            "mode": "saved_writer_segment_review_continuation_v1",
            "writer_parent_job_id": prior_job_id,
            "writer_remote_batch_id": prior["remote_id"],
            "writer_manifest_sha256": _sha(original_manifest_bytes),
            "writer_draft_file_sha256": _sha(original_draft_bytes),
            "normalized_draft_sha256": _sha(_json_bytes(draft)),
            "source_sha256": source_sha,
            "quality_policy_version": SEGMENT_REVIEW_QUALITY_POLICY_VERSION,
            "judge_workspace_id": judge_scope,
            "review_prompt_sha256": _sha(prompt_path.read_bytes()),
            "review_schema_sha256": _sha(_json_bytes(schema)),
            "verify_prompt_sha256": _sha(RECONCILE_PROMPT_PATH.read_bytes()),
            "verify_schema_sha256": _sha(_json_bytes(RECONCILE_SCHEMA)),
            "segmenter_version": INVENTORY_SEGMENTER_VERSION,
            "segment_review_output_cap": SEGMENT_REVIEW_OUTPUT_CAP,
            "segment_review_third_segment_after_chars": SEGMENT_REVIEW_THIRD_SEGMENT_AFTER_CHARS,
        }
        if reused is not None:
            report, reuse_provenance, _ = reused
            identity["reused_segment_1"] = {
                "source_stage_job_id": reused_segment_stage_id,
                "source_native_response_sha256": reuse_provenance["source_artifact_sha256"]["native_response.json"],
                "normalized_report_sha256": reuse_provenance["normalized_report_sha256"],
                "form_normalizer_version": SAVED_SEGMENT_REUSE_VERSION,
            }
        semantic_key = _sha(_json_bytes(identity))
        existing = ledger.by_semantic_key(semantic_key)
        if existing is not None and existing["status"] != "preparing":
            return {"status": existing["status"], "job_id": existing["id"],
                    "semantic_key": semantic_key, "new_writer_generations": 0}
        store = _credential_store(private_root)
        guard_context = credential_dispatch_guard(
            getattr(store, "path", Path(private_root) / "credentials.sqlite3"))
        guard_context.__enter__()
        guard = guard_context
        selected = next((candidate for candidate in store.dispatch_candidates(role="judge")
                         if candidate.get("workspace_id") == judge_scope), None)
        if selected is None:
            return {"status": "credential_required", "reason": "pinned_judge_workspace_unavailable"}
        first_stage_number = 2 if reused is not None else 1
        first_request = _quality_request_body(
            source_text, {"segment": segments[first_stage_number - 1], "draft": draft},
            kind=f"segment_{first_stage_number}", prior_findings=[], provider="gemini",
            policy=SEGMENT_REVIEW_QUALITY_POLICY_VERSION, source_index=source_index,
        )
        token = store.reveal_for_dispatch(selected["id"], selected["version"], role="judge")
        route = verify_gemini_batch_route(
            gemini_client_factory(token), first_request,
            max_output_tokens=first_request["max_completion_tokens"],
        )
        if route.workspace_id != judge_scope:
            raise ValueError("saved_writer_judge_workspace_changed")
        first_reserve = route.reserve_microusd()
        group = prior["billing_group_id"]
        rows = ledger.db.execute("""SELECT status,dispatches,reserved_microusd,billed_microusd
            FROM jobs WHERE billing_group_id=?""", (group,)).fetchall()
        spent = sum((row["billed_microusd"] if row["billed_microusd"] is not None
                     else row["reserved_microusd"])
                    for row in rows if row["status"] not in
                    {"rejected_before_submit", "cancelled_before_submit"})
        dispatches = sum(row["dispatches"] for row in rows)
        group_cap = ledger.logical_group_cap_microusd(group)
        if ((group_cap is not None and spent + first_reserve > group_cap)
                or dispatches >= MAX_DISPATCHES_PER_JOB):
            return {"status": "budget_blocked", "reason": "saved_logical_job_cap",
                    "already_counted_microusd": spent,
                    "first_review_reserve_microusd": first_reserve,
                    "already_counted_dispatches": dispatches}
        decision = ledger.reserve(
            semantic_key=semantic_key, source_sha256=source_sha,
            output_dir=Path(prior["output_dir"]), credential_id=prior["credential_id"],
            credential_version=prior["credential_version"], workspace_id=prior["workspace_id"],
            max_cost_microusd=0, continuation_of_job_id=prior_job_id,
        )
        if decision.kind not in {"new", "pending"}:
            return {"status": decision.kind, "reason": decision.reason,
                    "job_id": decision.job_id}
        job = ledger.get(decision.job_id)
        if job["status"] != "preparing":
            return {"status": job["status"], "job_id": job["id"],
                    "new_writer_generations": 0}
        artifacts = Path(job["artifact_dir"])
        manifest = {**original_manifest,
            "job_id": job["id"], "semantic_key": semantic_key,
            "quality_policy_version": SEGMENT_REVIEW_QUALITY_POLICY_VERSION,
            "judge_workspace_id": judge_scope,
            "audit_prompt_sha256": identity["review_prompt_sha256"],
            "audit_schema_sha256": identity["review_schema_sha256"],
            "segment_review_prompt_sha256": identity["review_prompt_sha256"],
            "segment_review_schema_sha256": identity["review_schema_sha256"],
            "segment_review_output_cap": SEGMENT_REVIEW_OUTPUT_CAP,
            "segment_review_third_segment_after_chars": SEGMENT_REVIEW_THIRD_SEGMENT_AFTER_CHARS,
            "writer_parent_job_id": prior_job_id,
            "writer_remote_batch_id": prior["remote_id"],
            "writer_manifest_sha256": identity["writer_manifest_sha256"],
            "writer_draft_file_sha256": identity["writer_draft_file_sha256"],
            "normalized_draft_sha256": identity["normalized_draft_sha256"],
            "reserve_microusd": 0,
            "created_at": datetime.fromtimestamp(job["created_at"], timezone.utc).isoformat(),
        }
        if reused is not None:
            manifest["reused_segment_1"] = {
                "source_stage_job_id": reused_segment_stage_id,
                "source_remote_batch_id": reuse_provenance["source_remote_batch_id"],
                "source_artifact_sha256": reuse_provenance["source_artifact_sha256"],
                "normalized_report_sha256": reuse_provenance["normalized_report_sha256"],
                "form_normalizer_version": SAVED_SEGMENT_REUSE_VERSION,
            }
        _pin_private_json(artifacts / "draft_document.json", draft)
        if reused is not None:
            _pin_private_json(artifacts / "reused_segment_1_report.json", report)
            _pin_private_json(artifacts / "reused_segment_1_provenance.json", reuse_provenance)
        _pin_private_json(artifacts / "manifest.json", manifest)
        _pin_private_json(artifacts / "continuation_provenance.json", {
            **identity, "writer_error_code": prior["error_code"],
            "original_draft_sha256": _sha(_json_bytes(original_draft)),
            "normalization_removals": removals, "normalization_additions": additions,
            "already_counted_microusd": spent,
            "already_counted_dispatches": dispatches,
            "first_review_reserve_microusd": first_reserve,
            "first_review_kind": f"segment_{first_stage_number}",
        })
        ledger.mark_continuation_ready(job["id"])
        return {"status": "quality_pending", "job_id": job["id"],
                "semantic_key": semantic_key, "writer_parent_job_id": prior_job_id,
                "new_writer_generations": 0, "first_review_reserve_microusd": first_reserve,
                "first_review_kind": f"segment_{first_stage_number}",
                "reused_segment_1_source_job_id": reused_segment_stage_id,
                "already_counted_microusd": spent,
                "already_counted_dispatches": dispatches}
    except (CredentialError, GeminiRouteBlocked, GeminiBatchError) as exc:
        return {"status": "preflight_blocked", "reason": str(exc)[:120]}
    finally:
        if guard is not None:
            guard.__exit__(None, None, None)
        ledger.close()


def _cost_micros(usage: dict) -> int | None:
    # BYOK batches report only OpenRouter's fee; the provider bills inference
    # separately. Keeping the original reserve prevents that partial receipt
    # from releasing application budget as if it were total spend.
    if usage.get("is_byok") is True:
        return None
    value = usage.get("cost")
    if value is None:
        return None
    try:
        amount = Decimal(str(value))
    except InvalidOperation:
        return None
    if not amount.is_finite() or amount < 0:
        return None
    return int((amount * 1_000_000).to_integral_value(rounding=ROUND_UP))


def _append_poll_event(artifacts: Path, value: dict) -> None:
    """Retain each physical GET/DELETE attempt without copying a secret to logs."""
    artifacts.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = _json_bytes({"at": datetime.now(timezone.utc).isoformat(), **value}) + b"\n"
    fd = os.open(artifacts / "batch_poll_events.jsonl", os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "ab") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        raise


def _poll_backoff(retry_after: str | None) -> int:
    if retry_after:
        try:
            return max(30, min(3600, int(float(retry_after))))
        except ValueError:
            try:
                seconds = (parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds()
                return max(30, min(3600, math.ceil(seconds)))
            except (TypeError, ValueError, OverflowError):
                pass
    return 300


def _finish_raw(ledger: Ledger, job: dict) -> dict:
    if job.get("kind") in DIAGNOSTIC_KINDS:
        from summary.gemini_v1.diagnostic import finish_raw as finish_diagnostic_raw
        return finish_diagnostic_raw(ledger, job)
    artifacts = Path(job["artifact_dir"])
    batch = json.loads((artifacts / "batch_terminal.json").read_text(encoding="utf-8"))
    submission = json.loads((artifacts / "submit_response.json").read_text(encoding="utf-8"))
    request_manifest = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
    if request_manifest.get("provider") == OPUS_QUALITY_PROVIDER:
        allowed_models = OPUS_BATCH_MODEL_IDS
        if (submission.get("id") != job["remote_id"]
                or batch.get("id") != job["remote_id"]
                or submission.get("model") not in allowed_models
                or batch.get("model") != submission.get("model")):
            raise ValueError("opus_batch_submission_identity_mismatch")
        saved_request = json.loads((artifacts / "request.json").read_text(encoding="utf-8"))
        body, usage = extract_one_opus_completed(
            batch, job["custom_id"], expected_batch_id=job["remote_id"],
            manifest=request_manifest, saved_request=saved_request)
        if body.get("model") not in allowed_models:
            raise ValueError("opus_response_model_mismatch")
        choices = body.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise ValueError("opus_response_choices_invalid")
        choice = choices[0]
        if choice.get("finish_reason") != "stop":
            raise ValueError("opus_response_not_terminal_stop")
        message = choice.get("message") or {}
        if message.get("refusal") or not isinstance(message.get("content"), str):
            raise ValueError("opus_response_text_invalid")
        raw_text = message["content"]
    elif request_manifest.get("provider") == QUALITY_PROVIDER:
        if (submission.get("id") != job["remote_id"]
                or batch.get("id") != job["remote_id"]
                or submission.get("model") not in GEMINI_BATCH_MODEL_IDS
                or batch.get("model") != submission.get("model")):
            raise ValueError("gemini_batch_submission_identity_mismatch")
        saved_request = json.loads((artifacts / "request.json").read_text(encoding="utf-8"))
        body, usage = extract_one_gemini_completed(
            batch, job["custom_id"], expected_batch_id=job["remote_id"],
            manifest=request_manifest, saved_request=saved_request)
        if body.get("model") not in GEMINI_BATCH_MODEL_IDS | {GEMINI_MODEL.removesuffix(":batch")}:
            raise ValueError("gemini_response_model_mismatch")
        choices = body.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise ValueError("gemini_response_choices_invalid")
        choice = choices[0]
        if choice.get("finish_reason") != "stop":
            raise ValueError("gemini_response_not_terminal_stop")
        message = choice.get("message") or {}
        if message.get("refusal") or not isinstance(message.get("content"), str):
            raise ValueError("gemini_response_text_invalid")
        raw_text = message["content"]
    else:
        if submission.get("id") != job["remote_id"] or batch.get("model") != submission.get("model"):
            raise ValueError("batch_submission_identity_mismatch")
        body, usage = extract_one_completed(batch, job["custom_id"])
        resolved_model = submission.get("model")
        allowed_models = {MODEL, "openai/gpt-6-luna"}
        if isinstance(resolved_model, str) and resolved_model.startswith("openai/gpt-6-luna-"):
            allowed_models.add(resolved_model)
        if body.get("model") not in allowed_models:
            raise ValueError("response_model_mismatch")
        choices = body.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise ValueError("response_choices_invalid")
        choice = choices[0]
        if choice.get("finish_reason") != "stop":
            raise ValueError("response_not_terminal_stop")
        message = choice.get("message") or {}
        if message.get("refusal") or not isinstance(message.get("content"), str):
            raise ValueError("response_refusal_or_content_invalid")
        raw_text = message["content"]
    write_private_json(artifacts / "native_response.json", {"text": raw_text, "usage": usage})
    document = json.loads(raw_text)
    transcript_path = Path(job["output_dir"]) / "transcript.json"
    _, source_index, source_sha = load_source(transcript_path)
    if source_sha != job["source_sha256"]:
        raise ValueError("source_revision_changed")
    if job["kind"] == "summary":
        if (_sha(PROMPT_PATH.read_bytes()) != request_manifest["prompt_sha256"]
                or _sha(_json_bytes(SCHEMA)) != request_manifest["schema_sha256"]):
            raise ValueError("code_or_prompt_changed_while_pending")
        if "quality_policy_version" not in request_manifest:
            # Complete a pre-migration Batch with its original one-call
            # contract; do not change an already submitted request's policy.
            validate_document(document, source_index)
            write_private_json(artifacts / "candidate_document.json", document)
            generation_id, generation_path = _publish_verified_document(
                document=document, job=job, request_manifest=request_manifest,
                source_index=source_index, transcript_path=transcript_path,
                output_dir=Path(job["output_dir"]), private_root=ledger.root,
            )
            ledger.accepted(job["id"], artifacts / "candidate_document.json", generation_id)
            ledger.mark_consumer(job["semantic_key"], Path(job["output_dir"]), generation_id=generation_id)
            return {"status": "accepted_legacy", "job_id": job["id"],
                    "generation_id": generation_id, "generation_path": str(generation_path),
                    "consumer_results": _publish_accepted_consumers(ledger, ledger.get(job["id"]))}
        provider = _quality_route(request_manifest)
        quality_prompt, quality_schema, _ = _quality_contract(
            provider, request_manifest["quality_policy_version"])
        if (request_manifest.get("audit_prompt_sha256") != _sha(quality_prompt.read_bytes())
                or request_manifest.get("audit_schema_sha256") != _sha(_json_bytes(quality_schema))):
            raise ValueError("code_or_prompt_changed_while_pending")
        if not isinstance(document, dict):
            raise ValueError("draft_document_not_object")
        try:
            validate_document(document, source_index)
        except (ValueError, KeyError, TypeError) as exc:
            # A parseable draft can still be repaired by the separate audit.
            # Preserve the exact failure and never treat it as an accepted hit.
            original_reason = type(exc).__name__ + ":" + str(exc)[:200]
            normalized, removals = _normalize_null_optional_field_sources(document, source_index)
            normalized, additions = _normalize_known_field_source_membership(normalized, source_index)
            if removals or additions:
                write_private_json(artifacts / "draft_normalization.json", {
                    "rule": "known_task_field_source_coordinates_v2",
                    "original_document_sha256": _sha(_json_bytes(document)),
                    "normalized_document_sha256": _sha(_json_bytes(normalized)),
                    "original_validation_error": original_reason,
                    "removals": removals,
                    "additions": additions,
                })
                document = normalized
            try:
                validate_document(document, source_index)
            except (ValueError, KeyError, TypeError) as current_exc:
                write_private_json(artifacts / "draft_validation.json", {
                    "status": "invalid", "original_reason": original_reason,
                    "reason": type(current_exc).__name__ + ":" + str(current_exc)[:200]})
            else:
                write_private_json(artifacts / "draft_validation.json", {
                    "status": "normalized", "original_reason": original_reason,
                    "normalization": "draft_normalization.json"})
        write_private_json(artifacts / "draft_document.json", document)
        ledger.mark_quality_pending(job["id"])
        return {"status": "draft_ready", "job_id": job["id"],
                "reserved_usd": job["reserved_microusd"] / 1_000_000}
    new_quality_kind = job["kind"] in {"inventory_1", "inventory_2", "inventory_3",
                                       "reconcile", "reconcile_1", "reconcile_2",
                                       "segment_1", "segment_2", "segment_3"}
    if job["kind"] not in {"audit", "verify"} and not new_quality_kind:
        raise ValueError("unknown_quality_stage")
    provider = ("opus" if request_manifest.get("provider") == OPUS_QUALITY_PROVIDER else
                "gemini" if request_manifest.get("provider") == QUALITY_PROVIDER else "luna")
    stage_policy = request_manifest.get("quality_policy_version")
    stage_prompt, stage_schema, _ = _quality_stage_contract(provider, stage_policy, job["kind"])
    if (_sha(stage_prompt.read_bytes()) != request_manifest["prompt_sha256"]
            or _sha(_json_bytes(stage_schema)) != request_manifest["schema_sha256"]):
        raise ValueError("audit_code_or_prompt_changed_while_pending")
    root = ledger.get(job["root_job_id"])
    if root is None or root["source_sha256"] != source_sha:
        raise ValueError("audit_root_source_mismatch")
    if provider == "opus":
        target_path = artifacts / "target.json"
        if target_path.is_symlink() or not target_path.is_file():
            raise ValueError("opus_target_unavailable")
        target = json.loads(target_path.read_text(encoding="utf-8"))
        if _sha(_json_bytes(target)) != request_manifest["target_document_sha256"]:
            raise ValueError("opus_target_changed")
        if stage_policy in OPUS_PARTITIONED_POLICIES:
            if (job["kind"] not in {"segment_1", "segment_2", "segment_3"}
                    or not isinstance(target, dict)
                    or set(target) != {"segment", "draft"}
                    or not isinstance(target["segment"], dict)
                    or not isinstance(target["draft"], dict)
                    or target["segment"].get("segment_id") !=
                    "S" + job["kind"].split("_")[1].zfill(2)):
                raise ValueError("opus_segment_target_invalid")
            validate_opus_segment_report(
                document, target["segment"], target["draft"], source_index,
                expected_schema_id=_opus_partition_profile(stage_policy)[2])
            report_file = artifacts / "opus_segment_report.json"
            write_private_json(report_file, document)
            ledger.stage_completed(job["id"], report_file)
            return {"status": "stage_complete", "stage": job["kind"],
                    "job_id": job["id"], "root_job_id": root["id"],
                    "findings": len(document["findings"])}
        verify_input = (json.loads(saved_request["messages"][1]["content"])
                        if job["kind"] == "verify" else None)
        expected_windows = verify_input["SOURCE_WINDOWS"] if verify_input else None
        verify_source_ids = ({item["id"] for item in verify_input["TRANSCRIPT_SOURCE"]["utterances"]}
                             if verify_input else None)
        validate_opus_audit(document, target, source_index, mode=job["kind"],
                            expected_windows=expected_windows,
                            verify_source_ids=verify_source_ids)
        warnings = coverage_warnings_opus(document, source_index)
        if warnings:
            write_private_json(artifacts / "coverage_warnings.json", {"warnings": warnings})
        report_file = artifacts / ("opus_audit_report.json" if job["kind"] == "audit"
                                   else "opus_verify_report.json")
        write_private_json(report_file, document)
        ledger.stage_completed(job["id"], report_file)
        return {"status": "stage_complete", "stage": job["kind"], "job_id": job["id"],
                "root_job_id": root["id"], "findings": len(document["findings"])}
    if stage_policy in SEGMENT_REVIEW_POLICIES:
        target = json.loads((artifacts / "target.json").read_text(encoding="utf-8"))
        if _sha(_json_bytes(target)) != request_manifest["target_document_sha256"]:
            raise ValueError("segment_review_target_changed")
        if job["kind"].startswith("segment_"):
            expected_schema_id = _segment_review_schema_id(stage_policy)
            if not isinstance(document, dict) or document.get("schema_version") != expected_schema_id:
                raise ValueError("segment_review_schema_version_mismatch")
            form_changes = []
            if stage_policy == SEGMENT_REVIEW_QUALITY_POLICY_VERSION:
                raw_report_sha = _sha(_json_bytes(document))
                document, form_changes = normalize_v3_segment_review_report(
                    document, target["segment"], target["draft"], source_index)
                if form_changes:
                    write_private_json(artifacts / "segment_report_form_normalization.json", {
                        "version": SAVED_SEGMENT_REUSE_VERSION,
                        "raw_report_sha256": raw_report_sha,
                        "normalized_report_sha256": _sha(_json_bytes(document)),
                        "changes": form_changes,
                    })
            validate_segment_review_report(document, target["segment"],
                                           target["draft"], source_index)
            warnings = segment_review_warnings(document, target["segment"],
                                               target["draft"], source_index)
            warnings.extend(segment_form_warnings(form_changes))
            report_file = artifacts / "segment_review_report.json"
            count = len(document["items"])
        elif job["kind"] == "verify":
            validate_reconciliation_report(document, target["draft"],
                                           target["inventory"], source_index,
                                           mode="verify")
            warnings = reconciliation_warnings(
                document, target["draft"], target["inventory"], source_index,
                mode="verify", prior_findings=target["prior_findings"])
            report_file = artifacts / "segment_verify_report.json"
            count = len(document["findings"])
        else:
            raise ValueError("unknown_segment_quality_stage")
        if warnings:
            write_private_json(artifacts / "coverage_warnings.json", {"warnings": warnings})
        write_private_json(report_file, document)
        ledger.stage_completed(job["id"], report_file)
        return {"status": "stage_complete", "stage": job["kind"], "job_id": job["id"],
                "root_job_id": root["id"], "items_or_findings": count}
    if stage_policy in {INVENTORY_RECONCILE_QUALITY_POLICY_VERSION,
                        EVIDENCE_RECONCILE_QUALITY_POLICY_VERSION}:
        target_path = artifacts / "target.json"
        target = json.loads(target_path.read_text(encoding="utf-8"))
        if _sha(_json_bytes(target)) != request_manifest["target_document_sha256"]:
            raise ValueError("quality_target_changed")
        if job["kind"].startswith("inventory_"):
            if stage_policy == EVIDENCE_RECONCILE_QUALITY_POLICY_VERSION:
                validate_inventory_report_v2(document, target)
                warnings = inventory_risk_warnings_v2(document, target)
                if warnings:
                    write_private_json(artifacts / "coverage_warnings.json", {"warnings": warnings})
            else:
                validate_inventory_report(document, target)
            report_file = artifacts / "inventory_report.json"
            count = len(document["items"])
        elif (stage_policy == EVIDENCE_RECONCILE_QUALITY_POLICY_VERSION
              and job["kind"] in {"reconcile_1", "reconcile_2"}):
            validate_reconciliation_target_report_v2(document, target, source_index)
            warnings = reconciliation_target_warnings_v2(document, target, source_index)
            if warnings:
                write_private_json(artifacts / "coverage_warnings.json", {"warnings": warnings})
            report_file = artifacts / "reconciliation_report.json"
            count = len(document["findings"])
        elif job["kind"] in {"reconcile", "verify"}:
            validator = (validate_reconciliation_report_v2
                         if stage_policy == EVIDENCE_RECONCILE_QUALITY_POLICY_VERSION
                         else validate_reconciliation_report)
            warning_check = (reconciliation_warnings_v2
                             if stage_policy == EVIDENCE_RECONCILE_QUALITY_POLICY_VERSION
                             else reconciliation_warnings)
            validator(document, target["draft"], target["inventory"], source_index,
                      mode="verify" if job["kind"] == "verify" else "reconcile")
            warnings = warning_check(
                document, target["draft"], target["inventory"], source_index,
                mode="verify" if job["kind"] == "verify" else "reconcile",
                prior_findings=target.get("prior_findings", []))
            if warnings:
                write_private_json(artifacts / "coverage_warnings.json", {"warnings": warnings})
            report_file = artifacts / "reconciliation_report.json"
            count = len(document["findings"])
        else:
            raise ValueError("unknown_inventory_quality_stage")
        write_private_json(report_file, document)
        ledger.stage_completed(job["id"], report_file)
        return {"status": "stage_complete", "stage": job["kind"], "job_id": job["id"],
                "root_job_id": root["id"], "items_or_findings": count}
    target_file = Path(root["artifact_dir"]) / ("draft_document.json" if job["kind"] == "audit" else "revised_document.json")
    target_bytes = target_file.read_bytes()
    target = json.loads(target_bytes)
    if _sha(_json_bytes(target)) != request_manifest["target_document_sha256"]:
        raise ValueError("audit_target_changed")
    if provider == "gemini" and stage_policy == INVENTORY_V2_QUALITY_POLICY_VERSION:
        validate_gemini_audit_v2(document, target, source_index, mode=job["kind"])
        warnings = coverage_warnings_gemini_v2(document, source_index, target)
    else:
        validate_audit(document, target, source_index, mode=job["kind"])
        warnings = coverage_warnings(document, source_index)
    if warnings:
        write_private_json(artifacts / "coverage_warnings.json", {"warnings": warnings})
    report_file = artifacts / "audit_report.json"
    write_private_json(report_file, document)
    ledger.stage_completed(job["id"], report_file)
    return {"status": "stage_complete", "stage": job["kind"], "job_id": job["id"],
            "root_job_id": root["id"], "findings": len(document["findings"])}


def _quality_request_body(source_text: str, target: dict, *, kind: str,
                          prior_findings: list[dict], provider: str,
                          policy: str | None = None,
                          source_index: dict | None = None) -> dict:
    if kind not in {"audit", "verify", "inventory_1", "inventory_2", "inventory_3",
                    "reconcile", "reconcile_1", "reconcile_2",
                    "segment_1", "segment_2", "segment_3"}:
        raise ValueError("invalid quality request kind")
    if provider == "opus":
        if policy == OPUS_QUALITY_POLICY_VERSION and kind in {"audit", "verify"}:
            return build_opus_audit_request(source_text, target, mode=kind,
                                            prior_findings=prior_findings)
        if (policy in OPUS_PARTITIONED_POLICIES
                and kind in {"segment_1", "segment_2", "segment_3"}
                and isinstance(target, dict)
                and set(target) == {"segment", "draft"}):
            return build_opus_segment_audit_request(
                source_text, target["draft"], target["segment"],
                profile=_opus_partition_profile(policy)[5])
        raise ValueError("invalid_opus_quality_stage")
    if provider == "gemini":
        # OpenRouter Chat Completions carries a Gemini-specific instruction.
        # The transcript stays in a separate user message and receives no
        # tools, web plugin, audio, or renderer output.
        prompt_path, schema, schema_id = _quality_stage_contract(provider, policy, kind)
        if policy in SEGMENT_REVIEW_POLICIES:
            if kind.startswith("segment_"):
                if source_index is None:
                    raise ValueError("segment source index missing")
                content = build_segment_review_input(
                    source_text, target["segment"], target["draft"], source_index,
                    schema_version=_segment_review_schema_id(policy))
                output_cap = _segment_review_output_cap(policy)
            elif kind == "verify":
                content = build_reconcile_input(
                    source_text, target["draft"], target["inventory"],
                    mode="verify", prior_findings=prior_findings)
                output_cap = FOCUSED_VERIFY_OUTPUT_CAP
            else:
                raise ValueError("invalid segment quality stage")
        elif policy in {INVENTORY_RECONCILE_QUALITY_POLICY_VERSION,
                        EVIDENCE_RECONCILE_QUALITY_POLICY_VERSION}:
            if kind.startswith("inventory_"):
                content = (build_inventory_input_v2(target)
                           if policy == EVIDENCE_RECONCILE_QUALITY_POLICY_VERSION
                           else build_inventory_input(target))
                output_cap = _inventory_profile(policy)[1]
            elif (policy == EVIDENCE_RECONCILE_QUALITY_POLICY_VERSION
                  and kind in {"reconcile_1", "reconcile_2"}):
                content = build_reconcile_target_input_v2(source_text, target)
                output_cap = EVIDENCE_RECONCILE_OUTPUT_CAP
            elif kind in {"reconcile", "verify"}:
                builder = (build_reconcile_input_v2
                           if policy == EVIDENCE_RECONCILE_QUALITY_POLICY_VERSION
                           else build_reconcile_input)
                content = builder(
                    source_text, target["draft"], target["inventory"],
                    mode="verify" if kind == "verify" else "reconcile",
                    prior_findings=prior_findings,
                )
                output_cap = _inventory_profile(policy)[3 if kind == "verify" else 2]
            else:
                raise ValueError("invalid inventory quality stage")
        else:
            content = build_gemini_audit_input(
                source_text, target, mode=kind, prior_findings=prior_findings)
            output_cap = 10_000 if kind == "audit" else 8_000
        return {
            "messages": [
                {"role": "system", "content": prompt_path.read_text(encoding="utf-8")},
                {"role": "user", "content": content},
            ],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": schema_id, "strict": True, "schema": schema,
            }},
            "max_completion_tokens": output_cap,
            "reasoning": {"effort": "medium"},
            "plugins": [],
        }
    if provider != "luna":
        raise ValueError("unknown_quality_provider")
    return {
        "messages": [
            {"role": "system", "content": AUDIT_PROMPT_PATH.read_text(encoding="utf-8")},
            {"role": "user", "content": build_audit_input(
                source_text, target, mode=kind, prior_findings=prior_findings)},
        ],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": AUDIT_SCHEMA_ID, "strict": True, "schema": AUDIT_SCHEMA,
        }},
        "max_completion_tokens": 12_000 if kind == "audit" else 6_000,
        "reasoning": {"effort": REASONING_EFFORT},
    }


def _quality_stage_key(root: dict, kind: str, request_body: dict, provider: str,
                       policy: str | None = None) -> str:
    prompt_path, schema, _ = _quality_stage_contract(provider, policy, kind)
    material = {
        "root_semantic_key": root["semantic_key"], "kind": kind,
        "request_sha256": _sha(_json_bytes(request_body)),
        "prompt_sha256": _sha(prompt_path.read_bytes()),
        "schema_sha256": _sha(_json_bytes(schema)),
        "quality_policy_version": (
            (policy or (SEGMENT_REVIEW_QUALITY_POLICY_VERSION if provider == "gemini"
                        else OPUS_QUALITY_POLICY_VERSION)) if provider in {"gemini", "opus"}
            else LEGACY_QUALITY_POLICY_VERSION
        ),
        "provider": provider,
    }
    if provider == "opus":
        material["privacy_mode"] = OPUS_PRIVACY_MODE
        material["workspace_io_logging_enabled"] = OPUS_WORKSPACE_IO_LOGGING_ENABLED
    return _sha(_json_bytes(material))


def _start_quality_stage(ledger: Ledger, root: dict, kind: str, target: dict,
                         prior_findings: list[dict], client_factory,
                         gemini_client_factory=GeminiBatchClient) -> dict:
    """Reserve, record and submit one dependent call, never duplicating a POST."""
    root_manifest = json.loads((Path(root["artifact_dir"]) / "manifest.json").read_text(encoding="utf-8"))
    provider = _quality_route(root_manifest)
    if provider == "gemini":
        return _start_gemini_stage(ledger, root, kind, target, prior_findings,
                                   gemini_client_factory)
    transcript_path = Path(root["output_dir"]) / "transcript.json"
    source_text, _, source_sha = load_source(transcript_path)
    if source_sha != root["source_sha256"]:
        raise ValueError("source_revision_changed_before_audit")
    request_body = _quality_request_body(source_text, target, kind=kind,
                                         prior_findings=prior_findings, provider="luna")
    request_bytes = _json_bytes(request_body)
    target_sha = _sha(_json_bytes(target))
    semantic_key = _quality_stage_key(root, kind, request_body, "luna")
    existing = ledger.stage(root["id"], kind)
    if existing is not None and existing["semantic_key"] != semantic_key:
        raise ValueError("quality_stage_identity_changed")
    guard = None
    stage = existing

    def cancel_reserved(code: str) -> None:
        current = ledger.get(stage["id"]) if stage is not None else None
        if current is not None and current["status"] == "reserved":
            ledger.cancel_before_submit(current["id"], code)

    try:
        store = _credential_store(ledger.root)
        guard_context = credential_dispatch_guard(getattr(store, "path", ledger.root / "credentials.sqlite3"))
        guard_context.__enter__()
        guard = guard_context
        if existing is None:
            candidates = store.dispatch_candidates()
            if not candidates:
                return {"status": "unavailable", "reason": "credential_required"}
            selected = candidates[0]
        else:
            selected = {"id": existing["credential_id"], "version": existing["credential_version"],
                        "workspace_id": existing["workspace_id"]}
        if selected.get("workspace_id") != root["workspace_id"]:
            cancel_reserved("audit_workspace_changed")
            return {"status": "unavailable", "reason": "audit_workspace_changed"}
        token = store.reveal_for_dispatch(selected["id"], selected["version"])
        route = verify_batch_route(client_factory(token))
        if route.workspace_id != root["workspace_id"]:
            cancel_reserved("audit_route_workspace_changed")
            return {"status": "unavailable", "reason": "audit_route_workspace_changed"}
        reserve = route.reserve_microusd(request_bytes,
                                         max_completion_tokens=request_body["max_completion_tokens"])
        decision = ledger.reserve(
            semantic_key=semantic_key, source_sha256=source_sha,
            output_dir=Path(root["output_dir"]), credential_id=selected["id"],
            credential_version=selected["version"], workspace_id=route.workspace_id,
            max_cost_microusd=reserve, kind=kind, root_job_id=root["id"],
        )
        if decision.kind == "blocked":
            return {"status": "unavailable", "reason": decision.reason}
        stage = ledger.get(decision.job_id)
        artifacts = Path(stage["artifact_dir"])
        manifest = {
            "job_id": stage["id"], "root_job_id": root["id"], "stage": kind,
            "semantic_key": semantic_key, "source_sha256": source_sha,
            "target_document_sha256": target_sha,
            "request_sha256": _sha(request_bytes),
            "prompt_sha256": _sha(AUDIT_PROMPT_PATH.read_bytes()),
            "schema_sha256": _sha(_json_bytes(AUDIT_SCHEMA)),
            "credential_id": stage["credential_id"],
            "credential_version": stage["credential_version"],
            "workspace_id": stage["workspace_id"], "model": MODEL,
            "privacy_mode": PRIVACY_MODE,
            "reserve_microusd": stage["reserved_microusd"],
            "max_completion_tokens": request_body["max_completion_tokens"],
        }
        _pin_stage_files(artifacts, manifest, request_body)
        if stage["status"] != "reserved":
            return {"status": stage["status"], "job_id": stage["id"]}
        try:
            token = store.reveal_for_dispatch(stage["credential_id"], stage["credential_version"])
        except CredentialError:
            ledger.cancel_before_submit(stage["id"], "credential_changed_before_post")
            return {"status": "unavailable", "reason": "credential_changed_before_post", "job_id": stage["id"]}
        if not ledger.mark_submitting(stage["id"]):
            return {"status": "submitting", "job_id": stage["id"]}
        client = client_factory(token)
        try:
            reply = client.submit(stage["custom_id"], request_body)
            write_private_json(artifacts / "submit_response.json", reply.body)
            remote_id = reply.body.get("id")
            if reply.status_code != 202 or not valid_batch_id(remote_id):
                ledger.submission_result(stage["id"], remote_id=None,
                                         error_code="unexpected_submit_response")
                return {"status": "submission_unknown", "job_id": stage["id"]}
            ledger.submission_result(stage["id"], remote_id=remote_id)
            return {"status": "submitted", "job_id": stage["id"], "remote_id": remote_id}
        except BatchError as exc:
            definite = exc.code in {400, 401, 402, 403, 404, 422, 429}
            ledger.submission_result(stage["id"], remote_id=None, error_code=exc.reason,
                                     definite_rejection=definite)
            write_private_json(artifacts / "submit_error.json", {
                "code": exc.code, "reason": exc.reason, "retry_after": exc.retry_after,
                "definite_rejection": definite,
            })
            return {"status": "rejected_before_submit" if definite else "submission_unknown",
                    "job_id": stage["id"], "reason": exc.reason}
    except (CredentialError, RouteBlocked, BatchError) as exc:
        cancel_reserved("stage_preflight_unavailable")
        return {"status": "unavailable", "reason": str(exc)[:120]}
    finally:
        if guard is not None:
            guard.__exit__(None, None, None)


def _start_gemini_stage(ledger: Ledger, root: dict, kind: str, target: dict,
                        prior_findings: list[dict], client_factory) -> dict:
    """Use a separate OpenRouter key for Gemini audit and bounded repair.

    A Batch POST is never repeated after an unknown outcome. The writer's
    credential and request remain separate from this stage.
    """
    from summary.gemini_v1.batch import BatchError as GeminiBatchError, batch_id_from_submit

    root_manifest = json.loads((Path(root["artifact_dir"]) / "manifest.json").read_text(encoding="utf-8"))
    existing = ledger.stage(root["id"], kind)
    if (QUALITY_POLICY_VERSION in OPUS_QUALITY_POLICIES
            and OPUS_WORKSPACE_IO_LOGGING_ENABLED
            and (existing is None or existing["status"] == "reserved")):
        # Saved Gemini roots require logging OFF. The active Opus policy has it
        # ON, so a restart must not create a new Gemini request. Already sent
        # stages still follow their saved remote IDs through normal polling.
        if existing is not None:
            ledger.cancel_before_submit(existing["id"], "gemini_workspace_logging_policy_mismatch")
        return {"status": "unavailable", "reason": "gemini_workspace_logging_policy_mismatch"}
    policy = root_manifest["quality_policy_version"]
    prompt_path, schema, _ = _quality_stage_contract("gemini", policy, kind)
    pinned_scope = root_manifest.get("judge_workspace_id")
    if not isinstance(pinned_scope, str) or pinned_scope == "judge-unavailable":
        return {"status": "unavailable", "reason": "judge_credential_required_at_writer_dispatch"}
    transcript_path = Path(root["output_dir"]) / "transcript.json"
    source_text, source_index, source_sha = load_source(transcript_path)
    if source_sha != root["source_sha256"]:
        raise ValueError("source_revision_changed_before_audit")
    request_body = _quality_request_body(source_text, target, kind=kind,
                                         prior_findings=prior_findings, provider="gemini",
                                         policy=policy, source_index=source_index)
    request_bytes = _json_bytes(request_body)
    target_sha = _sha(_json_bytes(target))
    semantic_key = _quality_stage_key(root, kind, request_body, "gemini", policy)
    if existing is not None and existing["semantic_key"] != semantic_key:
        raise ValueError("quality_stage_identity_changed")
    guard = None
    stage = existing

    def cancel_reserved(code: str) -> None:
        current = ledger.get(stage["id"]) if stage is not None else None
        if current is not None and current["status"] == "reserved":
            ledger.cancel_before_submit(current["id"], code)

    try:
        store = _credential_store(ledger.root)
        guard_context = credential_dispatch_guard(getattr(store, "path", ledger.root / "credentials.sqlite3"))
        guard_context.__enter__()
        guard = guard_context
        if existing is None:
            candidates = store.dispatch_candidates(role="judge")
            selected = next((candidate for candidate in candidates
                             if candidate.get("workspace_id") == pinned_scope), None)
            if selected is None:
                return {"status": "unavailable", "reason": "judge_credential_required"}
        else:
            public = store.list()
            selected = next((key for key in public["keys"] if key["id"] == existing["credential_id"]
                             and key["version"] == existing["credential_version"]), None)
            if selected is None:
                cancel_reserved("judge_credential_changed")
                return {"status": "unavailable", "reason": "judge_credential_changed"}
        if selected.get("workspace_id") != pinned_scope:
            cancel_reserved("judge_policy_or_scope_changed")
            return {"status": "unavailable", "reason": "judge_policy_or_scope_changed"}
        token = store.reveal_for_dispatch(selected["id"], selected["version"], role="judge")
        client = client_factory(token)
        output_cap = request_body["max_completion_tokens"]
        route = verify_gemini_batch_route(client, request_body, max_output_tokens=output_cap)
        if route.workspace_id != pinned_scope:
            cancel_reserved("judge_route_workspace_changed")
            return {"status": "unavailable", "reason": "judge_route_workspace_changed"}
        reserve = route.reserve_microusd()
        decision = ledger.reserve(
            semantic_key=semantic_key, source_sha256=source_sha,
            output_dir=Path(root["output_dir"]), credential_id=selected["id"],
            credential_version=selected["version"], workspace_id=pinned_scope,
            max_cost_microusd=reserve, kind=kind, root_job_id=root["id"],
        )
        if decision.kind == "blocked":
            return {"status": "unavailable", "reason": decision.reason}
        stage = ledger.get(decision.job_id)
        artifacts = Path(stage["artifact_dir"])
        manifest = {
            "job_id": stage["id"], "root_job_id": root["id"], "stage": kind,
            "semantic_key": semantic_key, "source_sha256": source_sha,
            "target_document_sha256": target_sha,
            "request_sha256": _sha(request_bytes),
            "custom_id": stage["custom_id"], "inline_request_count": 1,
            "prompt_sha256": _sha(prompt_path.read_bytes()),
            "schema_sha256": _sha(_json_bytes(schema)),
            "credential_id": stage["credential_id"],
            "credential_version": stage["credential_version"],
            "workspace_id": pinned_scope, "model": GEMINI_MODEL,
            "provider": QUALITY_PROVIDER, "quality_policy_version": policy,
            "privacy_mode": GEMINI_PRIVACY_MODE,
            "provider_only": ["google-vertex"],
            "cache_policy": "no_explicit_cache_full_miss_reserved",
            "reserve_microusd": stage["reserved_microusd"],
            "counted_input_tokens": route.input_tokens,
            "context_bound_tokens": route.context_bound_tokens,
            "max_output_tokens": output_cap,
        }
        _pin_stage_files(
            artifacts, manifest, request_body,
            target if policy in {INVENTORY_RECONCILE_QUALITY_POLICY_VERSION,
                                 EVIDENCE_RECONCILE_QUALITY_POLICY_VERSION,
                                 *SEGMENT_REVIEW_POLICIES} else None,
        )
        if stage["status"] != "reserved":
            return {"status": stage["status"], "job_id": stage["id"]}
        try:
            token = store.reveal_for_dispatch(stage["credential_id"], stage["credential_version"], role="judge")
        except CredentialError:
            ledger.cancel_before_submit(stage["id"], "judge_credential_changed_before_post")
            return {"status": "unavailable", "reason": "judge_credential_changed_before_post", "job_id": stage["id"]}
        if not ledger.mark_submitting(stage["id"]):
            return {"status": "submitting", "job_id": stage["id"]}
        client = client_factory(token)
        try:
            reply = client.submit(stage["custom_id"], request_body)
            write_private_json(artifacts / "submit_response.json", reply.body)
            try:
                remote_id = batch_id_from_submit(reply.body)
            except ValueError:
                remote_id = None
            if reply.status_code != 202 or remote_id is None:
                ledger.submission_result(stage["id"], remote_id=None,
                                         error_code="unexpected_submit_response")
                return {"status": "submission_unknown", "job_id": stage["id"]}
            ledger.submission_result(stage["id"], remote_id=remote_id)
            return {"status": "submitted", "job_id": stage["id"], "remote_id": remote_id}
        except GeminiBatchError as exc:
            definite = exc.code in {400, 401, 402, 403, 404, 422, 429}
            ledger.submission_result(stage["id"], remote_id=None, error_code=exc.reason,
                                     definite_rejection=definite)
            write_private_json(artifacts / "submit_error.json", {
                "code": exc.code, "reason": exc.reason, "retry_after": exc.retry_after,
                "definite_rejection": definite,
            })
            return {"status": "rejected_before_submit" if definite else "submission_unknown",
                    "job_id": stage["id"], "reason": exc.reason}
    except (CredentialError, GeminiRouteBlocked, GeminiBatchError) as exc:
        cancel_reserved("stage_preflight_unavailable")
        return {"status": "unavailable", "reason": str(exc)[:120]}
    finally:
        if guard is not None:
            guard.__exit__(None, None, None)


def _start_opus_stage(ledger: Ledger, root: dict, kind: str, target: dict,
                      prior_findings: list[dict], client_factory) -> dict:
    """Submit one pinned Opus Batch review using the judge key.

    Existing Gemini jobs retain their own stage code and manifests. Unknown
    POST outcomes retain their reservation and are never retried here.
    """
    from summary.opus_v1.batch import batch_id_from_submit

    root_manifest = json.loads((Path(root["artifact_dir"]) / "manifest.json").read_text(encoding="utf-8"))
    policy = root_manifest.get("quality_policy_version")
    permitted_kinds = ({"audit", "verify"} if policy == OPUS_QUALITY_POLICY_VERSION
                       else {"segment_1", "segment_2", "segment_3"}
                       if policy in OPUS_PARTITIONED_POLICIES else set())
    if kind not in permitted_kinds:
        raise ValueError("invalid_opus_quality_stage")
    if _quality_route(root_manifest) != "opus":
        raise ValueError("opus_root_policy_mismatch")
    if policy in OPUS_PARTITIONED_POLICIES:
        prompt, schema, _id, output_cap, effort, _profile = _opus_partition_profile(policy)
        if (root_manifest.get("partition_version") != OPUS_PARTITION_VERSION
                or root_manifest.get("segment_output_cap") != output_cap
                or root_manifest.get("audit_prompt_sha256") != _sha(prompt.read_bytes())
                or root_manifest.get("audit_schema_sha256") != _sha(_json_bytes(schema))
                or (policy == OPUS_PARTITIONED_QUALITY_POLICY_VERSION
                    and root_manifest.get("segment_reasoning_effort") != effort)):
            raise ValueError("opus_partition_contract_changed")
    if (root_manifest.get("privacy_mode") != OPUS_WRITER_PRIVACY_MODE
            or root_manifest.get("workspace_io_logging_enabled") is not True
            or root_manifest.get("audit_privacy_mode") != OPUS_PRIVACY_MODE
            or root_manifest.get("audit_workspace_io_logging_enabled") is not True):
        return {"status": "unavailable", "reason": "opus_workspace_logging_policy_mismatch"}
    pinned_scope = root_manifest.get("judge_workspace_id")
    if not isinstance(pinned_scope, str) or pinned_scope == "judge-unavailable":
        return {"status": "unavailable", "reason": "judge_credential_required_at_writer_dispatch"}
    transcript_path = Path(root["output_dir"]) / "transcript.json"
    source_text, _, source_sha = load_source(transcript_path)
    if source_sha != root["source_sha256"]:
        raise ValueError("source_revision_changed_before_opus_review")
    request_body = _quality_request_body(source_text, target, kind=kind,
                                         prior_findings=prior_findings,
                                         provider="opus", policy=policy)
    request_bytes = _json_bytes(request_body)
    target_sha = _sha(_json_bytes(target))
    semantic_key = _quality_stage_key(root, kind, request_body, "opus", policy)
    existing = ledger.stage(root["id"], kind)
    if existing is not None and existing["semantic_key"] != semantic_key:
        raise ValueError("opus_stage_identity_changed")
    stage = existing
    guard = None

    def cancel_reserved(code: str) -> None:
        current = ledger.get(stage["id"]) if stage is not None else None
        if current is not None and current["status"] == "reserved":
            ledger.cancel_before_submit(current["id"], code)

    try:
        store = _credential_store(ledger.root)
        guard_context = credential_dispatch_guard(getattr(store, "path", ledger.root / "credentials.sqlite3"))
        guard_context.__enter__()
        guard = guard_context
        if existing is None:
            candidates = store.dispatch_candidates(role="judge")
            selected = next((item for item in candidates
                             if item.get("workspace_id") == pinned_scope), None)
            if selected is None:
                return {"status": "unavailable", "reason": "judge_credential_required"}
        else:
            selected = next((item for item in store.list()["keys"]
                             if item["id"] == existing["credential_id"]
                             and item["version"] == existing["credential_version"]), None)
            if selected is None:
                cancel_reserved("judge_credential_changed")
                return {"status": "unavailable", "reason": "judge_credential_changed"}
        if selected.get("workspace_id") != pinned_scope:
            cancel_reserved("judge_policy_or_scope_changed")
            return {"status": "unavailable", "reason": "judge_policy_or_scope_changed"}
        token = store.reveal_for_dispatch(selected["id"], selected["version"], role="judge")
        output_cap = request_body["max_completion_tokens"]
        route = verify_opus_batch_route(client_factory(token), request_body,
                                        max_output_tokens=output_cap)
        if route.workspace_id != pinned_scope:
            cancel_reserved("judge_route_workspace_changed")
            return {"status": "unavailable", "reason": "judge_route_workspace_changed"}
        reserve = route.reserve_microusd()
        decision = ledger.reserve(
            semantic_key=semantic_key, source_sha256=source_sha,
            output_dir=Path(root["output_dir"]), credential_id=selected["id"],
            credential_version=selected["version"], workspace_id=pinned_scope,
            max_cost_microusd=reserve, kind=kind, root_job_id=root["id"],
        )
        if decision.kind == "blocked":
            return {"status": "unavailable", "reason": decision.reason}
        stage = ledger.get(decision.job_id)
        artifacts = Path(stage["artifact_dir"])
        prompt_path, schema, _ = _quality_stage_contract("opus", policy, kind)
        manifest = {
            "job_id": stage["id"], "root_job_id": root["id"], "stage": kind,
            "semantic_key": semantic_key, "source_sha256": source_sha,
            "target_document_sha256": target_sha,
            "request_sha256": _sha(request_bytes),
            "custom_id": stage["custom_id"], "inline_request_count": 1,
            "prompt_sha256": _sha(prompt_path.read_bytes()),
            "schema_sha256": _sha(_json_bytes(schema)),
            "credential_id": stage["credential_id"],
            "credential_version": stage["credential_version"],
            "workspace_id": pinned_scope, "model": OPUS_MODEL,
            "provider": OPUS_QUALITY_PROVIDER,
            "quality_policy_version": policy,
            "privacy_mode": OPUS_PRIVACY_MODE,
            "workspace_io_logging_enabled": OPUS_WORKSPACE_IO_LOGGING_ENABLED,
            "provider_only": [OPUS_BATCH_PROVIDER],
            "cache_policy": "no_explicit_cache_full_miss_reserved",
            "reserve_microusd": stage["reserved_microusd"],
            "counted_input_tokens": route.input_tokens,
            "context_bound_tokens": route.context_bound_tokens,
            "max_output_tokens": output_cap,
        }
        _pin_stage_files(artifacts, manifest, request_body, target)
        if stage["status"] != "reserved":
            return {"status": stage["status"], "job_id": stage["id"]}
        try:
            token = store.reveal_for_dispatch(stage["credential_id"],
                                              stage["credential_version"], role="judge")
        except CredentialError:
            ledger.cancel_before_submit(stage["id"], "judge_credential_changed_before_post")
            return {"status": "unavailable", "reason": "judge_credential_changed_before_post",
                    "job_id": stage["id"]}
        if not ledger.mark_submitting(stage["id"]):
            return {"status": "submitting", "job_id": stage["id"]}
        try:
            reply = client_factory(token).submit(stage["custom_id"], request_body)
            write_private_json(artifacts / "submit_response.json", reply.body)
            try:
                remote_id = batch_id_from_submit(reply.body)
            except ValueError:
                remote_id = None
            if reply.status_code != 202 or remote_id is None:
                ledger.submission_result(stage["id"], remote_id=None,
                                         error_code="unexpected_submit_response")
                return {"status": "submission_unknown", "job_id": stage["id"]}
            ledger.submission_result(stage["id"], remote_id=remote_id)
            return {"status": "submitted", "job_id": stage["id"], "remote_id": remote_id}
        except OpusBatchError as exc:
            definite = exc.code in {400, 401, 402, 403, 404, 422, 429}
            ledger.submission_result(stage["id"], remote_id=None, error_code=exc.reason,
                                     definite_rejection=definite)
            write_private_json(artifacts / "submit_error.json", {
                "code": exc.code, "reason": exc.reason, "retry_after": exc.retry_after,
                "definite_rejection": definite,
            })
            return {"status": "rejected_before_submit" if definite else "submission_unknown",
                    "job_id": stage["id"], "reason": exc.reason}
    except (CredentialError, OpusRouteBlocked, OpusBatchError) as exc:
        cancel_reserved("stage_preflight_unavailable")
        return {"status": "unavailable", "reason": str(exc)[:120]}
    finally:
        if guard is not None:
            guard.__exit__(None, None, None)


def _finalize_quality(ledger: Ledger, root: dict, document: dict, source_index: dict,
                      *, status: str, unresolved_count: int = 0,
                      reason: str | None = None) -> dict:
    artifacts = Path(root["artifact_dir"])
    transcript_path = Path(root["output_dir"]) / "transcript.json"
    request_manifest = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
    audit = ledger.stage(root["id"], "audit")
    verify = ledger.stage(root["id"], "verify")
    reconcile = ledger.stage(root["id"], "reconcile")
    reconciles = [ledger.stage(root["id"], f"reconcile_{number}") for number in (1, 2)]
    inventories = [ledger.stage(root["id"], f"inventory_{number}") for number in (1, 2, 3)]
    segments = [ledger.stage(root["id"], f"segment_{number}") for number in (1, 2, 3)]
    coverage_warning_count = 0
    for stage in (audit, reconcile, *reconciles, verify, *inventories, *segments):
        if stage is None:
            continue
        warning_file = Path(stage["artifact_dir"]) / "coverage_warnings.json"
        if warning_file.exists():
            coverage_warning_count += len(json.loads(warning_file.read_text(encoding="utf-8"))["warnings"])
    merge_warning_file = artifacts / "segment_merge_warnings.json"
    if merge_warning_file.exists():
        coverage_warning_count += len(json.loads(merge_warning_file.read_text(encoding="utf-8"))["warnings"])
    reused_stage = request_manifest.get("reused_segment_1")
    if reused_stage is not None:
        reused_provenance = json.loads(
            (artifacts / "reused_segment_1_provenance.json").read_text(encoding="utf-8"))
        coverage_warning_count += len(reused_provenance.get("coverage_warnings", []))
    reused_inventories = request_manifest.get("reused_inventories")
    reused_inventory_stage_ids = []
    parser_correction_warning_count = 0
    inventory_risk_warning_count = 0
    if reused_inventories is not None:
        for number, entry in enumerate(reused_inventories, 1):
            provenance_path = artifacts / f"reused_inventory_{number}_provenance.json"
            if provenance_path.is_symlink() or not provenance_path.is_file():
                raise ValueError("reused_inventory_provenance_unavailable")
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            if (_sha(_json_bytes(provenance)) != entry["provenance_sha256"]
                    or provenance.get("source_stage_job_id") != entry["source_stage_job_id"]):
                raise ValueError("reused_inventory_provenance_changed")
            reused_inventory_stage_ids.append(entry["source_stage_job_id"])
            parser_correction_warning_count += len(provenance["corrections"])
            inventory_risk_warning_count += len(provenance["risk_warnings"])
        coverage_warning_count += parser_correction_warning_count + inventory_risk_warning_count
    if coverage_warning_count and status in {"checked", "model_audit_unverified",
                                             "model_reconciled_unverified", "model_reconciled_checked",
                                             "model_segment_reviewed_unverified",
                                             "model_segment_reviewed_checked",
                                             "opus_audited_unverified", "opus_self_verified",
                                             "opus_segment_reviewed_unverified",
                                             "postverify_corrected_unchecked"}:
        status = "coverage_incomplete"
        reason = "coverage_report_inconsistent"
    quality_review = {
        "status": status, "unresolved_count": unresolved_count,
        "coverage_warning_count": coverage_warning_count,
        "reason": reason, "audit_job_id": audit["id"] if audit else None,
        "inventory_job_ids": ([stage["id"] for stage in inventories if stage]
                              or reused_inventory_stage_ids),
        "segment_job_ids": [stage["id"] for stage in segments if stage],
        "reconcile_job_id": (reconcile["id"] if reconcile else
                             next((stage["id"] for stage in reversed(reconciles) if stage), None)),
        "verify_job_id": verify["id"] if verify else None,
    }
    if request_manifest.get("quality_provider") == OPUS_QUALITY_PROVIDER:
        quality_review.update({
            "judge_model": OPUS_MODEL,
            "judge_provider": OPUS_QUALITY_PROVIDER,
            "verification_performed": (
                False if request_manifest.get("quality_policy_version") in
                OPUS_PARTITIONED_POLICIES else
                bool(verify and verify["status"] == "stage_complete")),
        })
        if request_manifest.get("quality_policy_version") in OPUS_PARTITIONED_POLICIES:
            quality_review["segment_job_ids"] = [stage["id"] for stage in segments if stage]
            plan = json.loads((artifacts / "opus_segment_plan.json").read_text(encoding="utf-8"))
            quality_review["segment_count"] = len(plan["segments"])
    if any(reconciles):
        quality_review["reconcile_job_ids"] = [stage["id"] for stage in reconciles if stage]
        quality_review["verification_performed"] = False
        quality_review["verification_reason"] = "two_part_reconciliation_without_verification"
    if root.get("writer_parent_id") is not None:
        quality_review["writer_parent_job_id"] = root["writer_parent_id"]
    if reused_stage is not None:
        quality_review["reused_segment_1_source_job_id"] = reused_stage["source_stage_job_id"]
        quality_review["reused_segment_1_has_new_dispatch"] = False
    if reused_inventories is not None:
        quality_review["reused_inventory_stage_ids"] = reused_inventory_stage_ids
        quality_review["reused_inventory_has_new_dispatch"] = False
        quality_review["inventory_correction_source"] = "saved_terminal_native_response"
        quality_review["parser_correction_warning_count"] = parser_correction_warning_count
        quality_review["inventory_risk_warning_count"] = inventory_risk_warning_count
    write_private_json(artifacts / "candidate_document.json", document)
    write_private_json(artifacts / "quality_decision.json", {"quality_review": quality_review})
    generation_id, generation_path = _publish_verified_document(
        document=document, job=root, request_manifest=request_manifest,
        source_index=source_index, transcript_path=transcript_path,
        output_dir=Path(root["output_dir"]), private_root=ledger.root,
    )
    ledger.accepted(root["id"], artifacts / "candidate_document.json", generation_id)
    ledger.mark_consumer(root["semantic_key"], Path(root["output_dir"]), generation_id=generation_id)
    consumer_results = _publish_accepted_consumers(ledger, ledger.get(root["id"]))
    return {"status": "accepted", "job_id": root["id"], "generation_id": generation_id,
            "generation_path": str(generation_path), "quality_review": quality_review,
            "consumer_results": consumer_results}


def _stage_timeout_reason(stage: dict) -> str | None:
    age = time.time() - stage["created_at"]
    if stage["status"] == "submitting" and time.time() - stage["updated_at"] > 300:
        return "submission_outcome_unknown"
    if stage["status"] == "credential_required" and age > QUALITY_CREDENTIAL_WAIT_SECONDS:
        return "credential_required_timeout"
    if stage["status"] in {"submitted", "polling", "credential_required"} and age > QUALITY_BATCH_WAIT_SECONDS:
        return "batch_result_timeout"
    return None


def _inventory_plan(root: dict, source_text: str, source_index: dict) -> list[dict]:
    """Freeze a source-only partition before the first inventory dispatch."""
    artifacts = Path(root["artifact_dir"])
    root_manifest = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
    policy = root_manifest.get("quality_policy_version")
    inventory_prompt_path, inventory_schema = _inventory_contract_for_policy(policy)
    third_after, inventory_cap, reconcile_cap, verify_cap = _inventory_profile(policy)
    if (root_manifest.get("inventory_segmenter_version") != INVENTORY_SEGMENTER_VERSION
            or root_manifest.get("inventory_third_segment_after_chars") != third_after
            or root_manifest.get("inventory_context_overlap") != INVENTORY_CONTEXT_OVERLAP
            or root_manifest.get("inventory_windows_per_segment") != INVENTORY_WINDOWS_PER_SEGMENT
            or root_manifest.get("inventory_output_cap") != inventory_cap
            or root_manifest.get("reconcile_output_cap") != reconcile_cap
            or root_manifest.get("focused_verify_output_cap") != verify_cap
            or root_manifest.get("inventory_prompt_sha256") != _sha(inventory_prompt_path.read_bytes())
            or root_manifest.get("inventory_schema_sha256") != _sha(_json_bytes(inventory_schema))):
        raise ValueError("inventory_code_or_prompt_changed_while_pending")
    count = 3 if len(source_text) > third_after else 2
    segments = plan_inventory_segments(
        source_text, count=count, overlap=INVENTORY_CONTEXT_OVERLAP,
        windows_per_segment=INVENTORY_WINDOWS_PER_SEGMENT,
    )
    validate_inventory_plan(segments, source_index)
    path = artifacts / "inventory_plan.json"
    frozen = {"version": INVENTORY_SEGMENTER_VERSION,
              "source_sha256": root["source_sha256"], "segments": segments}
    if path.exists():
        if _sha(_json_bytes(json.loads(path.read_text(encoding="utf-8")))) != _sha(_json_bytes(frozen)):
            raise ValueError("inventory_plan_changed_after_dispatch")
    else:
        write_private_json(path, frozen)
    return segments


def _segment_review_plan(root: dict, source_text: str, source_index: dict) -> list[dict]:
    """Freeze the complete primary partition before any segment review POST."""
    artifacts = Path(root["artifact_dir"])
    manifest = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
    policy = manifest.get("quality_policy_version")
    if policy not in SEGMENT_REVIEW_POLICIES:
        raise ValueError("unknown_segment_review_policy")
    prompt_path, schema, _ = _quality_contract("gemini", policy)
    third_after = _segment_review_third_after(policy)
    output_cap = _segment_review_output_cap(policy)
    if ((policy != SEGMENT_REVIEW_QUALITY_POLICY_VERSION_V1
         and manifest.get("segment_review_third_segment_after_chars") != third_after)
            or manifest.get("inventory_segmenter_version") != INVENTORY_SEGMENTER_VERSION
            or manifest.get("inventory_third_segment_after_chars") != INVENTORY_THIRD_SEGMENT_AFTER_CHARS
            or manifest.get("inventory_context_overlap") != INVENTORY_CONTEXT_OVERLAP
            or manifest.get("inventory_windows_per_segment") != INVENTORY_WINDOWS_PER_SEGMENT
            or manifest.get("segment_review_output_cap") != output_cap
            or manifest.get("focused_verify_output_cap") != FOCUSED_VERIFY_OUTPUT_CAP
            or manifest.get("segment_review_prompt_sha256") != _sha(prompt_path.read_bytes())
            or manifest.get("segment_review_schema_sha256") != _sha(_json_bytes(schema))
            or manifest.get("verify_prompt_sha256") != _sha(RECONCILE_PROMPT_PATH.read_bytes())
            or manifest.get("verify_schema_sha256") != _sha(_json_bytes(RECONCILE_SCHEMA))):
        raise ValueError("segment_review_code_or_prompt_changed_while_pending")
    count = 3 if len(source_text) > third_after else 2
    segments = plan_inventory_segments(
        source_text, count=count, overlap=INVENTORY_CONTEXT_OVERLAP,
        windows_per_segment=INVENTORY_WINDOWS_PER_SEGMENT,
    )
    validate_inventory_plan(segments, source_index)
    path = artifacts / "segment_review_plan.json"
    frozen = {"version": INVENTORY_SEGMENTER_VERSION,
              "source_sha256": root["source_sha256"], "segments": segments}
    if path.exists():
        if _sha(_json_bytes(json.loads(path.read_text(encoding="utf-8")))) != _sha(_json_bytes(frozen)):
            raise ValueError("segment_review_plan_changed_after_dispatch")
    else:
        write_private_json(path, frozen)
    return segments


def _quality_stage_outcome(stage: dict) -> str:
    if stage["status"] == "stage_complete":
        return "complete"
    if (stage["status"] == "reserved"
            and time.time() - stage["created_at"] > QUALITY_CREDENTIAL_WAIT_SECONDS):
        return "failed"
    if stage["status"] in {"submission_unknown", "rejected_before_submit",
                           "cancelled_before_submit", "failed_validation",
                           "remote_failed", "remote_expired", "remote_cancelled"}:
        return "failed"
    if _stage_timeout_reason(stage):
        return "failed"
    return "pending"


def _advance_evidence_reconciliation(ledger: Ledger, root: dict, source_text: str,
                                     source_index: dict, draft: dict, inventory: dict,
                                     gemini_client_factory) -> dict:
    """Reconcile two complete source halves within the six-dispatch limit.

    The second request is built only from the locally applied first result, so
    its target and request can be pinned as one immutable stage identity.
    Source-window and draft-unit accounting belongs to the scoped v2 contract.
    """
    artifacts = Path(root["artifact_dir"])
    partition_path = artifacts / "reconcile_partition_plan.json"
    current = draft
    unresolved_count = 0
    for number in (1, 2):
        kind = f"reconcile_{number}"
        try:
            frozen_cut = None
            if partition_path.exists():
                if partition_path.is_symlink():
                    raise ValueError("reconcile partition plan is a symlink")
                frozen_plan = json.loads(partition_path.read_text(encoding="utf-8"))
                frozen_cut = frozen_plan["parts"][0]["primary_window_ids"][-1]
            targets = partition_reconcile_targets(
                source_text, source_index, current, inventory,
                cut_after_window_id=frozen_cut)
            if len(targets) != 2:
                raise ValueError("evidence_reconcile_partition_count_changed")
            for label, complete, scoped in (
                ("source", list(source_index["by_id"]),
                 [source_id for part in targets
                  for source_id in part["scope"]["primary_source_ids"]]),
                ("window", [row["window_id"] for row in inventory["coverage"]],
                 [row["window_id"] for part in targets
                  for row in part["inventory"]["coverage"]]),
            ):
                if sorted(complete) != sorted(scoped):
                    raise ValueError(f"evidence_reconcile_{label}_partition_incomplete")
            scoped_items = {row["item_id"] for part in targets
                            for row in part["inventory"]["items"]}
            if scoped_items != {row["item_id"] for row in inventory["items"]}:
                raise ValueError("evidence_reconcile_item_partition_incomplete")
            _pin_private_json(partition_path, {
                "version": EVIDENCE_RECONCILE_PARTITION_VERSION,
                "source_sha256": root["source_sha256"],
                "parts": [{
                    "primary_source_ids": part["scope"]["primary_source_ids"],
                    "primary_window_ids": part["scope"]["primary_window_ids"],
                    "inventory_item_ids": [item["item_id"] for item in part["inventory"]["items"]],
                } for part in targets],
            })
            target = targets[number - 1]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            write_private_json(artifacts / "reconcile_partition_error.json",
                               {"stage": kind, "reason": str(exc)[:200]})
            return _finalize_quality(ledger, root, current, source_index,
                status="reconcile_unavailable", unresolved_count=unresolved_count,
                reason="invalid_reconciliation_partition")
        stage = ledger.stage(root["id"], kind)
        if stage is None or stage["status"] == "reserved":
            try:
                started = _start_gemini_stage(
                    ledger, root, kind, target, [], gemini_client_factory)
            except ValueError as exc:
                started = {"status": "unavailable", "reason": str(exc)[:120]}
            if started["status"] == "unavailable":
                return _finalize_quality(ledger, root, current, source_index,
                    status="reconcile_unavailable", unresolved_count=unresolved_count,
                    reason=f"{kind}:{started.get('reason')}")
            stage = ledger.stage(root["id"], kind)
        if stage is None or _quality_stage_outcome(stage) == "pending":
            return {"status": "reconcile_pending", "job_id": root["id"],
                    "stage": kind, "stage_job_id": stage["id"] if stage else None}
        if _quality_stage_outcome(stage) != "complete":
            return _finalize_quality(ledger, root, current, source_index,
                status="reconcile_unavailable", unresolved_count=unresolved_count,
                reason=f"{kind}:{stage['status']}")
        report = json.loads(Path(stage["accepted_document_path"]).read_text(encoding="utf-8"))
        try:
            validate_reconciliation_target_report_v2(report, target, source_index)
            revised, unresolved = apply_reconciliation_report_v2(
                current, report, target["inventory"], source_index)
        except (ValueError, json.JSONDecodeError) as exc:
            write_private_json(artifacts / f"{kind}_apply_error.json",
                               {"reason": str(exc)[:200]})
            return _finalize_quality(ledger, root, current, source_index,
                status="reconcile_unavailable", unresolved_count=unresolved_count,
                reason=f"{kind}:invalid_reconciliation_patch")
        _pin_private_json(artifacts / f"{kind}_revised_document.json", revised)
        current = revised
        unresolved_count += len(unresolved)
    _pin_private_json(artifacts / "revised_document.json", current)
    return _finalize_quality(ledger, root, current, source_index,
        status="unresolved" if unresolved_count else "model_reconciled_unverified",
        unresolved_count=unresolved_count,
        reason="verification_not_run_after_two_part_reconciliation")


def _advance_quality_inventory(ledger: Ledger, root: dict, source_text: str,
                               source_index: dict, draft: dict,
                               gemini_client_factory) -> dict:
    """Inventory every source segment, then reconcile both ways with the draft.

    Each external result has its own durable stage and immutable raw. A missing
    stage never causes a second POST for an already submitted stage.
    """
    artifacts = Path(root["artifact_dir"])
    root_manifest = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
    evidence_mode = (root_manifest.get("quality_policy_version") ==
                     EVIDENCE_RECONCILE_QUALITY_POLICY_VERSION)
    segments = _inventory_plan(root, source_text, source_index)
    stages = []
    reports = []
    reused = root_manifest.get("reused_inventories")
    if reused is not None:
        prior = ledger.get(root_manifest.get("source_inventory_root_job_id"))
        if (not evidence_mode or not isinstance(reused, list)
                or len(reused) != len(segments) or prior is None
                or prior["kind"] != "summary" or prior["status"] != "accepted"
                or prior["id"] != root["writer_parent_id"]
                or prior["source_sha256"] != root["source_sha256"]
                or prior["output_dir"] != root["output_dir"]
                or prior["billing_group_id"] != root["billing_group_id"]):
            raise ValueError("reused_inventory_root_identity_changed")
        for number, (segment, entry) in enumerate(zip(segments, reused), 1):
            if ledger.stage(root["id"], f"inventory_{number}") is not None:
                raise ValueError("reused_inventory_unexpected_new_stage")
            stage = ledger.get(entry.get("source_stage_job_id"))
            if stage is None:
                raise ValueError("reused_inventory_stage_missing")
            report, provenance = _reusable_saved_inventory(
                ledger, root=prior, stage=stage, segment=segment,
                source_text=source_text, source_index=source_index)
            report_path = artifacts / f"reused_inventory_{number}_report.json"
            provenance_path = artifacts / f"reused_inventory_{number}_provenance.json"
            if (report_path.is_symlink() or provenance_path.is_symlink()
                    or not report_path.is_file() or not provenance_path.is_file()
                    or json.loads(report_path.read_text(encoding="utf-8")) != report
                    or json.loads(provenance_path.read_text(encoding="utf-8")) != provenance
                    or entry.get("normalized_report_sha256") != _sha(_json_bytes(report))
                    or entry.get("provenance_sha256") != _sha(_json_bytes(provenance))
                    or entry.get("source_native_response_sha256") !=
                    provenance["source_artifact_sha256"]["native_response.json"]):
                raise ValueError("reused_inventory_report_or_provenance_changed")
            reports.append(report)
    else:
        for number, segment in enumerate(segments, 1):
            kind = f"inventory_{number}"
            stage = ledger.stage(root["id"], kind)
            if stage is None or stage["status"] == "reserved":
                try:
                    started = _start_gemini_stage(ledger, root, kind, segment, [], gemini_client_factory)
                except ValueError as exc:
                    started = {"status": "unavailable", "reason": str(exc)[:120]}
                if started["status"] == "unavailable":
                    write_private_json(artifacts / "inventory_unavailable.json",
                                       {"stage": kind, "reason": started.get("reason")})
                    return _finalize_quality(ledger, root, draft, source_index,
                        status="inventory_unavailable", reason=f"{kind}:{started.get('reason')}")
                stage = ledger.stage(root["id"], kind)
            stages.append(stage)
            if stage is None or _quality_stage_outcome(stage) == "pending":
                return {"status": "inventory_pending", "job_id": root["id"],
                        "stage": kind, "stage_job_id": stage["id"] if stage else None,
                        "completed_segments": len(reports), "total_segments": len(segments)}
            if _quality_stage_outcome(stage) != "complete":
                write_private_json(artifacts / "inventory_unavailable.json",
                                   {"stage": kind, "reason": stage["status"]})
                return _finalize_quality(ledger, root, draft, source_index,
                    status="inventory_unavailable", reason=f"{kind}:{stage['status']}")
            reports.append(json.loads(Path(stage["accepted_document_path"]).read_text(encoding="utf-8")))
    try:
        inventory = (merge_inventory_reports_v2(reports, segments, source_index)
                     if evidence_mode else merge_inventory_reports(reports, segments, source_index))
    except ValueError as exc:
        write_private_json(artifacts / "inventory_merge_error.json", {"reason": str(exc)[:200]})
        return _finalize_quality(ledger, root, draft, source_index,
            status="inventory_unavailable", reason="source_inventory_invalid")
    inventory_path = artifacts / "merged_inventory.json"
    if inventory_path.exists():
        saved = json.loads(inventory_path.read_text(encoding="utf-8"))
        if _sha(_json_bytes(saved)) != _sha(_json_bytes(inventory)):
            raise ValueError("merged_inventory_changed_while_pending")
    else:
        write_private_json(inventory_path, inventory)
    # An already sealed historical v2 reconcile stage keeps its saved request
    # and recovery path. New v2 jobs use two bounded stages.
    if evidence_mode and ledger.stage(root["id"], "reconcile") is None:
        return _advance_evidence_reconciliation(
            ledger, root, source_text, source_index, draft, inventory,
            gemini_client_factory)
    target = {"draft": draft, "inventory": inventory}
    reconciliation = ledger.stage(root["id"], "reconcile")
    if reconciliation is None or reconciliation["status"] == "reserved":
        try:
            started = _start_gemini_stage(ledger, root, "reconcile", target, [], gemini_client_factory)
        except ValueError as exc:
            started = {"status": "unavailable", "reason": str(exc)[:120]}
        if started["status"] == "unavailable":
            return _finalize_quality(ledger, root, draft, source_index,
                status="reconcile_unavailable", reason=started.get("reason"))
        reconciliation = ledger.stage(root["id"], "reconcile")
    if reconciliation is None or _quality_stage_outcome(reconciliation) == "pending":
        return {"status": "reconcile_pending", "job_id": root["id"],
                "stage_job_id": reconciliation["id"] if reconciliation else None}
    if _quality_stage_outcome(reconciliation) == "failed":
        return _finalize_quality(ledger, root, draft, source_index,
            status="reconcile_unavailable", reason=reconciliation["status"])
    report = json.loads(Path(reconciliation["accepted_document_path"]).read_text(encoding="utf-8"))
    try:
        apply_report = (apply_reconciliation_report_v2 if evidence_mode
                        else apply_reconciliation_report)
        revised, unresolved = apply_report(draft, report, inventory, source_index)
    except (ValueError, json.JSONDecodeError) as exc:
        write_private_json(artifacts / "reconcile_apply_error.json", {"reason": str(exc)[:200]})
        return _finalize_quality(ledger, root, draft, source_index,
            status="reconcile_unavailable", reason="invalid_reconciliation_patch")
    write_private_json(artifacts / "revised_document.json", revised)
    if not report["patches"]:
        return _finalize_quality(ledger, root, revised, source_index,
            status="unresolved" if unresolved else "model_reconciled_unverified",
            unresolved_count=len(unresolved))

    verify_target = {"draft": revised, "inventory": inventory,
                     "prior_findings": report["findings"]}
    verification = ledger.stage(root["id"], "verify")
    if verification is None or verification["status"] == "reserved":
        try:
            started = _start_gemini_stage(ledger, root, "verify", verify_target,
                                          report["findings"], gemini_client_factory)
        except ValueError as exc:
            started = {"status": "unavailable", "reason": str(exc)[:120]}
        if started["status"] == "unavailable":
            return _finalize_quality(ledger, root, revised, source_index,
                status="verify_unavailable", unresolved_count=len(unresolved),
                reason=started.get("reason"))
        verification = ledger.stage(root["id"], "verify")
    if verification is None or _quality_stage_outcome(verification) == "pending":
        return {"status": "verify_pending", "job_id": root["id"],
                "stage_job_id": verification["id"] if verification else None}
    if _quality_stage_outcome(verification) == "failed":
        return _finalize_quality(ledger, root, revised, source_index,
            status="verify_unavailable", unresolved_count=len(unresolved),
            reason=verification["status"])
    verify_report = json.loads(Path(verification["accepted_document_path"]).read_text(encoding="utf-8"))
    try:
        checked, later_unresolved = apply_report(
            revised, verify_report, inventory, source_index, mode="verify")
    except (ValueError, json.JSONDecodeError) as exc:
        write_private_json(artifacts / "verify_apply_error.json", {"reason": str(exc)[:200]})
        return _finalize_quality(ledger, root, revised, source_index,
            status="verify_unavailable", unresolved_count=len(unresolved),
            reason="invalid_verify_report")
    if verify_report["patches"]:
        write_private_json(artifacts / "postverify_document.json", checked)
    return _finalize_quality(ledger, root, checked, source_index,
        status=("postverify_corrected_unchecked" if verify_report["patches"] else
                "unresolved" if unresolved or later_unresolved else "model_reconciled_checked"),
        unresolved_count=len(unresolved) + len(later_unresolved),
        reason="final_bounded_correction" if verify_report["patches"] else None)


def _advance_quality_segments(ledger: Ledger, root: dict, source_text: str,
                              source_index: dict, draft: dict,
                              gemini_client_factory) -> dict:
    """Review each source part against the full draft, within one shared cap.

    The next segment is submitted only after the preceding one is terminal and
    its actual usage replaces its reservation. This prevents simultaneous
    pessimistic reserves from consuming the logical $0.10 cap without
    weakening either the per-call cost estimate or cache-miss assumption.
    """
    artifacts = Path(root["artifact_dir"])
    segments = _segment_review_plan(root, source_text, source_index)
    reports = []
    root_manifest = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
    reused = root_manifest.get("reused_segment_1")
    if reused is not None:
        if (not isinstance(reused, dict)
                or reused.get("form_normalizer_version") != SAVED_SEGMENT_REUSE_VERSION
                or ledger.stage(root["id"], "segment_1") is not None):
            raise ValueError("reused_segment_1_manifest_or_stage_differs")
        report_path = artifacts / "reused_segment_1_report.json"
        provenance_path = artifacts / "reused_segment_1_provenance.json"
        if (report_path.is_symlink() or not report_path.is_file()
                or provenance_path.is_symlink() or not provenance_path.is_file()):
            raise ValueError("reused_segment_1_report_unavailable")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        fresh_report, fresh_provenance, _ = _reusable_saved_segment_1(
            ledger, stage_job_id=reused["source_stage_job_id"],
            writer_job_id=root["writer_parent_id"], source_text=source_text,
            source_index=source_index, source_sha=root["source_sha256"],
            output_dir=root["output_dir"], draft=draft, segment=segments[0])
        if (report != fresh_report or provenance != fresh_provenance
                or _sha(_json_bytes(report)) != reused.get("normalized_report_sha256")
                or reused.get("source_artifact_sha256") != provenance.get("source_artifact_sha256")
                or reused.get("source_remote_batch_id") != provenance.get("source_remote_batch_id")):
            raise ValueError("reused_segment_1_evidence_changed")
        reports.append(report)

    def partial(reason: str) -> dict:
        revised = draft
        unresolved = []
        if reports:
            try:
                combined, warnings = merge_segment_review_reports(
                    reports, segments, draft, source_index, allow_partial=True)
                write_private_json(artifacts / "segment_partial_report.json", combined)
                if warnings:
                    write_private_json(artifacts / "segment_merge_warnings.json",
                                       {"warnings": warnings})
                revised, unresolved = apply_audit(draft, combined, source_index)
            except (ValueError, json.JSONDecodeError) as exc:
                write_private_json(artifacts / "segment_partial_apply_error.json",
                                   {"reason": type(exc).__name__ + ":" + str(exc)[:200]})
        write_private_json(artifacts / "segment_partial_document.json", revised)
        return _finalize_quality(ledger, root, revised, source_index,
            status="segment_review_unavailable", unresolved_count=len(unresolved),
            reason=reason)

    for number, segment in enumerate(segments, 1):
        if number == 1 and reused is not None:
            continue
        kind = f"segment_{number}"
        stage = ledger.stage(root["id"], kind)
        if stage is None or stage["status"] == "reserved":
            target = {"segment": segment, "draft": draft}
            try:
                started = _start_gemini_stage(ledger, root, kind, target, [],
                                               gemini_client_factory)
            except ValueError as exc:
                started = {"status": "unavailable", "reason": str(exc)[:120]}
            if started["status"] == "unavailable":
                write_private_json(artifacts / "segment_unavailable.json", {
                    "stage": kind, "reason": started.get("reason"),
                    "completed_segments": len(reports),
                })
                return partial(f"{kind}:{started.get('reason')}")
            stage = ledger.stage(root["id"], kind)
        if stage is None or _quality_stage_outcome(stage) == "pending":
            return {"status": "segment_review_pending", "job_id": root["id"],
                    "stage": kind, "stage_job_id": stage["id"] if stage else None,
                    "completed_segments": len(reports), "total_segments": len(segments)}
        if _quality_stage_outcome(stage) == "failed":
            write_private_json(artifacts / "segment_unavailable.json", {
                "stage": kind, "reason": stage["status"],
                "completed_segments": len(reports),
            })
            return partial(f"{kind}:{stage['status']}")
        reports.append(json.loads(Path(stage["accepted_document_path"]).read_text(encoding="utf-8")))

    try:
        combined, warnings = merge_segment_review_reports(
            reports, segments, draft, source_index)
        inventory = merge_inventory_reports(
            [segment_inventory_view(report) for report in reports],
            segments, source_index)
        revised, unresolved = apply_audit(draft, combined, source_index)
    except (ValueError, json.JSONDecodeError) as exc:
        write_private_json(artifacts / "segment_merge_error.json",
                           {"reason": type(exc).__name__ + ":" + str(exc)[:200]})
        return partial("segment_reports_could_not_be_merged")
    write_private_json(artifacts / "segment_combined_report.json", combined)
    write_private_json(artifacts / "segment_merged_inventory.json", inventory)
    write_private_json(artifacts / "revised_document.json", revised)
    if warnings:
        write_private_json(artifacts / "segment_merge_warnings.json", {"warnings": warnings})
    if not combined["patches"]:
        return _finalize_quality(ledger, root, revised, source_index,
            status="unresolved" if unresolved else "model_segment_reviewed_unverified",
            unresolved_count=len(unresolved))

    target = {"draft": revised, "inventory": inventory,
              "prior_findings": combined["findings"]}
    verification = ledger.stage(root["id"], "verify")
    if verification is None or verification["status"] == "reserved":
        try:
            started = _start_gemini_stage(ledger, root, "verify", target,
                                          combined["findings"], gemini_client_factory)
        except ValueError as exc:
            started = {"status": "unavailable", "reason": str(exc)[:120]}
        if started["status"] == "unavailable":
            return _finalize_quality(ledger, root, revised, source_index,
                status="verify_unavailable", unresolved_count=len(unresolved),
                reason=started.get("reason"))
        verification = ledger.stage(root["id"], "verify")
    if verification is None or _quality_stage_outcome(verification) == "pending":
        return {"status": "verify_pending", "job_id": root["id"],
                "stage_job_id": verification["id"] if verification else None}
    if _quality_stage_outcome(verification) == "failed":
        return _finalize_quality(ledger, root, revised, source_index,
            status="verify_unavailable", unresolved_count=len(unresolved),
            reason=verification["status"])
    report = json.loads(Path(verification["accepted_document_path"]).read_text(encoding="utf-8"))
    try:
        checked, later_unresolved = apply_reconciliation_report(
            revised, report, inventory, source_index, mode="verify")
    except (ValueError, json.JSONDecodeError) as exc:
        write_private_json(artifacts / "segment_verify_apply_error.json",
                           {"reason": type(exc).__name__ + ":" + str(exc)[:200]})
        return _finalize_quality(ledger, root, revised, source_index,
            status="verify_unavailable", unresolved_count=len(unresolved),
            reason="invalid_verify_report")
    if report["patches"]:
        write_private_json(artifacts / "postverify_document.json", checked)
    return _finalize_quality(ledger, root, checked, source_index,
        status=("postverify_corrected_unchecked" if report["patches"] else
                "unresolved" if unresolved or later_unresolved else
                "model_segment_reviewed_checked"),
        unresolved_count=len(unresolved) + len(later_unresolved),
        reason="final_bounded_correction" if report["patches"] else None)


def _opus_partition_plan(root: dict, source_text: str, source_index: dict,
                         draft: dict) -> list[dict]:
    """Seal the exact primary source partition before the first Opus POST."""
    artifacts = Path(root["artifact_dir"])
    manifest = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
    policy = manifest.get("quality_policy_version")
    if policy not in OPUS_PARTITIONED_POLICIES:
        raise ValueError("opus_partition_contract_changed")
    prompt, schema, _id, output_cap, effort, _profile = _opus_partition_profile(policy)
    if (manifest.get("partition_version") != OPUS_PARTITION_VERSION
            or manifest.get("segment_output_cap") != output_cap
            or manifest.get("audit_prompt_sha256") != _sha(prompt.read_bytes())
            or manifest.get("audit_schema_sha256") != _sha(_json_bytes(schema))
            or (policy == OPUS_PARTITIONED_QUALITY_POLICY_VERSION
                and manifest.get("segment_reasoning_effort") != effort)):
        raise ValueError("opus_partition_contract_changed")
    segments = plan_opus_audit_segments(source_text)
    validate_inventory_plan(segments, source_index)
    _pin_private_json(artifacts / "opus_segment_plan.json", {
        "version": OPUS_PARTITION_VERSION,
        "source_sha256": root["source_sha256"],
        "draft_sha256": _sha(_json_bytes(draft)),
        "segments": segments,
    })
    return segments


def _advance_quality_opus_partitioned(ledger: Ledger, root: dict, source_text: str,
                                      source_index: dict, draft: dict,
                                      opus_client_factory) -> dict:
    """Review each primary part against one pinned full draft and merge once.

    The scheduler submits the next part only after the previous part is
    terminal. This permits actual charges to replace conservative holds within
    the common weekly and logical-job caps. No stage changes the source or
    reuses a partially patched draft as another stage's comparison target.
    """
    artifacts = Path(root["artifact_dir"])
    segments = _opus_partition_plan(root, source_text, source_index, draft)
    policy = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))[
        "quality_policy_version"]
    expected_schema_id = _opus_partition_profile(policy)[2]
    reports: list[dict] = []

    def incomplete(stage_name: str, reason: str) -> dict:
        write_private_json(artifacts / "opus_partition_incomplete.json", {
            "stage": stage_name, "reason": reason,
            "completed_segments": len(reports),
            "total_segments": len(segments),
            "completed_report_sha256": [_sha(_json_bytes(row)) for row in reports],
        })
        # The completed reports remain sealed for diagnosis. A partial set is
        # not applied as if the entire source had been checked.
        return _finalize_quality(ledger, root, draft, source_index,
                                 status="coverage_incomplete",
                                 reason=f"{stage_name}:{reason}")

    for number, segment in enumerate(segments, 1):
        kind = f"segment_{number}"
        stage = ledger.stage(root["id"], kind)
        if stage is None or stage["status"] == "reserved":
            target = {"segment": segment, "draft": draft}
            started = _start_opus_stage(ledger, root, kind, target, [],
                                        opus_client_factory)
            if started["status"] == "unavailable":
                return incomplete(kind, started.get("reason") or "unavailable")
            stage = ledger.stage(root["id"], kind)
        if stage is None or _quality_stage_outcome(stage) == "pending":
            return {"status": "opus_segment_pending", "job_id": root["id"],
                    "stage": kind, "stage_job_id": stage["id"] if stage else None,
                    "completed_segments": len(reports),
                    "total_segments": len(segments)}
        if _quality_stage_outcome(stage) == "failed":
            return incomplete(kind, stage["status"])
        report = json.loads(Path(stage["accepted_document_path"]).read_text(encoding="utf-8"))
        validate_opus_segment_report(
            report, segment, draft, source_index,
            expected_schema_id=expected_schema_id)
        reports.append(report)

    try:
        combined, warnings = merge_opus_segment_reports(
            reports, segments, draft, source_index,
            expected_schema_id=expected_schema_id)
        write_private_json(artifacts / "opus_partition_combined_report.json", combined)
        if warnings:
            write_private_json(artifacts / "segment_merge_warnings.json",
                               {"warnings": warnings})
        revised, unresolved = apply_audit(draft, combined, source_index)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        write_private_json(artifacts / "opus_partition_merge_error.json", {
            "reason": type(exc).__name__ + ":" + str(exc)[:200],
        })
        return incomplete("merge", "invalid_report_or_patch")
    write_private_json(artifacts / "revised_document.json", revised)
    return _finalize_quality(
        ledger, root, revised, source_index,
        status=("unresolved" if unresolved else
                "coverage_incomplete" if warnings else
                "opus_segment_reviewed_unverified"),
        unresolved_count=len(unresolved),
        reason="segment_merge_warnings" if warnings else None,
    )


def _advance_quality_opus(ledger: Ledger, root: dict, source_index: dict,
                          draft: dict, opus_client_factory) -> dict:
    """One full-source judge call; verify only actual proposed corrections."""
    artifacts = Path(root["artifact_dir"])
    audit = ledger.stage(root["id"], "audit")
    if audit is None or audit["status"] == "reserved":
        started = _start_opus_stage(ledger, root, "audit", draft, [], opus_client_factory)
        if started["status"] == "unavailable":
            return _finalize_quality(ledger, root, draft, source_index,
                status="opus_audit_unavailable", reason=started.get("reason"))
        return {"status": "opus_audit_" + started["status"], "job_id": root["id"],
                "stage_job_id": started.get("job_id")}
    if audit["status"] != "stage_complete":
        outcome = _quality_stage_outcome(audit)
        if outcome == "failed":
            return _finalize_quality(ledger, root, draft, source_index,
                status="opus_audit_unavailable", reason=audit["status"])
        timeout = _stage_timeout_reason(audit)
        if timeout:
            return _finalize_quality(ledger, root, draft, source_index,
                status="opus_audit_unavailable", reason=timeout)
        return {"status": "opus_audit_pending", "job_id": root["id"],
                "stage_job_id": audit["id"]}
    report = json.loads(Path(audit["accepted_document_path"]).read_text(encoding="utf-8"))
    try:
        revised, unresolved = apply_opus_audit(draft, report, source_index, mode="audit")
    except (ValueError, json.JSONDecodeError) as exc:
        write_private_json(artifacts / "opus_apply_error.json", {
            "reason": type(exc).__name__ + ":" + str(exc)[:200]})
        return _finalize_quality(ledger, root, draft, source_index,
            status="opus_audit_unavailable", reason="invalid_opus_patch")
    write_private_json(artifacts / "revised_document.json", revised)
    if not report["patches"]:
        return _finalize_quality(ledger, root, revised, source_index,
            status="unresolved" if unresolved else "opus_audited_unverified",
            unresolved_count=len(unresolved))

    verify = ledger.stage(root["id"], "verify")
    if verify is None or verify["status"] == "reserved":
        started = _start_opus_stage(ledger, root, "verify", revised,
                                    report["findings"], opus_client_factory)
        if started["status"] == "unavailable":
            return _finalize_quality(ledger, root, revised, source_index,
                status="opus_verify_unavailable", unresolved_count=len(unresolved),
                reason=started.get("reason"))
        return {"status": "opus_verify_" + started["status"], "job_id": root["id"],
                "stage_job_id": started.get("job_id")}
    if verify["status"] != "stage_complete":
        outcome = _quality_stage_outcome(verify)
        if outcome == "failed":
            return _finalize_quality(ledger, root, revised, source_index,
                status="opus_verify_unavailable", unresolved_count=len(unresolved),
                reason=verify["status"])
        timeout = _stage_timeout_reason(verify)
        if timeout:
            return _finalize_quality(ledger, root, revised, source_index,
                status="opus_verify_unavailable", unresolved_count=len(unresolved),
                reason=timeout)
        return {"status": "opus_verify_pending", "job_id": root["id"],
                "stage_job_id": verify["id"]}
    verification = json.loads(Path(verify["accepted_document_path"]).read_text(encoding="utf-8"))
    try:
        verify_request = json.loads((Path(verify["artifact_dir"]) / "request.json").read_text(encoding="utf-8"))
        verify_input = json.loads(verify_request["messages"][1]["content"])
        expected_windows = verify_input["SOURCE_WINDOWS"]
        verify_source_ids = {item["id"] for item in verify_input["TRANSCRIPT_SOURCE"]["utterances"]}
        checked, later_unresolved = apply_opus_audit(
            revised, verification, source_index, mode="verify",
            expected_windows=expected_windows,
            verify_source_ids=verify_source_ids)
    except (ValueError, json.JSONDecodeError) as exc:
        write_private_json(artifacts / "opus_verify_apply_error.json", {
            "reason": type(exc).__name__ + ":" + str(exc)[:200]})
        return _finalize_quality(ledger, root, revised, source_index,
            status="opus_verify_unavailable", unresolved_count=len(unresolved),
            reason="invalid_opus_verify_patch")
    if verification["patches"]:
        write_private_json(artifacts / "postverify_document.json", checked)
    return _finalize_quality(ledger, root, checked, source_index,
        status=("postverify_corrected_unchecked" if verification["patches"] else
                "unresolved" if unresolved or later_unresolved else
                "opus_self_verified"),
        unresolved_count=len(unresolved) + len(later_unresolved),
        reason="final_bounded_correction" if verification["patches"] else None)


def _advance_quality(ledger: Ledger, root: dict, client_factory,
                     gemini_client_factory=GeminiBatchClient,
                     opus_client_factory=OpusBatchClient) -> dict:
    """Advance one saved workflow; semantic uncertainty is published visibly."""
    artifacts = Path(root["artifact_dir"])
    transcript_path = Path(root["output_dir"]) / "transcript.json"
    source_text, source_index, source_sha = load_source(transcript_path)
    if source_sha != root["source_sha256"]:
        raise ValueError("source_revision_changed_during_quality_review")
    draft = json.loads((artifacts / "draft_document.json").read_text(encoding="utf-8"))
    root_manifest_path = artifacts / "manifest.json"
    root_manifest = (json.loads(root_manifest_path.read_text(encoding="utf-8"))
                     if root_manifest_path.exists() else {})
    if root.get("writer_parent_id") is not None:
        if (root_manifest.get("writer_parent_job_id") != root["writer_parent_id"]
                or root_manifest.get("normalized_draft_sha256") != _sha(_json_bytes(draft))):
            raise ValueError("saved_writer_continuation_draft_changed")
        original = ledger.get(root["writer_parent_id"])
        if (original is None or original["kind"] != "summary"
                or original["source_sha256"] != source_sha
                or original["remote_id"] != root_manifest.get("writer_remote_batch_id")):
            raise ValueError("saved_writer_continuation_identity_changed")
        if (original["status"] == "accepted"
                and (root_manifest.get("continuation_mode") !=
                     "accepted_inventory_unavailable"
                     or root_manifest.get("source_inventory_root_job_id") != original["id"]
                     or not isinstance(root_manifest.get("reused_inventories"), list))):
            raise ValueError("saved_inventory_continuation_manifest_changed")
        original_artifacts = Path(original["artifact_dir"])
        if (_sha((original_artifacts / "manifest.json").read_bytes())
                != root_manifest.get("writer_manifest_sha256")
                or _sha((original_artifacts / "draft_document.json").read_bytes())
                != root_manifest.get("writer_draft_file_sha256")):
            raise ValueError("saved_writer_evidence_changed")
    if root_manifest.get("quality_policy_version") in OPUS_PARTITIONED_POLICIES:
        if _quality_route(root_manifest) != "opus":
            raise ValueError("opus_partitioned_quality_root_changed")
        return _advance_quality_opus_partitioned(
            ledger, root, source_text, source_index, draft,
            opus_client_factory)
    if root_manifest.get("quality_policy_version") == OPUS_QUALITY_POLICY_VERSION:
        if _quality_route(root_manifest) != "opus":
            raise ValueError("opus_quality_root_changed")
        return _advance_quality_opus(ledger, root, source_index, draft,
                                     opus_client_factory)
    if root_manifest.get("quality_policy_version") in SEGMENT_REVIEW_POLICIES:
        return _advance_quality_segments(ledger, root, source_text, source_index,
                                         draft, gemini_client_factory)
    if root_manifest.get("quality_policy_version") in {
            INVENTORY_RECONCILE_QUALITY_POLICY_VERSION,
            EVIDENCE_RECONCILE_QUALITY_POLICY_VERSION}:
        return _advance_quality_inventory(ledger, root, source_text, source_index,
                                          draft, gemini_client_factory)
    audit = ledger.stage(root["id"], "audit")
    if audit is None or audit["status"] == "reserved":
        started = _start_quality_stage(ledger, root, "audit", draft, [], client_factory,
                                       gemini_client_factory)
        if started["status"] == "unavailable":
            return _finalize_quality(ledger, root, draft, source_index,
                status="audit_unavailable", reason=started.get("reason"))
        return {"status": "audit_" + started["status"], "job_id": root["id"],
                "stage_job_id": started.get("job_id")}
    if audit["status"] != "stage_complete":
        if audit["status"] in {"submission_unknown", "rejected_before_submit",
                               "cancelled_before_submit", "failed_validation",
                               "remote_failed", "remote_expired", "remote_cancelled"}:
            return _finalize_quality(ledger, root, draft, source_index,
                status="audit_unavailable", reason=audit["status"])
        timeout = _stage_timeout_reason(audit)
        if timeout:
            return _finalize_quality(ledger, root, draft, source_index,
                status="audit_unavailable", reason=timeout)
        return {"status": "audit_pending", "job_id": root["id"], "stage_job_id": audit["id"]}
    report = json.loads(Path(audit["accepted_document_path"]).read_text(encoding="utf-8"))
    inventory_contract = (isinstance(report, dict)
                          and report.get("schema_version") == GEMINI_AUDIT_SCHEMA_ID_V2)
    if inventory_contract:
        report = legacy_report_v1(report)
    try:
        revised, unresolved = apply_audit(draft, report, source_index, mode="audit")
    except (ValueError, json.JSONDecodeError) as exc:
        failure_origin = "invalid_audit_patch"
        if not report.get("patches"):
            try:
                validate_document(draft, source_index)
            except (ValueError, KeyError, TypeError):
                failure_origin = "invalid_draft_unrepaired"
        write_private_json(artifacts / "audit_apply_error.json", {
            "reason": str(exc)[:200], "failure_origin": failure_origin})
        return _finalize_quality(ledger, root, draft, source_index,
            status="audit_unavailable", reason=failure_origin)
    write_private_json(artifacts / "revised_document.json", revised)
    if not report["patches"]:
        return _finalize_quality(ledger, root, revised, source_index,
            status=("unresolved" if unresolved else
                    "model_audit_unverified" if inventory_contract else "checked"),
            unresolved_count=len(unresolved))
    verify = ledger.stage(root["id"], "verify")
    if verify is None or verify["status"] == "reserved":
        started = _start_quality_stage(ledger, root, "verify", revised,
                                       report["findings"], client_factory,
                                       gemini_client_factory)
        if started["status"] == "unavailable":
            return _finalize_quality(ledger, root, revised, source_index,
                status="verify_unavailable", unresolved_count=len(unresolved),
                reason=started.get("reason"))
        return {"status": "verify_" + started["status"], "job_id": root["id"],
                "stage_job_id": started.get("job_id")}
    if verify["status"] != "stage_complete":
        if verify["status"] in {"submission_unknown", "rejected_before_submit",
                                "cancelled_before_submit", "failed_validation",
                                "remote_failed", "remote_expired", "remote_cancelled"}:
            return _finalize_quality(ledger, root, revised, source_index,
                status="verify_unavailable", unresolved_count=len(unresolved),
                reason=verify["status"])
        timeout = _stage_timeout_reason(verify)
        if timeout:
            return _finalize_quality(ledger, root, revised, source_index,
                status="verify_unavailable", unresolved_count=len(unresolved),
                reason=timeout)
        return {"status": "verify_pending", "job_id": root["id"], "stage_job_id": verify["id"]}
    verification = json.loads(Path(verify["accepted_document_path"]).read_text(encoding="utf-8"))
    if isinstance(verification, dict) and verification.get("schema_version") == GEMINI_AUDIT_SCHEMA_ID_V2:
        verification = legacy_report_v1(verification)
    try:
        checked, later_unresolved = apply_audit(revised, verification, source_index, mode="verify")
    except (ValueError, json.JSONDecodeError) as exc:
        write_private_json(artifacts / "verify_apply_error.json", {"reason": str(exc)[:200]})
        return _finalize_quality(ledger, root, revised, source_index,
            status="verify_unavailable", unresolved_count=len(unresolved),
            reason="invalid_verify_report")
    if verification["patches"]:
        write_private_json(artifacts / "postverify_document.json", checked)
    return _finalize_quality(ledger, root, checked, source_index,
        status=("postverify_corrected_unchecked" if verification["patches"] else
                "unresolved" if unresolved or later_unresolved else
                "model_audit_unverified" if inventory_contract else "checked"),
        unresolved_count=len(unresolved) + len(later_unresolved),
        reason="final_bounded_correction" if verification["patches"] else None)


def _publish_accepted_consumers(ledger: Ledger, job: dict) -> list[dict]:
    """Fan out one accepted raw result locally; never ask the model again."""
    manifest = json.loads((Path(job["artifact_dir"]) / "manifest.json").read_text(encoding="utf-8"))
    document = json.loads(Path(job["accepted_document_path"]).read_text(encoding="utf-8"))
    results = []
    for consumer in ledger.consumer_rows(job["semantic_key"], status="pending"):
        output_dir = Path(consumer["output_dir"])
        transcript_path = output_dir / "transcript.json"
        try:
            _, source_index, source_sha = load_source(transcript_path)
            if source_sha != job["source_sha256"]:
                raise ValueError("consumer_source_revision_changed")
            generation_id, _ = _publish_verified_document(
                document=document, job=job, request_manifest=manifest,
                source_index=source_index, transcript_path=transcript_path,
                output_dir=output_dir, private_root=ledger.root,
            )
            ledger.mark_consumer(job["semantic_key"], output_dir, generation_id=generation_id)
            results.append({"output_dir": str(output_dir), "status": "published", "generation_id": generation_id})
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            # An I/O interruption can be retried from the accepted document.
            # A changed source is a per-consumer failure, not a failed model.
            if isinstance(exc, (OSError, RevisionConflict)):
                status = "pending_retry"
            else:
                ledger.mark_consumer(job["semantic_key"], output_dir,
                                     error_code=type(exc).__name__ + ":" + str(exc)[:100])
                status = "failed_source_revision" if str(exc) == "consumer_source_revision_changed" else "failed_consumer"
            results.append({"output_dir": str(output_dir), "status": status,
                            "reason": type(exc).__name__ + ":" + str(exc)[:120]})
    return results


def poll_once(*, private_root: Path, client_factory=BatchClient,
              gemini_client_factory=GeminiBatchClient,
              opus_client_factory=OpusBatchClient) -> list[dict]:
    """One bounded scheduler tick; never sleeps or submits a second POST."""
    ledger = Ledger(private_root)
    outcomes = []
    try:
        for job in ledger.raw_ready():
            try:
                outcomes.append(_finish_raw(ledger, job))
            except (OSError, RevisionConflict) as exc:
                ledger.defer_raw(job["id"], delay_seconds=180, error_code=type(exc).__name__ + ":" + str(exc)[:100])
                outcomes.append({"status": "publication_pending_retry", "job_id": job["id"],
                                 "reason": type(exc).__name__})
            except (ValueError, json.JSONDecodeError) as exc:
                ledger.failed_validation(job["id"], type(exc).__name__ + ":" + str(exc)[:120])
                outcomes.append({"status": "failed_validation", "job_id": job["id"], "reason": str(exc)[:120]})
        for job in ledger.accepted_with_pending_consumers():
            outcomes.append({"status": "consumer_recovery", "job_id": job["id"],
                             "consumer_results": _publish_accepted_consumers(ledger, job)})
        for job in ledger.pending_remote()[:4]:
            artifacts = Path(job["artifact_dir"])
            manifest = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
            is_gemini = manifest.get("provider") == QUALITY_PROVIDER
            is_opus = manifest.get("provider") == OPUS_QUALITY_PROVIDER
            try:
                store = _credential_store(private_root)
                token = store.reveal_for_existing_job(job["credential_id"], job["credential_version"])
            except CredentialError:
                ledger.credential_required(job["id"])
                _append_poll_event(artifacts, {"operation": "credential_lookup", "status": "credential_required",
                                               "remote_id": job["remote_id"]})
                outcomes.append({"status": "credential_required", "job_id": job["id"], "remote_id": job["remote_id"]})
                continue
            selected_factory = (opus_client_factory if is_opus else
                                gemini_client_factory if is_gemini else client_factory)
            client = selected_factory(token)
            try:
                reply = client.get(job["remote_id"])
            except (BatchError, GeminiBatchError, OpusBatchError) as exc:
                delay = _poll_backoff(exc.retry_after)
                ledger.defer_poll(job["id"], delay_seconds=delay, error_code=exc.reason)
                _append_poll_event(artifacts, {"operation": "GET", "status": "error", "remote_id": job["remote_id"],
                                               "code": exc.code, "reason": exc.reason, "retry_after": exc.retry_after,
                                               "next_delay_seconds": delay})
                outcomes.append({"status": "poll_error", "job_id": job["id"], "reason": exc.reason})
                continue
            batch = reply.body
            if batch.get("id") != job["remote_id"]:
                ledger.defer_poll(job["id"], delay_seconds=600, error_code="remote_identity_mismatch")
                _append_poll_event(artifacts, {"operation": "GET", "status": "remote_identity_mismatch",
                                               "remote_id": job["remote_id"]})
                outcomes.append({"status": "remote_identity_mismatch", "job_id": job["id"]})
                continue
            remote_status = batch.get("status")
            if remote_status not in {"pending", "running", "validating", "in_progress",
                                     "finalizing", "cancelling", *TERMINAL}:
                ledger.defer_poll(job["id"], delay_seconds=600, error_code="unknown_remote_status")
                _append_poll_event(artifacts, {"operation": "GET", "status": "unknown_remote_status",
                                               "remote_id": job["remote_id"]})
                outcomes.append({"status": "unknown_remote_status", "job_id": job["id"]})
                continue
            batch_stats = batch.get("request_counts")
            _append_poll_event(artifacts, {"operation": "GET", "status": remote_status,
                                           "remote_id": job["remote_id"],
                                           "request_counts": batch_stats,
                                           "usage": batch.get("usage")})
            if remote_status in TERMINAL:
                write_private_json(artifacts / "batch_terminal.json", batch)
            else:
                write_private_json(artifacts / "batch_progress.json", {
                    "id": batch.get("id"), "status": remote_status,
                    "request_counts": batch_stats,
                    "usage": batch.get("usage"),
                })
            usage = batch.get("usage") or {}
            # Progress receipts may be partial. Spend the full reservation
            # until the remote batch is terminal and total billing is known.
            usage_cost = _cost_micros(usage) if remote_status in TERMINAL else None
            ledger.poll_result(job["id"], remote_status, usage_cost_microusd=usage_cost,
                               error_code=(None if remote_status == "completed" else
                                           ("opus_batch_" if is_opus else "gemini_batch_") + remote_status
                                           if is_opus or is_gemini else
                                           str(batch.get("error") or "")[:120] or None))
            if remote_status == "completed":
                try:
                    accepted = _finish_raw(ledger, ledger.get(job["id"]))
                    outcomes.append(accepted)
                except (OSError, RevisionConflict) as exc:
                    ledger.defer_raw(job["id"], delay_seconds=180, error_code=type(exc).__name__ + ":" + str(exc)[:100])
                    outcomes.append({"status": "publication_pending_retry", "job_id": job["id"],
                                     "reason": type(exc).__name__})
                except (ValueError, json.JSONDecodeError) as exc:
                    ledger.failed_validation(job["id"], type(exc).__name__ + ":" + str(exc)[:120])
                    outcomes.append({"status": "failed_validation", "job_id": job["id"], "reason": str(exc)[:120]})
            else:
                outcomes.append({"status": remote_status, "job_id": job["id"], "remote_id": job["remote_id"]})
            if remote_status in TERMINAL:
                try:
                    deletion = client.delete(job["remote_id"])
                    write_private_json(artifacts / "batch_delete.json", deletion.body)
                    _append_poll_event(artifacts, {"operation": "DELETE", "status": "completed",
                                                   "remote_id": job["remote_id"], "code": deletion.status_code})
                except (BatchError, GeminiBatchError, OpusBatchError) as exc:
                    write_private_json(artifacts / "batch_delete_error.json", {"code": exc.code, "reason": exc.reason})
                    _append_poll_event(artifacts, {"operation": "DELETE", "status": "error",
                                                   "remote_id": job["remote_id"], "code": exc.code, "reason": exc.reason})
        for root in ledger.quality_pending():
            try:
                outcomes.append(_advance_quality(ledger, root, client_factory,
                                                 gemini_client_factory,
                                                 opus_client_factory))
            except (OSError, RevisionConflict) as exc:
                ledger.defer_quality(root["id"], delay_seconds=180,
                                     error_code=type(exc).__name__ + ":" + str(exc)[:100])
                outcomes.append({"status": "quality_pending_retry", "job_id": root["id"],
                                 "reason": type(exc).__name__})
            except (ValueError, json.JSONDecodeError) as exc:
                ledger.failed_quality(root["id"], type(exc).__name__ + ":" + str(exc)[:100])
                outcomes.append({"status": "failed_quality_integrity", "job_id": root["id"],
                                 "reason": str(exc)[:120]})
        return outcomes
    finally:
        ledger.close()
