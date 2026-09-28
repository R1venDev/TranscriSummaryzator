"""Versioned wire contracts for the Luna Batch summary stages.

These are model *reports*, not a replacement for the existing WriterSchema in
``contract.py``.  Runtime input envelopes, expected-ID membership, exact quote
matching, source identity, patch authorization and publication are validated
by application code after parsing.  In particular, a valid report is not
evidence that the model's interpretation is correct.

The public helpers intentionally return one schema and one developer message
per stage.  The caller supplies a separate serialized JSON user message with
only that stage's source/draft/units/expected IDs.  Import only loads the
existing local WriterSchema; no inference or network I/O occurs.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict

from .contract import (
    SCHEMA as WRITER_SCHEMA,
    SCHEMA_ID as WRITER_SCHEMA_ID,
    SCHEMA_PATH as WRITER_SCHEMA_PATH,
)


STAGE_PROMPT_VERSION = "luna_batch_stage_v1"
PROMPT_DIR = Path(__file__).with_name("prompts_batch_v1")
SCHEMA_DIR = Path(__file__).with_name("schemas_batch_v1")
REVIEW_SIDECAR_SCHEMA_PATH = SCHEMA_DIR / "review_sidecar_v1.json"
StageName: TypeAlias = Literal["writer", "extract", "audit", "global", "repair", "verify"]


class WireModel(BaseModel):
    """All supplied fields are required and extra model-generated keys fail."""

    model_config = ConfigDict(extra="forbid", strict=True)


class Evidence(WireModel):
    evidence_id: str
    u_id: str
    quote: str


class Facet(WireModel):
    facet_id: str
    axis: Literal[
        "action", "actor", "recipient", "object", "scope", "status",
        "condition", "alternatives", "conjunction", "negation", "quantity",
        "unit", "time", "sequence", "parallelism", "causality",
        "uncertainty", "correction",
    ]
    value: str
    evidence_ids: list[str]


class OpenLink(WireModel):
    unit_id: str
    relation_kind: str
    cue: str
    evidence_ids: list[str]


class SourceUnit(WireModel):
    unit_id: str
    kind: Literal[
        "statement", "action", "decision", "question", "hypothesis",
        "constraint", "rationale", "example",
    ]
    text: str
    evidence_ids: list[str]
    facets: list[Facet]


class SourceAccounting(WireModel):
    u_id: str
    disposition: Literal["content", "context_only", "non_material", "uncertain"]
    unit_ids: list[str]
    note: str | None


class InventoryReport(WireModel):
    schema_version: Literal["luna_inventory_v1"]
    complete: bool
    evidence: list[Evidence]
    units: list[SourceUnit]
    source_accounting: list[SourceAccounting]
    open_links: list[OpenLink]
    unprocessed_ids: list[str]


class DocumentSurface(WireModel):
    """App-issued registry row; never trusted merely because a model echoes it."""

    surface_id: str
    entity_id: str
    field_key: str
    text_or_scalar: str | int | float | bool | None
    source_ids: list[str]
    parent_context: str


class DocumentEvidence(WireModel):
    surface_id: str
    quote: str


class FacetCheck(WireModel):
    facet_id: str
    verdict: Literal["preserved", "missing", "distorted", "uncertain", "not_applicable"]
    document_evidence: list[DocumentEvidence]
    note: str | None


class SourceCheck(WireModel):
    unit_id: str
    inventory_verdict: Literal["source_supported", "inventory_error", "needs_context", "uncertain"]
    coverage: Literal["full", "partial", "absent", "uncertain", "not_applicable"]
    facet_checks: list[FacetCheck]
    finding_ids: list[str]


class ClaimCheck(WireModel):
    claim_text: str
    verdict: Literal["supported", "contradicted", "unsupported", "needs_context", "uncertain"]
    evidence_ids: list[str]
    note: str | None


class DocumentCheck(WireModel):
    surface_id: str
    claims: list[ClaimCheck]
    finding_ids: list[str]


class Finding(WireModel):
    finding_id: str
    kind: str
    severity: Literal["minor", "material", "critical"]
    affected_surface_ids: list[str]
    affected_unit_ids: list[str]
    evidence_ids: list[str]
    problem: str
    required_preservation: str
    proposed_resolution: str | None


class ContextRequest(WireModel):
    request_id: str
    affected_ids: list[str]
    question: str
    known_source_ids: list[str]
    reason: str


class AuditReport(WireModel):
    schema_version: Literal["luna_audit_v1"]
    complete: bool
    evidence: list[Evidence]
    source_checks: list[SourceCheck]
    document_checks: list[DocumentCheck]
    additional_units: list[SourceUnit]
    findings: list[Finding]
    context_requests: list[ContextRequest]
    unprocessed_ids: list[str]


class ContextResolution(WireModel):
    request_id: str
    status: Literal["resolved", "unresolved"]
    conclusion: str | None
    evidence_ids: list[str]
    affected_surface_ids: list[str]


class LinkCheck(WireModel):
    link_id: str
    unit_ids: list[str]
    relation: Literal[
        "same_action", "distinct_actions", "acceptance", "reported_commitment",
        "alternative", "conjunction", "sequential", "parallel", "correction",
        "retraction", "unresolved",
    ]
    evidence_ids: list[str]
    conclusion: str


class GlobalReport(WireModel):
    schema_version: Literal["luna_global_v1"]
    complete: bool
    evidence: list[Evidence]
    resolutions: list[ContextResolution]
    link_checks: list[LinkCheck]
    additional_units: list[SourceUnit]
    findings: list[Finding]
    affected_surfaces: list[str]
    unprocessed_ids: list[str]


class PatchOperation(WireModel):
    """A bounded operation envelope, not executable JSON Patch.

    ``target_id`` and ``field_key`` must be checked against app-issued IDs and
    the real WriterSchema allowlist.  ``value_json`` is data: the applier must
    decode it and validate the selected field/item against the existing
    WriterSchema or TaskSchema before making a copy.  Existing entity IDs must
    be retained; ``temp_id`` is only for app allocation of a new entity.
    ``expected_before_hash`` is deliberately absent: the app adds it after
    resolving an authorized target, rather than accepting it from the model.
    """

    kind: Literal[
        "replace_field", "add_section_item", "create_task", "update_task",
        "change_task_relation", "split_task", "merge_tasks",
    ]
    target_id: str | None
    field_key: str | None
    value_json: str | None
    position_after_id: str | None
    temp_id: str | None
    lineage_ids: list[str]


class PatchBundle(WireModel):
    bundle_id: str
    finding_ids: list[str]
    evidence_ids: list[str]
    affected_surface_ids: list[str]
    operations: list[PatchOperation]
    preservation_notes: str
    dependency_bundle_ids: list[str]


class PatchPlan(WireModel):
    schema_version: Literal["luna_patch_plan_v1"]
    complete: bool
    bundles: list[PatchBundle]
    unresolved: list[str]
    unprocessed_finding_ids: list[str]


class BundleCheck(WireModel):
    bundle_id: str
    verdict: Literal["accept", "reject", "unresolved"]
    evidence: list[Evidence]
    preserved_surface_ids: list[str]
    regressions: list[str]
    note: str


class VerificationReport(WireModel):
    schema_version: Literal["luna_verification_v1"]
    complete: bool
    bundle_checks: list[BundleCheck]
    new_findings: list[Finding]
    unprocessed_bundle_ids: list[str]


class ArtifactHash(WireModel):
    name: str
    sha256: str


class PublicationReviewSidecar(WireModel):
    """Local generation-owned review state; never a model response format."""

    schema_version: Literal["luna_review_sidecar_v1"]
    generation_id: str
    source_revision: str
    execution_status: str
    review_status: Literal[
        "review_pending", "review_completed", "reviewed_with_uncertainties",
        "review_incomplete", "generation_failed",
    ]
    source_scope_ids: list[str]
    checked_ids: list[str]
    unreviewed_ids: list[str]
    accepted_bundle_ids: list[str]
    rejected_bundle_ids: list[str]
    unresolved_bundle_ids: list[str]
    source_ambiguities: list[str]
    model_disagreements: list[str]
    technical_failures: list[str]
    billed_cost_microusd: int | None
    held_cost_microusd: int
    unknown_bill_ids: list[str]
    artifact_hashes: list[ArtifactHash]
    generation_lineage: list[str]


REPORT_MODELS: dict[str, type[WireModel]] = {
    "extract": InventoryReport,
    "audit": AuditReport,
    "global": GlobalReport,
    "repair": PatchPlan,
    "verify": VerificationReport,
}
_PROMPTS = {
    "writer": "10_writer.md",
    "extract": "20_extract.md",
    "audit": "30_audit.md",
    "global": "40_global.md",
    "repair": "50_repair.md",
    "verify": "60_verify.md",
}
_SCHEMA_FILES = {
    "extract": "inventory_v1.json",
    "audit": "audit_v1.json",
    "global": "global_v1.json",
    "repair": "patch_plan_v1.json",
    "verify": "verification_v1.json",
}


def _stage(stage: str) -> str:
    if stage not in _PROMPTS:
        raise ValueError(f"unknown Luna Batch stage: {stage}")
    return stage


def _compact_strict_schema(node: object) -> None:
    """Trim Pydantic presentation metadata while retaining strict shapes."""
    if isinstance(node, list):
        for child in node:
            _compact_strict_schema(child)
        return
    if not isinstance(node, dict):
        return
    for key in ("title", "description", "default", "examples"):
        node.pop(key, None)
    if "const" in node:
        node["enum"] = [node.pop("const")]
    if node.get("type") == "object":
        properties = node.get("properties", {})
        node["additionalProperties"] = False
        node["required"] = list(properties)
    for child in node.values():
        _compact_strict_schema(child)


def schema_for_stage(stage: StageName | str) -> dict:
    """Return a fresh JSON Schema for exactly one response stage.

    Writer uses the real, unmodified ``output_schema_v1.json`` contract.  The
    other five schemas are generated from these Pydantic models on demand.
    Source U-IDs and app-issued IDs remain strings; membership is checked by
    the caller against its immutable JobPlan, never encoded as huge enums.
    """
    stage = _stage(stage)
    if stage == "writer":
        return deepcopy(WRITER_SCHEMA)
    schema = REPORT_MODELS[stage].model_json_schema(ref_template="#/$defs/{model}")
    _compact_strict_schema(schema)
    return schema


def schema_path_for_stage(stage: StageName | str) -> Path:
    """Path of the versioned schema artifact; sidecars regenerate from models."""
    stage = _stage(stage)
    return WRITER_SCHEMA_PATH if stage == "writer" else SCHEMA_DIR / _SCHEMA_FILES[stage]


def review_sidecar_schema() -> dict:
    """Generated schema for local canonical review metadata, not model I/O."""
    schema = PublicationReviewSidecar.model_json_schema(ref_template="#/$defs/{model}")
    _compact_strict_schema(schema)
    return schema


def response_format_for_stage(stage: StageName | str) -> dict:
    stage = _stage(stage)
    name = WRITER_SCHEMA_ID if stage == "writer" else {
        "extract": "luna_inventory_v1",
        "audit": "luna_audit_v1",
        "global": "luna_global_v1",
        "repair": "luna_patch_plan_v1",
        "verify": "luna_verification_v1",
    }[stage]
    return {"type": "json_schema", "json_schema": {
        "name": name, "strict": True, "schema": schema_for_stage(stage),
    }}


def developer_message_for_stage(stage: StageName | str) -> str:
    """Join the common instruction and exactly one stage instruction."""
    stage = _stage(stage)
    common = (PROMPT_DIR / "00_common.md").read_text(encoding="utf-8").strip()
    specific = (PROMPT_DIR / _PROMPTS[stage]).read_text(encoding="utf-8").strip()
    return common + "\n\n" + specific + "\n"


def prompt_sha256_for_stage(stage: StageName | str) -> str:
    return hashlib.sha256(developer_message_for_stage(stage).encode("utf-8")).hexdigest()


def parse_stage_report(stage: StageName | str, value: object) -> WireModel:
    """Parse a native sidecar report; writer stays on validate_document().

    This function checks shape and scalar types only.  The engine must next
    verify exact quotes, all expected IDs, source/contract identity and
    semantic dependency rules before trusting or publishing any result.
    """
    stage = _stage(stage)
    if stage == "writer":
        raise ValueError("writer uses the existing validate_document contract")
    return REPORT_MODELS[stage].model_validate(value)
