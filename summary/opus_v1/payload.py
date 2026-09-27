"""Opus source audit input and bounded Chat request construction."""

from __future__ import annotations

import hashlib
import json

from ..gemini_v1.inventory_contract import (
    _source_utterances, _windows, plan_inventory_segments,
    validate_inventory_plan,
)
from ..luna_v1.audit import build_audit_input
from ..luna_v1.audit import _working_draft
from .batch import OUTPUT_CAP_AUDIT, OUTPUT_CAP_VERIFY, REASONING_EFFORT
from .contract import (
    OPUS_AUDIT_PROMPT_PATH, OPUS_AUDIT_SCHEMA, OPUS_AUDIT_SCHEMA_ID,
    OPUS_SEGMENT_PROMPT_PATH, OPUS_SEGMENT_SCHEMA, OPUS_SEGMENT_SCHEMA_ID,
    OPUS_SEGMENT_PROMPT_PATH_V3, OPUS_SEGMENT_SCHEMA_V3, OPUS_SEGMENT_SCHEMA_ID_V3,
    risk_anchors_for_primary,
)

_TASKS = {
    "audit": "Проверь всю TRANSCRIPT_SOURCE против DRAFT_DOCUMENT; верни краткий отчёт с адресными исправлениями.",
    "verify": "Проверь отдельные адресные контексты TRANSCRIPT_SOURCE против исправленного DRAFT_DOCUMENT и PRIOR_FINDINGS; верни оставшиеся или новые ошибки и адресные исправления.",
}
OPUS_SEGMENT_OUTPUT_CAP = 6_000
OPUS_SEGMENT_OUTPUT_CAP_V3 = 7_000
OPUS_SEGMENT_EFFORT_V3 = "low"


def _focused_verify_input(source: dict, draft: dict, prior_findings: list[dict]) -> dict:
    """Keep prior source anchors with neighbors and the full revised draft.

    The prior finding indices refer to the pre-audit draft and can shift after
    insertions/removals. Showing the complete *revised* draft lets the verifier
    address the current index without silently selecting the wrong item.
    """
    if not prior_findings:
        raise ValueError("verify_requires_prior_findings")
    utterances = source["utterances"]
    positions = {item["id"]: number for number, item in enumerate(utterances)}
    anchors: set[int] = set()
    for finding in prior_findings:
        ids = finding.get("source_ids")
        if not isinstance(ids, list) or not ids:
            raise ValueError("verify_prior_source_ids_missing")
        for source_id in ids:
            if not isinstance(source_id, str) or source_id not in positions:
                raise ValueError("verify_prior_source_id_unknown")
            anchors.add(positions[source_id])
    selected = sorted({position + delta for position in anchors for delta in (-1, 0, 1)
                       if 0 <= position + delta < len(utterances)})
    if not selected:
        raise ValueError("verify_source_scope_empty")
    windows = []
    groups: list[list[int]] = []
    for position in selected:
        if not groups or position != groups[-1][-1] + 1:
            groups.append([])
        groups[-1].append(position)
    for number, group in enumerate(groups, 1):
        windows.append({
            "window_id": f"V{number:02d}",
            "start_id": utterances[group[0]]["id"],
            "end_id": utterances[group[-1]]["id"],
            "utterance_count": len(group),
        })

    scoped_source = dict(source)
    scoped_source["utterances"] = [utterances[position] for position in selected]
    return {"TRANSCRIPT_SOURCE": scoped_source, "SOURCE_WINDOWS": windows,
            "DRAFT_DOCUMENT": draft}


def build_opus_audit_input(
    source_text: str,
    draft: dict,
    *,
    mode: str = "audit",
    prior_findings: list[dict] | tuple[dict, ...] = (),
) -> str:
    """Keep the complete normalized transcript before the changing draft."""
    original = json.loads(build_audit_input(
        source_text, draft, mode=mode, prior_findings=prior_findings,
    ))
    if mode == "audit":
        ordered = {
            "TRANSCRIPT_SOURCE": original["TRANSCRIPT_SOURCE"],
            "SOURCE_WINDOWS": original["SOURCE_WINDOWS"],
            "DRAFT_DOCUMENT": original["DRAFT_DOCUMENT"],
            "PRIOR_FINDINGS": original["PRIOR_FINDINGS"],
            "MODE": mode,
            "TASK": _TASKS[mode],
        }
    else:
        ordered = {
            **_focused_verify_input(original["TRANSCRIPT_SOURCE"],
                                    original["DRAFT_DOCUMENT"],
                                    original["PRIOR_FINDINGS"]),
            "PRIOR_FINDINGS": original["PRIOR_FINDINGS"],
            "MODE": mode,
            "TASK": _TASKS[mode],
        }
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def build_opus_audit_request(
    source_text: str,
    draft: dict,
    *,
    mode: str = "audit",
    prior_findings: list[dict] | tuple[dict, ...] = (),
) -> dict:
    """Return the single text-only Chat body for OpenRouter Batch.

    Batch-level model and provider are pinned by BatchClient.submit. No
    temperature, tools, cache directive, or OpenRouter plugin is requested.
    """
    content = build_opus_audit_input(
        source_text, draft, mode=mode, prior_findings=prior_findings,
    )
    return {
        "messages": [
            {"role": "system", "content": OPUS_AUDIT_PROMPT_PATH.read_text(encoding="utf-8")},
            {"role": "user", "content": content},
        ],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": OPUS_AUDIT_SCHEMA_ID, "strict": True, "schema": OPUS_AUDIT_SCHEMA,
        }},
        "max_completion_tokens": OUTPUT_CAP_AUDIT if mode == "audit" else OUTPUT_CAP_VERIFY,
        "reasoning": {"effort": REASONING_EFFORT},
    }


def _canonical_segment_source(source_text: str) -> tuple[dict, list[dict], dict]:
    """Require the exact load_source projection, without inspecting audio/upstream."""
    source, utterances = _source_utterances(source_text)
    canonical = json.dumps(source, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if canonical != source_text:
        raise ValueError("Opus segment source is not canonical")
    required = {"source_kind", "source_name", "meeting_date", "duration_ms",
                "participants_by_transcript", "unattributed_speech", "utterances"}
    if set(source) != required or source["source_kind"] != "TRANSCRIPT_SOURCE":
        raise ValueError("Opus segment source metadata differs from canonical projection")
    source_index = {
        "source_name": source["source_name"],
        "meeting_date": source["meeting_date"],
        "duration_ms": source["duration_ms"],
        "participants": source["participants_by_transcript"],
        "unattributed_speech": source["unattributed_speech"],
        "by_id": {row["id"]: row for row in utterances},
    }
    return source, utterances, source_index


def plan_opus_audit_segments(source_text: str) -> list[dict]:
    """Cover each canonical utterance once, with bounded adjacent context."""
    _source, _utterances, source_index = _canonical_segment_source(source_text)
    segments = plan_inventory_segments(source_text, count=3, overlap=8,
                                       windows_per_segment=4)
    validate_inventory_plan(segments, source_index)
    return segments


def _validate_one_segment(source_text: str, source: dict,
                          utterances: list[dict], segment: dict) -> None:
    keys = {"segment_id", "source_fingerprint", "source_name", "meeting_date",
            "participants_by_transcript", "context_before", "primary_utterances",
            "context_after", "coverage_windows"}
    if not isinstance(segment, dict) or set(segment) != keys:
        raise ValueError("Opus segment fields differ from inventory plan")
    if segment["source_fingerprint"] != hashlib.sha256(source_text.encode("utf-8")).hexdigest():
        raise ValueError("Opus segment source fingerprint differs")
    if (segment["source_name"] != source["source_name"]
            or segment["meeting_date"] != source["meeting_date"]
            or segment["participants_by_transcript"] != source["participants_by_transcript"]):
        raise ValueError("Opus segment metadata differs from source")
    primary, before, after = (segment[name] for name in
                              ("primary_utterances", "context_before", "context_after"))
    if (not isinstance(primary, list) or not primary
            or not isinstance(before, list) or not isinstance(after, list)
            or len(before) > 12 or len(after) > 12):
        raise ValueError("Opus segment primary/context is invalid")
    positions = {row["id"]: position for position, row in enumerate(utterances)}
    first_id = primary[0].get("id") if isinstance(primary[0], dict) else None
    if first_id not in positions:
        raise ValueError("Opus segment primary ID is unknown")
    start = positions[first_id]
    end = start + len(primary)
    if (primary != utterances[start:end]
            or start < len(before)
            or before != utterances[start - len(before):start]
            or after != utterances[end:end + len(after)]):
        raise ValueError("Opus segment differs from canonical source")
    windows = segment["coverage_windows"]
    if (not isinstance(windows, list) or not windows
            or windows != _windows(primary, segment["segment_id"], len(windows))):
        raise ValueError("Opus segment coverage windows differ from primary source")


def build_opus_segment_audit_input(source_text: str, draft: dict,
                                   segment: dict, *,
                                   profile: str = "v2") -> str:
    """Place scoped source before complete draft and trusted task last."""
    source, utterances, _index = _canonical_segment_source(source_text)
    _validate_one_segment(source_text, source, utterances, segment)
    primary = segment["primary_utterances"]
    before = segment["context_before"]
    after = segment["context_after"]
    scoped = dict(source)
    scoped["utterances"] = before + primary + after
    if profile not in {"v2", "v3"}:
        raise ValueError("unknown Opus segment profile")
    schema_id = (OPUS_SEGMENT_SCHEMA_ID_V3 if profile == "v3"
                 else OPUS_SEGMENT_SCHEMA_ID)
    payload = {
        "TRANSCRIPT_SOURCE": scoped,
        "SEGMENT_ID": segment["segment_id"],
        "AUDIT_SCOPE": {
            "primary_start_id": primary[0]["id"],
            "primary_end_id": primary[-1]["id"],
            "primary_count": len(primary),
            "context_before_ids": [row["id"] for row in before],
            "context_after_ids": [row["id"] for row in after],
        },
        "SOURCE_WINDOWS": segment["coverage_windows"],
        "RISK_ANCHORS": risk_anchors_for_primary(primary),
        "DRAFT_DOCUMENT": _working_draft(draft),
        "TASK": ("Проверь только основной участок против всего черновика; "
                 "верни доказательные находки и адресные исправления "
                 f"по схеме {schema_id}."),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_opus_segment_audit_request(source_text: str, draft: dict,
                                     segment: dict, *,
                                     profile: str = "v2") -> dict:
    """Single Opus Batch item; no provider cache or speculative transport retry."""
    content = build_opus_segment_audit_input(source_text, draft, segment,
                                            profile=profile)
    if profile == "v3":
        prompt_path, schema, schema_id = (
            OPUS_SEGMENT_PROMPT_PATH_V3, OPUS_SEGMENT_SCHEMA_V3,
            OPUS_SEGMENT_SCHEMA_ID_V3)
        output_cap, effort = OPUS_SEGMENT_OUTPUT_CAP_V3, OPUS_SEGMENT_EFFORT_V3
    else:
        prompt_path, schema, schema_id = (
            OPUS_SEGMENT_PROMPT_PATH, OPUS_SEGMENT_SCHEMA,
            OPUS_SEGMENT_SCHEMA_ID)
        output_cap, effort = OPUS_SEGMENT_OUTPUT_CAP, REASONING_EFFORT
    return {
        "messages": [
            {"role": "system", "content": prompt_path.read_text(encoding="utf-8")},
            {"role": "user", "content": content},
        ],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": schema_id, "strict": True,
            "schema": schema,
        }},
        "max_completion_tokens": output_cap,
        "reasoning": {"effort": effort},
    }
