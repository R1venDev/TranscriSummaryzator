"""Full-source Opus audit input and bounded Chat request construction."""

from __future__ import annotations

import json

from ..luna_v1.audit import build_audit_input
from .batch import OUTPUT_CAP_AUDIT, OUTPUT_CAP_VERIFY, REASONING_EFFORT
from .contract import OPUS_AUDIT_PROMPT_PATH, OPUS_AUDIT_SCHEMA, OPUS_AUDIT_SCHEMA_ID

_TASKS = {
    "audit": "Проверь всю TRANSCRIPT_SOURCE против DRAFT_DOCUMENT; верни краткий отчёт с адресными исправлениями.",
    "verify": "Проверь отдельные адресные контексты TRANSCRIPT_SOURCE против исправленного DRAFT_DOCUMENT и PRIOR_FINDINGS; верни оставшиеся или новые ошибки и адресные исправления.",
}


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
