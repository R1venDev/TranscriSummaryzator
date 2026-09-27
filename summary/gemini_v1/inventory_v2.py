"""Source-only inventory v2 with exact excerpts and visible lexical gaps.

V1 stays available for saved jobs. Excerpts prove only that the model quoted a
cited primary utterance; they do not establish the truth of its paraphrase.
Risk anchors are a deterministic prompt to examine a span, never model facts.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from .inventory_contract import (
    INVENTORY_SCHEMA_ID,
    build_inventory_input,
    merge_inventory_reports,
    validate_inventory_report,
)


INVENTORY_SCHEMA_ID_V2 = "gemini_source_inventory_v2"
MERGED_INVENTORY_SCHEMA_ID_V2 = "gemini_source_inventory_merged_v2"
INVENTORY_PROMPT_PATH_V2 = Path(__file__).with_name("inventory_prompt_v2.md")
INVENTORY_SCHEMA_V2 = json.loads(
    Path(__file__).with_name("inventory_schema_v2.json").read_text(encoding="utf-8")
)

_RISK_PATTERNS = {
    "number": re.compile(r"(?<!\w)\d+(?:[\s\u00a0]\d{3})*(?:[.,]\d+)?(?!\w)", re.I),
    "time": re.compile(r"(?<!\d)\d{1,2}[:.]\d{2}(?!\d)|\b(?:час(?:а|ов)?|минут(?:а|ы)?|дн(?:я|ей)|недел(?:я|и|ь)|месяц(?:а|ев)?|завтра|вчера)\b", re.I),
    "negation": re.compile(r"\b(?:не|нет|ни|нельзя|без|отмен(?:ить|ил[аи]?|ено|яется))\b", re.I),
    "condition": re.compile(r"\b(?:если|когда|при\s+условии|в\s+случае|только\s+если)\b", re.I),
    "alternative": re.compile(r"\b(?:или|либо|вместо|иначе)\b", re.I),
    "correction": re.compile(r"\b(?:подожди|погоди|точнее|вернее|поправк\w*|исправл\w*|отменя\w*)\b|\bне\b[^.!?]{0,80}\bа\b", re.I),
    "actor": re.compile(r"(?<!\w)@[\w-]{2,}|\b(?:я|мы|ты|он|она|они|мне|нам|тебе|вам|миш[аеу]?)\b", re.I),
}


def build_inventory_input_v2(segment: dict) -> str:
    """Canonical v2 message still contains source only and ends with host task."""
    payload = json.loads(build_inventory_input(segment))
    payload["TASK"] = (
        "По PRIMARY_UTTERANCES независимо от черновика перечисли все существенные "
        "смыслы. Для каждого item приведи source_quote: короткий дословный "
        "фрагмент одной из цитируемых primary-реплик, точно как в SOURCE_SEGMENT. "
        "Отдельно проверь числа и время, отрицания, условия, альтернативы, "
        "поправки и участников действия. Заполни каждое COVERAGE_WINDOW. "
        "CONTEXT_BEFORE/AFTER поясняют связи, но не являются primary-репликами."
    )
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _as_v1(report: dict) -> dict:
    converted = copy.deepcopy(report)
    converted["schema_version"] = INVENTORY_SCHEMA_ID
    for item in converted["items"]:
        item.pop("source_quote", None)
    return converted


def validate_inventory_report_v2(report: dict, segment: dict) -> dict:
    """Check v1 coordinates and quote shape; exact mismatches become warnings.

    A complete, paid model response with a malformed quotation must remain
    inspectable instead of losing every other inventory item in the segment.
    The mismatch is carried as unresolved uncertainty by the merged report.
    """
    if (not isinstance(report, dict) or set(report) !=
            {"schema_version", "segment_id", "coverage", "items"}
            or report.get("schema_version") != INVENTORY_SCHEMA_ID_V2
            or not isinstance(report.get("items"), list)):
        raise ValueError("inventory v2 report shape or version differs")
    for number, item in enumerate(report["items"]):
        if (not isinstance(item, dict) or "source_quote" not in item
                or not isinstance(item["source_quote"], str)
                or not item["source_quote"].strip()
                or len(item["source_quote"]) > 500):
            raise ValueError(f"inventory items[{number}]: source_quote is invalid")
    validate_inventory_report(_as_v1(report), segment)
    return report


def inventory_risk_warnings_v2(report: dict, segment: dict) -> list[dict]:
    """Mark uncited source spans that carry lexical risk anchors.

    A warning is unresolved scope for reconciliation. It does not infer the
    omitted meaning or declare a false model claim. Ordinary uncited talk can
    still carry meaning; these patterns only prioritize visible uncertainty.
    """
    validate_inventory_report_v2(report, segment)
    cited = {source_id for item in report["items"] for source_id in item["source_ids"]}
    primary = segment["primary_utterances"]
    positions = {row["id"]: number for number, row in enumerate(primary)}
    window_for_id = {}
    for window in segment["coverage_windows"]:
        for row in primary[positions[window["start_id"]]:positions[window["end_id"]] + 1]:
            window_for_id[row["id"]] = window["window_id"]
    warnings = []
    primary_text = {row["id"]: row["text"] for row in primary}
    for number, item in enumerate(report["items"]):
        quote = item["source_quote"]
        cited_primary = [source_id for source_id in item["source_ids"]
                         if source_id in primary_text]
        if not any(quote in primary_text[source_id] for source_id in cited_primary):
            warnings.append({
                "code": "source_quote_mismatch",
                "segment_id": segment["segment_id"],
                "window_id": window_for_id[cited_primary[0]],
                "source_ids": cited_primary,
                "risk_kind": "quote",
                "anchors": [quote],
                "item_index": number,
                "status": "uncertain",
            })
    for row in primary:
        source_id = row["id"]
        if source_id in cited:
            continue
        for kind, pattern in _RISK_PATTERNS.items():
            anchors = list(dict.fromkeys(match.group(0) for match in pattern.finditer(row["text"])))
            if anchors:
                warnings.append({
                    "code": "source_risk_anchor_uncited",
                    "segment_id": segment["segment_id"],
                    "window_id": window_for_id[source_id],
                    "source_ids": [source_id],
                    "risk_kind": kind,
                    "anchors": anchors[:12],
                    "status": "uncertain",
                })
    return warnings


def merge_inventory_reports_v2(reports: list[dict], segments: list[dict],
                               source_index: dict) -> dict:
    """Preserve v2 quotes and uncertainty while reusing exact v1 partitioning."""
    if not isinstance(reports, list) or len(reports) != len(segments):
        raise ValueError("inventory v2 needs one report per segment")
    for report, segment in zip(reports, segments):
        validate_inventory_report_v2(report, segment)
    merged = merge_inventory_reports([_as_v1(report) for report in reports],
                                     segments, source_index)
    quote_by_item = {
        f"{segment['segment_id']}-I{number:03d}": item["source_quote"]
        for report, segment in zip(reports, segments)
        for number, item in enumerate(report["items"], 1)
    }
    for item in merged["items"]:
        item["source_quote"] = quote_by_item[item["item_id"]]
    merged["schema_version"] = MERGED_INVENTORY_SCHEMA_ID_V2
    merged["risk_warnings"] = [
        warning for report, segment in zip(reports, segments)
        for warning in inventory_risk_warnings_v2(report, segment)
    ]
    return merged
