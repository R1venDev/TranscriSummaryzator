"""Build one stable, source-first Gemini audit/repair message.

The full transcript is placed before the changing draft so consecutive audit
and verify requests share a useful prefix. The final task is host-authored;
utterances and draft text remain untrusted data in their own JSON fields.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..luna_v1.audit import AUDIT_SCHEMA, AUDIT_SCHEMA_ID, build_audit_input


GEMINI_AUDIT_PROMPT_PATH = Path(__file__).with_name("prompt_audit_v1.md")
GEMINI_AUDIT_SCHEMA = AUDIT_SCHEMA
GEMINI_AUDIT_SCHEMA_ID = AUDIT_SCHEMA_ID

_TASKS = {
    "audit": (
        "На основе всей TRANSCRIPT_SOURCE последовательно проверь SOURCE_WINDOWS "
        "и DRAFT_DOCUMENT. Верни source-audit report с адресными patches по схеме."
    ),
    "verify": (
        "На основе всей TRANSCRIPT_SOURCE проверь уже исправленный "
        "DRAFT_DOCUMENT, особенно PRIOR_FINDINGS и связанные разделы. "
        "Верни оставшиеся или новые находки и адресные patches по схеме."
    ),
}


def build_gemini_audit_input(
    source_text: str,
    draft: dict,
    *,
    mode: str = "audit",
    prior_findings: list[dict] | tuple[dict, ...] = (),
) -> str:
    """Return canonical JSON with source first and trusted task last.

    The existing builder validates the source, draft shape, mode, and prior
    findings. Reordering only top-level fields preserves its exact content.
    JSON string escaping keeps quoted transcript instructions inside data.
    """
    original = json.loads(
        build_audit_input(
            source_text, draft, mode=mode, prior_findings=prior_findings
        )
    )
    ordered = {
        "TRANSCRIPT_SOURCE": original["TRANSCRIPT_SOURCE"],
        "SOURCE_WINDOWS": original["SOURCE_WINDOWS"],
        "DRAFT_DOCUMENT": original["DRAFT_DOCUMENT"],
        "PRIOR_FINDINGS": original["PRIOR_FINDINGS"],
        "MODE": original["MODE"],
        "TASK": _TASKS[mode],
    }
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))
