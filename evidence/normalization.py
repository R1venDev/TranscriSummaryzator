"""Context-safe normalization that never overwrites raw evidence."""
from __future__ import annotations
from enum import StrEnum


class NormalizationPolicy(StrEnum):
    SAFE_EXACT = "SAFE_EXACT"
    CONTEXT_REQUIRED = "CONTEXT_REQUIRED"
    NEVER_AUTO = "NEVER_AUTO"


def normalize_text(raw_text, rules, *, asr_uncertain=False):
    normalized = raw_text
    operations = []
    lowered = raw_text.casefold()
    for rule in rules:
        source, target = rule.get("source", ""), rule.get("target", "")
        policy = rule.get("policy", NormalizationPolicy.NEVER_AUTO)
        allowed = policy == NormalizationPolicy.SAFE_EXACT
        if policy == NormalizationPolicy.CONTEXT_REQUIRED:
            context = [str(x).casefold() for x in rule.get("required_context", [])]
            allowed = any(x in lowered for x in context) and (asr_uncertain or not rule.get("require_asr_uncertainty", False))
        if not allowed or not source or source not in normalized:
            continue
        before = normalized
        normalized = normalized.replace(source, target)
        operations.append({"source": source, "target": target, "policy": str(policy), "before": before, "after": normalized})
    return {"raw_text": raw_text, "normalized_text": normalized, "normalization_operations": operations}
