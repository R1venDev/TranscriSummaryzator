"""Canonical proposition identity, entities, quantities and conditions."""
from __future__ import annotations
import hashlib
import json
import re
from .ontology import require_content_kind

TOKEN_RE = re.compile(r"(?iu)[a-zа-яё0-9]+")
# `без ошибок` negates an argument (errors), not the predicate `работает`.
# Predicate polarity and argument restrictions are therefore separate axes.
NEGATION_RE = re.compile(r"(?iu)\b(?:не|нет|нельзя|никогда)\b")
ARGUMENT_NEGATION_RE = re.compile(r"(?iu)\bбез\s+([a-zа-яё0-9_-]+)")
CONDITION_RE = re.compile(r"(?iu)\b(?:если|когда|после|перед|пока|при|до тех пор)\b")
MODALITY = {
    "asserted": "certain", "committed": "certain", "certain": "certain",
    "tentative": "possible", "possible": "possible", "hypothetical": "hypothetical",
    "uncertain": "unknown", "unknown": "unknown", "probable": "probable",
}
CONTENT_MAP = {kind: kind for kind in (
    "observation", "current_state", "problem", "definition", "metric",
    "experimental_result", "hypothesis", "proposal", "alternative", "decision",
    "action", "goal", "target", "constraint", "assumption", "trading_rule",
    "system_rule", "design_choice", "dataset", "resource", "risk", "dependency",
    "blocker", "follow_up", "correction", "rejected_option", "schedule", "question",
)}


def _clean(value):
    return " ".join(TOKEN_RE.findall(str(value or "").casefold()))


def normalize_entity(raw):
    if isinstance(raw, str):
        return {"entity_id": "ENT" + hashlib.sha256(_clean(raw).encode()).hexdigest()[:12], "canonical_name": raw, "type": "unknown", "aliases": []}
    name = raw.get("canonical_name") or raw.get("name") or raw.get("text") or raw.get("entity_id") or "unknown"
    return {"entity_id": raw.get("entity_id") or "ENT" + hashlib.sha256(_clean(name).encode()).hexdigest()[:12], "canonical_name": name, "type": raw.get("type", "unknown"), "aliases": list(raw.get("aliases", []))}


def normalize_quantity(raw, statement=""):
    if not isinstance(raw, dict):
        raw = {"value": raw}
    normalized = raw.get("normalized") if isinstance(raw.get("normalized"), dict) else {}
    value = raw.get("value") if raw.get("value") is not None else normalized.get("value")
    return {
        "value": value, "unit": raw.get("unit") or normalized.get("unit"), "entity_id": raw.get("entity_id") or raw.get("entity"),
        "role": raw.get("role") or raw.get("kind"), "operator": raw.get("operator") or normalized.get("operator") or "unknown",
        "direction": raw.get("direction") or normalized.get("direction"), "source_span": raw.get("source_span") or raw.get("raw_text") or str(value or ""),
        "evidence_ids": list(raw.get("evidence_ids", [])),
    }


def normalize_condition(raw):
    if isinstance(raw, dict):
        return {"condition_id": raw.get("condition_id"), "antecedent": raw.get("antecedent") or raw.get("predicate") or raw.get("text"), "consequent": raw.get("consequent") or raw.get("effect"), "relation": raw.get("relation", "if_then"), "evidence_ids": list(raw.get("evidence_ids", []))}
    return {"condition_id": None, "antecedent": str(raw), "consequent": None, "relation": "if_then", "evidence_ids": []}


def proposition_signature(record, registry=None):
    conditions = [normalize_condition(x) for x in record.get("conditions", [])]
    quantities = [normalize_quantity(x, record.get("statement")) for x in record.get("quantities", [])]
    entities = []
    for raw in record.get("entities", []):
        normalized = normalize_entity(raw)
        if registry is not None:
            candidates = registry.resolve_candidates(normalized["canonical_name"], normalized["type"])
            resolved = registry.resolve(normalized["canonical_name"], normalized["type"])
            if resolved:
                normalized = resolved
            elif len(candidates) > 1:
                normalized = {
                    **normalized,
                    "entity_id": "AMB" + hashlib.sha256(_clean(normalized["canonical_name"]).encode()).hexdigest()[:12],
                    "ambiguous_candidate_ids": sorted(item["entity_id"] for item in candidates),
                }
            else:
                normalized = registry.register(normalized["canonical_name"], normalized["type"], normalized["aliases"], normalized.get("entity_id"))
        entities.append(dict(normalized))
    statement = record.get("statement") or ""
    raw_kind = record.get("kind") or record.get("content_kind")
    signature = {
        "subject": _clean(record.get("subject")), "predicate": _clean(record.get("predicate")),
        "object": _clean(record.get("object")), "scope": record.get("scope") or {},
        "conditions": [{"antecedent": _clean(x["antecedent"]), "consequent": _clean(x["consequent"])} for x in conditions],
        "polarity": record.get("polarity") or ("negative" if NEGATION_RE.search(statement) else "positive"),
        "quantities": [{k: x.get(k) for k in ("value", "unit", "entity_id", "role", "operator", "direction")} for x in quantities],
        "time_scope": record.get("time_scope") or record.get("time_expression"),
        "entities": sorted(x["entity_id"] for x in entities),
        "negated_arguments": sorted(_clean(value) for value in ARGUMENT_NEGATION_RE.findall(statement)),
    }
    subject = _clean(record.get("subject"))
    participant_bound = raw_kind in {"action", "follow_up", "resource"} or subject in {"я", "i", "мне", "мы", "we"}
    if participant_bound:
        signature["actor"] = sorted(set(record.get("assignees", []) or record.get("attributed_speakers", []) or record.get("speaker_refs", [])))
    if raw_kind in {"hypothesis", "experimental_result"}:
        signature["epistemic_kind"] = raw_kind
    if not any((signature["subject"], signature["predicate"], signature["object"])):
        signature["lexical_fallback"] = _clean(statement)
    return signature, entities, quantities, conditions


def proposition_from_record(record, registry=None):
    signature, entities, quantities, conditions = proposition_signature(record, registry)
    digest = hashlib.sha256(json.dumps(signature, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    # `kind` is the lossless canonical axis. `content_kind` is accepted only
    # for records produced by the new schema that omit the legacy field.
    raw_kind = record.get("kind") or record.get("content_kind")
    content_kind = require_content_kind(CONTENT_MAP.get(raw_kind, raw_kind))
    return {
        "proposition_id": "P" + digest[:16], "semantic_signature": signature,
        "subject": record.get("subject"), "predicate": record.get("predicate"), "object": record.get("object"),
        "statement": record.get("statement") or "", "content_kind": content_kind,
        "claim_kind": record.get("kind") or content_kind,
        "entities": entities, "quantities": quantities, "conditions": conditions,
        "polarity": signature["polarity"], "scope": signature["scope"], "time_scope": signature["time_scope"],
        "evidence_ids": list(record.get("evidence_ids", [])), "source_record_ids": [record.get("record_id")],
    }


def epistemic_modality(record):
    return MODALITY.get(str(record.get("epistemic_modality") or record.get("modality") or "unknown").casefold(), "unknown")
