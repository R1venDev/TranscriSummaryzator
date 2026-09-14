"""Canonical proposition identity, entities, quantities and conditions."""
from __future__ import annotations
import hashlib
import json
import re

TOKEN_RE = re.compile(r"(?iu)[a-zа-яё0-9]+")
NEGATION_RE = re.compile(r"(?iu)\b(?:не|нет|нельзя|без|никогда)\b")
CONDITION_RE = re.compile(r"(?iu)\b(?:если|когда|после|перед|пока|при|до тех пор)\b")
MODALITY = {
    "asserted": "certain", "committed": "certain", "certain": "certain",
    "tentative": "possible", "possible": "possible", "hypothetical": "hypothetical",
    "uncertain": "unknown", "unknown": "unknown", "probable": "probable",
}
CONTENT_MAP = {
    "trading_rule": "rule", "system_rule": "rule", "definition": "rule",
    "metric": "metric", "experimental_result": "experiment", "hypothesis": "experiment",
    "design_choice": "design", "proposal": "design", "alternative": "design",
    "problem": "problem", "blocker": "problem", "risk": "problem",
    "resource": "resource", "dataset": "resource", "schedule": "schedule",
    "question": "question_content", "action": "action", "follow_up": "action",
    "current_state": "state", "decision": "state",
}


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
    value = raw.get("value")
    return {
        "value": value, "unit": raw.get("unit"), "entity_id": raw.get("entity_id") or raw.get("entity"),
        "role": raw.get("role") or raw.get("kind"), "operator": raw.get("operator", "eq"),
        "direction": raw.get("direction"), "source_span": raw.get("source_span") or str(value or ""),
        "evidence_ids": list(raw.get("evidence_ids", [])),
    }


def normalize_condition(raw):
    if isinstance(raw, dict):
        return {"condition_id": raw.get("condition_id"), "antecedent": raw.get("antecedent") or raw.get("text"), "consequent": raw.get("consequent"), "relation": raw.get("relation", "if_then"), "evidence_ids": list(raw.get("evidence_ids", []))}
    return {"condition_id": None, "antecedent": str(raw), "consequent": None, "relation": "if_then", "evidence_ids": []}


def proposition_signature(record, registry=None):
    conditions = [normalize_condition(x) for x in record.get("conditions", [])]
    quantities = [normalize_quantity(x, record.get("statement")) for x in record.get("quantities", [])]
    entities = []
    for raw in record.get("entities", []):
        normalized = normalize_entity(raw)
        if registry is not None:
            normalized = registry.resolve(normalized["canonical_name"], normalized["type"]) or registry.register(normalized["canonical_name"], normalized["type"], normalized["aliases"], normalized.get("entity_id"))
        entities.append(dict(normalized))
    statement = record.get("statement") or ""
    signature = {
        "subject": _clean(record.get("subject")), "predicate": _clean(record.get("predicate")),
        "object": _clean(record.get("object")), "scope": record.get("scope") or {},
        "conditions": [{"antecedent": _clean(x["antecedent"]), "consequent": _clean(x["consequent"])} for x in conditions],
        "polarity": record.get("polarity") or ("negative" if NEGATION_RE.search(statement) else "positive"),
        "quantities": [{k: x.get(k) for k in ("value", "unit", "entity_id", "role", "operator")} for x in quantities],
        "time_scope": record.get("time_scope") or record.get("time_expression"),
        "entities": sorted(x["entity_id"] for x in entities),
    }
    if not any((signature["subject"], signature["predicate"], signature["object"])):
        signature["lexical_fallback"] = _clean(statement)
    return signature, entities, quantities, conditions


def proposition_from_record(record, registry=None):
    signature, entities, quantities, conditions = proposition_signature(record, registry)
    digest = hashlib.sha256(json.dumps(signature, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "proposition_id": "P" + digest[:16], "semantic_signature": signature,
        "subject": record.get("subject"), "predicate": record.get("predicate"), "object": record.get("object"),
        "statement": record.get("statement") or "", "content_kind": CONTENT_MAP.get(record.get("kind"), record.get("content_kind") or "other"),
        "entities": entities, "quantities": quantities, "conditions": conditions,
        "polarity": signature["polarity"], "scope": signature["scope"], "time_scope": signature["time_scope"],
        "evidence_ids": list(record.get("evidence_ids", [])), "source_record_ids": [record.get("record_id")],
    }


def epistemic_modality(record):
    return MODALITY.get(str(record.get("epistemic_modality") or record.get("modality") or "unknown").casefold(), "unknown")
