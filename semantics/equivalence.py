"""Typed semantic equivalence used by selection and rendering.

Lexical overlap is only a final hint.  Opposite polarity, different bound
quantities, additional conditions, lifecycle revisions, actors, or action
objects always keep two claims distinct.
"""
from __future__ import annotations

import re


TOKEN_RE = re.compile(r"(?iu)[a-zа-яё0-9]+")


def _tokens(value):
    # Negation and one/two-character quantities are intentionally retained.
    return set(TOKEN_RE.findall(str(value or "").casefold()))


def _quantity_signature(item):
    return tuple(sorted(
        tuple(str(quantity.get(key)) for key in (
            "value", "unit", "dimension", "object_binding", "entity_id",
            "role", "operator", "direction", "metric_status",
        ))
        for quantity in item.get("quantities", []) if isinstance(quantity, dict)
    ))


def _condition_signature(item):
    return tuple(sorted(
        (str(condition.get("antecedent") or condition.get("predicate") or condition.get("text") or "").casefold(),
         str(condition.get("consequent") or condition.get("effect") or "").casefold())
        for condition in item.get("conditions", []) if isinstance(condition, dict)
    ))


def equivalent(left, right, threshold=.82):
    """Conservatively decide whether one presentation can suppress another."""
    for field in ("polarity", "lifecycle", "temporal_state", "commitment_state", "decision_status", "task_status"):
        a, b = left.get(field), right.get(field)
        if a and b and a != b:
            return False
    for field in ("assignee", "commitment_actor", "grammatical_actor", "recipient"):
        a, b = left.get(field), right.get(field)
        if a and b and a != b:
            return False
    if _quantity_signature(left) != _quantity_signature(right):
        if _quantity_signature(left) or _quantity_signature(right):
            return False
    if _condition_signature(left) != _condition_signature(right):
        if _condition_signature(left) or _condition_signature(right):
            return False
    a = _tokens(left.get("value") or left.get("text") or left.get("statement"))
    b = _tokens(right.get("value") or right.get("text") or right.get("statement"))
    return bool(a and b and len(a & b) / max(1, min(len(a), len(b))) >= threshold)
