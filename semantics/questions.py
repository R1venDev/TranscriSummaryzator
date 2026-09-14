"""Separate answer retrieval from slot-level entailment verification."""
from __future__ import annotations
import re


def _tokens(value):
    return {x for x in re.findall(r"(?iu)[a-zа-яё0-9]+", str(value or "").casefold()) if len(x) > 2}


def retrieve_answer_candidates(question, events, window_seconds=180.0, limit=8):
    """Broad, recall-oriented temporal and lexical retrieval; never resolves a slot."""
    start = float(question.get("timestamp", question.get("start", 0)))
    query = _tokens(question.get("statement") or question.get("text"))
    ranked = []
    for event in events:
        timestamp = float(event.get("timestamp", event.get("start", 0)))
        if timestamp < start or timestamp - start > window_seconds:
            continue
        text = event.get("statement") or event.get("text") or ""
        overlap = len(query & _tokens(text)) / max(1, len(query))
        speech = 1.0 if event.get("speech_act") in {"answer", "assert"} else .25
        ranked.append((.55 * speech + .30 * overlap + .15 * (1 - (timestamp - start) / window_seconds), event))
    return [item for _, item in sorted(ranked, key=lambda x: (-x[0], float(x[1].get("timestamp", x[1].get("start", 0)))))[:limit]]


def verify_slot_entailment(requested_slots, answer):
    """Conservative verifier: only structured or directly observable slots count."""
    structured = dict(answer.get("slots", {}))
    text = str(answer.get("statement") or answer.get("text") or "")
    numbers = re.findall(r"(?<!\w)\d+(?:[.,]\d+)?", text)
    entailed = {}
    for slot in requested_slots:
        if slot in structured and structured[slot] not in (None, "", []):
            entailed[slot] = structured[slot]
        elif slot in {"number", "quantity", "number_of_trades"} and len(numbers) == 1:
            entailed[slot] = numbers[0]
    return {"entailed_slots": entailed, "missing_slots": [x for x in requested_slots if x not in entailed], "passed": bool(requested_slots) and len(entailed) == len(requested_slots)}
