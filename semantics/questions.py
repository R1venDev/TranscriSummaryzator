"""Separate answer retrieval from slot-level entailment verification."""
from __future__ import annotations
import re


SLOT_ALIASES = {
    "closing": "yes_no", "decision": "yes_no", "possibility": "yes_no",
    "диапазон времени": "time_range", "exact_time": "exact_time",
    "time alternatives": "time_alternatives", "варианты времени": "time_alternatives",
    "should_misha_make_orderblock_labeler": "actor_commitment",
    "rhythmic entry success": "implementation_status",
    "rhythmic_entry_implementation": "implementation_status",
    "high_tf_result": "implementation_status", "result on higher timeframes": "implementation_status",
    "definition small imbalance": "threshold_value",
    "reversal zone bos condition": "reason_hypothesis",
    "cross day closure feasibility": "cross_day_closure",
    "cross_day_closure_feasibility": "cross_day_closure",
}


def normalize_slot(slot):
    value = re.sub(r"[_\s]+", " ", str(slot or "").strip().casefold())
    return SLOT_ALIASES.get(value, SLOT_ALIASES.get(str(slot or "").strip().casefold(), value.replace(" ", "_")))


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


def verify_slot_entailment(requested_slots, answer, question=None):
    """Conservative verifier: only structured or directly observable slots count."""
    structured = dict(answer.get("slots", {}))
    text = str(answer.get("statement") or answer.get("text") or "")
    question_text = str((question or {}).get("statement") or (question or {}).get("text") or "")
    topics = _tokens(question_text) - {"какие", "какой", "какая", "сколько", "будет", "нужно", "можно", "вопрос", "ответ", "есть", "этого"}
    relevant = not topics or bool(topics & _tokens(text)) or bool(structured)
    numbers = re.findall(r"(?<!\w)\d+(?:[.,]\d+)?", text)
    entailed = {}
    for original_slot in requested_slots:
        slot = normalize_slot(original_slot)
        structured_value = structured.get(original_slot, structured.get(slot))
        if structured_value not in (None, "", []):
            entailed[original_slot] = structured_value
        elif slot in {"number", "quantity", "number_of_trades", "threshold_value"} and len(numbers) == 1 and relevant and (
                slot not in {"number_of_trades", "threshold_value"} or re.search(r"(?iu)\b(?:сделк\w*|вход\w*|позици\w*|порог\w*|размер\w*|ширин\w*)\b", text)):
            entailed[original_slot] = numbers[0]
        elif slot == "time_range" and (relevant or answer.get("speech_act") == "answer") and re.search(r"(?iu)\b\d{1,2}(?::\d{2})?\s*(?:[-–—]|до)\s*\d{1,2}(?::\d{2})?\b", text):
            entailed[original_slot] = text
        elif slot == "time_alternatives" and (relevant or answer.get("speech_act") == "answer") and re.search(r"(?iu)\b\d{1,2}(?::\d{2})?\s*(?:или|либо)\s*\d{1,2}(?::\d{2})?\b", text):
            entailed[original_slot] = text
        elif slot == "exact_time" and (relevant or answer.get("speech_act") == "answer") and re.fullmatch(r"(?iu)\s*(?:в\s+)?\d{1,2}(?::\d{2})?\s*", text):
            entailed[original_slot] = text
        elif (slot in {"yes_no", "actor_commitment"} and
              re.match(r"(?iu)^\s*(?:нет|неа|(?:да[\s,!.—-]*)+|ага|угу|ну\s+ладно)\b", text) and
              answer.get("speech_act") in {"answer", "accept", "reject"}):
            entailed[original_slot] = "нет" if re.search(r"(?iu)\b(?:нет|неа)\b", text) else "да"
        elif slot == "implementation_status" and relevant and re.search(r"(?iu)\b(?:работа\w*|готов\w*|получил\w*|получен\w*|результат\w*|не\s+сработ\w*|неуспеш\w*|реализ\w*|задерж\w*|не\s+закончен\w*)\b", text):
            entailed[original_slot] = text
        elif slot == "reason_hypothesis" and relevant and re.search(r"(?iu)\b(?:потому|из-за|причин\w*|возможно|гипотез\w*)\b", text):
            entailed[original_slot] = text
        elif slot == "cross_day_closure" and re.search(r"(?iu)\b(?:автоматическ\w*\s+)?закрыва\w*\b", text) and re.search(r"(?iu)(?:00\s*:\s*00|полуноч\w*|следующ\w*\s+д(?:ень|ня))", text):
            entailed[original_slot] = text
        elif original_slot == "closing" and relevant and re.search(r"(?iu)\b(?:закрыва\w*|закро\w*)\b", text):
            entailed[original_slot] = text
    return {"entailed_slots": entailed, "missing_slots": [x for x in requested_slots if x not in entailed], "passed": bool(requested_slots) and len(entailed) == len(requested_slots)}
