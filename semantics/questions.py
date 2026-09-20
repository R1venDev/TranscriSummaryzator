"""Separate answer retrieval from typed answer verification.

``aspect`` is deliberately free text.  It is useful for readers and retrieval,
but never defines whether an answer is complete.  Completeness is decided by a
small answer-type vocabulary so renaming a model-generated slot cannot reopen
an already answered question.
"""
from __future__ import annotations
import re


ANSWER_TYPES = {
    "boolean", "entity", "time", "quantity", "explanation",
    "implementation_state", "action_choice", "free_text",
}


def infer_answer_type(slot, question=""):
    """Map an arbitrary aspect label to a stable, domain-neutral answer type."""
    value = re.sub(r"[_\s]+", " ", str(slot or "").strip().casefold())
    source = f"{value} {question}".casefold()
    if value in ANSWER_TYPES:
        return value
    if re.search(r"(?iu)\b(?:yes.?no|boolean|ли|should|whether|возможн\w*|решени\w*|closing|feasibility)\b", source):
        return "boolean"
    if re.search(r"(?iu)\b(?:кто|кому|чей|actor|owner|assignee|исполнител\w*)\b", source):
        return "entity"
    if re.search(r"(?iu)\b(?:когда|врем\w*|день|дат\w*|time|schedule|closing)\b", source):
        return "time"
    if re.search(r"(?iu)\b(?:сколько|числ\w*|количеств\w*|порог\w*|размер\w*|quantity|number|threshold|range)\b", source):
        return "quantity"
    if re.search(r"(?iu)\b(?:почему|зачем|причин\w*|объясн\w*|reason|explanation)\b", source):
        return "explanation"
    if re.search(r"(?iu)\b(?:статус|готов\w*|реализ\w*|работа\w*|результат\w*|implementation|state|status|result|success)\b", source):
        return "implementation_state"
    if re.search(r"(?iu)\b(?:что\s+делать|какой\s+вариант|действи\w*|commitment|choice|next step)\b", source):
        return "action_choice"
    return "free_text"


def normalize_slot(slot):
    return infer_answer_type(slot)


def infer_requested_slots(question):
    """Recover a narrow typed slot when extraction omitted an obvious one."""
    text = str(question or "")
    if re.search(r"(?iu)\b(?:есть|имеется|существует)\s+ли\b[^?.]{0,80}\b(?:проблем\w*\s+с\s+)?задерж\w*\b", text):
        return ["implementation_status"]
    if re.search(r"(?iu)\b(?:проблем\w*\s+с\s+задерж\w*|задерж\w*)\b[^?.]{0,50}\b(?:есть|имеется|существует)\b", text):
        return ["implementation_status"]
    return []


def _tokens(value):
    return {x for x in re.findall(r"(?iu)[a-zа-яё0-9]+", str(value or "").casefold()) if len(x) > 2}


def _stems(value):
    return {token[:5] for token in _tokens(value)}


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


def _value_supported(value, text, answer_type):
    """Return whether a structured value is independently visible in source text."""
    rendered = str(value).strip()
    if not rendered or not text.strip():
        return False
    if answer_type == "boolean":
        text_value = "нет" if re.match(r"(?iu)^\s*(?:нет|неа|нельзя|не\s+следует)\b", text) else "да" if re.match(r"(?iu)^\s*(?:(?:да[\s,!.—-]*)+|ага|угу|можно|следует|ну\s+ладно)\b", text) else None
        requested = "нет" if rendered.casefold() in {"нет", "no", "false", "0"} else "да" if rendered.casefold() in {"да", "yes", "true", "1"} else None
        return bool(text_value and requested and text_value == requested)
    if answer_type == "quantity":
        wanted = re.findall(r"(?<!\w)-?\d+(?:[.,]\d+)?", rendered)
        found = re.findall(r"(?<!\w)-?\d+(?:[.,]\d+)?", text)
        return bool(wanted and set(x.replace(",", ".") for x in wanted) <= set(x.replace(",", ".") for x in found))
    if answer_type == "time":
        wanted = re.findall(r"\d{1,2}(?::\d{2})?", rendered)
        found = re.findall(r"\d{1,2}(?::\d{2})?", text)
        return bool(wanted and set(wanted) <= set(found))
    wanted = _tokens(rendered)
    return bool(wanted and len(wanted & _tokens(text)) / len(wanted) >= .8)


def verify_slot_entailment(requested_slots, answer, question=None):
    """Verify values against the spoken text; structured JSON is never evidence."""
    structured = dict(answer.get("slots", {}))
    text = str(answer.get("statement") or answer.get("text") or "")
    question_text = str((question or {}).get("statement") or (question or {}).get("text") or "")
    topics = _tokens(question_text) - {"какие", "какой", "какая", "сколько", "будет", "нужно", "можно", "вопрос", "ответ", "есть", "этого"}
    relevant = not topics or bool(_stems(question_text) & _stems(text)) or bool(structured)
    numbers = re.findall(r"(?<!\w)\d+(?:[.,]\d+)?", text)
    entailed = {}
    contradictions = []
    answer_types = {}
    for original_slot in requested_slots:
        slot = infer_answer_type(original_slot, question_text)
        answer_types[original_slot] = slot
        structured_value = structured.get(original_slot, structured.get(slot))
        if structured_value not in (None, "", []) and _value_supported(structured_value, text, slot):
            entailed[original_slot] = structured_value
        elif structured_value not in (None, "", []) and text.strip():
            contradictions.append(original_slot)
        elif slot == "quantity" and numbers and relevant:
            entailed[original_slot] = numbers[0] if len(numbers) == 1 else text
        elif slot == "time" and (relevant or answer.get("speech_act") == "answer") and (
            re.search(r"(?iu)\b(?:exact\s*time|точн\w*\s+врем|^time$|час)\b", str(original_slot))
            and re.search(r"\b\d{1,2}:\d{2}\b", text)
            or re.search(r"(?iu)\b(?:day|date|день|дат\w*)\b", str(original_slot))
            and re.search(r"(?iu)\b(?:сегодня|завтра|понедельник|вторник|сред\w*|четверг|пятниц\w*|суббот\w*|воскресень\w*|\d{1,2}[./]\d{1,2})\b", text)
            or not re.search(r"(?iu)\b(?:exact\s*time|точн\w*\s+врем|^time$|час|day|date|день|дат\w*)\b", str(original_slot))
            and re.search(r"(?iu)(?:\b\d{1,2}(?::\d{2})?\b|\b(?:сегодня|завтра|понедельник|вторник|сред\w*|четверг|пятниц\w*|суббот\w*|воскресень\w*)\b)", text)
        ):
            entailed[original_slot] = text
        elif (slot == "boolean" and
              re.match(r"(?iu)^\s*(?:нет|неа|(?:да[\s,!.—-]*)+|ага|угу|ну\s+ладно)\b", text) and
              answer.get("speech_act") in {"answer", "accept", "reject"}):
            entailed[original_slot] = "нет" if re.search(r"(?iu)\b(?:нет|неа)\b", text) else "да"
        elif (slot == "boolean" and answer.get("speech_act") == "answer" and
              re.search(r"(?iu)\b(?:не\s+следует|нельзя|следует|можно)\b", text)):
            entailed[original_slot] = "нет" if re.search(r"(?iu)\b(?:не\s+следует|нельзя)\b", text) else "да"
        elif slot == "boolean" and answer.get("speech_act") == "answer" and relevant and text.strip():
            entailed[original_slot] = "нет" if re.search(r"(?iu)\b(?:не|нет|нельзя)\b", text) else "да"
        elif slot == "implementation_state" and re.search(r"(?iu)\b(?:работа\w*|работал\w*|готов\w*|получил\w*|получен\w*|результат\w*|не\s+сработ\w*|неуспеш\w*|реализ\w*|задерж\w*|не\s+закончен\w*)\b", text):
            entailed[original_slot] = text
        elif slot == "explanation" and relevant and re.search(r"(?iu)\b(?:потому|из-за|причин\w*|возможно|гипотез\w*|чтобы)\b", text):
            entailed[original_slot] = text
        elif slot == "entity" and relevant and re.search(r"@?[A-ZА-ЯЁ][\w.-]+", text):
            entailed[original_slot] = text
        elif slot == "action_choice" and relevant and re.search(r"(?iu)\b(?:сдела\w*|подготов\w*|провер\w*|переда\w*|попроб\w*|выбер\w*|буду\w*)\b", text):
            entailed[original_slot] = text
        elif slot == "free_text" and relevant and answer.get("speech_act") in {"answer", "assert"} and len(_tokens(text)) >= 2:
            entailed[original_slot] = text
    missing = [x for x in requested_slots if x not in entailed]
    return {
        "entailed_slots": entailed, "missing_slots": missing,
        "answer_types": answer_types, "contradicted_slots": contradictions,
        "verification_status": "contradicted" if contradictions else "supported" if requested_slots and not missing else "insufficient_evidence",
        "passed": bool(requested_slots) and not missing and not contradictions,
    }
