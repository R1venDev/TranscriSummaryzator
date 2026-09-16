#!/usr/bin/env python3
"""Deterministic speech-act signals shared by extraction and state building."""
from __future__ import annotations

import re


CORRECTION_CUE_RE = re.compile(
    r"(?iu)(?:\bа\s+нет\b|\bнет\s*[,—-]?\s*(?:подожди|постой)\b|\b(?:точнее|вернее|поправка|исправлюсь)\b|"
    r"\bя\s+(?:ошибся|ошиблась)\b|\bне\s+так\b|\bвместо\b|\bне\s+\S+\s*[,—-]+\s*а\b)"
)
COMMITMENT_RE = re.compile(
    r"(?iu)(?:\bя\b[^.!?\n]{0,100}\b(?:буду|сделаю|проверю|подготовлю|возьму|исправлю|отправлю|пришлю|передам|предоставлю|отдам|скину|кину|дам|залью|"
    r"размечу|протестирую|посмотрю|добавлю|соберу|покажу|встрою)\b|\b(?:сделаю|проверю|подготовлю|"
    r"исправлю|отправлю|пришлю|передам|предоставлю|скину|кину|размечу|протестирую|встрою)\b)"
    r"|\b(?:сделает|проверит|подготовит|исправит|отправит|пришл[её]т|передаст|предоставит|"
    r"разметит|протестирует|покажет|собер[её]т|добавит)\b"
)
DECISION_RE = re.compile(r"(?iu)\b(?:решили|решено|договорились|утвердили|фиксируем|оставляем)\b")
PROPOSAL_RE = re.compile(r"(?iu)\b(?:предлагаю|можно|давайте|стоит|имеет смысл|я бы)\b")
ACCEPT_RE = re.compile(
    r"(?iu)^\s*(?:(?:да[\s,!.—-]*)+|ага|угу|ок(?:ей)?|соглас(?:ен|на|ны)?|договорились|"
    r"(?:а[\s,]*)?(?:месяца?[\s,.:—-]*)?ну\s+ладно(?:[\s,]*хорошо)?)\W*(?:дальше\b.*)?$"
)
REJECT_RE = re.compile(r"(?iu)^\s*(?:нет|неа|не соглас(?:ен|на|ны)?)\W*$")
QUESTION_RE = re.compile(r"(?iu)\?|^\s*(?:кто|что|где|когда|почему|как|какой|какая|какие|нужно ли)\b")
ANSWER_RE = re.compile(r"(?iu)^\s*(?:да|нет|потому что|это|там|тогда|нужно|надо|можно)\b")
SCHEDULE_RE = re.compile(
    r"(?iu)(?:\b\d{1,2}[:.]\d{2}\b|\b(?:сегодня|завтра|послезавтра|понедельник|вторник|"
    r"сред[ау]|четверг|пятниц[ау]|суббот[ау]|воскресень[ея]|дедлайн|срок|к следующему разу)\b)"
)


def detect_speech_acts(text: str) -> list[str]:
    text = str(text or "").strip()
    acts = []
    for name, pattern in (
        ("correct", CORRECTION_CUE_RE), ("commit", COMMITMENT_RE),
        ("accept", ACCEPT_RE), ("reject", REJECT_RE), ("ask", QUESTION_RE),
        ("decide", DECISION_RE), ("propose", PROPOSAL_RE), ("schedule", SCHEDULE_RE),
    ):
        if pattern.search(text):
            acts.append(name)
    if not acts and ANSWER_RE.search(text):
        acts.append("answer")
    if not acts:
        acts.append("assert")
    return acts


def primary_speech_act(text: str, kind: str | None = None) -> str:
    acts = detect_speech_acts(text)
    for value in ("correct", "commit", "decide", "accept", "reject", "ask", "propose", "answer"):
        if value in acts:
            return value
    return {
        "question": "ask", "decision": "decide", "proposal": "propose", "action": "commit",
    }.get(str(kind or ""), "assert")


def content_kind(kind: str | None) -> str:
    return {
        "current_state": "state", "observation": "state", "problem": "state",
        "experimental_result": "experimental_result", "definition": "definition",
        "target": "goal", "constraint": "constraint", "assumption": "assumption",
        "trading_rule": "trading_rule", "system_rule": "system_rule",
        "dataset": "resource", "resource": "resource", "design_choice": "design_choice",
        "alternative": "alternative", "risk": "risk", "dependency": "dependency",
        "blocker": "blocker", "follow_up": "task", "correction": "correction",
        "rejected_option": "rejected_option",
        "metric": "metric", "proposal": "rule", "decision": "rule",
        "hypothesis": "rule", "action": "task", "goal": "goal",
        "schedule": "schedule", "question": "question",
    }.get(str(kind or ""), "state")


def modality_axis(value: str | None, certainty: str | None = None) -> str:
    return {
        "asserted": "certain", "committed": "certain", "question": "possible",
        "proposed": "possible", "tentative": "speculative", "conditional": "conditional",
        "certain": "certain", "probable": "probable", "possible": "possible",
        "speculative": "speculative",
    }.get(str(value or ""), "speculative" if certainty == "tentative" else "certain")
