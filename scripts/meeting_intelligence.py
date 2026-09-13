#!/usr/bin/env python3
"""Global dialogue resolution and utility-driven public meeting views.

The evidence registry is deliberately complete.  This module builds a smaller,
task-oriented view without deleting or rewriting any source claim.
"""
from __future__ import annotations

import hashlib
import json
import math
import re


QUESTION_STATES = {
    "answered", "partially_answered", "tentatively_answered", "unanswered",
    "deferred", "requires_external_verification", "superseded", "rhetorical",
    "misrecognized_question",
}

NOISE_RE = re.compile(
    r"(?iu)^\s*(?:угу|ага|понятно|хорошо|ясно|ладно|да|нет|ок(?:ей)?|"
    r"спасибо|слышно|видно)(?:[\s.,!?…]+(?:угу|ага|понятно|хорошо|ясно|"
    r"ладно|да|нет|ок(?:ей)?|спасибо))*[\s.,!?…]*$"
)
BANTER_RE = re.compile(
    r"(?iu)(?:когда\s+dow\s+jones\s+появ|начал[оа]?\s+(?:xx|xvii|\d{1,2})\s+век|"
    r"я\s+bitcoin\s+\d{4}\s+года\s+тебе\s+дам)"
)
FALSIFIABLE_RE = re.compile(
    r"(?iu)(?:если|может|должн|вероят|гипотез|предполож|провер|тест|сравн|"
    r"улучш|ухудш|повыс|сниз|уменьш|увелич|влияет|эффект|даст|фильтр)"
)
ACTIONABLE_RE = re.compile(
    r"(?iu)(?:провер|сдела|подготов|предостав|переда|реализ|исправ|добав|"
    r"сравн|запуст|размет|интегр|протест|симуляц|backtest|бэктест)"
)
SPECIFIC_RE = re.compile(r"(?iu)(?:\d|%|\b(?:m1|m15|h1|h4|tp|sl|bos|fbos|tpo|ote)\b)")

TYPE_WEIGHT = {
    "decision": 3.0, "action": 3.0, "experimental_result": 2.7,
    "metric": 2.6, "problem": 2.5, "rule": 2.45, "trading_rule": 2.45,
    "system_rule": 2.45, "proposal": 2.0, "design_choice": 2.0,
    "hypothesis": 1.8, "constraint": 1.7, "definition": 1.5,
    "current_state": 1.5, "goal": 1.4, "schedule": 1.4,
    "observation": 1.0, "question": 0.6,
}


def _tokens(value):
    ignored = {
        "это", "как", "для", "что", "при", "или", "уже", "ещё", "будет",
        "нужно", "надо", "так", "там", "тут", "тогда", "можно", "вот",
        "который", "которая", "которые", "иметь", "имеет", "есть",
    }
    return {
        item for item in re.findall(r"(?iu)[a-zа-яё0-9]+", str(value or "").casefold())
        if len(item) > 2 and item not in ignored
    }


def _overlap(left, right):
    a, b = _tokens(left), _tokens(right)
    return len(a & b) / max(1, min(len(a), len(b)))


def question_candidate_bundles(records, utterances=None, max_delay=900.0, max_candidates=16):
    """Retrieve local and episode-level candidates without requiring topic equality."""
    ordered = sorted(records, key=lambda item: (float(item.get("start", 0)), item.get("record_id", "")))
    bundles = []
    for index, question in enumerate(ordered):
        if question.get("kind") != "question":
            continue
        q_start = float(question.get("start", 0))
        candidates = []
        for offset, candidate in enumerate(ordered[index + 1:index + 61], 1):
            delay = float(candidate.get("start", 0)) - q_start
            if delay < 0 or delay > max_delay:
                continue
            if candidate.get("kind") == "question" and offset > 3:
                continue
            lexical = _overlap(
                " ".join(str(question.get(k) or "") for k in ("statement", "subject", "predicate", "object")),
                " ".join(str(candidate.get(k) or "") for k in ("statement", "subject", "predicate", "object")),
            )
            adjacency = 1.0 / (1.0 + offset)
            time_score = math.exp(-delay / 240.0)
            typed = 1.0 if question.get("subject") and question.get("subject") == candidate.get("subject") else 0.0
            answer_act = 1.0 if candidate.get("speech_act") in {"answer", "assert", "accept", "reject", "correct", "decide"} else 0.0
            score = 2.2 * adjacency + 1.2 * time_score + 1.5 * lexical + typed + 0.35 * answer_act
            # Always retain the immediate conversational window.  Real answers
            # often use no words from the question ("Тут есть два подхода...").
            if offset <= 12 or delay <= 90 or score >= 1.25:
                candidates.append((score, candidate))
        candidates.sort(key=lambda value: (-value[0], float(value[1].get("start", 0))))
        raw_candidates = []
        for turn in utterances or []:
            delay = float(turn.get("start", 0)) - q_start
            if delay < 0 or delay > max_delay:
                continue
            # The first 20 turns/90 seconds are preserved regardless of lexical
            # overlap; later candidates must share semantic surface terms.
            lexical = _overlap(question.get("statement"), turn.get("text"))
            if delay <= 90 or lexical >= 0.18:
                raw_candidates.append(turn)
            if len(raw_candidates) >= 24:
                break
        bundles.append({
            "question": question,
            "candidates": [item for _, item in candidates[:max_candidates]],
            "utterance_candidates": raw_candidates,
        })
    return bundles


def apply_question_resolutions(records, resolutions, utterances=None):
    """Apply only candidate-grounded global resolutions; fail closed otherwise."""
    result = [dict(item) for item in records]
    by_id = {item.get("record_id"): item for item in result}
    utterance_by_id = {item.get("id"): item for item in (utterances or []) if item.get("id")}
    for raw in resolutions:
        if not isinstance(raw, dict):
            continue
        question = by_id.get(raw.get("question_record_id"))
        if not question or question.get("kind") != "question":
            continue
        state = raw.get("status")
        if state not in QUESTION_STATES:
            continue
        answer_ids = [
            value for value in raw.get("answer_record_ids", [])
            if value in by_id and value != question.get("record_id")
            and float(by_id[value].get("start", 0)) >= float(question.get("start", 0))
        ]
        answer_evidence_ids = [
            value for value in raw.get("answer_evidence_ids", [])
            if value in utterance_by_id
            and float(utterance_by_id[value].get("start", 0)) >= float(question.get("start", 0))
        ]
        if state in {"answered", "partially_answered", "tentatively_answered"} and not (answer_ids or answer_evidence_ids):
            continue
        question["question_status"] = state
        question["answer_record_ids"] = list(dict.fromkeys(answer_ids))
        question["answer_evidence_ids"] = list(dict.fromkeys(answer_evidence_ids))
        question["answer_spans"] = [
            {
                "id": value,
                "start": float(utterance_by_id[value].get("start", 0)),
                "end": float(utterance_by_id[value].get("end", utterance_by_id[value].get("start", 0))),
                "speaker": utterance_by_id[value].get("speaker"),
                "text": utterance_by_id[value].get("text"),
                "source_word_ids": list(utterance_by_id[value].get("source_word_ids", [])),
            }
            for value in answer_evidence_ids
        ]
        question["answer_relation"] = {
            "answered": "answers", "partially_answered": "partially_answers",
            "tentatively_answered": "tentatively_answers",
        }.get(state)
        question["answer_resolution_basis"] = "global_dialogue_resolver"
        question["resolution_confidence"] = max(0.0, min(1.0, float(raw.get("confidence", 0.0) or 0.0)))
        question["resolution_reason"] = str(raw.get("reason_code") or "GLOBAL_DIALOGUE_RESOLUTION")
    return result


def is_noise(fact):
    text = str(fact.get("statement") or "").strip()
    return not text or bool(NOISE_RE.search(text)) or bool(BANTER_RE.search(text))


def valid_hypothesis(fact):
    if fact.get("type") != "hypothesis" or is_noise(fact):
        return False
    text = str(fact.get("statement") or "")
    return bool(FALSIFIABLE_RE.search(text) and (_tokens(text) - {"гипотеза"}))


def salience_score(fact, semantic_record=None, relation_count=0):
    """Transparent initial utility heuristic; gold labels can replace its weights."""
    semantic_record = semantic_record or {}
    kind = semantic_record.get("content_kind") or fact.get("type")
    score = TYPE_WEIGHT.get(kind, TYPE_WEIGHT.get(fact.get("type"), 0.8))
    text = str(fact.get("statement") or "")
    if ACTIONABLE_RE.search(text):
        score += 0.7
    if SPECIFIC_RE.search(text):
        score += 0.45
    if len(_tokens(text)) >= 8:
        score += 0.25
    score += min(0.75, relation_count * 0.2)
    uncertainty = fact.get("uncertainty", {})
    if uncertainty.get("needs_review"):
        score -= 1.2
    if is_noise(fact):
        score -= 5.0
    if fact.get("type") == "hypothesis" and not valid_hypothesis(fact):
        score -= 4.0
    if fact.get("type") == "question" and semantic_record.get("question_status") in {
        "answered", "partially_answered", "tentatively_answered", "rhetorical", "superseded",
    }:
        score += 1.6 if semantic_record.get("question_status") == "answered" else 0.8
    return round(score, 4)


def build_summary_plan(facts, state, max_units=32, max_chapters=12):
    """Select a public view while preserving the complete registry separately."""
    record_by_id = {
        item.get("source_record_id"): item for item in state.get("events", [])
    }
    relation_counts = {}
    for relation in state.get("relations", []):
        for event_id in (relation.get("source_event"), relation.get("target_event")):
            relation_counts[event_id] = relation_counts.get(event_id, 0) + 1
    ranked = []
    for fact in facts:
        record = record_by_id.get(fact.get("fact_id"), {})
        score = salience_score(fact, record, relation_counts.get(record.get("event_id"), 0))
        ranked.append((score, fact))

    mandatory_types = {"decision", "action"}
    open_question_ids = {
        item.get("source_record_id") for item in state.get("views", {}).get("questions", [])
        if item.get("state") in {"unanswered", "deferred", "requires_external_verification"}
    }
    selected = [
        fact for score, fact in ranked
        if score >= 1.55 or fact.get("type") in mandatory_types
        or fact.get("fact_id") in open_question_ids
    ]
    selected.sort(key=lambda item: (-salience_score(item, record_by_id.get(item.get("fact_id"), {})), float(item.get("start", 0))))
    selected = selected[:max_units]

    # Preserve answer evidence for selected questions and valuable definitions/rules.
    selected_ids = {item.get("fact_id") for item in selected}
    for question in state.get("views", {}).get("questions", []):
        if question.get("source_record_id") not in selected_ids:
            continue
        selected_ids.update(question.get("answer_record_ids", []))

    public = [item for item in facts if item.get("fact_id") in selected_ids and not is_noise(item)]
    chapter_ranked = sorted(
        public,
        key=lambda item: (-salience_score(item, record_by_id.get(item.get("fact_id"), {})), float(item.get("start", 0))),
    )
    chapters, used_topics, used_tokens = [], set(), []
    for fact in chapter_ranked:
        topic = str(fact.get("topic") or "").casefold().strip()
        tokens = _tokens(fact.get("statement"))
        redundant = any(len(tokens & prior) / max(1, min(len(tokens), len(prior))) >= 0.72 for prior in used_tokens)
        if redundant or (topic and topic in used_topics and len(chapters) >= 8):
            continue
        chapters.append(fact.get("fact_id"))
        used_tokens.append(tokens)
        if topic:
            used_topics.add(topic)
        if len(chapters) >= max_chapters:
            break
    chapters.sort(key=lambda fact_id: float(next(item for item in facts if item.get("fact_id") == fact_id).get("start", 0)))
    return {
        "schema_version": 1,
        "strategy": "utility-driven-v1",
        "selected_fact_ids": [item.get("fact_id") for item in public],
        "chapter_fact_ids": chapters,
        "evidence_fact_count": len(facts),
        "public_fact_count": len(public),
        "scores": {item.get("fact_id"): score for score, item in ranked},
    }


def consolidate_tasks(tasks):
    """Create aggregate task nodes without deleting their atomic source records."""
    ordered = sorted((dict(item) for item in tasks), key=lambda item: (float(item.get("start", 0)), item.get("task_id", "")))
    groups = []
    for task in ordered:
        text = " ".join(str(task.get(key) or "") for key in ("title", "description", "details"))
        owners = tuple(sorted(task.get("assignees", [])))
        matched = None
        for group in reversed(groups):
            anchor = group[0]
            anchor_text = " ".join(str(anchor.get(key) or "") for key in ("title", "description", "details"))
            same_owner = owners and owners == tuple(sorted(anchor.get("assignees", [])))
            close = abs(float(task.get("start", 0)) - float(anchor.get("start", 0))) <= 600
            related = _overlap(text, anchor_text) >= 0.22 or (
                ACTIONABLE_RE.search(text) and ACTIONABLE_RE.search(anchor_text)
                and bool(_tokens(text) & _tokens(anchor_text))
            )
            if same_owner and close and related:
                matched = group
                break
        if matched is None:
            groups.append([task])
        else:
            matched.append(task)

    result = []
    for index, group in enumerate(groups, 1):
        # Prefer the later, more concrete formulation (often the commitment that
        # refines an earlier general request).
        source = max(group, key=lambda item: (len(_tokens(" ".join(str(item.get(k) or "") for k in ("title", "description", "details")))), float(item.get("start", 0))))
        merged = dict(source)
        merged["task_id"] = f"TK{index:05d}"
        merged["source_record_ids"] = list(dict.fromkeys(item.get("source_record_id") for item in group if item.get("source_record_id")))
        merged["source_task_ids"] = list(dict.fromkeys(item.get("task_id") for item in group if item.get("task_id")))
        merged["evidence_ids"] = list(dict.fromkeys(value for item in group for value in item.get("evidence_ids", [])))
        merged["confirmation_evidence_ids"] = list(dict.fromkeys(value for item in group for value in item.get("confirmation_evidence_ids", [])))
        merged["context_fact_ids"] = list(dict.fromkeys(value for item in group for value in (item.get("context_fact_ids", []) or [item.get("source_record_id")]) if value))
        merged["consolidation"] = {"source_count": len(group), "method": "owner-time-semantic"}
        result.append(merged)
    return result


def plan_hash(plan):
    payload = json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
