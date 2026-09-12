#!/usr/bin/env python3
"""Shared, deterministic schemas for uncertainty and meeting semantics."""
from __future__ import annotations

import math
import re


SPEAKER_RISK_FLAGS = {"ambiguous", "no_diarization", "low_confidence", "overlap"}
INFERRED_SPEAKER_FLAGS = {"speaker_context", "speaker_smoothed", "clause_coherence"}
ASR_RISK_FLAGS = {"asr_boundary", "asr_alternative", "vocabulary_corrected"}
CONDITION_RE = re.compile(r"(?iu)^\s*(?:если|когда|после|перед|пока|при|до|в случае|при условии|только если|if|when|after|before|until|provided)(?:\W|$)")
NUMBER_WORDS = {
    "1": {"один", "одна", "одно", "одну", "одного"},
    "2": {"два", "две", "двух"},
    "3": {"три", "трёх", "трех"},
    "4": {"четыре", "четырёх", "четырех"},
    "5": {"пять", "пяти"},
    "6": {"шесть", "шести"},
    "7": {"семь", "семи"},
    "8": {"восемь", "восьми"},
    "9": {"девять", "девяти"},
    "10": {"десять", "десяти"},
}

UNRESOLVED_QUESTION_RE = re.compile(
    r"(?iu)(?:нужно уточнить|необходимо уточнить|требует проверки|не установлено|"
    r"определить нельзя|нельзя определить|остался вопрос|нет подтверждения|подтверждения\w*(?:\s+\w+){0,6}\s+нет|"
    r"спрашива(?:ет|ют|л|ли) о возможности)"
)
RESOLVED_QUESTION_RE = re.compile(
    r"(?iu)(?:в ответ|пояснил|пояснила|не рекомендовал|уточнил, что|уточнила, что|"
    r"далее обсуждались|затем предложили|в ответе)"
)
TRANSCRIPT_VERIFICATION_RE = re.compile(
    r"(?iu)(?:по аудио|в аудио|в стенограмме|распозн|реплик\w*\s+(?:оборван|обрыва)|"
    r"окончание фразы|точн\w*\s+формулировк\w*|конкретн\w*\s+значен\w*\s+определить нельзя|"
    r"нельзя уверенно (?:разобрать|восстановить|определить))"
)
DISCUSSION_QUESTION_RE = re.compile(
    r"(?iu)(?:следующ\w*\s+созвон|подтверждения времени.*нет|остался вопрос,?\s+где)"
)


def clamp(value, default=None):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, value)) if math.isfinite(value) else default


def word_uncertainty(word):
    flags = set(word.get("flags", []))
    speaker_confidence = clamp(word.get("speaker_confidence"))
    speaker_reasons = sorted(flags & (SPEAKER_RISK_FLAGS | INFERRED_SPEAKER_FLAGS))
    if word.get("speaker") is None and "no_speaker" not in speaker_reasons:
        speaker_reasons.append("no_speaker")
    recognition_reasons = sorted(flags & ASR_RISK_FLAGS)
    if word.get("vocabulary_corrected") and "vocabulary_corrected" not in recognition_reasons:
        recognition_reasons.append("vocabulary_corrected")
    return {
        "recognition": {
            "confidence": clamp(word.get("asr_confidence")),
            "confidence_source": "model" if word.get("asr_confidence") is not None else "unavailable",
            "reasons": recognition_reasons,
            "chunk_index": word.get("asr_chunk_index"),
        },
        "speaker": {
            "confidence": speaker_confidence,
            "confidence_source": "pipeline_heuristic" if speaker_confidence is not None else "unavailable",
            "reasons": speaker_reasons,
            "inferred": bool(flags & INFERRED_SPEAKER_FLAGS or word.get("speaker_inferred_from_context")),
        },
        "needs_review": bool(recognition_reasons or speaker_reasons or word.get("speaker") is None),
    }


def utterance_uncertainty(words):
    details = [word_uncertainty(word) for word in words]
    count = max(1, len(details))
    speaker_scores = [item["speaker"]["confidence"] for item in details if item["speaker"]["confidence"] is not None]
    recognition_scores = [item["recognition"]["confidence"] for item in details if item["recognition"]["confidence"] is not None]
    speaker_reasons = sorted({reason for item in details for reason in item["speaker"]["reasons"]})
    recognition_reasons = sorted({reason for item in details for reason in item["recognition"]["reasons"]})
    inferred = sum(item["speaker"]["inferred"] for item in details)
    review = sum(item["needs_review"] for item in details)
    return {
        "recognition": {
            "confidence": min(recognition_scores) if recognition_scores else None,
            "confidence_source": "model_minimum" if recognition_scores else "unavailable",
            "reasons": recognition_reasons,
        },
        "speaker": {
            "confidence": min(speaker_scores) if speaker_scores else None,
            "confidence_source": "pipeline_heuristic_minimum" if speaker_scores else "unavailable",
            "reasons": speaker_reasons,
            "inferred_word_ratio": round(inferred / count, 4),
        },
        "review_word_ratio": round(review / count, 4),
        "needs_review": bool(review),
    }


def evidence_uncertainty(evidence):
    uncertain = [item for item in evidence if item.get("uncertainty", {}).get("needs_review")]
    speaker_scores = [
        item.get("uncertainty", {}).get("speaker", {}).get("confidence")
        for item in evidence
    ]
    speaker_scores = [value for value in speaker_scores if value is not None]
    recognition_sources = {
        item.get("uncertainty", {}).get("recognition", {}).get("confidence_source", "unavailable")
        for item in evidence
    }
    reasons = sorted({
        reason
        for item in evidence
        for channel in ("recognition", "speaker")
        for reason in item.get("uncertainty", {}).get(channel, {}).get("reasons", [])
    })
    return {
        "evidence_utterances": len(evidence),
        "uncertain_evidence_utterances": len(uncertain),
        "uncertain_evidence_ratio": round(len(uncertain) / max(1, len(evidence)), 4),
        "speaker_confidence_min": min(speaker_scores) if speaker_scores else None,
        "recognition_confidence_available": bool(evidence) and recognition_sources != {"unavailable"},
        "reasons": reasons,
        "needs_review": bool(uncertain),
    }


def normalize_semantic_record(raw, fact):
    """Normalize model output without accepting references it cannot prove."""
    evidence_ids = list(fact.get("evidence_ids", []))
    evidence_set = set(evidence_ids)
    speakers = {item.get("speaker") for item in fact.get("evidence", []) if item.get("speaker")}
    raw = raw if isinstance(raw, dict) else {}
    confirmation_ids = [value for value in raw.get("confirmation_evidence_ids", []) if value in evidence_set]
    evidence_by_id = {
        item.get("id"): item for item in fact.get("evidence", [])
        if isinstance(item, dict) and item.get("id") in evidence_set
    }
    audited_owners = [value for value in fact.get("owner_refs", []) if value in speakers]
    owners = []
    if fact.get("type") == "action":
        # The semantic model may see several speakers in the evidence.  Only the
        # owner already proven by the action-policy pass may become an assignee.
        owners = audited_owners
        # An assignee is confirmed only by their own utterance.  Agreement from
        # another participant can confirm the plan, but cannot accept work on
        # somebody else's behalf.
        owner_set = set(owners)
        confirmation_ids = [
            evidence_id for evidence_id in confirmation_ids
            if evidence_by_id.get(evidence_id, {}).get("speaker") in owner_set
        ]
    proposed_by = [value for value in raw.get("proposed_by", []) if value in speakers]
    if fact.get("type") not in {"proposal", "action"}:
        proposed_by = []
    conditions = []
    for value in raw.get("conditions", []):
        if not isinstance(value, dict):
            continue
        text = str(value.get("text") or "").strip()
        ids = [item for item in value.get("evidence_ids", []) if item in evidence_set]
        if text and ids and CONDITION_RE.search(text):
            conditions.append({"text": text, "evidence_ids": ids})
    quantities = []
    for value in raw.get("quantities", []):
        if not isinstance(value, dict):
            continue
        ids = [item for item in value.get("evidence_ids", []) if item in evidence_set]
        amount = str(value.get("value") or "").strip()
        digits = re.findall(r"\d+(?:[.,]\d+)?", amount)
        cited_text = " ".join(str(evidence_by_id[item].get("text") or "") for item in ids if item in evidence_by_id).casefold()
        supported = any(
            digit in cited_text or any(word in cited_text.split() for word in NUMBER_WORDS.get(digit, set()))
            for digit in digits
        )
        if amount and ids and digits and supported:
            quantities.append({
                "value": amount,
                "unit": str(value.get("unit") or "").strip() or None,
                "evidence_ids": ids,
            })
    confirmations = [
        {
            "evidence_id": evidence_id,
            "speaker": evidence_by_id[evidence_id].get("speaker"),
            "text": evidence_by_id[evidence_id].get("text"),
        }
        for evidence_id in confirmation_ids
        if evidence_id in evidence_by_id
    ]
    answer_ids = [value for value in raw.get("answer_evidence_ids", []) if value in evidence_set]
    answer_record_ids = list(dict.fromkeys(
        str(value) for value in raw.get("answer_record_ids", [])
        if isinstance(value, str) and value.startswith("F") and value != fact.get("fact_id")
    ))
    question_status = "not_applicable"
    question_kind = "not_applicable"
    answer_resolution_basis = None
    if fact.get("type") == "question":
        statement = str(fact.get("statement") or "")
        question_kind = (
            "transcript_verification"
            if TRANSCRIPT_VERIFICATION_RE.search(statement) and not DISCUSSION_QUESTION_RE.search(statement)
            else "discussion"
        )
        if answer_ids:
            question_status = "resolved"
            answer_resolution_basis = "within_fact_evidence"
        elif raw.get("question_status") == "unresolved" or UNRESOLVED_QUESTION_RE.search(statement):
            question_status = "unresolved"
        elif RESOLVED_QUESTION_RE.search(statement):
            question_status = "resolved"
            answer_resolution_basis = "within_fact_statement"
        else:
            question_status = "unclear"
    uncertainty = dict(fact.get("uncertainty", {}))
    assignment_status = "not_applicable"
    if fact.get("type") == "action":
        if owners and confirmation_ids:
            assignment_status = "confirmed"
        elif owners:
            assignment_status = "unconfirmed"
        else:
            assignment_status = "unknown"
        if assignment_status != "confirmed":
            uncertainty = dict(uncertainty, needs_review=True)
            uncertainty["reasons"] = sorted(set(uncertainty.get("reasons", [])) | {"assignee_not_confirmed"})
    return {
        "record_id": fact["fact_id"],
        "kind": fact["type"],
        "statement": fact["statement"],
        "start": float(fact.get("start", 0)),
        "subject": str(raw.get("subject") or "").strip() or None,
        "predicate": str(raw.get("predicate") or "").strip() or None,
        "object": str(raw.get("object") or "").strip() or None,
        "polarity": "negative" if raw.get("polarity") == "negative" else "positive",
        "modality": raw.get("modality") if raw.get("modality") in {"asserted", "tentative", "proposed", "committed", "question"} else ("tentative" if fact.get("certainty") == "tentative" else "asserted"),
        "conditions": conditions,
        "quantities": quantities,
        "time_expression": (
            str(raw.get("time_expression", {}).get("text") or "").strip()
            if isinstance(raw.get("time_expression"), dict)
            else str(raw.get("time_expression") or "").strip()
        ) or None,
        "attributed_speakers": list(fact.get("speaker_refs", [])),
        "proposed_by": proposed_by,
        "assignees": owners,
        "assignment_status": assignment_status,
        "confirmation_evidence_ids": confirmation_ids,
        "confirmation_utterances": confirmations,
        "question_status": question_status,
        "question_kind": question_kind,
        "answer_evidence_ids": answer_ids,
        "answer_record_ids": answer_record_ids,
        "answer_resolution_basis": answer_resolution_basis,
        "evidence_ids": evidence_ids,
        "uncertainty": uncertainty,
    }


def task_records(records):
    result = []
    for record in sorted(records, key=lambda item: (float(item.get("start", 0)), item.get("record_id", ""))):
        if record["kind"] != "action":
            continue
        automation_eligible = (
            record["assignment_status"] == "confirmed"
            and not record.get("uncertainty", {}).get("needs_review")
        )
        result.append({
            "task_id": "T" + record["record_id"][1:],
            "source_record_id": record["record_id"],
            "description": record["statement"],
            "start": float(record.get("start", 0)),
            "proposed_by": record["proposed_by"],
            "assignees": record["assignees"],
            "assignment_status": record["assignment_status"],
            "automation_eligible": automation_eligible,
            "conditions": record["conditions"],
            "due": record["time_expression"],
            "evidence_ids": record["evidence_ids"],
            "confirmation_evidence_ids": record["confirmation_evidence_ids"],
            "confirmation_utterances": record["confirmation_utterances"],
            "uncertainty": record["uncertainty"],
        })
    return result
