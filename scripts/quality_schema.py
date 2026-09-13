#!/usr/bin/env python3
"""Shared, deterministic schemas for uncertainty and meeting semantics."""
from __future__ import annotations

import math
import re
import hashlib
import json

try:
    from speech_acts import CORRECTION_CUE_RE, content_kind, modality_axis, primary_speech_act
except ModuleNotFoundError:  # direct importlib loading in unit tests
    import importlib.util
    from pathlib import Path
    _speech_spec = importlib.util.spec_from_file_location("speech_acts", Path(__file__).with_name("speech_acts.py"))
    _speech = importlib.util.module_from_spec(_speech_spec)
    _speech_spec.loader.exec_module(_speech)
    CORRECTION_CUE_RE = _speech.CORRECTION_CUE_RE
    content_kind = _speech.content_kind
    modality_axis = _speech.modality_axis
    primary_speech_act = _speech.primary_speech_act


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
            source_span = str(value.get("source_span") or "").strip() or next(
                (str(evidence_by_id[item].get("text") or "").strip() for item in ids if item in evidence_by_id), ""
            )
            entity = str(value.get("entity") or "").strip() or None
            role = str(value.get("role") or "").strip() or None
            unit = str(value.get("unit") or "").strip() or None
            # A signed integer near index/window language is an offset, never a
            # timeframe merely because a model guessed that unit.
            if amount.startswith("-") and re.search(r"(?iu)\b(?:индекс|позици|окн[оа]|свеч[аи])\w*\b", cited_text):
                entity, role = entity or "window_index", role or "index_offset"
                if unit and re.search(r"(?iu)таймфрейм|time\s*frame", unit):
                    unit = None
            quantities.append({
                "value": amount,
                "unit": unit,
                "entity": entity,
                "role": role,
                "source_span": source_span or None,
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
    evidence_text = " ".join(str(item.get("text") or "") for item in fact.get("evidence", []))
    detected_act = primary_speech_act(evidence_text or fact.get("statement"), fact.get("type"))
    raw_act = raw.get("speech_act")
    if raw_act in {"assert", "propose", "ask", "answer", "commit", "accept", "reject", "correct", "decide"}:
        detected_act = raw_act if detected_act == "assert" else detected_act
    legacy_modality = raw.get("modality") if raw.get("modality") in {"asserted", "tentative", "proposed", "committed", "question"} else ("tentative" if fact.get("certainty") == "tentative" else "asserted")
    return {
        "record_id": fact["fact_id"],
        "kind": fact["type"],
        "topic": fact.get("topic") or "Прочее",
        "statement": fact["statement"],
        "start": float(fact.get("start", 0)),
        "subject": str(raw.get("subject") or "").strip() or None,
        "predicate": str(raw.get("predicate") or "").strip() or None,
        "object": str(raw.get("object") or "").strip() or None,
        "polarity": "negative" if raw.get("polarity") == "negative" else "positive",
        "modality": legacy_modality,
        "content_kind": raw.get("content_kind") or content_kind(fact.get("type")),
        "speech_act": detected_act,
        "modality_axis": modality_axis(legacy_modality, fact.get("certainty")),
        "lifecycle": "active",
        "revision_cue": bool(CORRECTION_CUE_RE.search(evidence_text or str(fact.get("statement") or ""))),
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
        "semantic_risks": list(fact.get("semantic_risks", [])),
        "risk_level": fact.get("risk_level", "LOW"),
        "source_word_ids": list(dict.fromkeys(
            word_id for item in fact.get("evidence", []) for word_id in item.get("source_word_ids", [])
        )),
        "model_provenance": fact.get("model_provenance"),
        "prompt_version": fact.get("prompt_version"),
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


def _semantic_id(prefix, value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return prefix + hashlib.sha256(encoded).hexdigest()[:16]


def _tokens(value):
    ignored = {"это", "как", "для", "что", "при", "или", "уже", "ещё", "будет", "нужно", "надо"}
    return {item for item in re.findall(r"(?iu)[a-zа-яё0-9]+", str(value or "").casefold()) if len(item) > 2 and item not in ignored}


def _same_proposition(left, right):
    left = {**left, **left.get("proposition", {})}
    right = {**right, **right.get("proposition", {})}
    a = (left.get("subject"), left.get("predicate"))
    b = (right.get("subject"), right.get("predicate"))
    if all(a) and a == b:
        return True
    left_tokens = _tokens(" ".join(str(left.get(key) or "") for key in ("subject", "predicate", "object", "presentation")))
    right_tokens = _tokens(" ".join(str(right.get(key) or "") for key in ("subject", "predicate", "object", "presentation")))
    return len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens))) >= 0.55


def adaptive_compute_plan(record):
    """Turn semantic risk into an explicit, auditable compute policy."""
    level = record.get("risk_level", "LOW")
    sensitive = record.get("kind") in {"decision", "action", "schedule", "metric"}
    if level == "CRITICAL" or (level == "HIGH" and sensitive):
        return {"tier": "CRITICAL", "passes": ["deterministic", "independent_verifier", "audio_repair", "speaker_recheck"], "fail_closed": True}
    if level == "HIGH":
        return {"tier": "HIGH", "passes": ["deterministic", "independent_verifier", "expanded_context"], "fail_closed": True}
    if level == "MEDIUM":
        return {"tier": "MEDIUM", "passes": ["deterministic", "verifier"], "fail_closed": False}
    return {"tier": "LOW", "passes": ["deterministic"], "fail_closed": False}


def meeting_state(records, *, provenance=None):
    """Build the canonical, immutable DialogueEvent graph and all state views."""
    provenance = provenance or {}
    events = []
    records = sorted(records, key=lambda item: (float(item.get("start", 0)), item.get("record_id", "")))
    for record in records:
        claim_id = record.get("claim_id") or _semantic_id("C", {
            "kind": record.get("kind"), "statement": record.get("statement"),
            "evidence_ids": record.get("evidence_ids", []),
        })
        event_id = _semantic_id("EV", claim_id)
        speech_act = record.get("speech_act") or primary_speech_act(record.get("statement"), record.get("kind"))
        act = {
            "assert": "assertion", "decide": "decision", "commit": "assignment",
            "ask": "question", "propose": "proposal", "correct": "correction",
        }.get(speech_act, {
            "action": "assignment", "current_state": "assertion", "observation": "assertion",
            "metric": "assertion", "goal": "proposal",
        }.get(record.get("kind"), record.get("kind", "assertion")))
        events.append({
            "event_id": event_id, "claim_id": claim_id, "source_record_id": record["record_id"], "act": act,
            "content_kind": record.get("content_kind") or content_kind(record.get("kind")),
            "speech_act": speech_act,
            "proposition": {"subject": record.get("subject"), "predicate": record.get("predicate"), "object": record.get("object")},
            "speaker_ids": list(record.get("attributed_speakers", [])),
            "mentioned_participant_ids": list(record.get("assignees", [])),
            "polarity": record.get("polarity", "positive"), "modality": record.get("modality_axis") or modality_axis(record.get("modality")),
            "lifecycle": "active", "revision_cue": bool(record.get("revision_cue")),
            "quantities": list(record.get("quantities", [])), "conditions": list(record.get("conditions", [])),
            "evidence_ids": list(record.get("evidence_ids", [])), "start": float(record.get("start", 0)),
            "risk": {"level": record.get("risk_level", "LOW"), "signals": list(record.get("semantic_risks", [])), **record.get("uncertainty", {})},
            "presentation": record.get("statement"),
            "topic": record.get("topic") or "Прочее",
            "question_status": record.get("question_status"),
            "question_kind": record.get("question_kind"),
            "provenance": {
                "audio_sha256": provenance.get("audio_sha256"),
                "source_word_ids": list(dict.fromkeys(record.get("source_word_ids", []))),
                "model": record.get("model_provenance"),
                "prompt_version": record.get("prompt_version"),
                "schema_version": 2,
            },
            "components": [
                {"component_id": _semantic_id("CC", [claim_id, "condition", index]), "kind": "precondition", **value}
                for index, value in enumerate(record.get("conditions", []), 1)
            ] + [
                {"component_id": _semantic_id("CQ", [claim_id, "quantity", index]), "kind": "quantity", **value}
                for index, value in enumerate(record.get("quantities", []), 1)
            ] + ([{"component_id": _semantic_id("CT", [claim_id, "time"]), "kind": "time", "text": record.get("time_expression"), "evidence_ids": list(record.get("evidence_ids", []))}] if record.get("time_expression") else []),
            "compute_plan": adaptive_compute_plan(record),
        })
    relations = []
    def add_relation(source, relation, target, evidence_ids=None, basis=None, metrics=None, **extra):
        payload = {
            "source_event": source["event_id"], "relation": relation,
            "target_event": target["event_id"],
            "evidence_ids": list(dict.fromkeys(evidence_ids or source["evidence_ids"])),
            "decision_basis": {"rule": basis or relation, "metrics": metrics or {}},
            **extra,
        }
        payload["relation_id"] = _semantic_id("R", payload)
        relations.append(payload)

    def quantity_signature(event):
        return {(str(q.get("value")), str(q.get("unit") or ""), str(q.get("entity") or ""), str(q.get("role") or "")) for q in event.get("quantities", [])}

    for index, current in enumerate(events):
        # A correction cue opens a short revision scope.  The replacement may
        # use entirely different words, so lexical similarity is deliberately
        # not required here.
        if current.get("revision_cue") or current.get("speech_act") == "correct":
            speakers = set(current.get("speaker_ids", []))
            candidates = [
                older for older in events[:index]
                if set(older.get("speaker_ids", [])) & speakers
                and current["start"] - older["start"] <= 180
                and older.get("speech_act") not in {"ask", "accept", "reject"}
                and (older.get("topic") == current.get("topic") or index - events.index(older) <= 3)
            ]
            if candidates:
                target = candidates[-1]
                add_relation(current, "corrects", target, current["evidence_ids"] + target["evidence_ids"], basis="explicit_revision_scope", metrics={"time_gap_seconds": current["start"] - target["start"], "same_speaker": True, "same_topic": target.get("topic") == current.get("topic"), "lexical_similarity_required": False})
        for older in events[:index]:
            if not _same_proposition(current, older):
                continue
            relation = None
            if current["polarity"] != older["polarity"]:
                relation = "contradicts"
            elif quantity_signature(current) != quantity_signature(older) and current.get("quantities") and older.get("quantities"):
                # A different number is an unresolved conflict unless the new
                # claim is explicitly framed as a correction.
                relation = "supersedes" if current.get("revision_cue") or current.get("speech_act") == "correct" else "conflicts_with"
            elif current["act"] == "decision" and older["act"] in {"proposal", "assertion"}:
                relation = "accepts"
            elif current.get("speech_act") == "answer" and older["act"] == "question":
                relation = "answers"
            elif "correction" in current.get("risk", {}).get("signals", []) or current.get("revision_cue"):
                relation = "corrects"
            if relation:
                basis = {
                    "contradicts": "same_proposition_opposite_polarity",
                    "supersedes": "same_proposition_quantity_changed_with_revision_cue",
                    "conflicts_with": "same_proposition_quantity_changed_without_revision_cue",
                    "accepts": "decision_accepts_prior_proposition",
                    "answers": "answer_matches_question_proposition",
                    "corrects": "same_proposition_with_correction_signal",
                }.get(relation, relation)
                add_relation(current, relation, older, basis=basis, metrics={"time_gap_seconds": current["start"] - older["start"], "same_proposition": True, "current_quantity_signature": sorted(quantity_signature(current)), "prior_quantity_signature": sorted(quantity_signature(older)), "current_polarity": current["polarity"], "prior_polarity": older["polarity"], "revision_cue": current.get("revision_cue", False)})

        # Q/A adjacency is resolved globally before graph construction.  Local
        # topic equality is intentionally not used: answer turns routinely use
        # different words and receive a different generated topic.

    # Explicit question links and assignee confirmations are already grounded by
    # normalize_semantic_record; materialize them in the same global graph.
    by_record = {item["source_record_id"]: item for item in events}
    for record in records:
        source = by_record.get(record.get("record_id"))
        if not source:
            continue
        for answer_id in record.get("answer_record_ids", []):
            target = by_record.get(answer_id)
            if target:
                answer_relation = record.get("answer_relation") or "answers"
                add_relation(target, answer_relation, source, target["evidence_ids"] + source["evidence_ids"], basis="global_or_explicit_answer_record_link", metrics={"source_record_id": record.get("record_id"), "answer_record_id": answer_id, "question_status": record.get("question_status")})
        if record.get("assignment_status") == "confirmed":
            add_relation(source, "accepted_by", source, record.get("confirmation_evidence_ids", []), basis="grounded_assignee_confirmation", metrics={"assignment_status": record.get("assignment_status")}, participant_ids=record.get("assignees", []))

    unique_relations = {item["relation_id"]: item for item in relations}
    relations = sorted(unique_relations.values(), key=lambda item: item["relation_id"])

    superseded = {item["target_event"] for item in relations if item["relation"] in {"supersedes", "corrects"}}
    conflicted = {value for item in relations if item["relation"] == "conflicts_with" for value in (item["source_event"], item["target_event"])}
    resolved_questions = {
        item["target_event"] for item in relations
        if item["relation"] in {"answers", "partially_answers", "tentatively_answers"}
    }
    for event in events:
        lifecycle_relations = [item["relation_id"] for item in relations if item["target_event"] == event["event_id"] or (item["relation"] == "conflicts_with" and item["source_event"] == event["event_id"])]
        if event["event_id"] in superseded:
            event["lifecycle"] = "superseded"
        elif event["event_id"] in conflicted:
            event["lifecycle"] = "conflicting"
        elif event["event_id"] in resolved_questions:
            event["lifecycle"] = "resolved"
        event["lifecycle_basis_relation_ids"] = lifecycle_relations
    decisions = [item for item in events if item["act"] == "decision" and item["lifecycle"] == "active"]
    questions = [
        {
            **item,
            "state": item.get("question_status") or ("answered" if item["event_id"] in resolved_questions else "unanswered"),
            "answer_record_ids": next((record.get("answer_record_ids", []) for record in records if record.get("record_id") == item.get("source_record_id")), []),
            "answer_resolution_basis": next((record.get("answer_resolution_basis") for record in records if record.get("record_id") == item.get("source_record_id")), None),
        }
        for item in events if item["act"] == "question"
    ]
    active_record_ids = {item["source_record_id"] for item in events if item["lifecycle"] == "active"}
    tasks = [item for item in task_records(records) if item["source_record_id"] in active_record_ids]
    active = [item for item in events if item["lifecycle"] == "active"]
    topic_states = []
    for topic in dict.fromkeys(item["topic"] for item in active):
        members = [item for item in active if item["topic"] == topic]
        topic_states.append({"topic": topic, "event_ids": [item["event_id"] for item in members], "claim_ids": [item["claim_id"] for item in members]})
    return {
        "schema_version": 4,
        "state_id": _semantic_id("MS", [item["claim_id"] for item in events]),
        "events": events,
        "relations": relations,
        "active_event_ids": [item["event_id"] for item in active],
        "topic_states": topic_states,
        "views": {
            "decisions": decisions, "tasks": tasks, "questions": questions,
            "timeline": sorted([item for item in active if item["content_kind"] in {"rule", "task", "goal", "schedule", "metric"}], key=lambda item: (item["start"], item["event_id"])),
            "full_timeline": sorted(events, key=lambda item: (item["start"], item["event_id"])),
            "summary": active,
            "history": [item for item in events if item["lifecycle"] != "active"],
        },
    }
