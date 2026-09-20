#!/usr/bin/env python3
"""Shared, deterministic schemas for uncertainty and meeting semantics."""
from __future__ import annotations

import math
import re
import hashlib
import json

try:
    from speech_acts import COMMITMENT_RE, CORRECTION_CUE_RE, content_kind, modality_axis, primary_speech_act
except ModuleNotFoundError:  # direct importlib loading in unit tests
    import importlib.util
    from pathlib import Path
    _speech_spec = importlib.util.spec_from_file_location("speech_acts", Path(__file__).with_name("speech_acts.py"))
    _speech = importlib.util.module_from_spec(_speech_spec)
    _speech_spec.loader.exec_module(_speech)
    COMMITMENT_RE = _speech.COMMITMENT_RE
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
SOURCE_ACTION_RE = re.compile(
    r"(?iu)\b(?:сдела(?:ть|ю|ем)|созда(?:ть|м|дим)|подготов(?:ить|лю|им)|"
    r"переда(?:ть|м|дим)|отправ(?:ить|лю|им)|предостав(?:ить|лю|им)|"
    r"размеч(?:ать|ивать|у|аю|иваю|аем|иваем)|встро(?:ить|ю|им)|провер(?:ить|ю|им)|"
    r"исправ(?:ить|лю|им)|продолж(?:ить|у|им)|экспериментир(?:овать|ую|уем)|"
    r"реализ(?:овать|ую|уем)|добав(?:ить|лю|им)|подключ(?:ить|у|им)|"
    r"скин(?:уть|у|ем)|показ(?:ать|жу|ем))\b"
)
SOURCE_FIRST_PERSON_COMMIT_RE = re.compile(
    r"(?iu)\bя\b[^.!?\n]{0,120}\b(?:буду|сделаю|создам|подготовлю|передам|"
    r"отправлю|предоставлю|пришлю|скину|кину|дам|размечу|встрою|проверю|"
    r"исправлю|продолжу|реализую|добавлю|подключу|покажу)\b|"
    r"\b(?:сделаю|создам|подготовлю|передам|отправлю|предоставлю|пришлю|"
    r"скину|кину|размечу|встрою|проверю|исправлю|продолжу|покажу)\b"
)
SOURCE_SELF_ASSIGNMENT_RE = re.compile(
    r"(?iu)\bмне\b[^.!?\n]{0,80}\b(?:сделать|создать|подготовить|передать|"
    r"отправить|предоставить|размечать|разметить|встроить|проверить|"
    r"исправить|продолжить|экспериментировать|реализовать|добавить|подключить)\b"
)
SOURCE_FIRST_PERSON_PROGRESS_RE = re.compile(
    r"(?iu)\bя\b[^.!?\n]{0,80}\b(?:делаю|создаю|готовлю|передаю|отправляю|"
    r"размечаю|встраиваю|проверяю|исправляю|продолжаю|экспериментирую|"
    r"реализую|добавляю|подключаю|анализирую|работаю|рисую|пишу)\b|"
    r"\bя\b(?:(?!\b(?:он|она|они|ты|вы)\b)[^.!?\n]){0,80}"
    r"\b(?:пош[её]л|пошла|начал|начала|приступил|приступила)\b"
    r"[^.!?\n]{0,30}\b[а-яё-]+(?:ть|ти)\b"
)
SOURCE_FIRST_PERSON_ONSET_RE = re.compile(
    r"(?iu)\bя\b(?:(?!\b(?:он|она|они|ты|вы)\b)[^.!?\n]){0,80}"
    r"\b(?:пош[её]л|пошла|начал|начала|приступил|приступила)\b"
    r"[^.!?\n]{0,30}?\b(?P<predicate>[а-яё-]+(?:ть|ти))\b"
)
SOURCE_FIRST_PERSON_PAST_RE = re.compile(
    r"(?iu)\bя\b[^.!?\n]{0,100}\b(?:пробовал(?:а)?|попробовал(?:а)?|"
    r"пытал(?:ся|ась)|делал(?:а)?|проверял(?:а)?|экспериментировал(?:а)?|"
    r"занимал(?:ся|ась)|реализовал(?:а)?|добавлял(?:а)?|анализировал(?:а)?)\b"
)
INTERNAL_PREDICATE_RE = re.compile(r"(?i)^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")
PROCEDURAL_RULE_SURFACE_RE = re.compile(
    r"(?iu)(?=.*\b(?:если|когда|при|как\s+только|после\s+того\s+как|до\s+того\s+как)\b)"
    r"(?=.*\b(?:нужно|надо|следует|требуется|разрешено|запрещено|обязательно|"
    r"[a-zа-яё-]{3,}(?:ется|ются|ается|яются|ится|ятся|ывается|иваются|ируется|ируются|уется|уются))\b)"
)


def source_action_clauses(fact, owners):
    """Recover independently reducible predicates from exact owner speech.

    This is deliberately a bounded syntactic fallback.  It never invents an
    actor: only a sole audited owner speaking the cited turn can supply it.
    """
    if len(owners) != 1:
        return []
    actor = owners[0]
    result = []
    for turn in fact.get("evidence", []):
        if turn.get("speaker") != actor:
            continue
        text = str(turn.get("text") or "")
        matches = []
        for match in SOURCE_ACTION_RE.finditer(text):
            local_start = max(text.rfind(mark, 0, match.start()) for mark in ".!?;,\n") + 1
            local_prefix = text[local_start:match.start()]
            # A plural subject is a team action, not a personal obligation of
            # the sole audited owner of the surrounding fact.
            if re.search(r"(?iu)\bмы\b[^.!?;]{0,40}$", local_prefix):
                continue
            matches.append({"start": match.start(), "end": match.end(), "predicate": match.group(0)})
        # Product/domain vocabularies cannot enumerate every useful verb.
        # A first-person onset construction (``я X начал/пошёл делать``)
        # supplies its own safe grammatical boundary, so recover its
        # infinitive without adding every Russian infinitive to the broad
        # action lexicon.  The regex rejects an intervening personal subject.
        for onset in SOURCE_FIRST_PERSON_ONSET_RE.finditer(text):
            predicate_start, predicate_end = onset.span("predicate")
            if not any(item["start"] == predicate_start for item in matches):
                matches.append({
                    "start": predicate_start,
                    "end": predicate_end,
                    "predicate": onset.group("predicate"),
                })
        matches.sort(key=lambda item: (item["start"], item["end"]))
        if not matches:
            continue
        # ``X или Y`` is one delivery alternative, while ``X и затем Y`` is
        # two independently trackable actions.  Preserve the alternative in a
        # single object instead of manufacturing two obligations.
        groups = []
        index = 0
        while index < len(matches):
            last = index
            while last + 1 < len(matches):
                bridge = text[matches[last]["end"]:matches[last + 1]["start"]]
                if not re.search(r"(?iu)\b(?:или|либо)\b", bridge):
                    break
                last += 1
            groups.append((index, last))
            index = last + 1
        for group_index, (first, last) in enumerate(groups):
            match = matches[first]
            end = matches[groups[group_index + 1][0]]["start"] if group_index + 1 < len(groups) else len(text)
            fragment = text[match["start"]:end]
            # Some languages naturally place the action object before the
            # finite verb (``объект тогда буду размечать``).  Preserve that
            # local object without relying on a domain vocabulary.
            sentence_start = max(text.rfind(mark, 0, match["start"]) for mark in ".!?;\n") + 1
            prefix = text[sentence_start:match["start"]].strip(" ,.;:!?—-\"")
            prefix = re.sub(
                r"(?iu)\b(?:я|мы|мне|нам|тогда|потом|затем|дальше|сейчас|"
                r"буду|будем|нужно|надо|хочу|пытаюсь|пош[её]л|пошла|"
                r"начал|начала|приступил|приступила)\b",
                " ", prefix,
            )
            prefix = re.sub(r"\s+", " ", prefix).strip(" ,.;:!?—-\"")
            if len(prefix.split()) > 5 or SOURCE_ACTION_RE.search(prefix):
                prefix = ""
            fragment = re.sub(r"(?iu)\s+(?:и\s+пойти|и|а|затем|потом|дальше|пойти)\s*$", "", fragment)
            fragment = fragment.strip(" ,.;:!?—-\"")
            if not fragment:
                continue
            predicate = match["predicate"]
            object_value = fragment[len(predicate):].strip(" ,.;:!?—-") or None
            if object_value:
                object_value = re.split(
                    r"(?iu)(?:^|[,;]?\s+)(?:и|а)\s+(?:параллельно\s+)?(?:я|мы|он|она|они|вы|ты)\b",
                    object_value,
                    maxsplit=1,
                )[0].strip(" ,.;:!?—-") or None
            if prefix:
                object_value = " ".join(part for part in (prefix, object_value) if part)
            source_temporal = (
                "past_attempt" if SOURCE_FIRST_PERSON_PAST_RE.search(text)
                else "in_progress" if SOURCE_FIRST_PERSON_PROGRESS_RE.search(text)
                else "planned"
            )
            source_commitment = (
                "intent_to_attempt" if re.search(r"(?iu)\bя\b[^.!?\n]{0,80}\b(?:попробую|попытаюсь|хочу\s+попробовать)\b", text)
                else "explicit_commitment" if SOURCE_FIRST_PERSON_COMMIT_RE.search(text)
                else "unknown"
            )
            result.append({
                "actor": actor, "predicate": predicate,
                "object": object_value, "recipient": None,
                "parallel": bool(re.search(r"(?iu)\bпараллельно\b", text[max(sentence_start, match["start"]-60):end+80])),
                "temporal_state": source_temporal, "commitment_state": source_commitment,
                "evidence_ids": [turn.get("id")] if turn.get("id") else [],
            })
    return result


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
    # Editorial correction may legitimately rewrite the public statement, but
    # it must not erase an actor spoken in the immutable source. Recover only a
    # first-person actor from that actor's own attributed turn; this is not a
    # name/role guess and works for arbitrary participants and domains.
    source_owner_turns = [
        turn for turn in fact.get("evidence", [])
        if turn.get("speaker") and (
            SOURCE_FIRST_PERSON_COMMIT_RE.search(str(turn.get("text") or ""))
            or SOURCE_SELF_ASSIGNMENT_RE.search(str(turn.get("text") or ""))
            or SOURCE_FIRST_PERSON_PROGRESS_RE.search(str(turn.get("text") or ""))
            or SOURCE_FIRST_PERSON_PAST_RE.search(str(turn.get("text") or ""))
        )
    ]
    source_owner_candidates = list(dict.fromkeys(turn.get("speaker") for turn in source_owner_turns))
    source_lifecycle_owner_candidates = list(dict.fromkeys(
        turn.get("speaker") for turn in fact.get("evidence", [])
        if turn.get("speaker") and (
            SOURCE_FIRST_PERSON_PROGRESS_RE.search(str(turn.get("text") or ""))
            or SOURCE_FIRST_PERSON_PAST_RE.search(str(turn.get("text") or ""))
        )
    ))
    action_text = str(fact.get("statement") or "")
    commitment_like = bool(COMMITMENT_RE.search(action_text))
    source_work_lifecycle = len(source_lifecycle_owner_candidates) == 1
    owners = []
    if fact.get("type") == "action" or commitment_like:
        # The semantic model may see several speakers in the evidence.  Only the
        # owner already proven by the action-policy pass may become an assignee.
        owners = audited_owners or (source_owner_candidates if len(source_owner_candidates) == 1 else [])
        # An assignee is confirmed only by their own utterance.  Agreement from
        # another participant can confirm the plan, but cannot accept work on
        # somebody else's behalf.
        owner_set = set(owners)
        confirmation_ids = [
            evidence_id for evidence_id in confirmation_ids
            if evidence_by_id.get(evidence_id, {}).get("speaker") in owner_set
        ]
    elif source_work_lifecycle:
        # A model/editor may retype ongoing or attempted work as a proposal or
        # observation.  Only the one speaker whose own words encode that
        # lifecycle may be restored; a self-assignment question remains merely
        # a proposal until accepted and is handled by its atomic action frames.
        owners = audited_owners or source_lifecycle_owner_candidates
    proposed_by = [value for value in raw.get("proposed_by", []) if value in speakers]
    if fact.get("type") not in {"proposal", "action"}:
        proposed_by = []
    conditions = []
    for value in raw.get("conditions", []):
        if not isinstance(value, dict):
            continue
        text = str(value.get("predicate") or value.get("text") or "").strip()
        ids = [item for item in value.get("evidence_ids", []) if item in evidence_set]
        if text and ids and CONDITION_RE.search(text):
            conditions.append({"predicate": text, "effect": str(value.get("effect") or "").strip() or None, "evidence_ids": ids})
    quantities = []
    for value in raw.get("quantities", []):
        if not isinstance(value, dict):
            continue
        ids = [item for item in value.get("evidence_ids", []) if item in evidence_set]
        normalized = value.get("normalized", {}) if isinstance(value.get("normalized"), dict) else {}
        amount = str(normalized.get("value") if normalized.get("value") is not None else value.get("value") or "").strip()
        digits = re.findall(r"\d+(?:[.,]\d+)?", amount)
        cited_text = " ".join(str(evidence_by_id[item].get("text") or "") for item in ids if item in evidence_by_id).casefold()
        cited_numbers = [float(token.replace(",", ".")) for token in re.findall(r"(?<!\d)\d+(?:[.,]\d+)?(?!\d)", cited_text)]
        supported = any(
            any(abs(float(digit.replace(",", ".")) - cited) < 1e-9 for cited in cited_numbers)
            or any(re.search(rf"(?iu)\b{re.escape(word)}(?:часов\w*|дневн\w*|месячн\w*|летн\w*)?\b", cited_text)
                   for word in NUMBER_WORDS.get(str(float(digit.replace(",", "."))).rstrip("0").rstrip("."), set()))
            for digit in digits
        )
        if amount and ids and digits and supported:
            source_span = str(value.get("source_span") or "").strip() or next(
                (str(evidence_by_id[item].get("text") or "").strip() for item in ids if item in evidence_by_id), ""
            )
            entity = str(value.get("entity") or "").strip() or None
            role = str(value.get("role") or "").strip() or None
            unit = str(normalized.get("unit") or value.get("unit") or "").strip() or None
            # A signed integer near index/window language is an offset, never a
            # timeframe merely because a model guessed that unit.
            if amount.startswith("-") and re.search(r"(?iu)\b(?:индекс|позици|окн[оа]|свеч[аи])\w*\b", cited_text):
                entity, role = entity or "window_index", role or "index_offset"
                if unit and re.search(r"(?iu)таймфрейм|time\s*frame", unit):
                    unit = None
            dimension = value.get("dimension")
            numeric = float(amount.replace(",", "."))
            if not dimension:
                if numeric.is_integer() and 1900 <= numeric <= 2100 and re.search(r"(?iu)\b(?:год|года|году|данн\w*|истори\w*)\b", cited_text):
                    dimension = "calendar_year"
                elif re.search(r"(?iu)\b(?:таймфрейм|свеч\w*|h\d+|m\d+|часов\w*)\b", cited_text):
                    dimension = "timeframe"
                elif re.search(r"(?iu)\b(?:месяц|недел|д(?:ень|ня|ней)|длительн|отрезок)\b", cited_text):
                    dimension = "duration"
                elif re.search(r"(?iu)\bпроцент|%", cited_text):
                    dimension = "percentage"
                else:
                    dimension = "count"
            metric_status = value.get("metric_status") or ("approximate" if re.search(r"(?iu)\b(?:примерно|около|где-то|порядка)\b", cited_text) else "defined")
            quantities.append({"quantity_id": value.get("quantity_id") or f"N{len(quantities)+1:03d}", "raw_text": value.get("raw_text") or source_span or amount, "normalized": {"value": numeric, "unit": unit, "operator": normalized.get("operator", "exact"), "direction": normalized.get("direction")}, "entity": entity, "object_binding": value.get("object_binding") or entity, "dimension": dimension, "metric_status": metric_status, "evidence_ids": ids, "status": value.get("status", "accepted"), "value": amount, "unit": unit, "role": role, "source_span": source_span or None})
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
    if fact.get("type") == "action" or commitment_like or source_work_lifecycle:
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
    owner_source_text = " ".join(
        str(item.get("text") or "") for item in fact.get("evidence", [])
        if len(owners) == 1 and item.get("speaker") == owners[0]
    )
    source_first_person_commitment = bool(
        owner_source_text and SOURCE_FIRST_PERSON_COMMIT_RE.search(owner_source_text)
    )
    source_first_person_progress = bool(
        owner_source_text and SOURCE_FIRST_PERSON_PROGRESS_RE.search(owner_source_text)
    )
    source_first_person_past = bool(
        owner_source_text and SOURCE_FIRST_PERSON_PAST_RE.search(owner_source_text)
    )
    # Detect the act from the words themselves.  Treating every extractor item
    # typed ``action`` as a commitment turned observations and questions into
    # tasks.  Type-based defaults remain safe for inherently dialogic kinds.
    detected_act = primary_speech_act(action_text, None)
    # Third-person future wording in an editorial statement ("X сделает") is
    # not itself a commitment. Only the actor's own source turn can upgrade it.
    if detected_act == "commit" and not source_first_person_commitment:
        detected_act = "propose" if fact.get("type") in {"action", "proposal", "follow_up"} else "assert"
    if source_first_person_commitment:
        detected_act = "commit"
    if detected_act == "assert" and fact.get("type") == "question":
        detected_act = "ask"
    elif detected_act == "assert" and fact.get("type") == "proposal":
        detected_act = "propose"
    elif detected_act == "assert" and fact.get("type") == "decision":
        detected_act = "decide"
    raw_act = raw.get("speech_act")
    if raw_act in {"assert", "propose", "ask", "answer", "commit", "accept", "reject", "correct", "decide"}:
        # A model-only ``commit`` cannot override source wording; an explicit
        # promise or the later acceptance reducer must establish obligation.
        if detected_act == "assert" and (raw_act != "commit" or commitment_like):
            detected_act = raw_act
    source_self_proposal = bool(
        fact.get("type") in {"action", "follow_up"}
        and len(owners) == 1
        and any(
            turn.get("speaker") == owners[0]
            and "?" in str(turn.get("text") or "")
            and (SOURCE_SELF_ASSIGNMENT_RE.search(str(turn.get("text") or ""))
                 or SOURCE_ACTION_RE.search(str(turn.get("text") or "")))
            for turn in fact.get("evidence", [])
        )
    )
    if source_self_proposal and detected_act in {"assert", "answer", "ask", "commit"}:
        detected_act = "propose"
    legacy_modality = raw.get("modality") if raw.get("modality") in {"asserted", "tentative", "proposed", "committed", "question"} else ("tentative" if fact.get("certainty") == "tentative" else "asserted")
    explicit_commitment = source_first_person_commitment and len(owners) == 1
    raw_time_expression = (
        str(raw.get("time_expression", {}).get("raw_text") or raw.get("time_expression", {}).get("text") or "").strip()
        if isinstance(raw.get("time_expression"), dict)
        else str(raw.get("time_expression") or "").strip()
    ) or None
    source_ambiguous_clock = re.search(
        r"(?iu)\bпосле\s+0{1,2}(?::0{2})?\b", evidence_text,
    )
    if source_ambiguous_clock:
        # Preserve the exact source expression.  A model-normalized 00:00
        # would falsely imply a resolved cross-day boundary.
        raw_time_expression = source_ambiguous_clock.group(0)
    ambiguous_clock = bool(raw_time_expression and re.search(r"(?iu)\bпосле\s+0{1,2}(?::0{2})?\b", raw_time_expression)
                           and not (isinstance(raw.get("time_expression"), dict) and raw["time_expression"].get("timezone")))
    known_term_text = " ".join(str(value or "") for value in (
        fact.get("topic"),
        *fact.get("topic_entities", []),
        *fact.get("entities", []),
    )).casefold()
    unresolved_terms = sorted(set(
        token for token in re.findall(r"(?u)\b[A-Z][A-Za-z-]{2,}(?:\s+[A-Z][A-Za-z-]{1,})+\b", action_text)
        if token.casefold() not in known_term_text
    ))
    semantic_verification = fact.get("verification_status", "supported")
    if semantic_verification == "supported" and (
        ambiguous_clock and fact.get("type") in {"action", "schedule", "system_rule", "trading_rule"}
        or unresolved_terms and fact.get("type") in {"decision", "proposal", "design_choice", "system_rule", "trading_rule"}
    ):
        semantic_verification = "insufficient_evidence"
    actions = []
    for index, value in enumerate(raw.get("actions", []), 1):
        if not isinstance(value, dict):
            continue
        ids = [item for item in value.get("evidence_ids", []) if item in evidence_set]
        predicate = str(value.get("predicate") or "").strip()
        if not predicate or not ids:
            continue
        cited = [evidence_by_id[item] for item in ids if item in evidence_by_id]
        cited_text = " ".join(str(item.get("text") or "") for item in cited)
        cited_speakers = {item.get("speaker") for item in cited if item.get("speaker")}
        actor = str(value.get("actor") or "").strip() or None
        object_value = str(value.get("object") or "").strip() or None
        recipient = str(value.get("recipient") or "").strip() or None
        # A role must be bound by a speaker-attributed first-person span or be
        # mentioned literally.  Seeing two names somewhere in a broad evidence
        # window is not field-level support.
        if actor and actor not in cited_speakers and actor not in cited_text:
            actor = None
        if recipient and recipient not in cited_text:
            recipient = None
        if object_value:
            object_tokens = {token[:5] for token in re.findall(r"(?iu)[a-zа-яё0-9]+", object_value) if len(token) > 2}
            cited_tokens = {token[:5] for token in re.findall(r"(?iu)[a-zа-яё0-9]+", cited_text) if len(token) > 2}
            if object_tokens and len(object_tokens & cited_tokens) / len(object_tokens) < .8:
                object_value = None
        field_ids = lambda key: [item for item in value.get(key, []) if item in ids]
        temporal_state = value.get("temporal_state") if value.get("temporal_state") in {"planned", "in_progress", "past_attempt", "completed", "unknown"} else "unknown"
        commitment_state = value.get("commitment_state") if value.get("commitment_state") in {"none", "intent_to_attempt", "explicit_commitment", "accepted_assignment", "unknown"} else "unknown"
        if source_first_person_past:
            temporal_state, commitment_state = "past_attempt", "none"
        elif source_first_person_progress:
            temporal_state = "in_progress"
        elif source_first_person_commitment:
            temporal_state, commitment_state = "planned", "explicit_commitment"
        elif temporal_state in {"planned", "in_progress", "past_attempt", "completed"}:
            # Model lifecycle labels cannot create work state without exact
            # source tense/commitment evidence.
            temporal_state, commitment_state = "unknown", "unknown"
        actions.append({
            "action_id": f"A{index:02d}", "model_action_id": str(value.get("action_id") or "") or None,
            "actor": actor, "predicate": predicate,
            "object": object_value,
            "recipient": recipient, "temporal_state": temporal_state,
            "parallel": bool(value.get("parallel")),
            "commitment_state": commitment_state, "evidence_ids": ids,
            "field_evidence": {
                "actor": field_ids("actor_evidence_ids") or (ids if actor else []),
                "predicate": field_ids("predicate_evidence_ids") or ids,
                "object": field_ids("object_evidence_ids") or (ids if object_value else []),
                "recipient": field_ids("recipient_evidence_ids") or (ids if recipient else []),
            },
        })
    if not actions and (fact.get("type") == "action" or commitment_like):
        actions.append({
            "action_id": "A01", "actor": owners[0] if len(owners) == 1 else None,
            "predicate": str(raw.get("predicate") or action_text).strip(),
            "object": str(raw.get("object") or "").strip() or None,
            "recipient": None,
            "temporal_state": "planned" if detected_act in {"commit", "propose"} else "unknown",
            "commitment_state": "explicit_commitment" if explicit_commitment else "unknown",
            "evidence_ids": evidence_ids,
            "field_evidence": {"actor": source_ids if (source_ids := [item for item in evidence_ids if evidence_by_id.get(item, {}).get("speaker") in owners]) else [], "predicate": evidence_ids, "object": [], "recipient": []},
        })
    # A single semantic frame must not swallow a second action from the same
    # exact owner utterance.  Map opaque model predicates to the corresponding
    # source clause by order, then materialize every remaining source clause.
    recovered_clauses = source_action_clauses(fact, owners)
    covered_clause_indexes = set()
    mapped_action_indexes = set()
    for action_index, action in enumerate(actions):
        predicate = str(action.get("predicate") or "")
        match_index = next((
            index for index, clause in enumerate(recovered_clauses)
            if index not in covered_clause_indexes
            and clause["predicate"].casefold()[:5] == predicate.casefold()[:5]
        ), None)
        if match_index is None and INTERNAL_PREDICATE_RE.fullmatch(predicate) and action_index < len(recovered_clauses):
            match_index = action_index
        if match_index is None:
            continue
        clause = recovered_clauses[match_index]
        covered_clause_indexes.add(match_index)
        mapped_action_indexes.add(action_index)
        action["predicate"] = clause["predicate"]
        action["object"] = action.get("object") or clause.get("object")
        action["actor"] = action.get("actor") or clause.get("actor")
        action["parallel"] = bool(action.get("parallel") or clause.get("parallel"))
        action["evidence_ids"] = list(dict.fromkeys(action.get("evidence_ids", []) + clause.get("evidence_ids", [])))
        action["field_evidence"]["predicate"] = clause.get("evidence_ids", [])
        if action.get("actor"):
            action["field_evidence"]["actor"] = clause.get("evidence_ids", [])
        if action.get("object"):
            action["field_evidence"]["object"] = clause.get("evidence_ids", [])
    # Once an exact source clause has been recovered, additional opaque model
    # labels are not independent obligations.  Keeping them would manufacture
    # duplicate tasks from a single source delivery alternative.
    if recovered_clauses:
        actions = [
            action for index, action in enumerate(actions)
            if index in mapped_action_indexes
            or not INTERNAL_PREDICATE_RE.fullmatch(str(action.get("predicate") or ""))
        ]
    for clause_index, clause in enumerate(recovered_clauses):
        if clause_index in covered_clause_indexes or not clause.get("evidence_ids"):
            continue
        actions.append({
            "action_id": f"A{len(actions)+1:02d}", "model_action_id": None,
            **clause,
            "field_evidence": {
                "actor": clause["evidence_ids"], "predicate": clause["evidence_ids"],
                "object": clause["evidence_ids"] if clause.get("object") else [], "recipient": [],
            },
        })
    derived_content_kind = raw.get("content_kind") or content_kind(fact.get("type"))
    if (
        detected_act not in {"ask", "commit", "accept", "reject"}
        and fact.get("type") not in {"action", "follow_up", "question", "schedule"}
        and PROCEDURAL_RULE_SURFACE_RE.search(action_text)
    ):
        # The source does not have to belong to any predefined business
        # domain.  A generic conditional instruction is a system rule; an
        # extractor may still explicitly classify a domain-specific rule.
        derived_content_kind = "system_rule"
    return {
        "record_id": fact["fact_id"],
        "kind": fact["type"],
        "topic": fact.get("topic") or "Прочее",
        "statement": fact["statement"],
        "start": float(fact.get("start", 0)),
        "primary_evidence_start": float(fact.get("primary_evidence_start", fact.get("start", 0))),
        "subject": str(raw.get("subject") or "").strip() or None,
        "predicate": str(raw.get("predicate") or "").strip() or None,
        "object": str(raw.get("object") or "").strip() or None,
        "polarity": "negative" if raw.get("polarity") == "negative" else "positive",
        "modality": legacy_modality,
        "content_kind": derived_content_kind,
        "speech_act": detected_act,
        "modality_axis": modality_axis(legacy_modality, fact.get("certainty")),
        "lifecycle": "active",
        "revision_cue": bool(CORRECTION_CUE_RE.search(evidence_text or str(fact.get("statement") or ""))),
        "conditions": conditions,
        "quantities": quantities,
        "time_expression": raw_time_expression,
        "time_contract": {"raw": raw_time_expression, "timezone": raw.get("time_expression", {}).get("timezone") if isinstance(raw.get("time_expression"), dict) else None, "resolution_status": "ambiguous_clock" if ambiguous_clock else "source_raw" if raw_time_expression else "not_provided", "execution_safe": bool(raw_time_expression and not ambiguous_clock)},
        "unresolved_terms": unresolved_terms,
        "term_resolution_status": "requires_clarification" if unresolved_terms else "resolved_or_not_applicable",
        "attributed_speakers": list(fact.get("speaker_refs", [])),
        "proposed_by": proposed_by,
        "assignees": owners,
        "assignment_status": assignment_status,
        "confirmation_evidence_ids": confirmation_ids,
        "confirmation_utterances": confirmations,
        "commitment_strength": "explicit" if explicit_commitment else "implicit" if detected_act == "commit" else "none",
        "commitment_actor": owners[0] if explicit_commitment else None,
        "assignment_actor": proposed_by[0] if len(proposed_by) == 1 else None,
        "assignment_target": owners[0] if len(owners) == 1 else None,
        "question_status": question_status,
        "question_kind": question_kind,
        "answer_evidence_ids": answer_ids,
        "answer_record_ids": answer_record_ids,
        "answer_resolution_basis": answer_resolution_basis,
        "requested_slots": list(dict.fromkeys(str(value) for value in raw.get("requested_slots", []) if str(value).strip())),
        "answered_slots": list(dict.fromkeys(str(value) for value in raw.get("answered_slots", []) if str(value).strip())),
        "actions": actions,
        "origin_id": fact.get("origin_id") or fact.get("fact_id"),
        "revision_id": fact.get("revision_id"),
        "reported_content_support": fact.get("verification_status", "supported"),
        "world_truth_status": "not_evaluated",
        "interpretation_status": "requires_clarification" if semantic_verification == "insufficient_evidence" else "typed",
        "evidence_ids": evidence_ids,
        # ``fact.evidence`` is already the immutable, speaker-attributed source
        # window.  Preserve it as dialogue evidence when an older extractor did
        # not duplicate the same turns under ``dialogue_evidence``; reducers
        # need those exact words for first-person intent and adjacency pairs.
        "dialogue_evidence": list(fact.get("dialogue_evidence") or fact.get("evidence", [])),
        "uncertainty": uncertainty,
        "semantic_risks": list(fact.get("semantic_risks", [])),
        "risk_level": fact.get("risk_level", "LOW"),
        "protected_outcome": bool(fact.get("protected_outcome")),
        "verification_status": semantic_verification,
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
        automation_status = (
            "eligible" if automation_eligible else
            "unknown" if record.get("uncertainty", {}).get("needs_review") else
            "human_only"
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
            "automation_status": automation_status,
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
    # Answers may live in transcript turns that the atomic extractor correctly
    # treated as contextual rather than standalone facts. Materialize grounded
    # answer-span events so Q/A remains first-class without polluting the fact
    # registry or losing immutable word provenance.
    answer_span_events = {}
    for record in records:
        if record.get("kind") != "question":
            continue
        for span in record.get("answer_spans", []):
            evidence_id = span.get("id")
            if not evidence_id:
                continue
            claim_id = _semantic_id("CA", [record.get("record_id"), evidence_id, span.get("text")])
            event = {
                "event_id": _semantic_id("EV", claim_id), "claim_id": claim_id,
                "source_record_id": f'answer:{record.get("record_id")}:{evidence_id}',
                "act": "answer", "content_kind": "answer", "speech_act": "answer",
                "proposition": {"subject": None, "predicate": None, "object": None},
                "speaker_ids": [span.get("speaker")] if span.get("speaker") else [],
                "mentioned_participant_ids": [], "polarity": "positive", "modality": "certain",
                "lifecycle": "active", "revision_cue": False, "quantities": [], "conditions": [],
                "evidence_ids": [evidence_id], "start": float(span.get("start", record.get("start", 0))),
                "risk": {"level": "LOW", "signals": []}, "presentation": span.get("text"),
                "topic": record.get("topic") or "Прочее", "question_status": "not_applicable",
                "question_kind": "not_applicable",
                "provenance": {
                    "audio_sha256": provenance.get("audio_sha256"),
                    "source_word_ids": list(span.get("source_word_ids", [])),
                    "model": "global_dialogue_resolver", "prompt_version": "global-dialogue-v1",
                    "schema_version": 2,
                },
                "components": [], "compute_plan": {"tier": "LOW", "passes": ["global_dialogue_resolver"], "fail_closed": False},
            }
            events.append(event)
            answer_span_events[(record.get("record_id"), evidence_id)] = event
    events.sort(key=lambda item: (float(item.get("start", 0)), item.get("event_id", "")))
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
        for evidence_id in record.get("answer_evidence_ids", []):
            target = answer_span_events.get((record.get("record_id"), evidence_id))
            if target:
                answer_relation = record.get("answer_relation") or "answers"
                add_relation(target, answer_relation, source, [evidence_id] + source["evidence_ids"], basis="global_answer_evidence_link", metrics={"source_record_id": record.get("record_id"), "answer_evidence_id": evidence_id, "question_status": record.get("question_status")})
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
            "answer_evidence_ids": next((record.get("answer_evidence_ids", []) for record in records if record.get("record_id") == item.get("source_record_id")), []),
            "answer_spans": next((record.get("answer_spans", []) for record in records if record.get("record_id") == item.get("source_record_id")), []),
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
