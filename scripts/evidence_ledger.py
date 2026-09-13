#!/usr/bin/env python3
"""Immutable word evidence and deterministic dialogue-risk primitives."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

try:
    from speech_acts import COMMITMENT_RE, CORRECTION_CUE_RE, SCHEDULE_RE
except ModuleNotFoundError:  # direct importlib loading in unit tests
    import importlib.util
    from pathlib import Path
    _speech_spec = importlib.util.spec_from_file_location("speech_acts", Path(__file__).with_name("speech_acts.py"))
    _speech = importlib.util.module_from_spec(_speech_spec)
    _speech_spec.loader.exec_module(_speech)
    COMMITMENT_RE, CORRECTION_CUE_RE, SCHEDULE_RE = _speech.COMMITMENT_RE, _speech.CORRECTION_CUE_RE, _speech.SCHEDULE_RE
try:
    from diagnostics import decision as diagnostic_decision
except ModuleNotFoundError:
    diagnostic_decision = lambda *args, **kwargs: None


RISK_PATTERNS = {
    "agreement": re.compile(r"(?iu)^\s*(?:да|ага|угу|ок(?:ей)?|соглас(?:ен|на|ны)?)\W*$"),
    "disagreement": re.compile(r"(?iu)^\s*(?:нет|неа|не соглас(?:ен|на|ны)?)\W*$"),
    "negation": re.compile(r"(?iu)(?:^|\W)(?:не|нет|нельзя|никогда|без)(?:\W|$)"),
    "quantity": re.compile(r"(?iu)(?:\d+(?:[.,:]\d+)*|\b(?:один|два|три|четыре|пять|шесть|семь|восемь|девять|десять)\b)"),
    "date_time": SCHEDULE_RE,
    "commitment": COMMITMENT_RE,
    "correction": CORRECTION_CUE_RE,
    "question": re.compile(r"(?iu)\?|^\s*(?:кто|что|где|когда|почему|как|какой|нужно ли)\b"),
}


def stable_id(prefix: str, payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return prefix + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def attach_word_ids(words: list[dict]) -> list[dict]:
    """Copy raw ASR words and attach stable, position-preserving IDs."""
    result = []
    for index, source in enumerate(words, 1):
        item = dict(source)
        item["word_id"] = item.get("word_id") or f"W{index:08d}"
        item.setdefault("raw", {
            "text": str(source.get("text", "")),
            "start": round(float(source.get("start", 0)), 3),
            "end": round(float(source.get("end", source.get("start", 0))), 3),
        })
        item.setdefault("source_word_ids", [item["word_id"]])
        result.append(item)
    return result


def record_resolution(word: dict, speaker, method: str, score=None, risk: str = "medium") -> None:
    """Append an interpretation without erasing the original observation."""
    previous = word.get("speaker")
    history = word.setdefault("speaker_resolution_history", [])
    if not history:
        history.append({"speaker": previous, "reason": "acoustic_resolution"})
    if previous != speaker or not history:
        history.append({"speaker": speaker, "reason": method})
    word["speaker"] = speaker
    if score is not None:
        word["speaker_confidence"] = round(float(score), 4)
    word["resolution"] = {"selected": speaker, "method": method, "risk": risk}
    diagnostic_decision(
        "speaker_resolution", speaker if speaker is not None else "unresolved",
        candidates=[previous, speaker], metrics={"score": score, "risk": risk, "start": word.get("start"), "end": word.get("end")},
        reasons=[method], refs={"word_id": word.get("word_id"), "source_word_ids": word.get("source_word_ids", [])},
    )


def semantic_risks(text: str, *, flags=(), speakers=()) -> list[str]:
    risks = [name for name, pattern in RISK_PATTERNS.items() if pattern.search(str(text or ""))]
    flags = set(flags or ())
    if flags & {"asr_boundary", "asr_alternative"}:
        risks.append("asr_boundary")
    if flags & {"ambiguous", "no_diarization", "speaker_context", "speaker_smoothed", "clause_coherence"}:
        risks.append("speaker_inferred")
    if len(set(value for value in speakers or () if value)) > 1:
        risks.append("multiple_speakers")
    return sorted(set(risks))


def risk_level(risks: list[str], kind: str | None = None) -> str:
    score = sum({
        "agreement": 2, "disagreement": 3, "negation": 3, "quantity": 3,
        "date_time": 3, "commitment": 4, "correction": 4, "question": 1,
        "asr_boundary": 3, "speaker_inferred": 3, "multiple_speakers": 3,
    }.get(value, 1) for value in set(risks))
    if kind in {"decision", "action", "schedule", "metric"}:
        score += 4
    return "CRITICAL" if score >= 8 else "HIGH" if score >= 5 else "MEDIUM" if score >= 2 else "LOW"


def build_evidence_spans(utterances: list[dict], words: list[dict]) -> list[dict]:
    """Create deterministic sentence/clause spans backed by exact word IDs."""
    spans = []
    for utterance_index, utterance in enumerate(utterances, 1):
        members = [
            word for word in words
            if float(word.get("end", 0)) >= float(utterance.get("start", 0))
            and float(word.get("start", 0)) <= float(utterance.get("end", 0))
            and word.get("speaker") == utterance.get("speaker")
        ]
        current = []
        for word in members:
            current.append(word)
            if str(word.get("text", "")).rstrip().endswith((".", "?", "!", "…")):
                _append_span(spans, utterance_index, utterance, current)
                current = []
        if current:
            _append_span(spans, utterance_index, utterance, current)
    return spans


def _append_span(spans, utterance_index, utterance, words):
    text = " ".join(str(item.get("text", "")) for item in words).strip()
    flags = sorted({flag for item in words for flag in item.get("flags", [])})
    risks = semantic_risks(text, flags=flags, speakers=[utterance.get("speaker")])
    spans.append({
        "evidence_id": f"E{len(spans) + 1:08d}",
        "utterance_id": f"U{utterance_index:05d}",
        "word_ids": list(dict.fromkeys(value for item in words for value in item.get("source_word_ids", [item["word_id"]]))),
        "start": round(float(words[0]["start"]), 3),
        "end": round(float(words[-1]["end"]), 3),
        "text": text,
        "speaker_id": utterance.get("speaker"),
        "semantic_risks": risks,
        "risk_level": risk_level(risks),
    })


def ledger_document(raw_words: list[dict], resolved_words: list[dict], utterances: list[dict]) -> dict:
    raw_by_id = {item["word_id"]: item for item in attach_word_ids(raw_words)}
    words = []
    normalized_tokens = []
    for item in resolved_words:
        source_ids = list(item.get("source_word_ids") or [item.get("word_id")])
        raw = [raw_by_id[value]["raw"] for value in source_ids if value in raw_by_id]
        for source_id in source_ids:
            source = raw_by_id.get(source_id)
            if source is None or any(value["word_id"] == source_id for value in words):
                continue
            words.append({
                "word_id": source_id,
                "raw": source["raw"],
                "speaker_evidence": item.get("speaker_evidence", []),
                "resolution": item.get("resolution", {"selected": item.get("speaker"), "method": "legacy", "risk": "unknown"}),
                "history": item.get("speaker_resolution_history", []),
                "flags": list(item.get("flags", [])),
            })
        normalized_tokens.append({
            "token_id": f"N{len(normalized_tokens) + 1:08d}",
            "normalized_text": str(item.get("text", "")),
            "source_word_ids": source_ids,
            "start": item.get("start"), "end": item.get("end"),
        })
    # Raw words that never reached a normalized token remain visible as evidence.
    present = {item["word_id"] for item in words}
    for source_id, source in raw_by_id.items():
        if source_id not in present:
            words.append({"word_id": source_id, "raw": source["raw"], "speaker_evidence": [], "resolution": {"selected": None, "method": "unresolved", "risk": "high"}, "history": [], "flags": list(source.get("flags", []))})
    words.sort(key=lambda item: item["word_id"])
    spans = build_evidence_spans(utterances, resolved_words)
    return {"schema_version": 1, "words": words, "normalized_tokens": normalized_tokens, "evidence_spans": spans}
