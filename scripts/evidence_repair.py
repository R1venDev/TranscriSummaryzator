#!/usr/bin/env python3
"""Plan and reconcile active audio-evidence repair for semantic claims."""
from __future__ import annotations

import re

try:
    from speech_acts import CORRECTION_CUE_RE, SCHEDULE_RE
except ModuleNotFoundError:  # direct importlib loading in unit tests
    import importlib.util
    from pathlib import Path
    _spec = importlib.util.spec_from_file_location("speech_acts", Path(__file__).with_name("speech_acts.py"))
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    CORRECTION_CUE_RE, SCHEDULE_RE = _module.CORRECTION_CUE_RE, _module.SCHEDULE_RE


NUMBER = re.compile(r"(?<!\w)\d+(?:[.,:]\d+)*(?:\s*%)?(?!\w)")
NEGATION = re.compile(r"(?iu)(?:^|\W)(?:не|нет|нельзя|никогда|без)(?:\W|$)")
UNIT = re.compile(r"(?iu)(?:%|процент\w*|секунд\w*|минут\w*|час\w*|дн(?:я|ей)|пункт\w*|свеч\w*|таймфрейм\w*)")
PROFILE = re.compile(r"(?iu)@[\w.-]+")
TECHNICAL = re.compile(r"(?iu)\b(?:bos|fbos|smc|tpo|order\s*block|tradingview|binance|[a-zа-яё]+\d+|\d+[a-zа-яё]+)\b")
MODAL = re.compile(r"(?iu)\b(?:точно|вероятно|возможно|может|если|при условии|обязательно|нельзя|нужно|надо)\b")
DIRECTION = re.compile(r"(?iu)\b(?:выше|ниже|до|после|перед|раньше|позже|больше|меньше|рост|падение|лонг|шорт)\b")


def semantic_signature(pattern, text):
    return {re.sub(r"\s+", " ", item.casefold()).strip() for item in pattern.findall(str(text or ""))}


def repair_mismatch_reasons(before, after):
    """Compare semantic anchors, not just surface similarity/WER."""
    reasons = []
    for name, pattern in (
        ("quantity", NUMBER), ("unit", UNIT), ("date_time", SCHEDULE_RE),
        ("participant", PROFILE), ("technical_entity", TECHNICAL),
        ("modality", MODAL), ("direction", DIRECTION),
    ):
        expected, actual = semantic_signature(pattern, before), semantic_signature(pattern, after)
        if not expected.issubset(actual):
            reasons.append(name)
    if bool(NEGATION.search(before)) != bool(NEGATION.search(after)):
        reasons.append("negation")
    if bool(CORRECTION_CUE_RE.search(before)) != bool(CORRECTION_CUE_RE.search(after)):
        reasons.append("correction_cue")
    return sorted(set(reasons))


def audio_chunk_ranges(total_samples, sample_rate, max_seconds=20.0):
    """Return contiguous chunks that stay below GigaAM's 25-second limit."""
    total_samples = max(0, int(total_samples))
    chunk_samples = max(1, int(float(sample_rate) * float(max_seconds)))
    return [
        (offset, min(total_samples, offset + chunk_samples))
        for offset in range(0, total_samples, chunk_samples)
    ]


def repair_requests(facts, padding_before=2.0, padding_after=4.0, limit=24):
    result, by_evidence = [], {}
    ranked = sorted(facts, key=lambda item: (item.get("risk_level") != "CRITICAL", float(item.get("start", 0))))
    for fact in ranked:
        if fact.get("risk_level") != "CRITICAL":
            continue
        for evidence in fact.get("evidence", []):
            key = evidence.get("id")
            if not key:
                continue
            if key in by_evidence:
                by_evidence[key]["fact_ids"] = list(dict.fromkeys(by_evidence[key]["fact_ids"] + [fact.get("fact_id")]))
                continue
            request = {
                "request_id": "AR" + str(key).lstrip("U"), "evidence_id": key,
                "fact_ids": [fact.get("fact_id")],
                "start": max(0.0, float(evidence.get("start", 0)) - padding_before),
                "end": float(evidence.get("end", 0)) + padding_after,
                "original_text": evidence.get("text", ""),
                "speaker": evidence.get("speaker"),
            }
            result.append(request); by_evidence[key] = request
            if len(result) >= limit:
                return result
    return result


def reconcile_repairs(facts, repairs):
    by_id = {item.get("evidence_id"): item for item in repairs}
    report = {"requested": len(repairs), "agreed": 0, "disagreed": 0, "missing": 0, "items": []}
    for fact in facts:
        relevant = [by_id[item.get("id")] for item in fact.get("evidence", []) if item.get("id") in by_id]
        if not relevant:
            continue
        disagreements = []
        for item in relevant:
            before, after = str(item.get("original_text", "")), str(item.get("text", ""))
            mismatch_reasons = repair_mismatch_reasons(before, after)
            mismatch = bool(mismatch_reasons)
            status = "disagreed" if mismatch else ("agreed" if after.strip() else "missing")
            report[status] += 1
            report["items"].append({"evidence_id": item["evidence_id"], "status": status, "mismatch_reasons": mismatch_reasons, "audio_clip_sha256": item.get("audio_clip_sha256"), "model": item.get("model")})
            if mismatch:
                disagreements.append(item["evidence_id"])
        fact["audio_repairs"] = relevant
        if disagreements:
            fact["semantic_risks"] = sorted(set(fact.get("semantic_risks", [])) | {"repair_disagreement"})
            fact["risk_level"] = "CRITICAL"
            uncertainty = dict(fact.get("uncertainty", {}), needs_review=True)
            uncertainty["reasons"] = sorted(set(uncertainty.get("reasons", [])) | {"audio_repair_disagreement"})
            fact["uncertainty"] = uncertainty
    return facts, report
