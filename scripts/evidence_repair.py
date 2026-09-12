#!/usr/bin/env python3
"""Plan and reconcile active audio-evidence repair for semantic claims."""
from __future__ import annotations

import re


NUMBER = re.compile(r"(?<!\w)\d+(?:[.,:]\d+)*(?:\s*%)?(?!\w)")
NEGATION = re.compile(r"(?iu)(?:^|\W)(?:не|нет|нельзя|никогда|без)(?:\W|$)")


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
            before_numbers, after_numbers = set(NUMBER.findall(before)), set(NUMBER.findall(after))
            mismatch = not before_numbers.issubset(after_numbers) or (bool(NEGATION.search(before)) and not bool(NEGATION.search(after)))
            status = "disagreed" if mismatch else ("agreed" if after.strip() else "missing")
            report[status] += 1
            report["items"].append({"evidence_id": item["evidence_id"], "status": status, "audio_clip_sha256": item.get("audio_clip_sha256"), "model": item.get("model")})
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
