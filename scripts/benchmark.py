#!/usr/bin/env python3
"""Lightweight RTTM benchmark for diarization variants.

It reports DER components over exact event intervals and overlap detection.
No model dependency is required, so post-processing thresholds can be tuned
against real labelled meetings without rerunning inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.consensus import track_matrix
except ModuleNotFoundError:  # direct execution: python scripts/benchmark.py
    from consensus import track_matrix


def read_rttm(path):
    segments = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 8 and fields[0] == "SPEAKER":
            start, duration = float(fields[3]), float(fields[4])
            segments.append({"start": start, "end": start + duration, "speaker": fields[7]})
    return segments


def active(segments, start, end):
    return {item["speaker"] for item in segments if min(end, item["end"]) > max(start, item["start"])}


def score(reference, hypothesis):
    # Map anonymous hypothesis labels onto reference identities by maximum
    # total temporal agreement before computing confusion.
    assignment = track_matrix(reference, hypothesis)["mapping"]
    reverse = {hyp: ref for ref, hyp in assignment.items()}
    points = sorted({float(item[key]) for item in reference + hypothesis for key in ("start", "end")})
    miss = false_alarm = confusion = reference_time = 0.0
    overlap_ref = overlap_hyp = overlap_hit = 0.0
    for start, end in zip(points, points[1:]):
        duration = end - start
        if duration <= 0:
            continue
        refs = active(reference, start, end)
        hyps = {reverse.get(value, "hyp:" + value) for value in active(hypothesis, start, end)}
        reference_time += duration * len(refs)
        common = len(refs & hyps)
        miss += duration * max(0, len(refs) - len(hyps))
        false_alarm += duration * max(0, len(hyps) - len(refs))
        confusion += duration * max(0, min(len(refs), len(hyps)) - common)
        if len(refs) > 1:
            overlap_ref += duration
        if len(hyps) > 1:
            overlap_hyp += duration
        if len(refs) > 1 and len(hyps) > 1:
            overlap_hit += duration
    denominator = max(reference_time, 1e-9)
    return {
        "DER": round((miss + false_alarm + confusion) / denominator, 6),
        "missed_speech": round(miss / denominator, 6),
        "false_alarm": round(false_alarm / denominator, 6),
        "speaker_confusion": round(confusion / denominator, 6),
        "reference_speaker_seconds": round(reference_time, 3),
        "overlap_recall": round(overlap_hit / max(overlap_ref, 1e-9), 6),
        "overlap_precision": round(overlap_hit / max(overlap_hyp, 1e-9), 6),
        "speaker_mapping": reverse,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--hypothesis", action="append", required=True, help="NAME=path.rttm")
    parser.add_argument("--output")
    args = parser.parse_args()
    reference = read_rttm(args.reference)
    result = {}
    for value in args.hypothesis:
        name, path = value.split("=", 1)
        result[name] = score(reference, read_rttm(path))
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
