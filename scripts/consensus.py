#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

def overlap(a, b):
    return max(0.0, min(float(a["end"]), float(b["end"])) - max(float(a["start"]), float(b["start"])))


def duration(items):
    return sum(max(0.0, float(x["end"]) - float(x["start"])) for x in items)


def by_speaker(items):
    result = {}
    for item in items:
        result.setdefault(str(item["speaker"]), []).append(item)
    return result


def intersect_duration(left, right):
    return sum(overlap(a, b) for a in left for b in right)


def track_matrix(primary, verifier):
    left, right = by_speaker(primary), by_speaker(verifier)
    left_ids, right_ids = sorted(left), sorted(right)
    matrix = [[0.0 for _ in right_ids] for _ in left_ids]
    details = {}
    for i, a in enumerate(left_ids):
        da = duration(left[a])
        for j, b in enumerate(right_ids):
            db = duration(right[b])
            inter = intersect_duration(left[a], right[b])
            union = max(1e-9, da + db - inter)
            coverage_a = inter / max(1e-9, da)
            coverage_b = inter / max(1e-9, db)
            score = 0.5 * inter / union + 0.25 * coverage_a + 0.25 * coverage_b
            matrix[i][j] = score
            details[f"{a}|{b}"] = {"intersection": round(inter, 3), "union": round(union, 3), "iou": round(inter / union, 4), "primary_coverage": round(coverage_a, 4), "verifier_coverage": round(coverage_b, 4), "agreement": round(score, 4)}
    mapping = {}
    if matrix and right_ids:
        # Exact maximum-weight assignment. At most eight Ultra slots means the
        # bitmask state space is tiny and avoids a runtime SciPy dependency.
        states = {(0, 0): (0.0, [])}
        for row in range(len(left_ids)):
            next_states = {}
            for (done, mask), (total, chosen) in states.items():
                skip = (done + 1, mask)
                if skip not in next_states or total > next_states[skip][0]:
                    next_states[skip] = (total, chosen + [None])
                for column in range(len(right_ids)):
                    if mask & (1 << column):
                        continue
                    key = (done + 1, mask | (1 << column))
                    candidate = (total + matrix[row][column], chosen + [column])
                    if key not in next_states or candidate[0] > next_states[key][0]:
                        next_states[key] = candidate
            states = next_states
        _, columns = max(states.values(), key=lambda item: item[0])
        mapping = {left_ids[i]: right_ids[column] for i, column in enumerate(columns) if column is not None and matrix[i][column] > 0}
    return {"primary_speakers": left_ids, "verifier_speakers": right_ids, "matrix": [[round(value, 4) for value in row] for row in matrix], "mapping": mapping, "pairs": details}


def active(items, start, end):
    values = {}
    for item in items:
        amount = max(0.0, min(end, float(item["end"])) - max(start, float(item["start"])))
        if amount:
            values[str(item["speaker"])] = values.get(str(item["speaker"]), 0.0) + amount
    return values


def consensus(primary, verifier, tolerance=0.3):
    # Treat sub-tolerance boundary jitter as one event before comparing tracks.
    raw_points = sorted({round(float(x[key]), 3) for x in primary + verifier for key in ("start", "end")})
    groups = []
    for point in raw_points:
        if groups and point - groups[-1][0] <= tolerance:
            groups[-1].append(point)
        else:
            groups.append([point])
    snapped = {point: values[len(values) // 2] for values in groups for point in values}
    def snap_interval(item):
        start = snapped[round(float(item["start"]), 3)]
        end = snapped[round(float(item["end"]), 3)]
        # A short acknowledgement must not disappear inside the tolerance window.
        if end <= start and float(item["end"]) > float(item["start"]):
            return dict(item)
        return dict(item, start=start, end=end)
    primary = [snap_interval(item) for item in primary]
    verifier = [snap_interval(item) for item in verifier]
    primary = [item for item in primary if item["end"] > item["start"]]
    verifier = [item for item in verifier if item["end"] > item["start"]]
    match = track_matrix(primary, verifier)
    reverse = {v: k for k, v in match["mapping"].items()}
    points = sorted({round(float(x[key]), 3) for x in primary + verifier for key in ("start", "end")})
    result = []
    for start, end in zip(points, points[1:]):
        if end - start <= 1e-4:
            continue
        pa, ua = active(primary, start, end), active(verifier, start, end)
        mapped_ultra = {reverse.get(s, f"ultra:{s}") for s in ua}
        pset = set(pa)
        agreed = pset & mapped_ultra
        union = pset | mapped_ultra
        if not union:
            continue
        confidence = 0.96 if union == agreed else (0.78 if agreed else 0.45)
        decision = "agreement" if union == agreed else ("partial_agreement" if agreed else "disagreement")
        # A single-vs-single disagreement is not overlap. Keep both candidates
        # for the Voice-ID arbiter, while exposing the primary hypothesis as
        # the provisional track. Real overlap is retained if either diarizer
        # independently reports simultaneous speakers.
        provisional = union if agreed else (pset or mapped_ultra)
        segment = {
            "start": start,
            "end": end,
            "clusters": sorted(provisional),
            "candidates": sorted(union),
            "primary": sorted(pset),
            "verifier": sorted(ua),
            "verifier_mapped": sorted(mapped_ultra),
            "agreement": round(len(agreed) / max(1, len(union)), 3),
            "confidence": confidence,
            "confidence_level": "HIGH" if confidence >= 0.9 else ("MEDIUM" if confidence >= 0.7 else "LOW"),
            "overlap": len(pset) > 1 or len(mapped_ultra) > 1,
            "decision": decision,
        }
        if result and all(result[-1].get(k) == segment.get(k) for k in ("clusters", "primary", "verifier", "decision", "overlap")) and start - result[-1]["end"] <= tolerance:
            result[-1]["end"] = end
        else:
            result.append(segment)
    return match, result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", required=True)
    parser.add_argument("--verifier", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--boundary-tolerance", type=float, default=0.3)
    args = parser.parse_args()
    primary = json.loads(Path(args.primary).read_text())["intervals"]
    verifier = json.loads(Path(args.verifier).read_text())["intervals"]
    mapping, timeline = consensus(primary, verifier, args.boundary_tolerance)
    Path(args.mapping).write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n")
    Path(args.output).write_text(json.dumps({"intervals": timeline}, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()

