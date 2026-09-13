"""Release-gate metrics for v19/v20 failure modes."""
from __future__ import annotations


def ratio(numerator, denominator):
    return numerator / max(1, denominator)


def public_metrics(expected, produced):
    expected_ids = {x["claim_id"] for x in expected}
    produced_ids = {x["claim_id"] for x in produced}
    mandatory = {x["claim_id"] for x in expected if x.get("mandatory")}
    episodes = {x.get("episode_id") for x in expected if x.get("episode_id")}
    covered = {x.get("episode_id") for x in produced if x.get("episode_id")}
    return {"mandatory_claim_recall": ratio(len(mandatory & produced_ids), len(mandatory)), "atomic_public_precision": ratio(len(expected_ids & produced_ids), len(produced_ids)), "episode_coverage": ratio(len(episodes & covered), len(episodes)), "unsupported_claims": sorted(produced_ids - expected_ids)}


def calibration(samples):
    if not samples:
        return {"ece": 0.0, "brier": 0.0}
    brier = sum((float(x["confidence"]) - int(bool(x["correct"]))) ** 2 for x in samples) / len(samples)
    bins = [[] for _ in range(10)]
    for x in samples:
        bins[min(9, int(float(x["confidence"]) * 10))].append(x)
    ece = sum(len(bucket) / len(samples) * abs(sum(float(x["confidence"]) for x in bucket) / len(bucket) - sum(bool(x["correct"]) for x in bucket) / len(bucket)) for bucket in bins if bucket)
    return {"ece": ece, "brier": brier}
