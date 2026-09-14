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


def classification(gold, predicted):
    gold, predicted = set(gold), set(predicted); tp = len(gold & predicted)
    precision, recall = ratio(tp, len(predicted)), ratio(tp, len(gold))
    return {"precision": precision, "recall": recall, "f1": 2*precision*recall/max(1e-12, precision+recall)}


def architecture_metrics(gold, produced):
    """Executable v21 release specification; callers provide normalized gold sets."""
    names = ("atomic_claims", "canonical_propositions", "relations", "decisions", "task_commitments", "question_slots", "latest_state", "corrections", "conditions", "quantity_bindings", "episodes", "threads", "technical_rules", "open_questions")
    result = {name: classification(gold.get(name, []), produced.get(name, [])) for name in names}
    result.update({
        "importance_weighted_recall": ratio(sum(gold.get("importance", {}).get(x, 1) for x in set(gold.get("atomic_claims", [])) & set(produced.get("atomic_claims", []))), sum(gold.get("importance", {}).values()) or len(gold.get("atomic_claims", []))),
        "public_factual_precision": result["atomic_claims"]["precision"],
        "unsupported_synthesis_rate": ratio(len(set(produced.get("atomic_claims", []))-set(gold.get("atomic_claims", []))), len(produced.get("atomic_claims", []))),
        "cross_episode_merge_error": ratio(len(produced.get("invalid_cross_episode_merges", [])), len(produced.get("cross_episode_merges", []))),
        "redundancy": float(produced.get("redundancy", 0)), "compression": float(produced.get("compression", 0)), "usefulness": float(produced.get("usefulness", 0)),
        "DER": float(produced.get("DER", 0)), "JER": float(produced.get("JER", 0)), "critical_asr_errors": int(produced.get("critical_asr_errors", 0)),
    })
    return result


def release_gate(metrics, policy=None):
    policy = policy or {"decision_false_positive": .01, "question_false_resolution": .02, "public_factual_precision": .99, "importance_weighted_recall_regression": .02}
    failures = []
    if metrics.get("decision_false_positive", 0) > policy["decision_false_positive"]: failures.append("decision_false_positive")
    if metrics.get("question_false_resolution", 0) > policy["question_false_resolution"]: failures.append("question_false_resolution")
    if metrics.get("public_factual_precision", 0) < policy["public_factual_precision"]: failures.append("public_factual_precision")
    if metrics.get("importance_weighted_recall_regression", 0) > policy["importance_weighted_recall_regression"]: failures.append("importance_weighted_recall_regression")
    return {"passed": not failures, "failures": failures, "policy": policy}
