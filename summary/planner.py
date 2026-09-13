"""Mandatory-first, coverage-constrained adaptive summary planning."""
from __future__ import annotations
import math
import re
from semantics.graph import cross_episode_allowed

MANDATORY_KINDS = {"decision", "action", "schedule", "blocker", "correction"}
TECHNICAL_KINDS = {"trading_rule", "system_rule", "experimental_result", "metric", "design_choice"}


def adaptive_budget(claims, episodes, *, minimum=20, maximum=120):
    active = [x for x in claims if x.get("lifecycle", "active") == "active"]
    decisions = sum(x.get("kind") == "decision" for x in active)
    tasks = sum(x.get("kind") == "action" for x in active)
    open_questions = sum(x.get("kind") == "question" and x.get("question_status") not in {"answered", "rhetorical", "superseded"} for x in active)
    rules = sum(x.get("kind") in TECHNICAL_KINDS for x in active)
    return max(minimum, min(maximum, 12 + 2 * len(episodes) + decisions + tasks + open_questions + math.ceil(rules / 2)))


def _tokens(value):
    return {x for x in re.findall(r"(?iu)[a-zа-яё0-9]+", str(value or "").casefold()) if len(x) > 2}


def _mandatory(claim):
    kind = claim.get("kind")
    if kind in MANDATORY_KINDS:
        if kind == "decision":
            return claim.get("decision_status", "accepted") == "accepted"
        if kind == "action":
            return claim.get("task_status") in {"accepted", "explicit_self_commitment", "in_progress", "blocked", "completed"}
        return True
    if kind == "question":
        return claim.get("question_status") not in {"answered", "rhetorical", "superseded"}
    return kind in {"experimental_result", "metric"} and bool(claim.get("quantities"))


def plan(claims, episodes, relations, score_fn, max_units=None):
    active = [x for x in claims if x.get("lifecycle", "active") == "active"]
    budget = max_units or adaptive_budget(active, episodes)
    mandatory = [x for x in active if _mandatory(x)]
    selected = {x["claim_id"]: x for x in mandatory}
    episode_ids = {x.get("episode_id") for x in selected.values()}
    used_tokens = [_tokens(x.get("statement")) for x in selected.values()]

    ranked = sorted(active, key=lambda x: (-score_fn(x), float(x.get("start", 0))))
    # Coverage constraint: one useful claim from every episode before global utility.
    for episode in episodes:
        if episode["episode_id"] in episode_ids:
            continue
        candidate = next((x for x in ranked if x.get("episode_id") == episode["episode_id"]), None)
        if candidate and len(selected) < budget:
            selected[candidate["claim_id"]] = candidate
            used_tokens.append(_tokens(candidate.get("statement")))
    for candidate in ranked:
        if len(selected) >= budget or candidate["claim_id"] in selected:
            continue
        tokens = _tokens(candidate.get("statement"))
        if any(len(tokens & prior) / max(1, min(len(tokens), len(prior))) >= .75 for prior in used_tokens):
            continue
        selected[candidate["claim_id"]] = candidate
        used_tokens.append(tokens)

    ordered = sorted(selected.values(), key=lambda x: float(x.get("start", 0)))
    channels = {
        "mandatory": [x["claim_id"] for x in ordered if _mandatory(x)],
        "balanced_core": [x["claim_id"] for x in ordered if not _mandatory(x)],
        "optional_detail": [x["claim_id"] for x in ranked if x["claim_id"] not in selected],
    }
    sentence_plans = [{
        "sentence_id": f"S{i:05d}", "episode_id": x.get("episode_id"),
        "claim_ids": [x["claim_id"]], "relation_ids": [], "intent": "state_claim",
        "allowed_numbers": re.findall(r"(?<!\w)\d+(?:[.,:]\d+)*(?:\s*%)?", x.get("statement", "")),
        "allowed_speakers": list(x.get("speaker_refs", [])), "max_sentences": 1,
    } for i, x in enumerate(ordered, 1)]
    paragraph_plans = [{
        "paragraph_id": f"P{i:02d}", "episode_ids": [x.get("episode_id")],
        "claim_ids": [x["claim_id"]], "relation_ids": [], "role": "central_result",
    } for i, x in enumerate(ordered[:10], 1)]
    assert all(cross_episode_allowed(x["claim_ids"], x["relation_ids"], active, relations) for x in sentence_plans)
    return {"schema": "SummaryPlanSchema", "schema_version": 2, "strategy": "mandatory-coverage-greedy-v2", "adaptive_budget": budget, "selected_claim_ids": [x["claim_id"] for x in ordered], "channels": channels, "episode_coverage": sorted({x.get("episode_id") for x in ordered if x.get("episode_id")}), "sentence_plans": sentence_plans, "paragraph_plans": paragraph_plans}
