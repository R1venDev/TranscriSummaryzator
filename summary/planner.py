"""Purpose-specific, relation-aware semantic-subgraph planning."""
from __future__ import annotations
import math, re
from semantics.graph import cross_episode_allowed
from summary.verifier import relation_markers

TECHNICAL = {"rule", "trading_rule", "system_rule", "experiment", "experimental_result", "metric", "design", "design_choice"}
VIEW_KINDS = {
    "executive": {"decision", "state", "problem", "blocker", "action", "question", "question_content", "rule", "design"},
    "technical": TECHNICAL, "tasks": {"action"},
    "experiments": {"experiment", "hypothesis", "experimental_result", "metric"},
    "questions": {"question", "question_content", "blocker"}, "minutes": set(),
}
VIEW_BOOST = {"executive": {"decision": 5, "state": 5, "problem": 4, "blocker": 5, "action": 4}, "technical": {x: 5 for x in TECHNICAL}, "tasks": {"action": 8}, "experiments": {"experiment": 7, "hypothesis": 7, "experimental_result": 7}, "questions": {"question": 8, "question_content": 8}, "minutes": {}}


def _kind(claim): return claim.get("content_kind") or claim.get("kind")
def _tokens(value): return {x for x in re.findall(r"(?iu)[a-zа-яё0-9]+", str(value or "").casefold()) if len(x) > 2}


def _mandatory(claim):
    kind = _kind(claim)
    if kind in {"decision", "state"}: return claim.get("decision_status") == "accepted"
    if kind == "action": return claim.get("task_status") in {"accepted", "self_committed", "explicit_self_commitment", "in_progress", "blocked", "completed"}
    if kind in {"question", "question_content"}: return claim.get("question_status") not in {"answered", "rhetorical", "superseded"}
    return kind in {"blocker", "correction", "experimental_result", "metric"}


def adaptive_budget(claims, episodes, *, minimum=7, maximum=120, view="minutes"):
    active = [x for x in claims if x.get("lifecycle", "active") == "active"]
    minutes = max([float(x.get("end", x.get("start", 0))) for x in active] or [0]) / 60
    threads = len({x.get("thread_id") for x in active if x.get("thread_id")}) or len(episodes)
    base = math.ceil(math.sqrt(max(1, minutes)) + 1.5 * threads + sum(_mandatory(x) for x in active) + len({_kind(x) for x in active}))
    return max(minimum, min(maximum, math.ceil(base * {"executive": .55, "technical": .85, "tasks": .7, "experiments": .8, "questions": .7, "minutes": 1.35}.get(view, 1))))


def _utility(claim, score_fn, view):
    risk = max([float(v or 0) for v in claim.get("risk", {}).values() if isinstance(v, (int, float))] or [0])
    return float(score_fn(claim)) + VIEW_BOOST[view].get(_kind(claim), 0) + 2 * (1-risk) + 2 * _mandatory(claim)


def _select(claims, score_fn, view, budget):
    eligible = [x for x in claims if x.get("lifecycle", "active") == "active" and (view == "minutes" or _kind(x) in VIEW_KINDS[view])]
    ranked = sorted(eligible, key=lambda x: (-_utility(x, score_fn, view), float(x.get("start", 0))))
    selected, token_sets = [], []
    for item in ranked:
        tokens = _tokens(item.get("statement"))
        if max((len(tokens & old) / max(1, min(len(tokens), len(old))) for old in token_sets), default=0) >= .78 and not _mandatory(item): continue
        if len(selected) >= budget and not _mandatory(item): continue
        selected.append(item); token_sets.append(tokens)
    return selected


def _units(selected, claims, relations):
    selected_ids, by_id, consumed = {x["claim_id"] for x in selected}, {x["claim_id"]: x for x in claims}, set()
    result = []
    for central in sorted(selected, key=lambda x: float(x.get("start", 0))):
        if central["claim_id"] in consumed: continue
        claim_ids, relation_ids = [central["claim_id"]], []
        linked = [r for r in relations if central["claim_id"] in {r.get("source_claim_id"), r.get("target_claim_id")}]
        for relation in linked:
            other = relation.get("target_claim_id") if relation.get("source_claim_id") == central["claim_id"] else relation.get("source_claim_id")
            if other in selected_ids and other in by_id and len(claim_ids) < 4:
                claim_ids.append(other); relation_ids.append(relation["relation_id"])
        consumed.update(claim_ids)
        result.append({"unit_id": f"SU{len(result)+1:04d}", "central_claim_id": central["claim_id"], "claim_ids": claim_ids, "relation_ids": relation_ids, "role": "state_change" if _mandatory(central) else "technical_context", "outcome": central.get("social_state"), "unresolved_edges": [r["relation_id"] for r in linked if r["relation_id"] not in relation_ids]})
    return result


def _sentence(index, unit, by_id):
    items = [by_id[x] for x in unit["claim_ids"]]
    return {"sentence_id": f"S{index:05d}", "summary_unit_id": unit["unit_id"], "episode_id": items[0].get("episode_id"), "claim_ids": unit["claim_ids"], "relation_ids": unit["relation_ids"], "intent": unit["role"], "allowed_numbers": [n for x in items for n in re.findall(r"(?<!\w)\d+(?:[.,:]\d+)*(?:\s*%)?", x.get("statement", ""))], "allowed_quantities": [q for x in items for q in x.get("quantities", [])], "allowed_entities": [e.get("entity_id") for x in items for e in x.get("entities", []) if isinstance(e, dict)], "allowed_relation_markers": sorted(set().union(*(relation_markers(x.get("statement")) for x in items))), "allowed_speakers": sorted({s for x in items for s in x.get("speaker_refs", [])}), "allowed_assignees": sorted({x.get("assignee") for x in items if x.get("assignee")}), "polarity": [x.get("polarity") for x in items], "modality": [x.get("modality") for x in items], "conditions": [c for x in items for c in x.get("conditions", [])], "time_scope": [x.get("time_scope") for x in items if x.get("time_scope")], "decision_state": [x.get("decision_status") for x in items if x.get("decision_status")], "task_state": [x.get("task_status") for x in items if x.get("task_status")], "question_slots": [x.get("question_slots") for x in items if x.get("question_slots")], "forbidden_inferences": ["modality_upgrade", "condition_drop", "new_assignee", "new_quantity_binding", "unsupported_causality", "superseded_claim"], "max_sentences": max(1, len(items))}


def plan(claims, episodes, relations, score_fn, max_units=None):
    view_plans, union = {}, {}
    for view in VIEW_KINDS:
        budget = max_units or adaptive_budget(claims, episodes, view=view)
        selected = _select(claims, score_fn, view, budget)
        view_plans[view] = {"objective": view, "budget": budget, "selected_claim_ids": [x["claim_id"] for x in selected], "summary_units": _units(selected, claims, relations)}
        if view in {"executive", "minutes"}: union.update({x["claim_id"]: x for x in selected})
    ordered = sorted(union.values(), key=lambda x: float(x.get("start", 0)))
    units, by_id = _units(ordered, claims, relations), {x["claim_id"]: x for x in claims}
    sentences = [_sentence(i, unit, by_id) for i, unit in enumerate(units, 1)]
    assert all(cross_episode_allowed(x["claim_ids"], x["relation_ids"], claims, relations) for x in sentences)
    return {"schema": "SummaryPlanSchema", "schema_version": 3, "strategy": "purpose-specific-subgraph-utility-v3", "adaptive_budget": view_plans["minutes"]["budget"], "selected_claim_ids": [x["claim_id"] for x in ordered], "channels": {"mandatory": [x["claim_id"] for x in ordered if _mandatory(x)], "balanced_core": [x["claim_id"] for x in ordered if not _mandatory(x)], "optional_detail": []}, "episode_coverage": sorted({x.get("episode_id") for x in ordered if x.get("episode_id")}), "summary_units": units, "sentence_plans": sentences, "paragraph_plans": [{"paragraph_id": f"P{i:02d}", **u} for i, u in enumerate(units, 1)], "view_plans": view_plans}
