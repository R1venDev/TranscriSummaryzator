"""Purpose-specific, relation-aware semantic-subgraph planning."""
from __future__ import annotations
import math, re
from semantics.graph import cross_episode_allowed
from summary.verifier import relation_markers
from summary.policy import TECHNICAL_KINDS

TECHNICAL = set(TECHNICAL_KINDS)
VIEW_KINDS = {
    "executive": {"decision", "proposal", "current_state", "problem", "blocker", "action", "follow_up", "question", "trading_rule", "system_rule", "design_choice"},
    "technical": TECHNICAL, "tasks": {"action", "follow_up"},
    "experiments": {"hypothesis", "experimental_result", "metric"},
    "questions": {"question", "blocker", "schedule"}, "minutes": set(),
}
VIEW_BOOST = {"executive": {"decision": 5, "current_state": 5, "problem": 4, "blocker": 5, "action": 4}, "technical": {x: 5 for x in TECHNICAL}, "tasks": {"action": 8, "follow_up": 8}, "experiments": {"hypothesis": 7, "experimental_result": 7}, "questions": {"question": 8, "schedule": 6}, "minutes": {}}


def _kind(claim): return claim.get("content_kind") or claim.get("kind")
def _tokens(value): return {x for x in re.findall(r"(?iu)[a-zа-яё0-9]+", str(value or "").casefold()) if len(x) > 2}


def _mandatory(claim):
    kind = _kind(claim)
    if kind in {"decision", "proposal"}: return claim.get("decision_status") == "accepted"
    if kind in {"action", "follow_up"}: return claim.get("task_status") in {"accepted", "self_committed", "explicit_self_commitment", "in_progress", "blocked", "completed"} and claim.get("canonical_task_anchor", True)
    if kind == "question": return claim.get("question_status") not in {"answered", "rhetorical", "superseded"}
    return kind in {"blocker", "correction", "experimental_result"}


def adaptive_budget(claims, episodes, *, minimum=7, maximum=120, view="minutes"):
    active = [x for x in claims if x.get("lifecycle", "active") == "active"]
    minutes = max([float(x.get("end", x.get("start", 0))) for x in active] or [0]) / 60
    threads = len({x.get("thread_id") for x in active if x.get("thread_id")}) or len(episodes)
    base = math.ceil(math.sqrt(max(1, minutes)) + 1.5 * threads + sum(_mandatory(x) for x in active) + len({_kind(x) for x in active}))
    configured = {"executive": (4, 5), "technical": (4, 6), "tasks": (3, 120), "experiments": (2, 5), "questions": (3, 5), "minutes": (16, 32)}
    low, high = configured.get(view, (minimum, maximum))
    return max(low, min(high, math.ceil(base * {"executive": .55, "technical": .85, "tasks": .7, "experiments": .8, "questions": .7, "minutes": 2.0}.get(view, 1))))


def _utility(claim, score_fn, view):
    risk = max([float(v or 0) for v in claim.get("risk", {}).values() if isinstance(v, (int, float))] or [0])
    closing_schedule = 8 if view == "questions" and _kind(claim) == "schedule" else 0
    text = str(claim.get("statement") or "")
    definition_only = bool(re.search(r"(?iu)\b(?:называется|определяется|это\s+когда|ширина\s*[—–-]\s*ширина)\b", text))
    non_work = bool(re.search(r"(?iu)\b(?:шутк|ха-ха|смешно|камышов|табзон)\w*", text))
    no_deliverable = _kind(claim) in {"action", "follow_up"} and not re.search(r"(?iu)\b(?:показ|переда|отправ|сдела|размет|провер|исправ|встро|подготов)\w*", text)
    raw_slot = bool(re.search(r"(?u)\b[a-z]+_[a-z_]+\b", text))
    penalty = 4 * definition_only + 8 * non_work + 4 * no_deliverable + 8 * raw_slot + 4 * (claim.get("verification_status") == "verification_unavailable")
    return float(score_fn(claim)) + VIEW_BOOST[view].get(_kind(claim), 0) + 2 * (1-risk) + 2 * _mandatory(claim) + closing_schedule - penalty


def _select(claims, score_fn, view, budget):
    eligible = [x for x in claims if not x.get("dialogue_only") and x.get("lifecycle", "active") == "active" and (view == "minutes" or _kind(x) in VIEW_KINDS[view] or (view == "tasks" and x.get("canonical_task_state_id")))]
    if view == "technical":
        eligible = [x for x in eligible if not re.search(r"(?iu)\b(?:ширина\s*[—–-]\s*ширина|называется|определяется)\b", str(x.get("statement") or ""))]
    if view == "tasks":
        eligible = [x for x in eligible if x.get("canonical_task_anchor", True)]
    if view == "questions":
        eligible = [x for x in eligible if x.get("question_status") not in {"answered", "rhetorical", "superseded"}]
    ranked = sorted(eligible, key=lambda x: (-_utility(x, score_fn, view), float(x.get("start", 0))))
    selected, token_sets = [], []
    for item in ranked:
        tokens = _tokens(item.get("statement"))
        if max((len(tokens & old) / max(1, min(len(tokens), len(old))) for old in token_sets), default=0) >= .78 and not _mandatory(item): continue
        if len(selected) >= budget: continue
        selected.append(item); token_sets.append(tokens)
    if view == "minutes":
        selected.sort(key=lambda x: (float(x.get("start", 0)), x.get("claim_id", "")))
    return selected, max(0, len(eligible) - len(selected))


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
    sources = [str(x.get("statement", "")) + " " + str(x.get("time_scope") or "") for x in items]
    return {"sentence_id": f"S{index:05d}", "summary_unit_id": unit["unit_id"], "episode_id": items[0].get("episode_id"), "claim_ids": unit["claim_ids"], "relation_ids": unit["relation_ids"], "intent": unit["role"], "allowed_numbers": [n for source in sources for n in re.findall(r"(?<!\w)\d+(?:[.,:]\d+)*(?:\s*%)?", source)], "allowed_quantities": [q for x in items for q in x.get("quantities", [])], "allowed_entities": [e.get("entity_id") for x in items for e in x.get("entities", []) if isinstance(e, dict)], "allowed_relation_markers": sorted(set().union(*(relation_markers(x.get("statement")) for x in items))), "allowed_speakers": sorted({s for x in items for s in x.get("speaker_refs", [])}), "allowed_assignees": sorted({x.get("assignee") for x in items if x.get("assignee")}), "polarity": [x.get("polarity") for x in items], "modality": [x.get("modality") for x in items], "conditions": [c for x in items for c in x.get("conditions", [])], "time_scope": [x.get("time_scope") for x in items if x.get("time_scope")], "decision_state": [x.get("decision_status") for x in items if x.get("decision_status")], "task_state": [x.get("task_status") for x in items if x.get("task_status")], "question_slots": [x.get("question_slots") for x in items if x.get("question_slots")], "forbidden_inferences": ["modality_upgrade", "condition_drop", "new_assignee", "new_quantity_binding", "unsupported_causality", "superseded_claim"], "max_sentences": max(1, len(items))}


def plan(claims, episodes, relations, score_fn, max_units=None):
    view_plans, union = {}, {}
    by_id = {x["claim_id"]: x for x in claims}
    for view in VIEW_KINDS:
        budget = max_units or adaptive_budget(claims, episodes, view=view)
        selected, overflow = _select(claims, score_fn, view, budget)
        view_units = _units(selected, claims, relations)
        view_plans[view] = {"objective": view, "budget": budget, "selected_count": len(selected), "overflow_count": overflow, "exclusion_reason": "hard_budget_or_duplicate" if overflow else None, "selected_claim_ids": [x["claim_id"] for x in selected], "summary_units": view_units, "sentence_plans": [_sentence(i, unit, by_id) for i, unit in enumerate(view_units, 1)], "dispositions": {x["claim_id"]: {"status": "selected", "section": view} for x in selected}}
        union.update({x["claim_id"]: x for x in selected})
    # A per-view editorial budget cannot silently drop a distinct canonical
    # work result. Include one anchor per task state, then let the renderer
    # label tentative states instead of pretending they were commitments.
    task_view = view_plans["tasks"]
    represented_states = {by_id[cid].get("canonical_task_state_id") for cid in task_view["selected_claim_ids"]}
    missing_tasks = [x for x in claims if x.get("lifecycle", "active") == "active"
                     and x.get("canonical_task_anchor", True)
                     and x.get("canonical_task_state_id")
                     and x.get("verification_status") != "verification_unavailable"
                     and x.get("canonical_task_state_id") not in represented_states]
    for claim in sorted(missing_tasks, key=lambda x: (float(x.get("start", 0)), x["claim_id"])):
        state_id = claim["canonical_task_state_id"]
        if state_id in represented_states: continue
        represented_states.add(state_id)
        task_view["selected_claim_ids"].append(claim["claim_id"])
        task_view["dispositions"][claim["claim_id"]] = {"status": "selected", "section": "tasks"}
        union[claim["claim_id"]] = claim
    task_view["selected_count"] = len(task_view["selected_claim_ids"])
    task_view["budget"] = max(task_view["budget"], task_view["selected_count"])
    task_view["overflow_count"] = 0
    task_view["exclusion_reason"] = None
    task_view["summary_units"] = _units([by_id[cid] for cid in task_view["selected_claim_ids"]], claims, relations)
    task_view["sentence_plans"] = [_sentence(i, unit, by_id) for i, unit in enumerate(task_view["summary_units"], 1)]
    # Fail-open verifier results are published only in an explicit quarantine
    # section. They still need the same sentence contract as every other
    # PublicItem, even when a view budget did not select the source claim.
    quarantine = [x for x in claims
                  if x.get("lifecycle", "active") == "active"
                  and x.get("verification_status") == "verification_unavailable"
                  and _kind(x) in {"action", "follow_up", "resource", "decision"}]
    quarantine_units = _units(quarantine, claims, relations)
    view_plans["requires_verification"] = {
        "objective": "requires_verification", "budget": len(quarantine),
        "selected_count": len(quarantine), "overflow_count": 0, "exclusion_reason": None,
        "selected_claim_ids": [x["claim_id"] for x in quarantine],
        "summary_units": quarantine_units,
        "sentence_plans": [_sentence(i, unit, by_id) for i, unit in enumerate(quarantine_units, 1)],
        "dispositions": {x["claim_id"]: {"status": "selected", "section": "requires_verification"} for x in quarantine},
    }
    union.update({x["claim_id"]: x for x in quarantine})
    ordered = sorted(union.values(), key=lambda x: float(x.get("start", 0)))
    units = _units(ordered, claims, relations)
    sentences = [_sentence(i, unit, by_id) for i, unit in enumerate(units, 1)]
    assert all(cross_episode_allowed(x["claim_ids"], x["relation_ids"], claims, relations) for x in sentences)
    public_sentence_plans = [sentence for view in view_plans.values() for sentence in view["sentence_plans"]]
    return {"schema": "SummaryPlanSchema", "schema_version": 4, "strategy": "canonical-state-hard-budget-v4", "adaptive_budget": view_plans["minutes"]["budget"], "selected_claim_ids": [x["claim_id"] for x in ordered], "channels": {"mandatory": [x["claim_id"] for x in ordered if _mandatory(x)], "balanced_core": [x["claim_id"] for x in ordered if not _mandatory(x)], "optional_detail": []}, "episode_coverage": sorted({x.get("episode_id") for x in ordered if x.get("episode_id")}), "summary_units": units, "sentence_plans": sentences, "public_sentence_plans": public_sentence_plans, "paragraph_plans": [{"paragraph_id": f"P{i:02d}", **u} for i, u in enumerate(units, 1)], "view_plans": view_plans}
