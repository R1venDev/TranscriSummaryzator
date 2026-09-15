"""Purpose-specific projections from one MeetingState; no new reasoning."""
from __future__ import annotations


def project_views(state, selected_ids):
    """Project only claims admitted by the planner for each public view."""
    claims = [x for x in state.get("claims", []) if x.get("lifecycle", "active") == "active"]
    if isinstance(selected_ids, dict):
        plans = selected_ids.get("view_plans", selected_ids)
        ids_for = lambda name: set(plans.get(name, {}).get("selected_claim_ids", []))
    else:
        shared = set(selected_ids)
        ids_for = lambda _name: shared
    def selected(name):
        allowed = ids_for(name)
        return [x for x in claims if x.get("claim_id") in allowed]
    def kinds(name, *values):
        return [x for x in selected(name) if (x.get("content_kind") or x.get("kind")) in values]
    decisions = [x for x in kinds("executive", "decision", "proposal") if x.get("decision_status") == "accepted" and x.get("evidence_ids")]
    return {
        "executive": selected("executive")[:10],
        "technical": kinds("technical", "trading_rule", "system_rule", "definition", "metric", "experimental_result", "design_choice", "constraint", "dependency"),
        "tasks": [
            {**task, "statement": task.get("deliverable") or task.get("description")}
            for task in state.get("task_states", state.get("tasks", []))
            if set(task.get("source_proposition_ids", [task.get("proposition_id")])) & {x.get("proposition_id") for x in selected("tasks")}
        ],
        "decisions": decisions,
        "mentioned_rules": [x for x in kinds("technical", "trading_rule", "system_rule") if x.get("decision_status") != "accepted"],
        "experiments": kinds("experiments", "experimental_result", "hypothesis", "metric"),
        "open_questions": [x for x in kinds("questions", "question", "schedule") if x.get("question_status") not in {"answered", "rhetorical", "superseded"}],
        "minutes": sorted(selected("minutes"), key=lambda x: (x.get("episode_id", ""), float(x.get("start", 0)))),
    }
