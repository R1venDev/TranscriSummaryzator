"""Purpose-specific projections from one MeetingState; no new reasoning."""
from __future__ import annotations


def project_views(state, selected_ids):
    claims = [x for x in state.get("claims", []) if x.get("lifecycle", "active") == "active"]
    selected = [x for x in claims if x.get("claim_id") in set(selected_ids)]
    kinds = lambda *values: [x for x in claims if (x.get("content_kind") or x.get("kind")) in values]
    return {
        "executive": selected[:10],
        "technical": kinds("rule", "trading_rule", "system_rule", "metric", "experiment", "experimental_result", "design", "design_choice"),
        "tasks": kinds("action"),
        "decisions": [x for x in kinds("decision", "state", "rule", "design") if x.get("decision_status") == "accepted"],
        "experiments": kinds("experiment", "experimental_result", "hypothesis", "metric"),
        "open_questions": [x for x in kinds("question", "question_content") if x.get("question_status") not in {"answered", "rhetorical", "superseded"}],
        "minutes": sorted(selected, key=lambda x: (x.get("episode_id", ""), float(x.get("start", 0)))),
    }
