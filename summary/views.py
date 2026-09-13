"""Purpose-specific projections from one MeetingState; no new reasoning."""
from __future__ import annotations


def project_views(state, selected_ids):
    claims = [x for x in state.get("claims", []) if x.get("lifecycle", "active") == "active"]
    selected = [x for x in claims if x.get("claim_id") in set(selected_ids)]
    kinds = lambda *values: [x for x in claims if x.get("kind") in values]
    return {
        "executive": selected[:10],
        "technical": [x for x in selected if x.get("kind") not in {"schedule", "question"}],
        "tasks": kinds("action"),
        "decisions": [x for x in kinds("decision") if x.get("decision_status", "accepted") == "accepted"],
        "experiments": kinds("experimental_result", "hypothesis", "metric"),
        "open_questions": [x for x in kinds("question") if x.get("question_status") not in {"answered", "rhetorical", "superseded"}],
        "minutes": sorted(selected, key=lambda x: (x.get("episode_id", ""), float(x.get("start", 0)))),
    }
