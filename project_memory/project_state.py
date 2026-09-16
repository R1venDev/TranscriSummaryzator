"""Evidence-separated ProjectState and meeting-to-meeting delta."""
from __future__ import annotations
import hashlib, json

CATEGORY = {"decision": "active_decisions", "action": "active_tasks", "definition": "definitions", "system_rule": "system_rules", "trading_rule": "trading_rules", "hypothesis": "open_experiments", "experimental_result": "known_results", "question": "open_questions", "blocker": "blockers"}


def update_project_state(previous, meeting):
    previous = previous or {"schema": "ProjectStateSchema", "schema_version": 1}
    state = {**previous, "schema": "ProjectStateSchema", "schema_version": 1, "source_meeting_ids": list(dict.fromkeys(previous.get("source_meeting_ids", []) + [meeting.get("meeting_id")]))}
    for field in set(CATEGORY.values()):
        state[field] = [dict(x) for x in previous.get(field, [])]
    for claim in meeting.get("claims", []):
        field = CATEGORY.get(claim.get("kind"))
        if not field or claim.get("lifecycle") != "active":
            continue
        if field == "open_questions" and claim.get("question_status") == "answered":
            continue
        item = {"claim_id": claim["claim_id"], "statement": claim["statement"], "meeting_id": meeting.get("meeting_id"), "evidence_ids": claim.get("evidence_ids", []), "context_only": True,
                "semantic_state": {key: claim.get(key) for key in ("task_status", "decision_status", "question_status", "time_scope", "assignee", "polarity", "modality", "lifecycle")},
                "transition_provenance": {"event_ids": claim.get("event_ids", []), "source_record_ids": claim.get("source_record_ids", [])}}
        state[field] = [x for x in state[field] if x.get("statement") != item["statement"]] + [item]
    payload = json.dumps(state, ensure_ascii=False, sort_keys=True).encode()
    state["project_state_id"] = "PS" + hashlib.sha256(payload).hexdigest()[:16]
    return state


def delta(previous, current):
    previous = previous or {}
    changes = []
    for field in set(CATEGORY.values()):
        old = {x.get("statement"): x for x in previous.get(field, [])}
        new = {x.get("statement"): x for x in current.get(field, [])}
        changes.extend({"status": "NEW", "category": field, **item} for key, item in new.items() if key not in old)
        changes.extend({"status": "RESOLVED" if field in {"open_questions", "blockers", "active_tasks"} else "CHANGED", "category": field, **item} for key, item in old.items() if key not in new)
        for key in new.keys() & old.keys():
            changed = old[key].get("semantic_state") != new[key].get("semantic_state")
            status = "CHANGED" if changed else "STILL_OPEN" if field in {"open_questions", "blockers", "active_tasks"} else "CONFIRMED"
            changes.append({"status": status, "category": field, "previous_semantic_state": old[key].get("semantic_state"), **new[key]})
    return {"schema_version": 1, "changes": changes}
