"""Build MeetingState v2 from the proven v18 event graph."""
from __future__ import annotations
import re
from .episodes import build_episodes, build_threads
from .graph import normalize_relations, reconcile_latest_state

RELATION_MAP = {"accepted_by": "accepts_assignment", "conflicts_with": "contradicts"}


def _confidence(event):
    risk = event.get("risk", {})
    recognition = risk.get("recognition", {}) if isinstance(risk.get("recognition"), dict) else {}
    speaker = risk.get("speaker", {}) if isinstance(risk.get("speaker"), dict) else {}
    return {
        "recognition": recognition.get("confidence"),
        "speaker_identity": speaker.get("confidence") or risk.get("speaker_confidence_min"),
        "semantic_support": event.get("semantic_support"),
        "relation_support": None,
        "modality": event.get("modality_confidence"),
        "quantity": event.get("quantity_confidence"),
    }


def _risk(event):
    signals = set(event.get("risk", {}).get("signals", []))
    confidence = _confidence(event)
    inverse = lambda value: 1.0 - value if isinstance(value, (int, float)) else 0.5
    text = str(event.get("presentation") or "")
    return {
        "recognition": inverse(confidence["recognition"]),
        "speaker": inverse(confidence["speaker_identity"]),
        "number": 0.7 if re.search(r"\d|%", text) and "number" in signals else 0.0,
        "negation": 0.7 if "negation" in signals else 0.0,
        "modality": 0.7 if "modality" in signals else inverse(confidence["modality"]),
        "relation": 0.7 if "relation" in signals else 0.0,
        "task_assignment": 0.8 if event.get("content_kind") == "task" and not event.get("mentioned_participant_ids") else 0.0,
    }


def _question(record, claim_id):
    requested = list(record.get("requested_slots", []))
    answered = list(record.get("answered_slots", []))
    status = record.get("question_status") or "unanswered"
    if status in {"resolved", "unclear"}:
        status = "answered" if status == "resolved" else "unanswered"
    return {"question_id": f"Q{claim_id[1:]}", "claim_id": claim_id, "requested_slots": requested, "answered_slots": answered, "missing_slots": [x for x in requested if x not in answered], "answer_claim_ids": list(record.get("answer_record_ids", [])), "status": status}


def _task(record, claim_id):
    modality = record.get("modality")
    assignees = list(record.get("assignees", []))
    confirmations = list(record.get("confirmation_evidence_ids", []))
    if confirmations and assignees:
        status, strength = "accepted", "explicit"
    elif modality == "committed" and assignees:
        status, strength = "explicit_self_commitment", "explicit"
    elif modality == "tentative" and assignees:
        status, strength = "tentative_self_commitment", "tentative"
    elif assignees:
        status, strength = "assigned", "none"
    else:
        status, strength = "proposed", "none"
    return {"task_id": f"T{claim_id[1:]}", "claim_id": claim_id, "description": record.get("statement"), "proposed_by": (record.get("proposed_by") or [None])[0], "assignee": assignees[0] if len(assignees) == 1 else None, "assignment_evidence_ids": list(record.get("evidence_ids", [])), "acceptance_evidence_ids": confirmations, "commitment_strength": strength, "deadline": record.get("time_expression"), "conditions": list(record.get("conditions", [])), "status": status}


def build_meeting_state(legacy_state, records, provenance=None):
    provenance = provenance or {}
    record_by_id = {x.get("record_id"): x for x in records}
    event_to_claim = {x.get("event_id"): x.get("claim_id") for x in legacy_state.get("events", [])}
    claims = []
    for event in legacy_state.get("events", []):
        record = record_by_id.get(event.get("source_record_id"), {})
        raw_kind = record.get("kind") or event.get("content_kind") or "observation"
        if raw_kind not in {"observation", "current_state", "problem", "definition", "metric", "experimental_result", "hypothesis", "proposal", "alternative", "decision", "action", "goal", "target", "constraint", "assumption", "trading_rule", "system_rule", "design_choice", "dataset", "resource", "risk", "dependency", "blocker", "follow_up", "correction", "rejected_option", "schedule", "question"}:
            raw_kind = "observation"
        claim = {
            "claim_id": event["claim_id"], "kind": raw_kind,
            "statement": event.get("presentation") or record.get("statement") or "",
            "evidence_ids": list(event.get("evidence_ids", [])), "speaker_refs": list(event.get("speaker_ids", [])),
            "modality": record.get("modality", "asserted"), "lifecycle": "active",
            "confidence": _confidence(event), "risk": _risk(event), "quantities": list(event.get("quantities", [])),
            "conditions": list(event.get("conditions", [])), "topic": event.get("topic"), "start": event.get("start", 0),
            "end": record.get("end", event.get("start", 0)), "source_record_id": event.get("source_record_id"),
        }
        if claim["kind"] == "question":
            claim["question_status"] = _question(record, claim["claim_id"])["status"]
        claims.append(claim)
    relation_input = []
    for relation in legacy_state.get("relations", []):
        kind = RELATION_MAP.get(relation.get("relation"), relation.get("relation"))
        relation_input.append({**relation, "type": kind, "source_claim_id": event_to_claim.get(relation.get("source_event")), "target_claim_id": event_to_claim.get(relation.get("target_event"))})
    relations = normalize_relations(relation_input, claims)
    reconcile_latest_state(claims, relations)
    episodes = build_episodes(claims)
    threads = build_threads(episodes, claims)
    questions = [_question(record_by_id.get(x.get("source_record_id"), {}), x["claim_id"]) for x in claims if x.get("kind") == "question"]
    tasks = [_task(record_by_id.get(x.get("source_record_id"), {}), x["claim_id"]) for x in claims if x.get("kind") == "action"]
    decisions = [{"decision_id": f"D{x['claim_id'][1:]}", "claim_id": x["claim_id"], "proposal_claim_ids": [], "acceptance_claim_ids": [], "decision_makers": x.get("speaker_refs", []), "scope": x.get("statement"), "conditions": x.get("conditions", []), "status": "accepted" if x.get("modality") in {"asserted", "committed"} else "candidate"} for x in claims if x.get("kind") == "decision"]
    for task in tasks:
        next(x for x in claims if x["claim_id"] == task["claim_id"])["task_status"] = task["status"]
    for decision in decisions:
        next(x for x in claims if x["claim_id"] == decision["claim_id"])["decision_status"] = decision["status"]
    return {"schema": "MeetingStateSchema", "schema_version": 2, "meeting_id": legacy_state.get("state_id"), "provenance": provenance, "episodes": episodes, "threads": threads, "claims": claims, "relations": relations, "tasks": tasks, "questions": questions, "decisions": decisions, "active_rules": [x for x in claims if x.get("kind") in {"trading_rule", "system_rule"} and x.get("lifecycle") == "active"], "experimental_results": [x for x in claims if x.get("kind") == "experimental_result" and x.get("lifecycle") == "active"], "open_threads": [x for x in threads if x.get("state") == "open"], "uncertainty_summary": {"claims_with_recognition_risk": sum(x["risk"]["recognition"] >= .5 for x in claims), "claims_with_speaker_risk": sum(x["risk"]["speaker"] >= .5 for x in claims)}}
