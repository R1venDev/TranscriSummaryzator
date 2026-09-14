"""Deterministic state machines over propositions, dialogue events and relations."""
from __future__ import annotations
from .questions import verify_slot_entailment


def _incoming(target, relations, kinds):
    return [x for x in relations if x.get("target_proposition_id") == target and x.get("type") in kinds]


def reduce_decisions(propositions, events, relations):
    by_prop = {x["proposition_id"]: x for x in propositions}
    result, processed = [], set()
    for event in events:
        if event["proposition_id"] in processed:
            continue
        processed.add(event["proposition_id"])
        prop = by_prop[event["proposition_id"]]
        prop_events = [x for x in events if x["proposition_id"] == prop["proposition_id"]]
        latest = max(prop_events, key=lambda x: x.get("timestamp", 0))
        acts = {x["speech_act"] for x in prop_events}
        candidates = bool(acts & {"propose", "decide"}) or prop["content_kind"] in {"proposal", "decision"}
        if not candidates:
            continue
        accepts = _incoming(prop["proposition_id"], relations, {"accepts", "confirms"})
        rejects = _incoming(prop["proposition_id"], relations, {"rejects"})
        supersedes = _incoming(prop["proposition_id"], relations, {"supersedes", "corrects"})
        if rejects: status = "rejected"
        elif supersedes: status = "superseded"
        elif "decide" in acts or accepts: status = "accepted"
        elif "defer" in acts: status = "deferred"
        else: status = "candidate"
        result.append({"decision_id": "D" + prop["proposition_id"][1:], "proposition_id": prop["proposition_id"], "source_record_id": latest.get("source_record_id"), "status": status, "acceptance_relation_ids": [x["relation_id"] for x in accepts], "decision_evidence_ids": list(dict.fromkeys(prop["evidence_ids"] + [e for x in accepts for e in x.get("evidence_ids", [])])), "acceptance_check": "entailed" if status == "accepted" else "not_entailed", "conditions": prop["conditions"], "evidence_ids": prop["evidence_ids"]})
    return result


def reduce_tasks(propositions, events, relations, records):
    by_record = {x.get("record_id"): x for x in records}
    result, processed = [], set()
    for event in events:
        if event["proposition_id"] in processed:
            continue
        processed.add(event["proposition_id"])
        prop = next(x for x in propositions if x["proposition_id"] == event["proposition_id"])
        if prop["content_kind"] not in {"action", "follow_up"}:
            continue
        prop_events = [x for x in events if x["proposition_id"] == prop["proposition_id"]]
        event = max(prop_events, key=lambda x: x.get("timestamp", 0))
        record = by_record.get(event.get("source_record_id"), {})
        owners = list(record.get("assignees", []))
        confirmations = list(record.get("confirmation_evidence_ids", []))
        commit_event = next((x for x in reversed(sorted(prop_events, key=lambda x: x.get("timestamp", 0))) if x["speech_act"] == "commit"), None)
        if commit_event and commit_event.get("speaker") and not owners:
            owners = [commit_event["speaker"]]
        if record.get("completion_evidence_ids"): status = "completed"
        elif record.get("blocked"): status = "blocked"
        elif confirmations and owners: status = "accepted"
        elif commit_event and owners: status = "self_committed"
        elif owners: status = "assigned"
        elif event["speech_act"] == "propose": status = "proposed"
        else: status = "idea"
        superseded_by = [x["source_proposition_id"] for x in _incoming(prop["proposition_id"], relations, {"corrects", "supersedes"})]
        scope_state = "superseded" if superseded_by else "ambiguous" if record.get("scope_ambiguous") else "active"
        if superseded_by:
            status = "superseded"
        explicit_automation = record.get("automation_eligible")
        automation = explicit_automation if isinstance(explicit_automation, bool) else (False if status in {"idea", "proposed", "assigned", "superseded"} else "unknown")
        result.append({"task_id": "T" + prop["proposition_id"][1:], "proposition_id": prop["proposition_id"], "source_record_id": event.get("source_record_id"), "description": prop["statement"], "deliverable": prop["statement"], "owner": owners[0] if len(owners) == 1 else None, "assignee": owners[0] if len(owners) == 1 else None, "assignees": owners, "assignee_confidence": record.get("assignee_confidence"), "proposed_by": event.get("speaker"), "acceptance_evidence_ids": confirmations, "deadline": record.get("time_expression"), "due": record.get("time_expression"), "conditions": prop["conditions"], "completion_criterion": record.get("completion_criterion"), "status": status, "task_status": status, "current_scope": record.get("scope") or record.get("time_scope") or record.get("time_expression"), "scope_state": scope_state, "superseded_by": superseded_by, "automation_eligible": automation})
    return result


def reduce_questions(propositions, events, relations, records):
    by_record = {x.get("record_id"): x for x in records}
    result, processed = [], set()
    for event in events:
        if event["proposition_id"] in processed:
            continue
        processed.add(event["proposition_id"])
        prop = next(x for x in propositions if x["proposition_id"] == event["proposition_id"])
        if event["speech_act"] != "ask" and prop["content_kind"] != "question":
            continue
        record = by_record.get(event.get("source_record_id"), {})
        requested = list(record.get("requested_slots", []))
        explicit_answered = list(record.get("answered_slots", []))
        checks = [verify_slot_entailment(requested, by_record.get(answer_id, {})) for answer_id in record.get("answer_record_ids", [])]
        inferred = {slot for check in checks for slot in check["entailed_slots"]} if checks else set()
        entailed = [x for x in requested if x in explicit_answered or x in inferred]
        missing = [x for x in requested if x not in entailed]
        answer_relations = _incoming(prop["proposition_id"], relations, {"answers", "partially_answers", "resolves"})
        if requested and not missing and answer_relations: status = "answered"
        elif entailed and answer_relations: status = "partially_answered"
        elif _incoming(prop["proposition_id"], relations, {"tentatively_answers"}): status = "tentatively_answered"
        elif record.get("question_status") in {"deferred", "requires_external_verification", "rhetorical", "superseded"}: status = record["question_status"]
        else: status = "unanswered"
        result.append({"question_id": "Q" + prop["proposition_id"][1:], "proposition_id": prop["proposition_id"], "source_record_id": event.get("source_record_id"), "intent": record.get("question_intent") or "unknown", "requested_slots": requested, "answered_slots": entailed, "missing_slots": missing, "candidate_answer_ids": list(record.get("answer_record_ids", [])), "answer_record_ids": list(record.get("answer_record_ids", [])), "answer_relation_ids": [x["relation_id"] for x in answer_relations], "status": status})
    return result


def reduce_rules_and_experiments(propositions, events, decisions):
    decision_by_prop = {x["proposition_id"]: x for x in decisions}
    rules, experiments = [], []
    for prop in propositions:
        if prop["content_kind"] in {"trading_rule", "system_rule"}:
            decision = decision_by_prop.get(prop["proposition_id"])
            rules.append({"rule_id": "RUL" + prop["proposition_id"][1:], "proposition_id": prop["proposition_id"], "rule": prop["statement"], "scope": prop["scope"], "conditions": prop["conditions"], "exceptions": [], "status": decision["status"] if decision else "described_existing_rule", "evidence_ids": prop["evidence_ids"]})
        if prop["content_kind"] in {"hypothesis", "experimental_result"}:
            experiments.append({"experiment_id": "EXP" + prop["proposition_id"][1:], "proposition_id": prop["proposition_id"], "content_kind": prop["content_kind"], "hypothesis": prop["statement"] if prop["content_kind"] == "hypothesis" else None, "result": prop["statement"] if prop["content_kind"] == "experimental_result" else None, "motivation": None, "method": None, "dataset": None, "independent_variable": None, "metrics": [x for x in prop["quantities"]], "baseline": None, "expected_direction": None, "owner": None, "status": "result" if prop["content_kind"] == "experimental_result" else "planned", "evidence_ids": prop["evidence_ids"]})
    return rules, experiments
