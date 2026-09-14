"""Deterministic state machines over propositions, dialogue events and relations."""
from __future__ import annotations
from .questions import verify_slot_entailment


def _incoming(target, relations, kinds):
    return [x for x in relations if x.get("target_proposition_id") == target and x.get("type") in kinds]


def reduce_decisions(propositions, events, relations):
    by_prop = {x["proposition_id"]: x for x in propositions}
    result = []
    for event in events:
        prop = by_prop[event["proposition_id"]]
        candidates = event["speech_act"] in {"propose", "decide"} or prop["content_kind"] in {"rule", "design"}
        if not candidates:
            continue
        accepts = _incoming(prop["proposition_id"], relations, {"accepts", "confirms"})
        rejects = _incoming(prop["proposition_id"], relations, {"rejects"})
        supersedes = _incoming(prop["proposition_id"], relations, {"supersedes", "corrects"})
        if rejects: status = "rejected"
        elif supersedes: status = "superseded"
        elif event["speech_act"] == "decide" or accepts: status = "accepted"
        elif event["speech_act"] == "defer": status = "deferred"
        else: status = "candidate"
        result.append({"decision_id": "D" + prop["proposition_id"][1:], "proposition_id": prop["proposition_id"], "status": status, "acceptance_relation_ids": [x["relation_id"] for x in accepts], "conditions": prop["conditions"], "evidence_ids": prop["evidence_ids"]})
    return result


def reduce_tasks(propositions, events, relations, records):
    by_record = {x.get("record_id"): x for x in records}
    result = []
    for event in events:
        prop = next(x for x in propositions if x["proposition_id"] == event["proposition_id"])
        if prop["content_kind"] != "action":
            continue
        record = by_record.get(event.get("source_record_id"), {})
        owners = list(record.get("assignees", []))
        confirmations = list(record.get("confirmation_evidence_ids", []))
        if event["speech_act"] == "commit" and event.get("speaker") and not owners:
            owners = [event["speaker"]]
        if record.get("completion_evidence_ids"): status = "completed"
        elif record.get("blocked"): status = "blocked"
        elif confirmations and owners: status = "accepted"
        elif event["speech_act"] == "commit" and owners: status = "self_committed"
        elif owners: status = "assigned"
        elif event["speech_act"] == "propose": status = "proposed"
        else: status = "idea"
        result.append({"task_id": "T" + prop["proposition_id"][1:], "proposition_id": prop["proposition_id"], "source_record_id": event.get("source_record_id"), "description": prop["statement"], "deliverable": prop["statement"], "owner": owners[0] if len(owners) == 1 else None, "assignee": owners[0] if len(owners) == 1 else None, "proposed_by": event.get("speaker"), "acceptance_evidence_ids": confirmations, "deadline": record.get("time_expression"), "conditions": prop["conditions"], "completion_criterion": record.get("completion_criterion"), "status": status, "automation_eligible": status in {"accepted", "self_committed"} and len(owners) == 1})
    return result


def reduce_questions(propositions, events, relations, records):
    by_record = {x.get("record_id"): x for x in records}
    result = []
    for event in events:
        prop = next(x for x in propositions if x["proposition_id"] == event["proposition_id"])
        if event["speech_act"] != "ask" and prop["content_kind"] != "question_content":
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
        elif record.get("question_status") in {"deferred", "requires_external_verification", "rhetorical", "superseded"}: status = record["question_status"]
        else: status = "unanswered"
        result.append({"question_id": "Q" + prop["proposition_id"][1:], "proposition_id": prop["proposition_id"], "source_record_id": event.get("source_record_id"), "intent": record.get("question_intent") or "unknown", "requested_slots": requested, "answered_slots": entailed, "missing_slots": missing, "candidate_answer_ids": list(record.get("answer_record_ids", [])), "answer_record_ids": list(record.get("answer_record_ids", [])), "answer_relation_ids": [x["relation_id"] for x in answer_relations], "status": status})
    return result


def reduce_rules_and_experiments(propositions, events, decisions):
    decision_by_prop = {x["proposition_id"]: x for x in decisions}
    rules, experiments = [], []
    for prop in propositions:
        if prop["content_kind"] == "rule":
            decision = decision_by_prop.get(prop["proposition_id"])
            rules.append({"rule_id": "RUL" + prop["proposition_id"][1:], "proposition_id": prop["proposition_id"], "rule": prop["statement"], "scope": prop["scope"], "conditions": prop["conditions"], "exceptions": [], "status": decision["status"] if decision else "described_existing_rule", "evidence_ids": prop["evidence_ids"]})
        if prop["content_kind"] == "experiment":
            experiments.append({"experiment_id": "EXP" + prop["proposition_id"][1:], "proposition_id": prop["proposition_id"], "hypothesis": prop["statement"], "motivation": None, "method": None, "dataset": None, "independent_variable": None, "metrics": [x for x in prop["quantities"]], "baseline": None, "expected_direction": None, "owner": None, "status": "result" if any(e["proposition_id"] == prop["proposition_id"] and e["speech_act"] == "assert" for e in events) else "planned", "evidence_ids": prop["evidence_ids"]})
    return rules, experiments
