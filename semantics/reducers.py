"""Deterministic state machines over propositions, dialogue events and relations."""
from __future__ import annotations
import hashlib
import re
from .questions import verify_slot_entailment


PROPOSAL_WORDING_RE = re.compile(r"(?iu)\b(?:предлагалось|предлагает|можно|стоит|нужно\s+бы|планируется|планирует)\b")
FIRST_PERSON_COMMIT_RE = re.compile(r"(?iu)\b(?:я\s+(?:сделаю|отправлю|передам|дам|кину|буду|возьмусь)|i\s+will)\b")
COUNT_WORDS = {"один": 1, "одного": 1, "одну": 1, "два": 2, "две": 2, "трех": 3, "трёх": 3}
SCOPE_VALUE_RE = re.compile(r"(?iu)\b(?P<count>\d+|один|одного|одну|два|две|тр[её]х)?\s*(?P<unit>месяц(?:а|ев|ем)?|недел(?:я|и|ь|ю)|д(?:ень|ня|ней)|год(?:а|ов)?)\b")
YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
TOKEN_RE = re.compile(r"(?iu)[a-zа-яё0-9]+")
TASK_STOP = {"говорящий", "предлагает", "планирует", "будут", "параллельно", "надо", "нужно", "чтобы", "участник"}


def _tokens(value):
    return {x for x in TOKEN_RE.findall(str(value or "").casefold()) if len(x) > 3 and x not in TASK_STOP}


def _task_similarity(left, right):
    a, b = _tokens(left), _tokens(right)
    return len(a & b) / max(1, min(len(a), len(b)))


def _incoming(target, relations, kinds):
    return [x for x in relations if x.get("target_proposition_id") == target and x.get("type") in kinds]


def _scope_value(text):
    match = SCOPE_VALUE_RE.search(str(text or ""))
    if not match:
        return None
    raw = (match.group("count") or "один").casefold()
    count = int(raw) if raw.isdigit() else COUNT_WORDS.get(raw, 1)
    unit = match.group("unit").casefold()
    stem = "месяц" if unit.startswith("месяц") else "неделя" if unit.startswith("недел") else "день" if unit.startswith("д") else "год"
    forms = {"месяц": ("месяц", "месяца", "месяцев"), "неделя": ("неделя", "недели", "недель"), "день": ("день", "дня", "дней"), "год": ("год", "года", "лет")}[stem]
    form = forms[0] if count % 10 == 1 and count % 100 != 11 else forms[1] if count % 10 in {2, 3, 4} and count % 100 not in {12, 13, 14} else forms[2]
    return f"{count} {form}"


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
    atomic, processed = [], set()
    for event in events:
        if event["proposition_id"] in processed:
            continue
        processed.add(event["proposition_id"])
        prop = next(x for x in propositions if x["proposition_id"] == event["proposition_id"])
        prop_events = [x for x in events if x["proposition_id"] == prop["proposition_id"]]
        event = max(prop_events, key=lambda x: x.get("timestamp", 0))
        record = by_record.get(event.get("source_record_id"), {})
        commitment_candidate = event["speech_act"] == "commit" or record.get("commitment_strength") in {"explicit", "implicit"} or prop["content_kind"] in {"action", "follow_up"}
        if not commitment_candidate:
            continue
        owners = list(record.get("assignees", []))
        confirmations = list(record.get("confirmation_evidence_ids", []))
        acceptance_relations = _incoming(prop["proposition_id"], relations, {"accepts", "confirms", "accepts_assignment"})
        event_by_prop = {x["proposition_id"]: x for x in events}
        accepted_by_owner = [x for x in acceptance_relations if event_by_prop.get(x.get("source_proposition_id"), {}).get("speaker") in owners]
        if accepted_by_owner:
            confirmations = list(dict.fromkeys(confirmations + [e for x in accepted_by_owner for e in x.get("evidence_ids", [])]))
        commit_event = next((x for x in reversed(sorted(prop_events, key=lambda x: x.get("timestamp", 0))) if x["speech_act"] == "commit"), None)
        if commit_event and commit_event.get("speaker") and not owners:
            owners = [commit_event["speaker"]]
        statement = str(prop.get("statement") or "")
        explicit_commitment = bool(
            record.get("commitment_strength") == "explicit"
            or FIRST_PERSON_COMMIT_RE.search(statement)
        ) and not PROPOSAL_WORDING_RE.search(statement)
        commitment_actor = record.get("commitment_actor") or (commit_event.get("speaker") if explicit_commitment and commit_event else None)
        assignment_status = str(record.get("assignment_status") or "unknown")
        uncertainty_reasons = list(record.get("uncertainty", {}).get("reasons", []))
        ambiguous_owner = len(owners) != 1 or len(event.get("speaker_candidates", [])) != 1
        if record.get("completion_evidence_ids"): status = "completed"
        elif record.get("blocked"): status = "blocked"
        elif confirmations and owners and (assignment_status in {"confirmed", "accepted"} or accepted_by_owner): status = "accepted"
        elif explicit_commitment and not ambiguous_owner and commitment_actor == owners[0]: status = "self_committed"
        elif owners and assignment_status in {"confirmed", "accepted", "assigned"}: status = "assigned"
        elif owners: status = "assigned_pending"
        elif event["speech_act"] == "propose": status = "proposed"
        else: status = "idea"
        superseded_by = [x["source_proposition_id"] for x in _incoming(prop["proposition_id"], relations, {"corrects", "supersedes"})]
        scope_state = "superseded" if superseded_by else "ambiguous" if record.get("scope_ambiguous") else "active"
        if superseded_by:
            status = "superseded"
        explicit_automation = record.get("automation_eligible")
        automation = explicit_automation if isinstance(explicit_automation, bool) else (False if status in {"idea", "proposed", "assigned", "assigned_pending", "superseded"} else "unknown")
        scope = record.get("scope") or record.get("time_scope")
        data_origin = record.get("data_origin") or next(iter(YEAR_RE.findall(statement)), None)
        scope_relations, scope_evidence, scope_words = [], [], []
        revisers = sorted(_incoming(prop["proposition_id"], relations, {"revises_scope"}), key=lambda x: x.get("confidence", 0))
        for relation in revisers:
            source_prop = next((x for x in propositions if x["proposition_id"] == relation.get("source_proposition_id")), None)
            value = _scope_value(source_prop.get("statement") if source_prop else "")
            if not value:
                continue
            scope = value
            scope_relations.append(relation["relation_id"])
            scope_evidence.extend(relation.get("evidence_ids", []))
            source_records = source_prop.get("source_record_ids", []) if source_prop else []
            scope_words.extend(w for rid in source_records for w in by_record.get(rid, {}).get("source_word_ids", []))
        atomic.append({"task_id": "T" + prop["proposition_id"][1:], "proposition_id": prop["proposition_id"], "source_proposition_ids": [prop["proposition_id"]], "source_record_id": event.get("source_record_id"), "source_record_ids": [event.get("source_record_id")], "description": prop["statement"], "deliverable": prop["statement"], "owner": owners[0] if len(owners) == 1 else None, "assignee": owners[0] if len(owners) == 1 else None, "assignees": owners, "assignee_confidence": record.get("assignee_confidence"), "proposed_by": event.get("speaker"), "commitment_strength": "explicit" if explicit_commitment else "implicit" if commit_event else "none", "commitment_actor": commitment_actor, "assignment_actor": event.get("speaker"), "assignment_target": owners[0] if len(owners) == 1 else None, "acceptance_relation_ids": [x["relation_id"] for x in accepted_by_owner], "acceptance_evidence_ids": confirmations, "scope_relation_ids": scope_relations, "uncertainty_reasons": sorted(set(uncertainty_reasons + (["ambiguous_owner"] if ambiguous_owner else []))), "deadline": record.get("time_expression"), "due": record.get("time_expression"), "conditions": prop["conditions"], "completion_criterion": record.get("completion_criterion"), "status": status, "task_status": status, "current_scope": scope, "data_origin": data_origin, "scope_state": scope_state, "scope_confidence": "high" if scope_relations else "unknown" if not scope else "source", "superseded_scopes": [], "superseded_by": superseded_by, "automation_eligible": automation, "evidence_ids": list(dict.fromkeys(prop.get("evidence_ids", []) + scope_evidence)), "source_word_ids": list(dict.fromkeys(list(record.get("source_word_ids", [])) + scope_words)), "start": float(record.get("start", 0))})
    # Canonical task envelopes: one state is consumed by every public/API view.
    groups = []
    for item in sorted(atomic, key=lambda x: x["start"]):
        match = next((g for g in reversed(groups) if item.get("owner") == g.get("owner") and item["start"] - g["start"] <= 120 and _task_similarity(item["description"], g["description"]) >= .34), None)
        if not match:
            groups.append(item)
            continue
        match["source_proposition_ids"].extend(item["source_proposition_ids"])
        match["source_record_ids"].extend(item["source_record_ids"])
        match["evidence_ids"] = list(dict.fromkeys(match["evidence_ids"] + item["evidence_ids"]))
        match["source_word_ids"] = list(dict.fromkeys(match["source_word_ids"] + item["source_word_ids"]))
        match["uncertainty_reasons"] = sorted(set(match["uncertainty_reasons"] + item["uncertainty_reasons"]))
        if item["status"] in {"accepted", "self_committed", "completed"}: match["status"] = match["task_status"] = item["status"]
        if item.get("current_scope"): match["current_scope"] = item["current_scope"]
    for item in groups:
        raw = "|".join(sorted(item["source_proposition_ids"]))
        item["task_id"] = "TS" + hashlib.sha256(raw.encode()).hexdigest()[:14]
        item["canonical"] = True
    return groups


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
        upstream = str(record.get("question_status") or "").casefold()
        answer_ids = list(record.get("answer_record_ids", []))
        answer_evidence = list(record.get("answer_evidence_ids", []))
        has_answer_support = bool(answer_relations or answer_ids or answer_evidence)
        if upstream in {"answered", "resolved"} and (has_answer_support or not requested):
            status, entailed, missing = "answered", requested or explicit_answered, []
        elif upstream == "partially_answered" and has_answer_support:
            status = "partially_answered"
        elif requested and not missing and answer_relations: status = "answered"
        elif entailed and answer_relations: status = "partially_answered"
        elif _incoming(prop["proposition_id"], relations, {"tentatively_answers"}): status = "tentatively_answered"
        elif upstream in {"deferred", "requires_external_verification", "rhetorical", "superseded"}: status = upstream
        else: status = "unanswered"
        display = {"additional_tools": "какие дополнительные инструменты нужны", "rhythmic_entry_implementation": "какой вариант ритмического входа работает", "high_tf_result": "какой результат получен на старших таймфреймах", "exact_time": "точное время", "day": "день"}
        result.append({"question_id": "Q" + prop["proposition_id"][1:], "proposition_id": prop["proposition_id"], "source_record_id": event.get("source_record_id"), "intent": record.get("question_intent") or "unknown", "requested_slots": requested, "answered_slots": entailed, "missing_slots": missing, "missing_slot_labels": [display.get(x) for x in missing if display.get(x)], "candidate_answer_ids": answer_ids, "answer_record_ids": answer_ids, "answer_evidence_ids": answer_evidence, "answer_relation_ids": [x["relation_id"] for x in answer_relations], "status": status, "start": float(record.get("start", 0)), "closing_schedule_priority": prop["content_kind"] == "schedule"})
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
