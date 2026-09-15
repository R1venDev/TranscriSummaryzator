"""Deterministic state machines over propositions, dialogue events and relations."""
from __future__ import annotations
import hashlib
import re
from .questions import verify_slot_entailment


PROPOSAL_WORDING_RE = re.compile(r"(?iu)\b(?:предлагалось|предлагает|можно|стоит|нужно\s+бы|планируется|планирует)\b")
FIRST_PERSON_COMMIT_RE = re.compile(r"(?iu)\b(?:я\s+(?:сделаю|отправлю|передам|дам|кину|буду|возьмусь)|i\s+will)\b")
WORK_PREDICATE_RE = re.compile(r"(?iu)\b(?:сдела\w*|созда\w*|подготов\w*|переда\w*|отправ\w*|предостав\w*|разме[тч]\w*|встраива\w*|встро\w*|провер\w*|исправ\w*|продолж\w*|эксперимент\w*|разработ\w*|написа\w*|провед\w*|реализ\w*|обработ\w*|собра\w*|запуст\w*|добав\w*|выгруз\w*|скин\w*|кин\w*|дам|даю|отдам|покаж\w*|заль\w*|deploy\w*|build\w*)\b")
META_ACTION_RE = re.compile(r"(?iu)^\s*(?:вопрос|уточнение|метаописание|участник\s+спрашивает)\b")
COUNT_WORDS = {"один": 1, "одного": 1, "одну": 1, "два": 2, "две": 2, "три": 3, "трех": 3, "трёх": 3, "четыре": 4, "пять": 5, "шесть": 6, "семь": 7, "восемь": 8, "девять": 9, "десять": 10}
SCOPE_VALUE_RE = re.compile(r"(?iu)\b(?P<count>\d+|один|одного|одну|два|две|три|тр[её]х|четыре|пять|шесть|семь|восемь|девять|десять)\s*(?P<unit>месяц(?:а|ев|ем)?|недел(?:я|и|ь|ю)|д(?:ень|ня|ней)|час(?:а|ов)?|минут(?:а|ы)?|год(?:а|ов)?)\b")
YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
TOKEN_RE = re.compile(r"(?iu)[a-zа-яё0-9]+")
TASK_STOP = {"говорящий", "предлагает", "планирует", "будут", "параллельно", "надо", "нужно", "чтобы", "участник"}


def _tokens(value):
    return {x for x in TOKEN_RE.findall(str(value or "").casefold()) if len(x) > 3 and x not in TASK_STOP}


def _task_similarity(left, right):
    a, b = _tokens(left), _tokens(right)
    return len(a & b) / max(1, min(len(a), len(b)))


def _task_object_tokens(value):
    stop = TASK_STOP | {"я", "мы", "сделаю", "отправлю", "передам", "дам", "буду", "сделать", "отправить", "передать", "дать"}
    return {x for x in TOKEN_RE.findall(str(value or "").casefold()) if len(x) > 2 and x not in stop}


def _task_predicates(value):
    return {match.group(0).casefold()[:5] for match in WORK_PREDICATE_RE.finditer(str(value or ""))}


def _incoming(target, relations, kinds):
    return [x for x in relations if x.get("target_proposition_id") == target and x.get("type") in kinds]


def _scope_value(text):
    match = SCOPE_VALUE_RE.search(str(text or ""))
    if not match:
        return None
    raw = match.group("count").casefold()
    count = int(raw) if raw.isdigit() else COUNT_WORDS[raw]
    unit = match.group("unit").casefold()
    stem = "месяц" if unit.startswith("месяц") else "неделя" if unit.startswith("недел") else "день" if unit.startswith("д") else "час" if unit.startswith("час") else "минута" if unit.startswith("минут") else "год"
    if stem == "год" and count >= 1900:
        return None
    forms = {"месяц": ("месяц", "месяца", "месяцев"), "неделя": ("неделя", "недели", "недель"), "день": ("день", "дня", "дней"), "час": ("час", "часа", "часов"), "минута": ("минута", "минуты", "минут"), "год": ("год", "года", "лет")}[stem]
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
        event_by_prop = {x["proposition_id"]: x for x in events}
        transitions = sorted(accepts + rejects, key=lambda rel: event_by_prop.get(rel.get("source_proposition_id"), {}).get("timestamp", 0))
        if supersedes: status = "superseded"
        elif transitions: status = "rejected" if transitions[-1]["type"] == "rejects" else "accepted"
        elif "decide" in acts: status = "accepted"
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
        source_statement = str(prop.get("statement") or "")
        if META_ACTION_RE.search(source_statement) or not WORK_PREDICATE_RE.search(source_statement):
            continue
        if event["speech_act"] == "assert" and record.get("commitment_strength") not in {"explicit", "implicit"} and not FIRST_PERSON_COMMIT_RE.search(source_statement):
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
        explicit_commitment = bool(record.get("commitment_strength") == "explicit" or FIRST_PERSON_COMMIT_RE.search(statement))
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
        task_transitions = sorted(
            acceptance_relations + _incoming(prop["proposition_id"], relations, {"rejects"}),
            key=lambda rel: event_by_prop.get(rel.get("source_proposition_id"), {}).get("timestamp", 0),
        )
        if task_transitions:
            status = "cancelled" if task_transitions[-1]["type"] == "rejects" else status
        superseded_by = [x["source_proposition_id"] for x in _incoming(prop["proposition_id"], relations, {"corrects", "supersedes"})]
        scope_state = "superseded" if superseded_by else "ambiguous" if record.get("scope_ambiguous") else "active"
        if superseded_by:
            status = "superseded"
        explicit_automation = record.get("automation_eligible")
        automation = explicit_automation if isinstance(explicit_automation, bool) else (False if status in {"idea", "proposed", "assigned", "assigned_pending", "superseded"} else "unknown")
        scope = record.get("scope") or record.get("time_scope")
        if isinstance(scope, str) and YEAR_RE.search(scope) and not SCOPE_VALUE_RE.search(scope):
            scope = None
        data_origin = record.get("data_origin") or next(iter(YEAR_RE.findall(statement)), None)
        scope_relations, scope_evidence, scope_words, scope_history, proposed_scopes = [], [], [], [], []
        scope_confirmed = bool(scope)
        revisers = sorted(_incoming(prop["proposition_id"], relations, {"revises_scope"}), key=lambda x: event_by_prop.get(x.get("source_proposition_id"), {}).get("timestamp", 0))
        for relation in revisers:
            source_prop = next((x for x in propositions if x["proposition_id"] == relation.get("source_proposition_id")), None)
            value = _scope_value(source_prop.get("statement") if source_prop else "")
            if not value:
                continue
            accepted = bool(_incoming(source_prop["proposition_id"], relations, {"accepts", "confirms"})) if source_prop else False
            source_events = [x for x in events if x["proposition_id"] == source_prop["proposition_id"]] if source_prop else []
            asserted = any(x["speech_act"] in {"assert", "answer", "decide"} for x in source_events)
            selected_scope = accepted or asserted or not scope
            if selected_scope:
                if scope and scope != value:
                    scope_history.append({"value": scope, "status": "superseded", "relation_id": relation["relation_id"]})
                scope = value
                scope_confirmed = accepted or asserted
                scope_relations.append(relation["relation_id"])
                scope_evidence.extend(relation.get("evidence_ids", []))
            else:
                proposed_scopes.append({"value": value, "status": "proposed_not_accepted", "relation_id": relation["relation_id"]})
            source_records = source_prop.get("source_record_ids", []) if source_prop else []
            if selected_scope:
                scope_words.extend(w for rid in source_records for w in by_record.get(rid, {}).get("source_word_ids", []))
        atomic.append({"task_id": "T" + prop["proposition_id"][1:], "proposition_id": prop["proposition_id"], "source_proposition_ids": [prop["proposition_id"]], "source_record_id": event.get("source_record_id"), "source_record_ids": [event.get("source_record_id")], "description": prop["statement"], "deliverable": prop["statement"], "owner": owners[0] if len(owners) == 1 else None, "assignee": owners[0] if len(owners) == 1 else None, "assignees": owners, "assignee_confidence": record.get("assignee_confidence"), "proposed_by": event.get("speaker"), "commitment_strength": "explicit" if explicit_commitment else "implicit" if commit_event else "none", "commitment_actor": commitment_actor, "assignment_actor": event.get("speaker"), "assignment_target": owners[0] if len(owners) == 1 else None, "acceptance_relation_ids": [x["relation_id"] for x in accepted_by_owner], "acceptance_evidence_ids": confirmations, "scope_relation_ids": scope_relations, "uncertainty_reasons": sorted(set(uncertainty_reasons + (["ambiguous_owner"] if ambiguous_owner else []))), "deadline": record.get("time_expression"), "due": record.get("time_expression"), "conditions": prop["conditions"], "completion_criterion": record.get("completion_criterion"), "status": status, "task_status": status, "current_scope": scope, "data_origin": data_origin, "scope_state": scope_state, "scope_confidence": "high" if scope_relations else "unknown" if not scope else "source", "scope_history": scope_history, "proposed_scopes": proposed_scopes, "superseded_scopes": [value["value"] for value in scope_history], "superseded_by": superseded_by, "automation_eligible": automation, "evidence_ids": list(dict.fromkeys(prop.get("evidence_ids", []) + scope_evidence)), "source_word_ids": list(dict.fromkeys(list(record.get("source_word_ids", [])) + scope_words)), "start": float(record.get("start", 0))})
        atomic[-1]["scope_confidence"] = "accepted" if scope_confirmed else "proposed" if scope else "unknown"
    # Canonical task envelopes: one state is consumed by every public/API view.
    groups = []
    for item in sorted(atomic, key=lambda x: x["start"]):
        item_object = _task_object_tokens(item["description"])
        match = next((g for g in reversed(groups)
                      if item.get("owner") == g.get("owner")
                      and item["start"] - g["start"] <= 120
                      and _task_similarity(item["description"], g["description"]) >= .34
                      and _task_predicates(item["description"]) == _task_predicates(g["description"])
                      and item_object and _task_object_tokens(g["description"])
                      and len(item_object & _task_object_tokens(g["description"])) / max(1, len(item_object | _task_object_tokens(g["description"]))) >= .65), None)
        if not match:
            groups.append(item)
            continue
        match["source_proposition_ids"].extend(item["source_proposition_ids"])
        match["source_record_ids"].extend(item["source_record_ids"])
        match["evidence_ids"] = list(dict.fromkeys(match["evidence_ids"] + item["evidence_ids"]))
        match["source_word_ids"] = list(dict.fromkeys(match["source_word_ids"] + item["source_word_ids"]))
        match["uncertainty_reasons"] = sorted(set(match["uncertainty_reasons"] + item["uncertainty_reasons"]))
        for field in ("acceptance_relation_ids", "acceptance_evidence_ids", "scope_relation_ids"):
            match[field] = list(dict.fromkeys(match[field] + item[field]))
        match["conditions"] += [value for value in item["conditions"] if value not in match["conditions"]]
        if item["status"] in {"cancelled", "superseded", "accepted", "self_committed", "completed"}:
            match["status"] = match["task_status"] = item["status"]
        if item.get("current_scope") and item["current_scope"] != match.get("current_scope"):
            if match.get("current_scope"):
                match["superseded_scopes"].append(match["current_scope"])
            match["current_scope"] = item["current_scope"]
        if not match.get("deadline") and item.get("deadline"):
            match["deadline"] = match["due"] = item["deadline"]
        if not match.get("completion_criterion") and item.get("completion_criterion"):
            match["completion_criterion"] = item["completion_criterion"]
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
        checks = [verify_slot_entailment(requested, by_record.get(answer_id, {}), record) for answer_id in record.get("answer_record_ids", [])]
        inferred = {slot for check in checks for slot in check["entailed_slots"]} if checks else set()
        entailed = [x for x in requested if x in explicit_answered or x in inferred]
        missing = [x for x in requested if x not in entailed]
        answer_relations = _incoming(prop["proposition_id"], relations, {"answers", "partially_answers", "resolves"})
        upstream = str(record.get("question_status") or "").casefold()
        answer_ids = list(record.get("answer_record_ids", []))
        answer_evidence = list(record.get("answer_evidence_ids", []))
        has_answer_support = bool(answer_relations or answer_ids or answer_evidence)
        if upstream in {"answered", "resolved"} and ((requested and not missing and has_answer_support) or (not requested and has_answer_support)):
            status, entailed, missing = "answered", requested or explicit_answered, []
        elif upstream == "partially_answered" and has_answer_support and entailed:
            status = "partially_answered"
        elif requested and not missing and answer_relations: status = "answered"
        elif entailed and answer_relations: status = "partially_answered"
        elif _incoming(prop["proposition_id"], relations, {"tentatively_answers"}): status = "tentatively_answered"
        elif upstream in {"deferred", "requires_external_verification", "rhetorical", "superseded"}: status = upstream
        else: status = "unanswered"
        display = {"additional_tools": "какие дополнительные инструменты нужны", "rhythmic_entry_implementation": "какой вариант ритмического входа работает", "high_tf_result": "какой результат получен на старших таймфреймах", "exact_time": "точное время", "day": "день"}
        known_parts = [str(by_record.get(answer_id, {}).get("statement") or "").strip() for answer_id in answer_ids]
        known_parts = [value for value in known_parts if value]
        remaining = "; ".join(display.get(x, x.replace("_", " ")) for x in missing)
        result.append({"question_id": "Q" + prop["proposition_id"][1:], "proposition_id": prop["proposition_id"], "source_record_id": event.get("source_record_id"), "intent": record.get("question_intent") or "unknown", "original_question": prop["statement"], "known_answer": " ".join(known_parts) or None, "remaining_question": remaining or (prop["statement"] if status in {"unanswered", "deferred", "requires_external_verification"} else None), "answer_support": answer_evidence, "residual_support": prop.get("evidence_ids", []), "requested_slots": requested, "answered_slots": entailed, "missing_slots": missing, "missing_slot_labels": [display.get(x, x.replace("_", " ")) for x in missing], "candidate_answer_ids": answer_ids, "answer_record_ids": answer_ids, "answer_evidence_ids": answer_evidence, "answer_relation_ids": [x["relation_id"] for x in answer_relations], "status": status, "start": float(record.get("start", 0)), "closing_schedule_priority": prop["content_kind"] == "schedule"})
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
