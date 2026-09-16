"""Deterministic state machines over propositions, dialogue events and relations."""
from __future__ import annotations
import hashlib
import re
from .questions import normalize_slot, verify_slot_entailment


PROPOSAL_WORDING_RE = re.compile(r"(?iu)\b(?:предлагалось|предлагает|можно|стоит|нужно\s+бы|планируется|планирует)\b")
ACCEPT_RE = re.compile(r"(?iu)^\s*(?:(?:да[\s,!.—-]*)+|согласен|делаем|ок(?:ей)?|подтверждаю|(?:а[\s,]*)?(?:месяца?[\s,.:—-]*)?ну\s+ладно)\b")
FIRST_PERSON_COMMIT_RE = re.compile(r"(?iu)\b(?:я\s+(?:сделаю|отправлю|передам|дам|кину|буду|возьмусь|встрою)|я\b.{1,80}\b(?:сделаю|отправлю|передам|дам|кину|буду|возьмусь|встрою)|(?:потом\s+)?встрою|это\s+за\s+мной|с\s+меня|беру|i\s+will)\b")
WORK_PREDICATE_RE = re.compile(r"(?iu)\b(?:сдела\w*|созда\w*|подготов\w*|переда\w*|отправ\w*|предостав\w*|разме[тч]\w*|встраива\w*|встро\w*|провер\w*|исправ\w*|продолж\w*|эксперимент\w*|разработ\w*|написа\w*|провед\w*|реализ\w*|обработ\w*|собра\w*|запуст\w*|добав\w*|выгруз\w*|скин\w*|кин\w*|дам|даю|отдам|покаж\w*|заль\w*|deploy\w*|build\w*)\b")
META_ACTION_RE = re.compile(r"(?iu)^\s*(?:вопрос|уточнение|метаописание|участник\s+спрашивает)\b")
COUNT_WORDS = {"один": 1, "одного": 1, "одну": 1, "два": 2, "две": 2, "три": 3, "трех": 3, "трёх": 3, "четыре": 4, "пять": 5, "шесть": 6, "семь": 7, "восемь": 8, "девять": 9, "десять": 10}
SCOPE_VALUE_RE = re.compile(r"(?iu)\b(?P<count>\d+|один|одного|одну|два|две|три|тр[её]х|четыре|пять|шесть|семь|восемь|девять|десять)\s*(?P<unit>месяц(?:а|ев|ем)?|недел(?:я|и|ь|ю)|д(?:ень|ня|ней)|час(?:а|ов)?|минут(?:а|ы)?|год(?:а|ов)?)\b")
BARE_SCOPE_RE = re.compile(r"(?iu)\b(?P<unit>месяц(?:а|ев|ем)?|недел(?:я|и|ь|ю)|д(?:ень|ня|ней)|час(?:а|ов)?|минут(?:а|ы)?|год(?:а|ов)?)\b")
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
    # "разметчик" is an object (labeling tool), not the verb "размечать".
    source = re.sub(r"(?iu)\bразметчик\w*\b", "", str(value or ""))
    return {match.group(0).casefold()[:5] for match in WORK_PREDICATE_RE.finditer(source)}


def _incoming(target, relations, kinds):
    return [x for x in relations if x.get("target_proposition_id") == target and x.get("type") in kinds]


def _scope_value(text):
    source = str(text or "")
    match = SCOPE_VALUE_RE.search(source)
    if not match:
        match = BARE_SCOPE_RE.search(source)
        if not match:
            return None
        prefix = source[max(0, match.start() - 16):match.start()]
        if re.search(r"(?iu)\b(?:не|вместо)\s*$", prefix):
            return None
        count = 1
    else:
        raw = match.group("count").casefold()
        count = int(raw) if raw.isdigit() else COUNT_WORDS[raw]
    unit = match.group("unit").casefold()
    stem = "месяц" if unit.startswith("месяц") else "неделя" if unit.startswith("недел") else "день" if unit.startswith("д") else "час" if unit.startswith("час") else "минута" if unit.startswith("минут") else "год"
    if stem == "год" and count >= 1900:
        return None
    forms = {"месяц": ("месяц", "месяца", "месяцев"), "неделя": ("неделя", "недели", "недель"), "день": ("день", "дня", "дней"), "час": ("час", "часа", "часов"), "минута": ("минута", "минуты", "минут"), "год": ("год", "года", "лет")}[stem]
    form = forms[0] if count % 10 == 1 and count % 100 != 11 else forms[1] if count % 10 in {2, 3, 4} and count % 100 not in {12, 13, 14} else forms[2]
    return f"{count} {form}"


def _commitment_matches_statement(statement, turn_text):
    """Require the recovered promise to express the same action, not just the same topic."""
    statement_predicates = _task_predicates(statement)
    turn_predicates = _task_predicates(turn_text)
    if statement_predicates & turn_predicates:
        return True
    transfer = re.compile(r"(?iu)\b(?:предостав\w*|переда\w*|отправ\w*|скин\w*|кин\w*|дам|даю|отдам)\b")
    return bool(transfer.search(str(statement or "")) and transfer.search(str(turn_text or "")))


def _source_commitment(record, owner=None, statement=None):
    """Find a first-person promise in the exact source dialogue."""
    turns = [turn for turn in record.get("dialogue_evidence", []) if FIRST_PERSON_COMMIT_RE.search(str(turn.get("text") or ""))]
    if statement:
        turns = [turn for turn in turns if _commitment_matches_statement(statement, turn.get("text"))]
    if owner:
        turns = [turn for turn in turns if turn.get("speaker") == owner]
    speakers = {turn.get("speaker") for turn in turns if turn.get("speaker")}
    if len(speakers) != 1:
        return [], None
    actor = next(iter(speakers))
    evidence = [turn.get("id") for turn in turns if turn.get("speaker") == actor]
    evidence = list(dict.fromkeys(value for value in evidence if value))
    return evidence, actor if evidence else None


def _committed_description(statement, actor):
    """Remove proposal wording only when exact dialogue proves a self-commitment."""
    value = str(statement or "").strip()
    replacements = {"предоставить": "предоставит", "подготовить": "подготовит", "передать": "передаст", "отправить": "отправит", "сделать": "сделает"}
    match = re.match(r"(?iu)^(@[\w.-]+)\s+предложил(?:а)?\s+(предоставить|подготовить|передать|отправить|сделать)\b(.*)$", value)
    if match and actor == match.group(1):
        return f"{match.group(1)} {replacements[match.group(2).casefold()]}{match.group(3)}"
    return value


def reduce_decisions(propositions, events, relations, records=None):
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
        event_by_id = {x["event_id"]: x for x in events}
        event_by_record = {x["source_record_id"]: x for x in events}
        event_by_prop = {x["proposition_id"]: x for x in events}
        def relation_event(rel):
            return event_by_id.get(rel.get("source_event_id")) or event_by_record.get(rel.get("source_record_id")) or event_by_prop.get(rel.get("source_proposition_id"), {})
        transitions = sorted(accepts + rejects, key=lambda rel: relation_event(rel).get("timestamp", 0))
        local_acceptance = []
        if records:
            local_records = [record for record in records if record.get("record_id") in prop.get("source_record_ids", [])]
            for record in local_records:
                dialogue = sorted(record.get("dialogue_evidence", []), key=lambda value: float(value.get("start", 0)))
                proposal_start = min((float(value.get("timestamp", 0)) for value in prop_events), default=0)
                for index, turn in enumerate(dialogue):
                    if (float(turn.get("start", 0)) < proposal_start
                            or turn.get("speaker") == latest.get("speaker")
                            or not ACCEPT_RE.search(str(turn.get("text") or ""))):
                        continue
                    if any(prior.get("speaker") and turn.get("speaker") and prior.get("speaker") != turn.get("speaker") for prior in dialogue[:index]):
                        local_acceptance.extend([value for value in (turn.get("id"),) if value])
        if supersedes: status = "superseded"
        elif transitions: status = "rejected" if transitions[-1]["type"] == "rejects" else "accepted"
        elif local_acceptance: status = "accepted"
        elif "decide" in acts: status = "accepted"
        elif "defer" in acts: status = "deferred"
        else: status = "candidate"
        accepted_by = list(dict.fromkeys(
            turn.get("speaker") for record in (local_records if records else [])
            for turn in record.get("dialogue_evidence", [])
            if turn.get("id") in local_acceptance and turn.get("speaker")
        ))
        acceptance_evidence = list(dict.fromkeys(local_acceptance + [e for x in accepts for e in x.get("evidence_ids", [])]))
        result.append({"decision_id": "D" + prop["proposition_id"][1:], "proposition_id": prop["proposition_id"], "source_record_id": latest.get("source_record_id"), "status": status, "acceptance_relation_ids": [x["relation_id"] for x in accepts], "acceptance_evidence_ids": acceptance_evidence, "decision_evidence_ids": list(dict.fromkeys(prop["evidence_ids"] + acceptance_evidence)), "acceptance_check": "entailed" if status == "accepted" else "not_entailed", "proposed_by": latest.get("speaker"), "accepted_by": accepted_by, "action_frame": {"speaker": latest.get("speaker"), "reporter": latest.get("speaker"), "accepted_by": accepted_by, "state": "accepted_plan" if status == "accepted" else "proposal"}, "conditions": prop["conditions"], "evidence_ids": prop["evidence_ids"]})
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
        source_statement = str(prop.get("statement") or "")
        source_commitment_evidence, source_commitment_actor = _source_commitment(record, statement=source_statement)
        commitment_candidate = event["speech_act"] == "commit" or record.get("commitment_strength") in {"explicit", "implicit"} or prop["content_kind"] in {"action", "follow_up"} or bool(source_commitment_evidence)
        if not commitment_candidate:
            continue
        if META_ACTION_RE.search(source_statement) or not WORK_PREDICATE_RE.search(source_statement):
            continue
        if event["speech_act"] == "assert" and record.get("commitment_strength") not in {"explicit", "implicit"} and not FIRST_PERSON_COMMIT_RE.search(source_statement) and not source_commitment_evidence:
            continue
        if re.search(r"(?iu)\b(?:сделать\s+упор|сосредоточиться|ещ[её]\s+над\s+этим\s+посидеть)\b", source_statement) and not re.search(r"(?iu)\b(?:подготов\w*|переда\w*|отправ\w*|предостав\w*|размет\w*|встро\w*|исправ\w*|созда\w*|провер\w*)\b", source_statement):
            continue
        owners = list(record.get("assignees", []))
        if not owners and source_commitment_actor:
            owners = [source_commitment_actor]
        confirmations = list(record.get("confirmation_evidence_ids", []))
        acceptance_relations = _incoming(prop["proposition_id"], relations, {"accepts", "confirms", "accepts_assignment"})
        event_by_id = {x["event_id"]: x for x in events}
        event_by_record = {x["source_record_id"]: x for x in events}
        event_by_prop = {x["proposition_id"]: x for x in events}
        accepted_by_owner = [x for x in acceptance_relations if (event_by_id.get(x.get("source_event_id")) or event_by_record.get(x.get("source_record_id")) or event_by_prop.get(x.get("source_proposition_id"), {})).get("speaker") in owners]
        if accepted_by_owner:
            confirmations = list(dict.fromkeys(confirmations + [e for x in accepted_by_owner for e in x.get("evidence_ids", [])]))
        commit_event = next((x for x in reversed(sorted(prop_events, key=lambda x: x.get("timestamp", 0))) if x["speech_act"] == "commit"), None)
        if commit_event and commit_event.get("speaker") and not owners:
            owners = [commit_event["speaker"]]
        statement = str(prop.get("statement") or "")
        sole_owner = owners[0] if len(owners) == 1 else None
        source_commitment_evidence, source_commitment_actor = _source_commitment(record, sole_owner, statement)
        explicit_commitment = bool(
            record.get("commitment_strength") == "explicit"
            or FIRST_PERSON_COMMIT_RE.search(statement)
            or source_commitment_evidence
        )
        commitment_actor = (
            record.get("commitment_actor")
            or source_commitment_actor
            or (commit_event.get("speaker") if explicit_commitment and commit_event else None)
        )
        speaker = event.get("speaker")
        direct_assignment = bool(re.search(r"(?iu)\b(?:должен|должна|сделай|сделайте|нужно\s+тебе|тебе\s+нужно)\b", statement))
        statement_people = list(dict.fromkeys(re.findall(r"@[\w.-]+", statement)))
        reported_actor_mismatch = bool(speaker and statement_people and speaker not in statement_people and not direct_assignment)
        reported_third_person = bool(
            (owners and speaker and speaker not in owners and not direct_assignment and not record.get("confirmation_evidence_ids"))
            or reported_actor_mismatch
        )
        mentioned_people = list(owners)
        if reported_actor_mismatch:
            mentioned_people = statement_people
            owners = []
            commitment_actor = None
            source_commitment_actor = None
            source_commitment_evidence = []
        assignment_status = str(record.get("assignment_status") or "unknown")
        uncertainty_reasons = list(record.get("uncertainty", {}).get("reasons", []))
        ambiguous_owner = len(owners) != 1 or len(event.get("speaker_candidates", [])) != 1
        if record.get("completion_evidence_ids"): status = "completed"
        elif record.get("blocked"): status = "blocked"
        elif confirmations and owners and (assignment_status in {"confirmed", "accepted"} or accepted_by_owner): status = "accepted"
        elif explicit_commitment and not ambiguous_owner and commitment_actor == owners[0]: status = "self_committed"
        elif owners and assignment_status in {"confirmed", "accepted", "assigned"}: status = "assigned"
        elif owners: status = "proposed" if reported_third_person else "assigned_pending"
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
            source_start = min([x.get("timestamp", 0) for x in events if source_prop and x.get("proposition_id") == source_prop.get("proposition_id")] or [0])
            scope_unit = value.split()[-1][:5].casefold()
            local_scope_acceptance = [turn.get("id") for turn in record.get("dialogue_evidence", [])
                                      if sole_owner and turn.get("speaker") == sole_owner
                                      and float(turn.get("start", 0)) >= float(source_start)
                                      and scope_unit in str(turn.get("text") or "").casefold()
                                      and ACCEPT_RE.search(str(turn.get("text") or ""))]
            if local_scope_acceptance:
                accepted = True
                scope_evidence.extend(value for value in local_scope_acceptance if value)
            source_events = [x for x in events if x["proposition_id"] == source_prop["proposition_id"]] if source_prop else []
            asserted = any(x["speech_act"] in {"assert", "answer", "decide"} for x in source_events)
            # A later technical constraint (for example, a four-hour
            # simulation window) is not automatically a revision of an
            # already established delivery scope (for example, one month of
            # source data). It may fill an empty scope, but replacing a scope
            # requires an acceptance or a non-constraint assertion.
            selected_scope = accepted or (asserted and source_prop.get("content_kind") != "constraint") or not scope
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
        action_state = "reported_plan" if reported_third_person else "commitment" if explicit_commitment else "accepted" if status == "accepted" else "assigned_pending" if status in {"assigned", "assigned_pending"} else "proposal"
        raw_actor_confidence = record.get("assignee_confidence")
        actor_confidence = float(raw_actor_confidence) if isinstance(raw_actor_confidence, (int, float)) else 0.0
        action_frame = {"speaker": speaker, "reporter": record.get("reporter") or speaker, "grammatical_actor": None if reported_third_person else commitment_actor or (owners[0] if len(owners) == 1 else None), "mentioned_people": mentioned_people, "beneficiary": record.get("beneficiary"), "recipient": record.get("recipient") or record.get("beneficiary"), "proposed_by": speaker, "proposed_for": owners[0] if len(owners) == 1 else None, "assignment_target": owners[0] if len(owners) == 1 and not reported_third_person else None, "explicit_acceptance_actor": owners[0] if accepted_by_owner and len(owners) == 1 else None, "utterance_ids": list(prop.get("evidence_ids", [])), "alias_resolution": record.get("alias_resolution", {}), "confidence": actor_confidence, "state": action_state}
        description = _committed_description(prop["statement"], source_commitment_actor) if source_commitment_evidence else prop["statement"]
        atomic.append({"task_id": "T" + prop["proposition_id"][1:], "proposition_id": prop["proposition_id"], "source_proposition_ids": [prop["proposition_id"]], "source_record_id": event.get("source_record_id"), "source_record_ids": [event.get("source_record_id")], "description": description, "deliverable": description, "owner": None if reported_third_person else owners[0] if len(owners) == 1 else None, "assignee": None if reported_third_person else owners[0] if len(owners) == 1 else None, "assignees": owners, "assignee_confidence": record.get("assignee_confidence"), "proposed_by": speaker, "commitment_strength": "explicit" if explicit_commitment else "implicit" if commit_event else "none", "commitment_actor": commitment_actor, "source_commitment_evidence_ids": source_commitment_evidence, "assignment_actor": speaker, "assignment_target": None if reported_third_person else owners[0] if len(owners) == 1 else None, "action_frame": action_frame, "acceptance_relation_ids": [x["relation_id"] for x in accepted_by_owner], "acceptance_evidence_ids": confirmations, "scope_relation_ids": scope_relations, "uncertainty_reasons": sorted(set(uncertainty_reasons + (["ambiguous_owner"] if ambiguous_owner else []) + (["reported_third_person"] if reported_third_person else []))), "deadline": record.get("time_expression"), "due": record.get("time_expression"), "conditions": prop["conditions"], "completion_criterion": record.get("completion_criterion"), "status": status, "task_status": status, "current_scope": scope, "data_origin": data_origin, "scope_state": scope_state, "scope_confidence": "high" if scope_relations else "unknown" if not scope else "source", "scope_history": scope_history, "proposed_scopes": proposed_scopes, "superseded_scopes": [value["value"] for value in scope_history], "superseded_by": superseded_by, "automation_eligible": automation, "evidence_ids": list(dict.fromkeys(prop.get("evidence_ids", []) + source_commitment_evidence + scope_evidence)), "source_word_ids": list(dict.fromkeys(list(record.get("source_word_ids", [])) + scope_words)), "start": float(record.get("start", 0))})
        atomic[-1]["scope_confidence"] = "accepted" if scope_confirmed else "proposed" if scope else "unknown"
    # Canonical task envelopes: one state is consumed by every public/API view.
    groups = []
    for item in sorted(atomic, key=lambda x: x["start"]):
        item_object = _task_object_tokens(item["description"])
        match = next((g for g in reversed(groups)
                      if item.get("owner") == g.get("owner")
                      and item["start"] - g["start"] <= 120
                      and (bool(set(item.get("source_commitment_evidence_ids", [])) & set(g.get("source_commitment_evidence_ids", [])))
                           or (_task_similarity(item["description"], g["description"]) >= .34
                               and _task_predicates(item["description"]) == _task_predicates(g["description"])
                               and item_object and _task_object_tokens(g["description"])
                               and len(item_object & _task_object_tokens(g["description"])) / max(1, len(item_object | _task_object_tokens(g["description"]))) >= .65))), None)
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
            for field in ("description", "deliverable", "commitment_actor", "assignment_actor", "assignment_target", "action_frame", "commitment_strength", "assignee", "owner"):
                if item.get(field) is not None:
                    match[field] = item[field]
        match["source_commitment_evidence_ids"] = list(dict.fromkeys(
            match.get("source_commitment_evidence_ids", []) + item.get("source_commitment_evidence_ids", [])
        ))
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
        if event.get("event_id") in processed:
            continue
        processed.add(event.get("event_id"))
        prop = next(x for x in propositions if x["proposition_id"] == event["proposition_id"])
        if event["speech_act"] != "ask" and prop["content_kind"] != "question":
            continue
        record = by_record.get(event.get("source_record_id"), {})
        requested = list(record.get("requested_slots", []))
        explicit_answered = list(record.get("answered_slots", []))
        answer_relations = _incoming(prop["proposition_id"], relations, {"answers", "partially_answers", "resolves"})
        events_by_prop = {}
        for candidate in events:
            events_by_prop.setdefault(candidate.get("proposition_id"), []).append(candidate)
        answer_ids = list(record.get("answer_record_ids", []))
        answer_ids.extend(
            candidate.get("source_record_id")
            for relation in answer_relations
            for candidate in events_by_prop.get(relation.get("source_proposition_id"), [])
            if candidate.get("source_record_id")
        )
        answer_ids = list(dict.fromkeys(answer_ids))
        checks = [verify_slot_entailment(requested, by_record.get(answer_id, {}), record) for answer_id in answer_ids]
        inferred = {slot for check in checks for slot in check["entailed_slots"]} if checks else set()
        entailed = [x for x in requested if x in explicit_answered or x in inferred]
        missing = [x for x in requested if x not in entailed]
        upstream = str(record.get("question_status") or "").casefold()
        answer_evidence = list(dict.fromkeys(list(record.get("answer_evidence_ids", [])) + [
            value for relation in answer_relations for value in relation.get("evidence_ids", [])
        ]))
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
        display = {"additional_tools": "решить, нужен ли дополнительный фильтр", "implementation_status": "проверить статус конкретной реализации", "time_range": "подтвердить точный диапазон времени", "day": "подтвердить день", "actor_commitment": "подтвердить действие и исполнителя", "threshold_value": "определить проверяемый порог", "reason_hypothesis": "проверить причинную гипотезу", "yes_no": "подтвердить решение"}
        known_parts = [str(by_record.get(answer_id, {}).get("statement") or "").strip() for answer_id in answer_ids]
        known_parts = [value for value in known_parts if value]
        typed_missing = [normalize_slot(x) for x in missing]
        remaining = "; ".join(dict.fromkeys(display.get(x, "уточнить недостающий результат") for x in typed_missing))
        original = str(prop["statement"])
        human_original = original if re.search(r"[а-яё]", original, re.I) and not re.search(r"\b[a-z]+_[a-z_]+\b", original) else None
        residual_text = human_original if human_original and status == "unanswered" else remaining or human_original
        result.append({"question_id": "Q" + prop["proposition_id"][1:], "proposition_id": prop["proposition_id"], "source_record_id": event.get("source_record_id"), "intent": record.get("question_intent") or "unknown", "original_question": original, "known_answer": " ".join(known_parts) or None, "remaining_question": residual_text if status in {"unanswered", "partially_answered", "deferred", "requires_external_verification"} else None, "residual_question_text": residual_text if status in {"unanswered", "partially_answered", "deferred", "requires_external_verification"} else None, "answer_support": answer_evidence, "residual_support": prop.get("evidence_ids", []), "requested_slots": [normalize_slot(x) for x in requested], "answered_slots": [normalize_slot(x) for x in entailed], "missing_slots": typed_missing, "missing_slot_labels": [display.get(x, "уточнить недостающий результат") for x in typed_missing], "candidate_answer_ids": answer_ids, "answer_record_ids": answer_ids, "answer_evidence_ids": answer_evidence, "answer_relation_ids": [x["relation_id"] for x in answer_relations], "status": status, "start": float(record.get("start", 0)), "closing_schedule_priority": prop["content_kind"] == "schedule"})
        result[-1].update({
            "question_id": "Q" + str(event.get("event_id") or prop["proposition_id"])[2:],
            "remaining_unknown": result[-1]["remaining_question"],
            "resolution_basis": "direct_answer" if status == "answered" else "partial_answer" if status == "partially_answered" else "no_entailed_answer",
        })
    # Semantic extraction can emit a generic wrapper immediately before the
    # actual question (for example, "Участник задаёт вопрос адресованный …").
    # Once that specific child question is answered, the wrapper is not an
    # independent open issue and must not leak into either questions or minutes.
    by_source_question = {item.get("source_record_id"): item for item in result}
    generic_wrapper = re.compile(r"(?iu)^\s*участник\s+зада[её]т\s+вопрос(?:\s+адресованн\w*\s+(?:@[\w.-]+(?:\s*/\s*@[\w.-]+)*))?[.!?]*\s*$")
    for item in result:
        if item.get("status") != "unanswered" or not generic_wrapper.fullmatch(str(item.get("original_question") or "")):
            continue
        answered_child = next((
            by_source_question.get(record_id) for record_id in item.get("answer_record_ids", [])
            if by_source_question.get(record_id, {}).get("status") in {"answered", "rhetorical", "superseded"}
        ), None)
        if answered_child:
            item.update(
                status="superseded",
                remaining_question=None,
                residual_question_text=None,
                remaining_unknown=None,
                resolution_basis="superseded_by_answered_specific_question",
                superseded_by_question_id=answered_child.get("question_id"),
            )
    return result


def reduce_rules_and_experiments(propositions, events, decisions):
    decision_by_prop = {x["proposition_id"]: x for x in decisions}
    rules, experiments = [], []
    for prop in propositions:
        if prop["content_kind"] in {"trading_rule", "system_rule"}:
            decision = decision_by_prop.get(prop["proposition_id"])
            rules.append({"rule_id": "RUL" + prop["proposition_id"][1:], "proposition_id": prop["proposition_id"], "rule": prop["statement"], "scope": prop["scope"], "conditions": prop["conditions"], "exceptions": [], "status": decision["status"] if decision else "described_existing_rule", "evidence_ids": prop["evidence_ids"]})
        if prop["content_kind"] in {"hypothesis", "experimental_result"}:
            experiments.append({"experiment_id": "EXP" + prop["proposition_id"][1:], "proposition_id": prop["proposition_id"], "content_kind": prop["content_kind"], "hypothesis": prop["statement"] if prop["content_kind"] == "hypothesis" else None, "result": prop["statement"] if prop["content_kind"] == "experimental_result" else None, "proposed_by": next(iter(prop.get("speaker_refs", [])), None), "testable_mechanism": None, "artifact_or_signal": None, "decision_after_test": None, "motivation": None, "method": None, "dataset": None, "independent_variable": None, "metrics": [x for x in prop["quantities"]], "baseline": None, "expected_direction": None, "owner": None, "status": "result" if prop["content_kind"] == "experimental_result" else "proposed_not_run", "evidence_ids": prop["evidence_ids"]})
    return rules, experiments
