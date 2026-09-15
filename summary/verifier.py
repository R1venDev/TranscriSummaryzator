"""Deterministic plan and atomic/relation surface guards."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import hashlib
import re
from semantics.graph import cross_episode_allowed
from summary.policy import RULE_KINDS, TECHNICAL_KINDS

CAUSAL_RE = re.compile(r"(?iu)\b(?:из-за|поэтому|привел[оа]? к|в результате|для этого)\b")
NUMBER_RE = re.compile(r"(?<!\w)\d+(?:[.,:]\d+)*(?:\s*%)?")
NEGATION_RE = re.compile(r"(?iu)\b(?:не|нет|нельзя|без|никогда)\b")
CERTAIN_RE = re.compile(r"(?iu)\b(?:точно|обязательно|гарантированно|решено|утверждено)\b")
COMPLETED_RE = re.compile(r"(?iu)\b(?:проверен[аоы]?|завершен[аоы]?|готов[аоы]?|выполнен[аоы]?|сделан[аоы]?)\b")
CONDITION_RE = re.compile(r"(?iu)\b(?:если|когда|после|перед|пока|при|до тех пор)\b")


@dataclass(frozen=True)
class PublicItem:
    public_id: str
    section: str
    text: str
    claim_ids: list[str]
    evidence_ids: list[str]
    source_word_ids: list[str]
    content_kind: str
    social_state: str
    lifecycle: str = "active"
    relation_ids: list[str] = field(default_factory=list)

    def as_dict(self):
        return asdict(self)


def can_publish_as_decision(item):
    return bool(item.get("lifecycle", "active") == "active"
                and item.get("decision_status") == "accepted"
                and item.get("decision_evidence_ids", item.get("evidence_ids"))
                and item.get("acceptance_check", "entailed") == "entailed")


def build_public_items(meeting_graph, summary_plan):
    """Create the complete public contract before formatting Markdown."""
    claims = {x["claim_id"]: x for x in meeting_graph.get("claims", [])}
    claims_by_source = {x.get("source_record_id"): x for x in claims.values()}
    view_plans = summary_plan.get("view_plans", {})
    task_states = {x["task_id"]: x for x in meeting_graph.get("task_states", [])}
    question_states = {x["proposition_id"]: x for x in meeting_graph.get("question_states", [])}
    planned_relations = {}
    for sentence in summary_plan.get("public_sentence_plans", []):
        for claim_id in sentence.get("claim_ids", []):
            planned_relations.setdefault(claim_id, []).extend(sentence.get("relation_ids", []))
    sections = []
    materialized = {}
    def add(section, claim, social_state=None, text=None, claim_ids=None, relation_ids=None, extra_evidence=None):
        if claim.get("lifecycle", "active") != "active" or not claim.get("evidence_ids"):
            return
        task_state = task_states.get(claim.get("canonical_task_state_id"), {})
        evidence_ids = task_state.get("evidence_ids", claim.get("evidence_ids", [])) if section == "tasks" else claim.get("evidence_ids", [])
        source_word_ids = task_state.get("source_word_ids", claim.get("source_word_ids", [])) if section == "tasks" else claim.get("source_word_ids", [])
        # Context turns are not automatically supporting evidence. Only cited
        # task/answer support is added to the public provenance contract.
        evidence_ids = list(dict.fromkeys(evidence_ids))
        source_word_ids = list(dict.fromkeys(source_word_ids))
        cited_ids = list(dict.fromkeys(claim_ids or [claim["claim_id"]]))
        retained_relations = list(relation_ids or [])
        retained_relations.extend(value for claim_id in cited_ids for value in planned_relations.get(claim_id, []))
        if section == "tasks":
            retained_relations.extend(task_state.get("scope_relation_ids", []))
            retained_relations.extend(task_state.get("acceptance_relation_ids", []))
        public = PublicItem(
            public_id=f"PI{len(sections)+1:05d}", section=section,
            text=str(text or claim.get("statement") or "").strip(), claim_ids=cited_ids,
            evidence_ids=list(dict.fromkeys(evidence_ids + list(extra_evidence or []))), source_word_ids=list(source_word_ids), content_kind=claim.get("content_kind") or claim.get("kind"),
            social_state=social_state or claim.get("social_state", "candidate"), lifecycle=claim.get("lifecycle", "active"), relation_ids=list(dict.fromkeys(retained_relations)),
        ).as_dict() | {"start": claim.get("primary_evidence_start", claim.get("start", 0)), "task_state_id": claim.get("canonical_task_state_id"), "task_state": task_state, "question_state": question_states.get(claim.get("proposition_id"), {}), "topic_entities": [x.get("canonical_name") or x.get("name") for x in claim.get("entities", []) if isinstance(x, dict) and (x.get("canonical_name") or x.get("name"))], "context_ids": claim.get("context_ids", []), "verification_status": claim.get("verification_status", "supported")}
        sections.append(public)
        for claim_id in cited_ids:
            materialized.setdefault(section, {})[claim_id] = {"status": "published", "public_id": public["public_id"]}
    def selected(view):
        return [claims[x] for x in view_plans.get(view, {}).get("selected_claim_ids", []) if x in claims]

    for claim in claims.values():
        if claim.get("verification_status") == "verification_unavailable" and claim.get("content_kind") in {"action", "follow_up", "resource", "decision"}:
            add("requires_verification", claim, "needs_verification", text=claim.get("statement"))

    def attributed_text(claim):
        text = str(claim.get("statement") or "").strip()
        if re.search(r"(?iu)(?:\b100\s*%|\b100\s+процент|\b(?:всегда|никогда|невозможно|нельзя|гарантированно)\b)", text) and len(claim.get("speaker_refs", [])) == 1:
            text = f"По словам {claim['speaker_refs'][0]}, {text[:1].lower() + text[1:]}"
        if claim.get("risk", {}).get("recognition", 0) >= .65:
            text += " ⚠ Формулировка или термин требуют проверки."
        return text
    # Overview is a compact mix of state, obstacle and next step. Repeating a
    # short task outcome across sections is useful; verbatim duplicates are not.
    overview = []
    executive_claims = selected("executive")
    category_order = (
        {"problem", "blocker", "constraint"},
        {"action", "follow_up", "decision"},
        set(TECHNICAL_KINDS),
        {"current_state", "observation", "experimental_result"},
    )
    overview_order, overview_seen = [], set()
    for kinds in category_order:
        candidate = next((x for x in executive_claims if x.get("content_kind") in kinds and x["claim_id"] not in overview_seen), None)
        if candidate:
            overview_order.append(candidate); overview_seen.add(candidate["claim_id"])
    overview_order.extend(x for x in executive_claims if x["claim_id"] not in overview_seen)
    for claim in overview_order:
        if claim.get("verification_status") == "verification_unavailable":
            continue
        if claim.get("content_kind") in {"question", "schedule"} or claim.get("risk", {}).get("recognition", 0) >= .65:
            continue
        if claim.get("risk", {}).get("number", 0) >= .45:
            continue
        if re.search(r"(?iu)^\s*(?:это|так|вот\s+эт\w+|они|он|она)\b", claim.get("statement", "")) and not claim.get("entities"):
            continue
        tokens = set(re.findall(r"(?iu)[a-zа-яё0-9]+", claim.get("statement", "").casefold()))
        if any(len(tokens & old) / max(1, min(len(tokens), len(old))) >= .55 for old in (x[1] for x in overview)):
            continue
        overview.append((claim, tokens))
        if len(overview) == 4: break
    for claim, _ in overview:
        add("overview", claim, text=attributed_text(claim))
    for claim in selected("executive"):
        if claim.get("verification_status") == "verification_unavailable":
            continue
        if can_publish_as_decision(claim):
            add("decisions", claim, "accepted", attributed_text(claim))
    for claim in selected("technical"):
        if claim.get("verification_status") == "verification_unavailable":
            continue
        if claim.get("content_kind") in RULE_KINDS:
            add("rules", claim, "accepted" if can_publish_as_decision(claim) else "described", attributed_text(claim))
        elif claim.get("content_kind") in TECHNICAL_KINDS:
            add("technical", claim, "observation", attributed_text(claim))
    proposed, emitted_tasks = 0, set()
    confirmed = {"self_committed", "explicit_self_commitment", "assigned", "accepted", "in_progress", "blocked", "completed"}
    for claim in selected("tasks"):
        if claim.get("verification_status") == "verification_unavailable":
            continue
        state = task_states.get(claim.get("canonical_task_state_id"), {})
        if not state or state.get("task_id") in emitted_tasks: continue
        emitted_tasks.add(state["task_id"])
        status = state.get("status", "idea")
        description = str(state.get("deliverable") or state.get("description") or claim.get("statement") or "")
        if re.search(r"(?iu)^\s*(?:вопрос|уточнение|метаописание)\b", description):
            continue
        if claim.get("content_kind") not in {"action", "follow_up", "resource"} and not re.search(r"(?iu)\b(?:сделать|подготовить|отправить|передать|предоставить|разметить|размечивать|проверить|продолжить|реализовать|встроить|экспериментировать)\b", description):
            continue
        details = [description]
        if state.get("assignee"):
            details.append(f"исполнитель: {state['assignee']}")
        if state.get("current_scope"):
            qualifier = " (предложен, не подтверждён)" if state.get("scope_confidence") == "proposed" else ""
            details.append(f"объём: {state['current_scope']}{qualifier}")
        if state.get("data_origin"):
            details.append(f"период данных: {state['data_origin']}")
        if state.get("deadline"):
            due = state["deadline"].get("text") if isinstance(state["deadline"], dict) else state["deadline"]
            if due: details.append(f"срок: {due}")
        conditions = [x.get("antecedent") or x.get("text") for x in state.get("conditions", []) if isinstance(x, dict) and (x.get("antecedent") or x.get("text"))]
        if conditions:
            details.append("условие: " + "; ".join(conditions))
        labels = {"self_committed": "участник взял на себя", "proposed": "предложено, не подтверждено", "idea": "идея, не подтверждена", "assigned_pending": "назначение ожидает подтверждения", "assigned": "назначено", "accepted": "согласовано", "completed": "выполнено", "blocked": "заблокировано", "needs_verification": "требует проверки источника"}
        if status in labels:
            details.append(f"статус: {labels[status]}")
        task_text = " — ".join(details)
        if status in confirmed or status == "needs_verification":
            add("tasks", claim, status, task_text)
        elif status in {"proposed", "idea", "assigned_pending"} and proposed < 3:
            add("tasks", claim, status, task_text); proposed += 1
    for claim in selected("questions"):
        if claim.get("verification_status") == "verification_unavailable":
            continue
        if claim.get("content_kind") in {"question", "schedule"} and claim.get("question_status") not in {"answered", "rhetorical", "superseded"}:
            state = question_states.get(claim.get("proposition_id"), {})
            missing = [x for x in state.get("missing_slot_labels", []) if x]
            suffix = f" — не уточнено: {', '.join(missing)}" if missing else ""
            question_text = attributed_text(claim)
            if state.get("known_answer") and state.get("remaining_question"):
                question_text = f"Известно: {state['known_answer']} Осталось уточнить: {state['remaining_question']}"
            elif state.get("remaining_question"):
                question_text = state["remaining_question"]
            speakers = list(claim.get("speaker_refs", []))
            if len(speakers) == 1 and speakers[0] not in question_text:
                question_text = re.sub(r"(?iu)^\s*(?:участник\s+)?(?:спрашивает|зада[её]т\s+вопрос)(?:\s+о\s+том)?[, :] *", "", question_text)
                question_text = f"{speakers[0]} спрашивает: {question_text[:1].lower() + question_text[1:]}"
            answer_claim_ids = [claims_by_source[value]["claim_id"] for value in state.get("answer_record_ids", []) if value in claims_by_source]
            add("questions", claim, claim.get("question_status", "unanswered"), question_text + suffix,
                claim_ids=[claim["claim_id"]] + answer_claim_ids,
                relation_ids=state.get("answer_relation_ids", []), extra_evidence=state.get("answer_evidence_ids", []))
    for claim in selected("experiments"):
        if claim.get("verification_status") == "verification_unavailable":
            continue
        if claim.get("content_kind") == "hypothesis" and re.search(r"(?iu)\b(?:нельзя|невозможно|ограничен)\b", claim.get("statement", "")):
            add("technical", claim, "constraint", attributed_text(claim))
        else:
            add("experiments", claim, text=attributed_text(claim))
    seen_minutes = set()
    for claim in sorted(selected("minutes"), key=lambda x: (float(x.get("start", 0)), x.get("claim_id", ""))):
        if claim.get("verification_status") == "verification_unavailable":
            continue
        key = (claim.get("proposition_id"), claim.get("social_state"), tuple(claim.get("evidence_ids", [])))
        if key not in seen_minutes:
            add("minutes", claim, text=attributed_text(claim)); seen_minutes.add(key)
    for view, view_plan in view_plans.items():
        for claim_id in view_plan.get("selected_claim_ids", []):
            published = next((values[claim_id] for values in materialized.values() if claim_id in values), None)
            view_plan.setdefault("dispositions", {})[claim_id] = published or {"status": "excluded", "reason": "editorial_route_or_dedup"}
    return sections


def relation_markers(text):
    """Return relation wording that was already present in a source claim."""
    return {match.casefold() for match in CAUSAL_RE.findall(text or "")}


def source_aware_plan(plan, claims):
    """Allow participant references that occur verbatim in cited source claims."""
    result = dict(plan)
    source_text = " ".join(str(claim.get("statement") or "") for claim in claims)
    result["allowed_speakers"] = sorted(
        set(plan.get("allowed_speakers", [])) | set(re.findall(r"@[\w.-]+", source_text))
    )
    source_polarity = {"negative" if NEGATION_RE.search(str(claim.get("statement") or "")) else "positive"
                       for claim in claims}
    result["polarity"] = sorted(source_polarity)
    return result


def verify_sentence_plan(plan, claims, relations):
    by_id = {x.get("claim_id"): x for x in claims}
    errors = []
    if any(x not in by_id for x in plan.get("claim_ids", [])):
        errors.append("unknown_claim")
    if not cross_episode_allowed(plan.get("claim_ids", []), plan.get("relation_ids", []), claims, relations):
        errors.append("cross_episode_without_relation")
    relation_ids = {x.get("relation_id") for x in relations}
    if any(x not in relation_ids for x in plan.get("relation_ids", [])):
        errors.append("unknown_relation")
    return {"passed": not errors, "errors": errors}


def audit_realization(text, plan):
    allowed_numbers = {x.replace(" ", "") for x in plan.get("allowed_numbers", [])}
    found_numbers = {x.replace(" ", "") for x in NUMBER_RE.findall(text or "")}
    errors = []
    if not found_numbers.issubset(allowed_numbers):
        errors.append("unplanned_number")
    found_relations = relation_markers(text)
    allowed_relations = {str(x).casefold() for x in plan.get("allowed_relation_markers", [])}
    if found_relations and not plan.get("relation_ids") and not found_relations.issubset(allowed_relations):
        errors.append("unsupported_relation_language")
    polarities = set(plan.get("polarity", []))
    semantic_text = re.sub(r"(?iu)\b(?:не\s+уточнено|не\s+подтвержд[её]н[оаы]?|ожидает\s+подтверждения)\b", "", text or "")
    if "negative" in polarities and not NEGATION_RE.search(text or ""):
        errors.append("negation_not_preserved")
    if polarities == {"positive"} and NEGATION_RE.search(semantic_text):
        errors.append("unsupported_negation")
    modalities = set(plan.get("modality", []))
    if modalities & {"possible", "hypothetical", "unknown", "tentative"} and CERTAIN_RE.search(text or ""):
        errors.append("modality_upgraded")
    if modalities & {"possible", "hypothetical", "unknown", "tentative"} and COMPLETED_RE.search(text or ""):
        errors.append("completion_status_upgraded")
    if plan.get("conditions") and not CONDITION_RE.search(text or ""):
        errors.append("condition_not_preserved")
    number_words = {"один": "1", "одного": "1", "одну": "1", "два": "2", "две": "2", "три": "3", "четыре": "4"}
    normalize_scope = lambda value: re.sub(r"(?iu)\b(?:один|одного|одну|два|две|три|четыре)\b", lambda m: number_words[m.group(0).casefold()], str(value).casefold()).replace("ё", "е")
    allowed_scopes = [normalize_scope(x) for x in plan.get("time_scope", []) if isinstance(x, str) and x.strip()]
    if allowed_scopes and not any(scope in normalize_scope(text or "") for scope in allowed_scopes):
        errors.append("time_scope_not_preserved")
    allowed_values = list(plan.get("allowed_speakers", [])) + list(plan.get("allowed_assignees", []))
    allowed_speakers = set(allowed_values) | set(re.findall(r"@[\w.-]+", " ".join(allowed_values)))
    mentioned = set(re.findall(r"@[\w.-]+", text or ""))
    if not mentioned.issubset(allowed_speakers):
        errors.append("speaker_or_assignee_not_preserved")
    return {"passed": not errors, "errors": sorted(set(errors)), "atomic_claims": list(plan.get("claim_ids", [])), "relations": list(plan.get("relation_ids", [])), "status": "SUPPORTED" if not errors else "ABSTAIN"}


def verify_generated_items(items, sentence_plans, claims):
    """Audit actual generated text against the union contract of cited claims."""
    by_claim = {x.get("claim_id"): x for x in claims}
    plan_by_claim = {claim_id: plan for plan in sentence_plans for claim_id in plan.get("claim_ids", [])}
    audits = []
    for item in items:
        text = str(item.get("text") or item.get("statement") or "")
        requested_claim_ids = list(item.get("claim_ids") or item.get("fact_ids", []))
        unknown_claim_ids = [x for x in requested_claim_ids if x not in by_claim]
        claim_ids = [x for x in requested_claim_ids if x in by_claim]
        plans = [plan_by_claim[x] for x in claim_ids if x in plan_by_claim]
        if not text or not claim_ids or not plans:
            errors = (["empty_public_text"] if not text else []) + (["orphan_public_item"] if not claim_ids else []) + (["claim_outside_plan"] if claim_ids and not plans else []) + (["unknown_claim"] if unknown_claim_ids else [])
            audits.append({"text": text, "claim_ids": claim_ids, "passed": False, "errors": errors, "atomic_claims": claim_ids, "relations": [], "status": "ABSTAIN", "qa": {"passed": False, "checks": {}}})
            continue
        source_mentions = {mention for claim_id in claim_ids for mention in re.findall(r"@[\w.-]+", str(by_claim[claim_id].get("statement") or ""))}
        merged = {"claim_ids": claim_ids, "relation_ids": sorted({r for p in plans for r in p.get("relation_ids", [])}), "allowed_numbers": [n for p in plans for n in p.get("allowed_numbers", [])], "allowed_relation_markers": sorted({r for p in plans for r in p.get("allowed_relation_markers", [])}), "allowed_speakers": sorted({s for p in plans for s in p.get("allowed_speakers", [])} | source_mentions), "allowed_assignees": sorted({s for p in plans for s in p.get("allowed_assignees", [])}), "polarity": [v for p in plans for v in p.get("polarity", [])], "modality": [v for p in plans for v in p.get("modality", [])], "conditions": [v for p in plans for v in p.get("conditions", [])], "time_scope": [v for p in plans for v in p.get("time_scope", [])]}
        task_state = item.get("task_state", {}) if item.get("section") == "tasks" else {}
        question_state = item.get("question_state", {}) if item.get("section") == "questions" else {}
        if task_state and item.get("task_state_id") in {by_claim[x].get("canonical_task_state_id") for x in claim_ids}:
            metadata_text = " ".join(str(task_state.get(field) or "") for field in ("description", "current_scope", "data_origin", "deadline", "assignee"))
            merged["allowed_numbers"].extend(NUMBER_RE.findall(metadata_text))
            merged["allowed_speakers"].extend(re.findall(r"@[\w.-]+", metadata_text))
            merged["allowed_assignees"].extend(re.findall(r"@[\w.-]+", metadata_text))
            if task_state.get("current_scope"):
                merged["time_scope"].append(str(task_state["current_scope"]))
        else:
            metadata_text = ""
        if question_state:
            metadata_text += " " + " ".join(str(question_state.get(field) or "") for field in
                                                ("original_question", "known_answer", "remaining_question"))
            metadata_text += " " + " ".join(map(str, question_state.get("missing_slot_labels", [])))
            merged["polarity"] = []  # Question-state labels are not predicate polarity.
        cited = [by_claim[x] for x in claim_ids]
        merged["allowed_numbers"].extend(NUMBER_RE.findall(" ".join(str(x.get("statement") or "") for x in cited)))
        merged["allowed_speakers"].extend(s for x in cited for s in x.get("speaker_refs", []))
        source_has_negation = any(NEGATION_RE.search(str(x.get("statement") or "")) for x in cited)
        if source_has_negation and "negative" not in merged["polarity"]:
            merged["polarity"].append("negative")
        realization = audit_realization(text, merged)
        if unknown_claim_ids:
            realization["errors"].append("unknown_claim")
        source_tokens = {v for x in cited for v in re.findall(r"(?iu)[a-zа-яё0-9]+", str(x.get("statement") or "").casefold()) if len(v) > 2}
        source_tokens.update(v for v in re.findall(r"(?iu)[a-zа-яё0-9]+", metadata_text.casefold()) if len(v) > 2)
        text_tokens = {v for v in re.findall(r"(?iu)[a-zа-яё0-9]+", text.casefold()) if len(v) > 2}
        editorial_tokens = {"спрашивает", "исполнитель", "статус", "объём", "срок", "условие", "участник", "назначение", "ожидает", "подтверждения", "предложен", "подтверждён", "уточнено", "известно", "осталось", "уточнить", "взял", "себя"}
        content_tokens = text_tokens - editorial_tokens
        overlap = len(source_tokens & content_tokens)
        if item.get("section") and overlap / max(1, len(content_tokens)) < .45:
            realization["errors"].append("cleaned_text_semantic_drift")
        unsupported = content_tokens - source_tokens
        if len(unsupported) >= 3 and overlap / max(1, len(content_tokens)) < .68:
            realization["errors"].append("unsupported_added_clause")
        task_like = item.get("section") == "tasks" or bool(re.search(r"(?iu)\b(?:должен|должна|сделает|подготовит|отправит)\b", text))
        expected_actor = task_state.get("assignee") if task_state else None
        rendered_actor = re.search(r"(?iu)исполнитель:\s*(@[\w.-]+)", text)
        if task_like and rendered_actor and (not expected_actor or rendered_actor.group(1) != expected_actor):
            realization["errors"].append("actor_recipient_swap")
        if any(x.get("lifecycle", "active") != "active" for x in cited):
            realization["errors"].append("inactive_claim_published")
        if item.get("section") == "decisions" and not all(can_publish_as_decision(x) for x in cited):
            realization["errors"].append("status_upgrade")
        if item.get("section") == "tasks" and any(x.get("task_status") in {"idea", "proposed", "superseded"} for x in cited) and item.get("social_state") in {"accepted", "self_committed", "assigned"}:
            realization["errors"].append("task_status_upgrade")
        qa = qa_verify(text, merged)
        realization["errors"] = sorted(set(realization["errors"]))
        realization["passed"] = not realization["errors"] and qa["passed"]
        if not qa["passed"]:
            realization["errors"] = sorted(set(realization["errors"] + ["qa_slot_failure"]))
            realization["status"] = "ABSTAIN"
        audits.append({"text": text, "claim_ids": claim_ids, **realization, "qa": qa})
    return {"passed": all(x["passed"] for x in audits), "audits": audits, "abstentions": [x for x in audits if not x["passed"]]}


def publication_audit(report, artifact_text, items=None, summary_plan=None, verified_hash=None):
    items = items or []
    summary_plan = summary_plan or {}
    audits = report.get("audits", [])
    counters = {
        "unsupported_public_items": sum(not x.get("passed") for x in audits),
        "orphan_public_items": sum("orphan_public_item" in x.get("errors", []) for x in audits),
        "status_upgrades": sum(bool({"status_upgrade", "task_status_upgrade"} & set(x.get("errors", []))) for x in audits),
        "superseded_items_published": sum("inactive_claim_published" in x.get("errors", []) for x in audits),
        "number_or_negation_mismatches": sum(bool({"unplanned_number", "negation_not_preserved", "unsupported_negation"} & set(x.get("errors", []))) for x in audits),
        "cross_episode_merges_without_relation": sum("cross_episode_without_relation" in x.get("errors", []) for x in audits),
        "unknown_assignee_publications": sum("speaker_or_assignee_not_preserved" in x.get("errors", []) for x in audits),
    }
    def tokens(value): return {x for x in re.findall(r"(?iu)[a-zа-яё0-9]+", str(value or "").casefold()) if len(x) > 3}
    duplicates = 0
    for index, left in enumerate(items):
        for right in items[index + 1:]:
            if left.get("section") != right.get("section"): continue
            a, b = tokens(left.get("text")), tokens(right.get("text"))
            duplicates += bool(a and b and len(a & b) / max(1, min(len(a), len(b))) >= .8)
    minute_starts = [float(x.get("start", 0)) for x in items if x.get("section") == "minutes"]
    task_ids = [x.get("task_state_id") for x in items if x.get("section") == "tasks"]
    internal = re.compile(r"(?iu)\b(?:self_committed|assigned_pending|additional_tools|rhythmic_entry_implementation|high_tf_result|stop_loss_options)\b")
    overview_tokens = set().union(*(tokens(x.get("text")) for x in items if x.get("section") == "overview"))
    task_tokens = set().union(*(tokens(x.get("text")) for x in items if x.get("section") == "tasks"))
    rendered_counts, current = {}, None
    heading_sections = {"Главное": "overview", "Краткое описание — что изменилось после встречи": "overview", "Принятые решения": "decisions", "Упомянутые действующие правила": "rules", "Договорённости и следующие шаги": "tasks", "Задачи и следующие шаги": "tasks", "Что осталось уточнить": "questions", "Открытые вопросы": "questions", "Технические выводы и ограничения": "technical", "Идеи и эксперименты": "experiments", "Гипотезы и эксперименты": "experiments", "Требует проверки источника": "requires_verification", "Хронология встречи": "minutes", "Подробная хронология встречи": "minutes"}
    for line in artifact_text.splitlines():
        if line.startswith("## "): current = heading_sections.get(line[3:].strip())
        elif line.startswith("- ") and current: rendered_counts[current] = rendered_counts.get(current, 0) + 1
    item_counts = {section: sum(x.get("section") == section for x in items) for section in sorted({x.get("section") for x in items})}
    title_line = next((line for line in artifact_text.splitlines() if line.startswith("# ")), "")
    authored_title = title_line.split("—", 1)[-1]
    overview_match = re.search(r"(?s)## (?:Главное|Краткое описание[^\n]*)\n(.*?)(?=\n## |\Z)", artifact_text)
    authored_surface = authored_title + "\n" + (overview_match.group(1) if overview_match else "")
    allowed_document_numbers = {x.replace(" ", "") for item in items for x in NUMBER_RE.findall(str(item.get("text") or ""))}
    found_document_numbers = {x.replace(" ", "") for x in NUMBER_RE.findall(authored_surface)}
    state_conflicts = sum(bool(
        x.get("section") == "tasks" and (not x.get("task_state_id") or x.get("social_state") != x.get("task_state", {}).get("status") or (x.get("task_state", {}).get("current_scope") and str(x["task_state"]["current_scope"]).casefold() not in str(x.get("text") or "").casefold()))
    )
        for x in items
    )
    counters.update({
        "duplicate_items": duplicates,
        "answered_questions_published_as_open": sum(x.get("section") == "questions" and x.get("question_state", {}).get("status") in {"answered", "rhetorical", "superseded"} for x in items),
        "unconfirmed_tasks_published_as_committed": sum(x.get("section") == "tasks" and x.get("social_state") == "self_committed" and x.get("task_state", {}).get("commitment_strength") != "explicit" for x in items),
        "duplicate_task_states": len([x for x in task_ids if x]) - len({x for x in task_ids if x}),
        "chronology_inversions": sum(a > b for a, b in zip(minute_starts, minute_starts[1:])),
        "internal_labels_exposed": sum(bool(internal.search(str(x.get("text") or ""))) for x in items),
        "missing_public_provenance": sum(not x.get("evidence_ids") or not x.get("source_word_ids") for x in items),
        "planner_budget_violations": sum(int(v.get("selected_count", len(v.get("selected_claim_ids", [])))) > int(v.get("budget", 0)) for v in summary_plan.get("view_plans", {}).values()),
        "state_conflicts": state_conflicts,
        "task_without_deliverable": sum(x.get("section") == "tasks" and not str(x.get("task_state", {}).get("deliverable") or "").strip() for x in items),
        "overview_task_overlap": (len(overview_tokens & task_tokens) / max(1, len(overview_tokens))) if overview_tokens else 0,
        # Overview items are intentionally merged into prose paragraphs rather
        # than rendered one bullet per PublicItem.
        "section_count_mismatches": sum(item_counts.get(k, 0) != rendered_counts.get(k, 0) for k in (set(item_counts) | set(rendered_counts)) - {"overview", "minutes"}),
        "unplanned_document_numbers": len(found_document_numbers - allowed_document_numbers),
        "navigation_missing": int(bool(item_counts.get("minutes")) and "## Таймкоды" not in artifact_text),
        "chronology_missing": int(bool(item_counts.get("minutes")) and "## Подробная хронология встречи" not in artifact_text and "## Хронология встречи" not in artifact_text),
        "title_missing": int(not title_line),
    })
    counters["section_counts"] = item_counts
    counters["rendered_section_counts"] = rendered_counts
    counters["planner_overflow"] = {name: value.get("overflow_count", 0) for name, value in summary_plan.get("view_plans", {}).items()}
    artifact_hash = hashlib.sha256(artifact_text.encode("utf-8")).hexdigest()
    counters["verified_artifact_hash"] = verified_hash or artifact_hash
    counters["final_artifact_hash"] = artifact_hash
    unexplained = sum(1 for view in summary_plan.get("view_plans", {}).values() for claim_id in view.get("selected_claim_ids", []) if claim_id not in view.get("dispositions", {}))
    public_claims = {claim_id for item in items for claim_id in item.get("claim_ids", [])}
    candidate_ids = set(summary_plan.get("commitment_candidate_ids", []))
    routed_work = {claim_id for item in items if item.get("section") in {"tasks", "requires_verification"} for claim_id in item.get("claim_ids", [])}
    counters.update({"unexplained_selected_claims": unexplained, "unexplained_commitment_candidates": len(candidate_ids - routed_work), "published_unique_claims": len(public_claims)})
    integrity_keys = {"unsupported_public_items", "orphan_public_items", "status_upgrades", "superseded_items_published", "number_or_negation_mismatches", "cross_episode_merges_without_relation", "unknown_assignee_publications", "duplicate_items", "answered_questions_published_as_open", "unconfirmed_tasks_published_as_committed", "duplicate_task_states", "chronology_inversions", "internal_labels_exposed", "missing_public_provenance", "planner_budget_violations", "state_conflicts", "task_without_deliverable", "section_count_mismatches", "unplanned_document_numbers", "navigation_missing", "chronology_missing", "title_missing"}
    integrity = all(counters.get(key, 0) == 0 for key in integrity_keys) and counters["verified_artifact_hash"] == artifact_hash
    grounding = counters["unsupported_public_items"] == counters["number_or_negation_mismatches"] == counters["missing_public_provenance"] == 0
    coverage = bool(items) and unexplained == 0 and counters["unexplained_commitment_candidates"] == 0
    readability = counters["duplicate_items"] == counters["internal_labels_exposed"] == 0
    return {"schema": "PublicationAudit", "schema_version": 3, "passed": integrity and grounding and coverage and readability, "dimensions": {"integrity": integrity, "grounding": grounding, "candidate_disposition_integrity": coverage, "readability": readability}, **counters}


def runtime_quality_gates(report, artifact_text, verified_hash=None, items=None, summary_plan=None):
    return publication_audit(report, artifact_text, items, summary_plan, verified_hash)


def verify_public_document(document, artifact_text, items):
    """Bind the final title, prose, navigation and sections to their sources."""
    errors = []
    source_ids = {claim for item in items for claim in item.get("claim_ids", [])}
    by_claim = {claim: item for item in items for claim in item.get("claim_ids", [])}
    tokens = lambda value: set(re.findall(r"(?iu)[a-zа-яё0-9]+", re.sub(r"[*`]", "", str(value or "")).casefold()))
    public_words = tokens(artifact_text)
    title = document.get("title", {})
    if not title.get("text") or not set(title.get("claim_ids", [])) <= source_ids or not title.get("evidence_ids"):
        errors.append("unsupported_title")
    if not artifact_text.splitlines() or not artifact_text.splitlines()[0].endswith(" — " + str(title.get("text") or "")):
        errors.append("title_not_rendered")
    title_source = set().union(*(tokens(by_claim[claim].get("text")) for claim in title.get("claim_ids", []) if claim in by_claim))
    if tokens(title.get("text")) - title_source - {"следующие", "шаги", "итоги", "встречи", "и"}:
        errors.append("title_semantic_drift")
    for node in document.get("overview", []):
        if not node.get("claim_ids") or not set(node["claim_ids"]) <= source_ids or not node.get("evidence_ids"):
            errors.append("unsupported_overview")
        if node.get("text") not in artifact_text:
            errors.append("overview_not_rendered")
        backing = [item for item in items if item.get("section") == "overview" and item.get("claim_ids") == node.get("claim_ids")]
        source_words = set().union(*(tokens(item.get("text")) for item in backing)) if backing else set()
        if not backing or len(tokens(node.get("text")) - source_words) > 2:
            errors.append("overview_semantic_drift")
    navigation = document.get("navigation", [])
    if document.get("chronology") and not navigation:
        errors.append("missing_navigation")
    if navigation and "## Таймкоды" not in artifact_text:
        errors.append("navigation_not_rendered")
    for chapter in navigation:
        if not chapter.get("claim_ids") or not set(chapter["claim_ids"]) <= source_ids or not chapter.get("evidence_ids"):
            errors.append("unsupported_navigation")
        if chapter.get("label") not in artifact_text:
            errors.append("navigation_not_rendered")
        source_words = set().union(*(tokens(item.get("text")) for item in items if item.get("section") == "minutes" and set(item.get("claim_ids", [])) & set(chapter.get("claim_ids", []))))
        if len(tokens(chapter.get("label")) - source_words) > 1:
            errors.append("navigation_semantic_drift")
    for section, section_items in document.get("sections", {}).items():
        for item in section_items:
            if item not in items or len(tokens(item.get("text")) - public_words) > 2:
                errors.append("section_not_rendered_from_verified_items")
    chronology_ids = {item.get("public_id") for chapter in document.get("chronology", []) for item in chapter.get("items", [])}
    minute_ids = {item.get("public_id") for item in items if item.get("section") == "minutes"}
    if chronology_ids != minute_ids:
        errors.append("chronology_item_loss")
    return {"passed": not errors, "errors": sorted(set(errors)), "title_claim_ids": title.get("claim_ids", []), "navigation_chapters": len(navigation)}


def diff_public_items(previous, current):
    """Non-blocking shadow diff for release review and regression triage."""
    key = lambda x: (x.get("section"), tuple(x.get("claim_ids", [])))
    before, after = {key(x): x for x in previous or []}, {key(x): x for x in current or []}
    return {
        "added": [after[x] for x in sorted(after.keys() - before.keys())],
        "removed": [before[x] for x in sorted(before.keys() - after.keys())],
        "changed": [{"before": before[x], "after": after[x]} for x in sorted(before.keys() & after.keys()) if before[x].get("text") != after[x].get("text") or before[x].get("social_state") != after[x].get("social_state")],
        "section_counts_before": {section: sum(x.get("section") == section for x in previous or []) for section in sorted({x.get("section") for x in previous or []})},
        "section_counts_after": {section: sum(x.get("section") == section for x in current or []) for section in sorted({x.get("section") for x in current or []})},
    }


def alignment_score(premise, hypothesis):
    """Cheap independent alignment stage; ambiguous cases are escalated by caller."""
    tokens = lambda x: {v for v in re.findall(r"(?iu)[a-zа-яё0-9]+", x or "") if len(v) > 2}
    left, right = tokens(premise), tokens(hypothesis)
    entailment = len(left & right) / max(1, len(right))
    contradiction = 1.0 if bool(NEGATION_RE.search(premise or "")) != bool(NEGATION_RE.search(hypothesis or "")) else 0.0
    return {"entailment": entailment, "contradiction": contradiction, "ambiguous": entailment < .72 or contradiction > 0}


def qa_verify(text, plan):
    """Independent slot checks for who/quantity/condition/state questions."""
    assignment_claimed = bool(re.search(r"(?iu)\b(?:поручено|ответственн(?:ый|ая)|должен|владелец)\b", text or ""))
    allowed_values = list(plan.get("allowed_assignees", [])) + list(plan.get("allowed_speakers", []))
    allowed_people = set(re.findall(r"@[\w.-]+", " ".join(allowed_values)))
    mentioned_people = set(re.findall(r"@[\w.-]+", text or ""))
    checks = {"who": not assignment_claimed or (bool(mentioned_people) and mentioned_people.issubset(allowed_people)), "quantity": not plan.get("allowed_numbers") or set(NUMBER_RE.findall(text)).issubset(set(plan["allowed_numbers"])), "condition": not plan.get("conditions") or bool(CONDITION_RE.search(text)), "decision_state": not plan.get("decision_state") or not ("решено" in text.casefold() and "accepted" not in plan["decision_state"])}
    return {"passed": all(checks.values()), "checks": checks}
