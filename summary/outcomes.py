"""Build field-grounded topic outcomes from canonical meeting state."""
from __future__ import annotations

import hashlib
import re
from semantics.equivalence import equivalent


def _dedupe(values):
    return list(dict.fromkeys(value for value in values if value))


def _pick(claims, kinds, predicate=lambda _item: True):
    candidates = [item for item in claims if item.get("content_kind") in kinds and predicate(item)
                  and item.get("lifecycle", "active") == "active"]
    # The resolved state wins over verbosity/evidence-window size.  Evidence
    # count is useful only as a final deterministic tie-breaker.
    rank = {"completed": 7, "accepted": 6, "self_committed": 5, "in_progress": 4,
            "intent_to_attempt": 3, "assigned": 3, "candidate": 2, "unknown": 1}
    return max(candidates, key=lambda item: (
        float(item.get("start", 0)),
        rank.get(item.get("task_status") or item.get("decision_status") or item.get("social_state") or "unknown", 0),
        len(item.get("evidence_ids", [])),
    ), default=None)


def _field(claim, text=None):
    if not claim:
        return None
    return {
        "value": str(text or claim.get("publication_text") or claim.get("statement") or "").strip(),
        "claim_ids": [claim.get("claim_id")],
        "evidence_ids": list(claim.get("evidence_ids", [])),
        "verification_status": claim.get("verification_status"),
        "content_kind": claim.get("content_kind"),
        "polarity": claim.get("polarity"),
        "modality": claim.get("modality"),
        "lifecycle": claim.get("lifecycle", "active"),
        "social_state": claim.get("task_status") or claim.get("decision_status") or claim.get("social_state"),
        "quantities": list(claim.get("quantities", [])),
        "conditions": list(claim.get("conditions", [])),
        "origin_ids": list(claim.get("origin_ids", [])),
        "uncertainty": {"status": claim.get("verification_status", "supported"), "risk": claim.get("risk", {})},
    }


def _near_duplicate(left, right, threshold=.82):
    return equivalent(left if isinstance(left, dict) else {"value": left}, right if isinstance(right, dict) else {"value": right}, threshold)


def build_outcome_cards(graph, allowed_claim_ids=None):
    """Produce useful topic cards whose every field has its own evidence."""
    by_claim = {item["claim_id"]: item for item in graph.get("claims", [])}
    task_by_prop = {
        prop_id: task for task in graph.get("task_states", [])
        for prop_id in task.get("source_proposition_ids", [task.get("proposition_id")])
    }
    question_by_prop = {item.get("proposition_id"): item for item in graph.get("question_states", [])}
    cards = []
    for bundle in graph.get("dialogue_bundles", []):
        allowed = set(allowed_claim_ids) if allowed_claim_ids is not None else None
        claims = [by_claim[value] for value in bundle.get("claim_ids", [])
                  if value in by_claim and (allowed is None or value in allowed)]
        if not claims:
            continue
        def not_open_question(item):
            if item.get("verification_status") in {"verification_unavailable", "insufficient_evidence", "contradicted"}:
                return False
            question = question_by_prop.get(item.get("proposition_id"))
            return not question or question.get("status") in {"answered", "rhetorical", "superseded"}
        state = _pick(claims, {"current_state", "experimental_result", "observation", "action"}, lambda item: not_open_question(item) and (
            item.get("content_kind") != "action" or item.get("temporal_state") in {"in_progress", "past_attempt"}
        ))
        constraint = _pick(claims, {"problem", "blocker", "constraint", "risk", "dependency"}, not_open_question)
        resolution = _pick(claims, {"decision", "proposal", "design_choice", "correction"}, lambda item: not_open_question(item) and item.get("lifecycle", "active") == "active" and (item.get("content_kind") != "proposal" or item.get("decision_status") == "accepted"))
        work = _pick(claims, {"experimental_result", "action", "follow_up"}, lambda item: not_open_question(item) and (
            item.get("task_status") == "completed" or item.get("temporal_state") == "completed" or item.get("content_kind") == "experimental_result"
        ))
        mentioned_resource = _pick(claims, {"resource", "dataset"}, not_open_question)
        described_rule = _pick(claims, {"definition", "trading_rule", "system_rule"}, not_open_question)
        next_claim = _pick(claims, {"action", "follow_up", "proposal"}, lambda item: not_open_question(item) and (
            item.get("content_kind") == "proposal" or item.get("task_status") not in {"past_attempt", "completed"}
            and item.get("temporal_state") not in {"past_attempt", "completed"}
        ))
        next_field = None
        if next_claim:
            task = task_by_prop.get(next_claim.get("proposition_id"), {})
            # ``publication_text`` has already passed the public-surface
            # grounding guard.  Prefer it to a semantic task deliverable,
            # which may still contain an internal snake_case model label.
            label = str(next_claim.get("publication_text") or task.get("deliverable") or next_claim.get("statement") or "").strip()
            if task.get("status"):
                labels = {"self_committed": "участник взял на себя", "intent_to_attempt": "участник намерен попробовать", "in_progress": "в работе", "past_attempt": "ранее выполнялось", "assigned_pending": "назначение ожидает подтверждения", "assigned": "назначено", "accepted": "согласовано", "completed": "выполнено", "blocked": "заблокировано", "proposed": "предложено, не подтверждено", "idea": "идея, не подтверждена"}
                label += f" (статус: {labels.get(task['status'], task['status'])})"
            next_field = _field(next_claim, label)
        resolution_field = _field(resolution)
        earlier_fields = [
            _field(candidate)
            for candidate in (state, constraint, resolution, work, mentioned_resource, described_rule)
            if candidate
        ]
        if next_field and any(_near_duplicate(field, next_field) for field in earlier_fields):
            next_field = None
        remaining = []
        for claim in claims:
            question = question_by_prop.get(claim.get("proposition_id"))
            if question and question.get("status") not in {"answered", "rhetorical", "superseded"}:
                text = question.get("remaining_unknown") or question.get("residual_question_text") or question.get("remaining_question")
                if text:
                    remaining.append({
                        "value": text, "claim_ids": [claim["claim_id"]],
                        # The residual field is grounded by the question event.
                        # Answer evidence belongs to its own claim and must not
                        # be smuggled into this field's evidence closure.
                        "evidence_ids": _dedupe(claim.get("evidence_ids", [])),
                    })
        fields = {
            "current_state": _field(state), "constraint": _field(constraint),
            "resolution": resolution_field, "work_result": _field(work),
            "mentioned_resource": _field(mentioned_resource),
            "described_rule": _field(described_rule),
            "next_step": next_field, "remaining_unknown": remaining,
        }
        populated = [value for key, value in fields.items() if value and key != "remaining_unknown"]
        populated += remaining
        if not populated:
            fallback = max((item for item in claims if not_open_question(item)), key=lambda item: len(str(item.get("statement") or "")), default=None)
            if fallback:
                fields["current_state"] = _field(fallback)
                populated = [fields["current_state"]]
        if not populated:
            continue
        all_claim_ids = _dedupe(item.get("claim_id") for item in claims)
        all_evidence_ids = _dedupe(value for item in claims for value in item.get("evidence_ids", []))
        raw = "|".join([str(bundle.get("bundle_id") or "")] + sorted(all_claim_ids))
        topic = str(bundle.get("topic") or "Тема встречи").strip()
        if re.search(r"(?iu)^участник\s+спрашивает", topic):
            topic = "Результат обсуждения"
        cards.append({
            "outcome_id": "OC" + hashlib.sha256(raw.encode()).hexdigest()[:12],
            "bundle_id": bundle.get("bundle_id"), "topic": topic,
            "ranges": list(bundle.get("ranges", [])), "fields": fields,
            # A card represents the full verified bundle, while each field
            # retains its narrower claim/evidence binding.
            "claim_ids": all_claim_ids,
            "evidence_ids": all_evidence_ids,
            "status": "verified_input", "verification": {"status": "pending", "errors": []},
        })
    return cards
