"""Build field-grounded topic outcomes from canonical meeting state."""
from __future__ import annotations

import hashlib
import re


def _dedupe(values):
    return list(dict.fromkeys(value for value in values if value))


def _pick(claims, kinds, predicate=lambda _item: True):
    candidates = [item for item in claims if item.get("content_kind") in kinds and predicate(item)]
    return max(candidates, key=lambda item: (len(item.get("evidence_ids", [])), len(str(item.get("statement") or ""))), default=None)


def _field(claim, text=None):
    if not claim:
        return None
    return {
        "value": str(text or claim.get("statement") or "").strip(),
        "claim_ids": [claim.get("claim_id")],
        "evidence_ids": list(claim.get("evidence_ids", [])),
    }


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
        state = _pick(claims, {"current_state", "experimental_result", "observation"})
        constraint = _pick(claims, {"problem", "blocker", "constraint", "risk", "dependency"})
        resolution = _pick(claims, {"decision", "design_choice", "correction"}, lambda item: item.get("lifecycle", "active") == "active")
        work = _pick(claims, {"resource", "dataset", "definition", "trading_rule", "system_rule"})
        next_claim = _pick(claims, {"action", "follow_up", "proposal"})
        next_field = None
        if next_claim:
            task = task_by_prop.get(next_claim.get("proposition_id"), {})
            label = str(task.get("deliverable") or next_claim.get("statement") or "").strip()
            if task.get("status"):
                label += f" (статус: {task['status']})"
            next_field = _field(next_claim, label)
        remaining = []
        for claim in claims:
            question = question_by_prop.get(claim.get("proposition_id"))
            if question and question.get("status") not in {"answered", "rhetorical", "superseded"}:
                text = question.get("remaining_unknown") or question.get("residual_question_text") or question.get("remaining_question")
                if text:
                    remaining.append({
                        "value": text, "claim_ids": [claim["claim_id"]],
                        "evidence_ids": _dedupe(claim.get("evidence_ids", []) + question.get("answer_evidence_ids", [])),
                    })
        fields = {
            "current_state": _field(state), "constraint": _field(constraint),
            "resolution": _field(resolution), "work_result": _field(work),
            "next_step": next_field, "remaining_unknown": remaining,
        }
        populated = [value for key, value in fields.items() if value and key != "remaining_unknown"]
        populated += remaining
        if not populated:
            fallback = max(claims, key=lambda item: len(str(item.get("statement") or "")))
            fields["current_state"] = _field(fallback)
            populated = [fields["current_state"]]
        raw = "|".join(sorted(value for field in populated for value in field.get("claim_ids", [])))
        topic = str(bundle.get("topic") or "Тема встречи").strip()
        if re.search(r"(?iu)^участник\s+спрашивает", topic):
            topic = "Результат обсуждения"
        cards.append({
            "outcome_id": "OC" + hashlib.sha256(raw.encode()).hexdigest()[:12],
            "bundle_id": bundle.get("bundle_id"), "topic": topic,
            "ranges": list(bundle.get("ranges", [])), "fields": fields,
            "claim_ids": _dedupe(value for field in populated for value in field.get("claim_ids", [])),
            "evidence_ids": _dedupe(value for field in populated for value in field.get("evidence_ids", [])),
            "status": "verified_input", "verification": {"status": "pending", "errors": []},
        })
    return cards
