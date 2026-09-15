"""Bounded candidate generation plus high-precision semantic relation resolution."""
from __future__ import annotations
import hashlib
import re
from .questions import verify_slot_entailment

ACCEPT_RE = re.compile(r"(?iu)^\s*(?:да|согласен|делаем|ок(?:ей)?|подтверждаю)\b")
REJECT_RE = re.compile(r"(?iu)^\s*(?:нет|не согласен|не делаем|отклоняем)\b")
CAUSE_RE = re.compile(r"(?iu)\b(?:из-за|поэтому|в результате|привел[оа]? к)\b")
CONDITION_RE = re.compile(r"(?iu)\b(?:если|когда|при условии|после того как)\b")
SCOPE_RE = re.compile(r"(?iu)\b(?:месяц\w*|год\w*|недел\w*|день|дня|дней|час\w*|минут\w*|период\w*|объ[её]м\w*)\b")
SCOPE_REPLY_RE = re.compile(r"(?iu)\b(?:для\s+(?:начала|проверки|этого)|достаточно|возьм[её]м|объ[её]м|период|нужн\w+\s+(?:данн\w*|выборк\w*|объ[её]м\w*|период\w*))")
DATA_RESULT_RE = re.compile(r"(?iu)\b(?:данн\w*|выборк\w*|выгруз\w*|отрезк\w*|файл\w*|истори\w*)\b")


def _tokens(value):
    return {x for x in re.findall(r"(?iu)[a-zа-яё0-9]+", str(value or "").casefold()) if len(x) > 2}


def _similarity(left, right):
    a, b = _tokens(left), _tokens(right)
    return len(a & b) / max(1, len(a | b))


def _relation(kind, source, target, evidence, confidence, basis):
    raw = f"{kind}|{source}|{target}|{'|'.join(evidence)}"
    return {"relation_id": "R" + hashlib.sha256(raw.encode()).hexdigest()[:16], "type": kind, "source_proposition_id": source, "target_proposition_id": target, "evidence_ids": list(dict.fromkeys(evidence)), "confidence": confidence, "basis": basis}


def resolve_relations(propositions, events, records):
    """Resolve explicit links first, then only bounded nearby candidates."""
    by_record = {x.get("record_id"): x for x in records}
    prop_by_record = {rid: prop for prop in propositions for rid in prop.get("source_record_ids", [])}
    ordered = sorted(events, key=lambda x: (x.get("timestamp", 0), x.get("event_id")))
    relations, seen = [], set()
    def add(kind, source, target, evidence, confidence=.9, basis="deterministic"):
        if not source or not target or source == target or not evidence:
            return
        key = (kind, source, target)
        if key in seen:
            return
        seen.add(key); relations.append(_relation(kind, source, target, evidence, confidence, basis))
    for record in records:
        source = prop_by_record.get(record.get("record_id"))
        if not source:
            continue
        for answer_id in record.get("answer_record_ids", []):
            answer = prop_by_record.get(answer_id)
            if answer:
                slot_check = verify_slot_entailment(record.get("requested_slots", []), by_record.get(answer_id, {}), record)
                kind = "answers" if slot_check["passed"] else "partially_answers"
                add(kind, answer["proposition_id"], source["proposition_id"], answer["evidence_ids"] + source["evidence_ids"], .98 if slot_check["passed"] else .75, "explicit_slot_entailment")
        for target_id in record.get("corrects_record_ids", []):
            target = prop_by_record.get(target_id)
            if target: add("corrects", source["proposition_id"], target["proposition_id"], source["evidence_ids"] + target["evidence_ids"], .99, "explicit_revision")
        for target_id in record.get("supersedes_record_ids", []) + record.get("revises_record_ids", []):
            target = prop_by_record.get(target_id)
            if target: add("supersedes", source["proposition_id"], target["proposition_id"], source["evidence_ids"] + target["evidence_ids"], .99, "explicit_revision")
    for index, event in enumerate(ordered):
        source = next(x for x in propositions if x["proposition_id"] == event["proposition_id"])
        text = source["statement"]
        # Short replies are only safe inside a very small dialogue window.
        prior_events = [x for x in ordered[max(0, index - 8):index]
                        if event.get("timestamp", 0) - x.get("timestamp", 0) <= 120]
        local_scope_targets = [x for x in prior_events if event.get("timestamp", 0) - x.get("timestamp", 0) <= 45
                               and next(p for p in propositions if p["proposition_id"] == x["proposition_id"])["content_kind"] in {"action", "follow_up", "resource"}]
        if event["speech_act"] in {"accept", "reject"} or ACCEPT_RE.search(text) or REJECT_RE.search(text):
            candidates = [x for x in prior_events if x["speech_act"] in {"propose", "ask", "commit"}]
            same_thread = [x for x in candidates if not event.get("thread_hint") or x.get("thread_hint") == event.get("thread_hint")]
            candidates = same_thread or candidates
            target_event = candidates[0] if len(candidates) == 1 else None
            # A speaker's own acknowledgement is not evidence that another
            # participant accepted the proposition.
            is_rejection = event["speech_act"] == "reject" or bool(REJECT_RE.search(text))
            if target_event and (is_rejection or not event.get("speaker") or event.get("speaker") != target_event.get("speaker")):
                kind = "rejects" if is_rejection else "accepts"
                add(kind, source["proposition_id"], target_event["proposition_id"], source["evidence_ids"], .96, "coreference_short_reply")
        for prior in reversed(prior_events):
            target = next(x for x in propositions if x["proposition_id"] == prior["proposition_id"])
            similarity = _similarity(text, target["statement"])
            shared_entities = {x["entity_id"] for x in source["entities"]} & {x["entity_id"] for x in target["entities"]}
            local_reply = event.get("timestamp", 0) - prior.get("timestamp", 0) <= 20 and bool(SCOPE_REPLY_RE.search(text))
            if (target["content_kind"] in {"action", "follow_up", "resource"}
                    and source["content_kind"] in {"proposal", "constraint", "correction", "decision"}
                    and SCOPE_RE.search(text)
                    and event.get("timestamp", 0) - prior.get("timestamp", 0) <= 45
                    and (shared_entities or local_reply or similarity >= .18 or
                         (len(local_scope_targets) == 1 and DATA_RESULT_RE.search(text) and DATA_RESULT_RE.search(target["statement"])))):
                add("revises_scope", source["proposition_id"], target["proposition_id"], source["evidence_ids"] + target["evidence_ids"], .86, "bounded_cross_kind_scope")
                continue
            if similarity < .18 and not shared_entities:
                continue
            if event["speech_act"] == "correct": add("corrects", source["proposition_id"], target["proposition_id"], source["evidence_ids"], .92, "bounded_semantic_candidate")
            elif source["polarity"] != target["polarity"]: add("contradicts", source["proposition_id"], target["proposition_id"], source["evidence_ids"] + target["evidence_ids"], .82, "polarity_conflict")
            elif CONDITION_RE.search(text): add("condition_for", source["proposition_id"], target["proposition_id"], source["evidence_ids"], .78, "condition_marker")
            elif CAUSE_RE.search(text): add("explains", source["proposition_id"], target["proposition_id"], source["evidence_ids"], .75, "causal_marker")
            elif source["content_kind"] in {"hypothesis", "experimental_result"} and target["content_kind"] in {"trading_rule", "system_rule", "design_choice"}: add("tests", source["proposition_id"], target["proposition_id"], source["evidence_ids"], .74, "experiment_candidate")
            break
    return relations


def conflict_sets(propositions, relations):
    by_id = {x["proposition_id"]: x for x in propositions}
    result = []
    for relation in relations:
        if relation["type"] != "contradicts":
            continue
        source, target = by_id[relation["source_proposition_id"]], by_id[relation["target_proposition_id"]]
        resolved = any(x["type"] in {"corrects", "supersedes"} and x["target_proposition_id"] in {source["proposition_id"], target["proposition_id"]} for x in relations)
        result.append({"conflict_id": "CF" + relation["relation_id"][1:], "members": [source["proposition_id"], target["proposition_id"]], "conflict_type": "numeric" if source["quantities"] or target["quantities"] else "polarity", "resolution": "resolved" if resolved else "unresolved", "evidence_ids": relation["evidence_ids"]})
    return result
