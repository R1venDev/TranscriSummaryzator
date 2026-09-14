"""Authoritative MeetingGraph built directly from semantic records."""
from __future__ import annotations
import hashlib
import json
from .episodes import build_episodes, build_threads
from .entities import EntityRegistry
from .propositions import epistemic_modality, proposition_from_record
from .relation_resolver import conflict_sets, resolve_relations
from .reducers import reduce_decisions, reduce_questions, reduce_rules_and_experiments, reduce_tasks

ACT_MAP = {"asserted": "assert", "assert": "assert", "ask": "ask", "answer": "answer", "propose": "propose", "proposal": "propose", "accept": "accept", "reject": "reject", "commit": "commit", "correct": "correct", "decide": "decide", "defer": "defer"}


def _event(record, proposition):
    raw_act = str(record.get("speech_act") or "assert").casefold()
    act = ACT_MAP.get(raw_act, "ask" if record.get("kind") == "question" else "propose" if record.get("kind") in {"proposal", "hypothesis"} else "commit" if record.get("kind") == "action" and record.get("modality") == "committed" else "decide" if record.get("kind") == "decision" else "assert")
    evidence = list(record.get("evidence_ids", []))
    raw = f"{record.get('record_id')}|{proposition['proposition_id']}|{act}|{'|'.join(evidence)}"
    speakers = list(record.get("attributed_speakers", []))
    return {"event_id": "EV" + hashlib.sha256(raw.encode()).hexdigest()[:16], "proposition_id": proposition["proposition_id"], "source_record_id": record.get("record_id"), "speaker": speakers[0] if len(speakers) == 1 else None, "speaker_candidates": speakers, "speech_act": act, "epistemic_modality": epistemic_modality(record), "timestamp": float(record.get("start", 0)), "evidence_ids": evidence, "reference_resolution": record.get("reference_resolution")}


def build_meeting_graph(records, provenance=None, meeting_id=None):
    """No legacy state is accepted: records/evidence are the only semantic input."""
    records = sorted(records, key=lambda x: (float(x.get("start", 0)), x.get("record_id", "")))
    registry = EntityRegistry((provenance or {}).get("entity_vocabulary", []))
    for record in records:
        for raw in record.get("entities", []):
            if isinstance(raw, str):
                registry.register(raw)
            else:
                registry.register(raw.get("canonical_name") or raw.get("name") or raw.get("text"), raw.get("type", "unknown"), raw.get("aliases", []), raw.get("entity_id"))
    proposition_map, record_to_prop = {}, {}
    for record in records:
        prop = proposition_from_record(record, registry)
        existing = proposition_map.get(prop["proposition_id"])
        if existing:
            existing["evidence_ids"] = list(dict.fromkeys(existing["evidence_ids"] + prop["evidence_ids"]))
            existing["source_record_ids"].append(record.get("record_id"))
            prop = existing
        else:
            proposition_map[prop["proposition_id"]] = prop
        record_to_prop[record.get("record_id")] = prop
    propositions = list(proposition_map.values())
    events = [_event(record, record_to_prop[record.get("record_id")]) for record in records]
    relations = resolve_relations(propositions, events, records)
    decisions = reduce_decisions(propositions, events, relations)
    tasks = reduce_tasks(propositions, events, relations, records)
    questions = reduce_questions(propositions, events, relations, records)
    rules, experiments = reduce_rules_and_experiments(propositions, events, decisions)
    event_by_prop = {x["proposition_id"]: x for x in events}
    decision_by_prop = {x["proposition_id"]: x for x in decisions}
    task_by_prop = {x["proposition_id"]: x for x in tasks}
    question_by_prop = {x["proposition_id"]: x for x in questions}
    claims = []
    for prop in propositions:
        event = event_by_prop.get(prop["proposition_id"], {})
        social = decision_by_prop.get(prop["proposition_id"], {}).get("status", "candidate")
        lifecycle = "superseded" if social == "superseded" else "rejected" if social == "rejected" else "active"
        claim = {"claim_id": "C" + prop["proposition_id"][1:], "proposition_id": prop["proposition_id"], "kind": prop["content_kind"], "content_kind": prop["content_kind"], "statement": prop["statement"], "semantic_signature": prop["semantic_signature"], "entities": prop["entities"], "quantities": prop["quantities"], "conditions": prop["conditions"], "polarity": prop["polarity"], "modality": event.get("epistemic_modality", "unknown"), "social_state": social, "lifecycle": lifecycle, "evidence_ids": prop["evidence_ids"], "speaker_refs": event.get("speaker_candidates", []), "source_record_id": prop["source_record_ids"][0], "source_record_ids": prop["source_record_ids"], "start": event.get("timestamp", 0), "end": event.get("timestamp", 0), "risk": {"recognition": .5, "speaker": .5, "number": .5 if prop["quantities"] else 0, "modality": .3}}
        if prop["proposition_id"] in decision_by_prop: claim["decision_status"] = decision_by_prop[prop["proposition_id"]]["status"]
        if prop["proposition_id"] in task_by_prop: claim["task_status"] = task_by_prop[prop["proposition_id"]]["status"]
        if prop["proposition_id"] in question_by_prop: claim["question_status"] = question_by_prop[prop["proposition_id"]]["status"]
        claims.append(claim)
    claim_relations = [{**x, "source_claim_id": "C" + x["source_proposition_id"][1:], "target_claim_id": "C" + x["target_proposition_id"][1:]} for x in relations]
    episodes = build_episodes(claims)
    threads = build_threads(episodes, claims, claim_relations)
    identity = meeting_id or "MG" + hashlib.sha256(json.dumps([[x["proposition_id"], x["evidence_ids"]] for x in propositions], sort_keys=True).encode()).hexdigest()[:16]
    graph = {"schema": "MeetingGraphSchema", "schema_version": 3, "meeting_id": identity, "authoritative": True, "provenance": provenance or {}, "propositions": propositions, "dialogue_events": events, "relations": claim_relations, "decision_states": decisions, "task_states": tasks, "question_states": questions, "rule_states": rules, "experiment_states": experiments, "conflict_sets": conflict_sets(propositions, relations), "episodes": episodes, "threads": threads, "claims": claims, "tasks": tasks, "questions": questions, "decisions": decisions, "active_rules": rules, "experimental_results": experiments, "open_threads": [x for x in threads if x["state"] == "open"], "uncertainty": {"abstentions": [], "conflicts": sum(x["resolution"] == "unresolved" for x in conflict_sets(propositions, relations))}}
    graph["entity_registry"] = registry.snapshot()
    return graph


def compatibility_state(graph):
    """Read-only adapter for legacy renderers; never used to derive graph state."""
    events = [{"event_id": x["event_id"], "claim_id": "C" + x["proposition_id"][1:], "source_record_id": x["source_record_id"], "act": x["speech_act"], "content_kind": next(c["content_kind"] for c in graph["claims"] if c["proposition_id"] == x["proposition_id"]), "speech_act": x["speech_act"], "modality": x["epistemic_modality"], "presentation": next(c["statement"] for c in graph["claims"] if c["proposition_id"] == x["proposition_id"]), "evidence_ids": x["evidence_ids"], "speaker_ids": x["speaker_candidates"], "start": x["timestamp"], "lifecycle": next(c["lifecycle"] for c in graph["claims"] if c["proposition_id"] == x["proposition_id"]), "provenance": {"audio_sha256": graph.get("provenance", {}).get("audio_sha256"), "source_word_ids": x["evidence_ids"]}} for x in graph["dialogue_events"]]
    active = [x for x in events if x["lifecycle"] == "active"]
    return {"schema": "LegacyMeetingStateAdapter", "schema_version": 3, "state_id": graph["meeting_id"], "authoritative_source": "meeting_graph.json", "events": events, "relations": [{"relation_id": x["relation_id"], "relation": x["type"], "source_event": next((e["event_id"] for e in events if e["claim_id"] == x["source_claim_id"]), None), "target_event": next((e["event_id"] for e in events if e["claim_id"] == x["target_claim_id"]), None), "evidence_ids": x["evidence_ids"]} for x in graph["relations"]], "active_event_ids": [x["event_id"] for x in active], "views": {"summary": active, "timeline": active, "decisions": graph["decision_states"], "tasks": graph["task_states"], "atomic_tasks": graph["task_states"], "questions": graph["question_states"]}}
