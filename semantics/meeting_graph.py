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
    confidence = record.get("confidence") or {}
    return {"event_id": "EV" + hashlib.sha256(raw.encode()).hexdigest()[:16], "proposition_id": proposition["proposition_id"], "source_record_id": record.get("record_id"), "speaker": speakers[0] if len(speakers) == 1 else None, "speaker_candidates": speakers, "speech_act": act, "epistemic_modality": epistemic_modality(record), "timestamp": float(record.get("start", 0)), "primary_evidence_start": float(record.get("primary_evidence_start", record.get("start", 0))), "end": float(record.get("end", record.get("start", 0))), "evidence_ids": evidence, "dialogue_evidence": list(record.get("dialogue_evidence", [])), "source_word_ids": list(record.get("source_word_ids", [])), "thread_hint": record.get("thread_id") or record.get("episode_id"), "confidence": {"recognition": confidence.get("recognition", record.get("asr_confidence")), "speaker": confidence.get("speaker_identity", record.get("speaker_confidence")), "number": confidence.get("quantity", record.get("number_confidence")), "semantic": confidence.get("semantic_support", record.get("confidence_score"))}, "reference_resolution": record.get("reference_resolution")}


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
    events_by_prop = {}
    for event in events:
        events_by_prop.setdefault(event["proposition_id"], []).append(event)
    decision_by_prop = {x["proposition_id"]: x for x in decisions}
    task_by_prop = {prop_id: task for task in tasks for prop_id in task.get("source_proposition_ids", [task["proposition_id"]])}
    question_by_prop = {x["proposition_id"]: x for x in questions}
    superseded_props = {x["target_proposition_id"] for x in relations if x.get("type") in {"corrects", "supersedes"}}
    rejected_props = {x["target_proposition_id"] for x in relations if x.get("type") == "rejects"}
    claims = []
    for prop in propositions:
        prop_events = sorted(events_by_prop.get(prop["proposition_id"], []), key=lambda x: (x.get("timestamp", 0), x.get("event_id", "")))
        event = prop_events[-1] if prop_events else {}
        social = decision_by_prop.get(prop["proposition_id"], {}).get("status", "candidate")
        lifecycle = "superseded" if social == "superseded" or prop["proposition_id"] in superseded_props else "rejected" if social == "rejected" or prop["proposition_id"] in rejected_props else "active"
        confidences = [x.get("confidence", {}) for x in prop_events]
        def risk(name, default=0.0):
            values = [x.get(name) for x in confidences if isinstance(x.get(name), (int, float))]
            return 1.0 - max(values) if values else default
        claim = {"claim_id": "C" + prop["proposition_id"][1:], "proposition_id": prop["proposition_id"], "event_ids": [x["event_id"] for x in prop_events], "kind": prop["claim_kind"], "content_kind": prop["content_kind"], "speech_act": event.get("speech_act", "assert"), "statement": prop["statement"], "semantic_signature": prop["semantic_signature"], "entities": prop["entities"], "quantities": prop["quantities"], "conditions": prop["conditions"], "polarity": prop["polarity"], "modality": event.get("epistemic_modality", "unknown"), "epistemic_modality": event.get("epistemic_modality", "unknown"), "social_state": social, "lifecycle": lifecycle, "evidence_ids": prop["evidence_ids"], "source_word_ids": list(dict.fromkeys(w for x in prop_events for w in x.get("source_word_ids", []))), "speaker_refs": sorted({s for x in prop_events for s in x.get("speaker_candidates", [])}), "source_record_id": prop["source_record_ids"][0], "source_record_ids": prop["source_record_ids"], "start": min([x.get("timestamp", 0) for x in prop_events] or [0]), "end": max([x.get("end", x.get("timestamp", 0)) for x in prop_events] or [0]), "risk": {"recognition": risk("recognition", .5), "speaker": risk("speaker", .5), "number": risk("number", .5 if prop["quantities"] else 0), "modality": risk("semantic", .3)}}
        claim["primary_evidence_start"] = min([x.get("primary_evidence_start", x.get("timestamp", 0)) for x in prop_events] or [0])
        claim["dialogue_evidence"] = [item for x in prop_events for item in x.get("dialogue_evidence", [])]
        if prop["proposition_id"] in decision_by_prop:
            decision = decision_by_prop[prop["proposition_id"]]
            claim.update(decision_status=decision["status"], decision_evidence_ids=decision.get("decision_evidence_ids", []), acceptance_check=decision.get("acceptance_check"))
        if prop["proposition_id"] in task_by_prop:
            task = task_by_prop[prop["proposition_id"]]
            claim.update(task_status=task["status"], assignee=task.get("assignee"), automation_eligible=task.get("automation_eligible"), scope_state=task.get("scope_state"), time_scope=task.get("current_scope"), canonical_task_state_id=task["task_id"], canonical_task_anchor=prop["proposition_id"] == task["proposition_id"], commitment_strength=task.get("commitment_strength"), commitment_actor=task.get("commitment_actor"))
        if prop["proposition_id"] in question_by_prop:
            question = question_by_prop[prop["proposition_id"]]
            claim.update(question_status=question["status"], question_slots=question.get("missing_slots", []))
        claims.append(claim)
    claim_relations = [{**x, "source_claim_id": "C" + x["source_proposition_id"][1:], "target_claim_id": "C" + x["target_proposition_id"][1:]} for x in relations]
    episodes = build_episodes(claims)
    threads = build_threads(episodes, claims, claim_relations)
    identity = meeting_id or "MG" + hashlib.sha256(json.dumps([[x["proposition_id"], x["evidence_ids"]] for x in propositions], sort_keys=True).encode()).hexdigest()[:16]
    graph = {"schema": "MeetingGraphSchema", "schema_version": 5, "meeting_id": identity, "authoritative": True, "provenance": provenance or {}, "propositions": propositions, "dialogue_events": events, "events_by_proposition": events_by_prop, "relations": claim_relations, "decision_states": decisions, "task_states": tasks, "question_states": questions, "rule_states": rules, "experiment_states": experiments, "conflict_sets": conflict_sets(propositions, relations), "episodes": episodes, "threads": threads, "claims": claims, "tasks": tasks, "questions": questions, "decisions": decisions, "active_rules": rules, "experimental_results": experiments, "open_threads": [x for x in threads if x["state"] == "open"], "uncertainty": {"abstentions": [], "conflicts": sum(x["resolution"] == "unresolved" for x in conflict_sets(propositions, relations))}}
    graph["entity_registry"] = registry.snapshot()
    return graph


def compatibility_state(graph):
    """Read-only adapter for legacy renderers; never used to derive graph state."""
    events = [{"event_id": x["event_id"], "claim_id": "C" + x["proposition_id"][1:], "source_record_id": x["source_record_id"], "act": x["speech_act"], "content_kind": next(c["content_kind"] for c in graph["claims"] if c["proposition_id"] == x["proposition_id"]), "speech_act": x["speech_act"], "modality": x["epistemic_modality"], "presentation": next(c["statement"] for c in graph["claims"] if c["proposition_id"] == x["proposition_id"]), "evidence_ids": x["evidence_ids"], "speaker_ids": x["speaker_candidates"], "start": x["timestamp"], "end": x.get("end", x["timestamp"]), "lifecycle": next(c["lifecycle"] for c in graph["claims"] if c["proposition_id"] == x["proposition_id"]), "provenance": {"audio_sha256": graph.get("provenance", {}).get("audio_sha256"), "source_word_ids": x.get("source_word_ids", [])}} for x in graph["dialogue_events"]]
    active = [x for x in events if x["lifecycle"] == "active"]
    return {"schema": "LegacyMeetingStateAdapter", "schema_version": 3, "state_id": graph["meeting_id"], "authoritative_source": "meeting_graph.json", "events": events, "relations": [{"relation_id": x["relation_id"], "relation": x["type"], "source_event": next((e["event_id"] for e in events if e["claim_id"] == x["source_claim_id"]), None), "target_event": next((e["event_id"] for e in events if e["claim_id"] == x["target_claim_id"]), None), "evidence_ids": x["evidence_ids"]} for x in graph["relations"]], "active_event_ids": [x["event_id"] for x in active], "views": {"summary": active, "timeline": active, "decisions": graph["decision_states"], "tasks": graph["task_states"], "atomic_tasks": graph["task_states"], "questions": graph["question_states"]}}
