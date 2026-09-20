"""Authoritative MeetingGraph built directly from semantic records."""
from __future__ import annotations
import hashlib
import json
import re
from .episodes import build_episodes, build_threads
from .bundles import build_dialogue_bundles
from .entities import EntityRegistry
from .propositions import epistemic_modality, proposition_from_record
from .relation_resolver import conflict_sets, resolve_relations
from .reducers import reduce_decisions, reduce_questions, reduce_rules_and_experiments, reduce_tasks

ACT_MAP = {"asserted": "assert", "assert": "assert", "ask": "ask", "answer": "answer", "propose": "propose", "proposal": "propose", "accept": "accept", "reject": "reject", "commit": "commit", "correct": "correct", "decide": "decide", "defer": "defer"}
SHORT_ACK_RE = re.compile(r"(?iu)^\s*(?:(?:да[\s,!.—-]*)+|ага|угу|ок(?:ей)?|согласен|делаем|подтверждаю|нет|неа|не\s+согласен)\s*(?:$|\b)")
INTERNAL_ACTION_RE = re.compile(r"(?iu)^@?[\w.-]+\s+[a-z][a-z0-9]*(?:_[a-z0-9]+)+(?:\s+.*)?$")


def _expand_action_records(records):
    """Make every predicate independently reducible and keep source lineage."""
    expanded = []
    for record in records:
        actions = [item for item in record.get("actions", []) if isinstance(item, dict) and item.get("predicate")]
        if not actions:
            expanded.append(record)
            continue
        if record.get("kind") not in {"action", "follow_up", "resource"}:
            # Proposition truth and action frames are orthogonal views.  A
            # proposal/decision remains present while its actionable clauses
            # become independently reducible children.
            expanded.append(record)
        origin = record.get("origin_id") or record.get("record_id")
        for index, action in enumerate(actions, 1):
            evidence = list(action.get("evidence_ids", [])) or list(record.get("evidence_ids", []))
            actor = action.get("actor")
            temporal = action.get("temporal_state", "unknown")
            commitment = action.get("commitment_state", "unknown")
            child = dict(record)
            turns = sorted(record.get("dialogue_evidence", []), key=lambda item: float(item.get("start", 0)))
            direct = [turn for turn in turns if turn.get("id") in set(evidence)]
            reference_surface = " ".join(
                str(value or "") for value in (
                    record.get("statement"), action.get("object"),
                    *(turn.get("text") for turn in direct),
                )
            )
            if len(actions) == 1 and re.search(
                r"(?iu)\b(?:это|этот|эта|эти|его|е[её]|их|тебе|вам)\b",
                reference_surface,
            ):
                if direct:
                    first = min(float(turn.get("start", 0)) for turn in direct)
                    direct_speakers = {turn.get("speaker") for turn in direct if turn.get("speaker")}
                    antecedents = [
                        turn for turn in turns
                        if turn.get("id") not in evidence
                        and 0 <= first - float(turn.get("start", 0)) <= 20
                        and turn.get("speaker") in direct_speakers
                        and len(str(turn.get("text") or "").split()) >= 4
                    ]
                    if antecedents:
                        antecedent = antecedents[-1]
                        evidence = list(dict.fromkeys([antecedent.get("id")] + evidence))
                        child["source_word_ids"] = list(dict.fromkeys(
                            list(child.get("source_word_ids", []))
                            + list(antecedent.get("source_word_ids", []))
                        ))
                        child["reference_resolution"] = {
                            "status": "resolved_from_adjacent_same_speaker_turn",
                            "antecedent_evidence_ids": [antecedent.get("id")],
                        }
            action_statement = " ".join(str(value).strip() for value in (actor, action.get("predicate"), action.get("object")) if value).strip()
            # Machine predicate identifiers are useful for identity, but are
            # not a human proposition and fail the language-aware task guard.
            # Keep the atomic fields while reducing the source statement.
            # A one-action semantic record already has an independently
            # reviewed, reader-facing statement.  Rebuilding it from the
            # model's predicate/object fields can reintroduce the raw spoken
            # turn (hesitations, second-person deixis and several sentences)
            # after that wording has been cleaned.  Atomic reconstruction is
            # needed only when several actions must be separated.
            public_statement = (
                record.get("statement")
                if len(actions) == 1 or INTERNAL_ACTION_RE.fullmatch(action_statement or "")
                else action_statement or record.get("statement")
            )
            child.update({
                "record_id": f"{record.get('record_id')}:A{index:02d}",
                "source_fact_id": record.get("source_fact_id") or record.get("record_id"),
                "origin_id": f"{origin}:{action.get('action_id') or f'A{index:02d}'}", "parent_record_id": record.get("record_id"),
                "action_id": action.get("action_id") or f"A{index:02d}",
                "kind": "action", "content_kind": "action",
                "subject": actor, "predicate": action.get("predicate"),
                "object": action.get("object"), "recipient": action.get("recipient"),
                "parallel": bool(action.get("parallel")),
                "statement": public_statement,
                "source_statement": record.get("statement"),
                "grammatical_actor": actor, "assignees": [actor] if actor else [],
                "temporal_state": temporal, "commitment_state": commitment,
                "commitment_strength": "explicit" if commitment == "explicit_commitment" else "implicit" if commitment == "intent_to_attempt" else "none",
                "commitment_actor": actor if commitment in {"explicit_commitment", "intent_to_attempt"} else None,
                "evidence_ids": evidence, "field_evidence": action.get("field_evidence", {}),
                "actions": [action],
            })
            expanded.append(child)
    return expanded


def _expand_embedded_replies(records):
    """Promote only immediate adjacency-pair replies hidden in a wide fact.

    Some extractors keep a proposal and its one-word reply in one semantic
    record.  Turning the adjacent reply into its own event lets the normal
    relation resolver bind it.  An intervening question prevents the binding.
    """
    expanded = list(records)
    # ``_expand_action_records`` keeps a proposal/decision parent alongside
    # its atomic action children.  An acknowledgement belongs to the work
    # clauses, not to the broad parent summary.  Let the first atomic child
    # claim it; the task reducer deliberately shares the resulting relation
    # with its same-parent siblings.
    expanded_action_parents = {
        record.get("parent_record_id")
        for record in records
        if record.get("parent_record_id")
    }
    covered_evidence = {evidence for record in records for evidence in record.get("evidence_ids", []) if record.get("record_id", "").find(":ACK:") >= 0}
    for record in records:
        if record.get("kind") not in {"proposal", "decision", "action", "follow_up", "question"}:
            continue
        if record.get("record_id") in expanded_action_parents:
            continue
        direct_evidence = set(record.get("evidence_ids", []))
        direct_evidence.update(
            evidence_id
            for action in record.get("actions", []) if isinstance(action, dict)
            for evidence_id in action.get("evidence_ids", [])
        )
        turns = sorted(record.get("dialogue_evidence", []), key=lambda item: float(item.get("start", 0)))
        for prior, reply in zip(turns, turns[1:]):
            reply_id = reply.get("id")
            if not reply_id or reply_id in covered_evidence or not SHORT_ACK_RE.search(str(reply.get("text") or "")):
                continue
            # ``dialogue_evidence`` is a context window, not proof that every
            # turn belongs to this record.  Bind an acknowledgement only when
            # the immediately preceding utterance is direct evidence for the
            # record/action.  Otherwise an earlier broad observation can steal
            # a later "да" from the actual two-party proposal.
            if prior.get("id") not in direct_evidence:
                continue
            if not prior.get("speaker") or not reply.get("speaker") or prior.get("speaker") == reply.get("speaker"):
                continue
            if float(reply.get("start", 0)) - float(prior.get("start", 0)) > 45:
                continue
            expected_speaker = set(record.get("attributed_speakers", []))
            if expected_speaker and prior.get("speaker") not in expected_speaker:
                continue
            is_rejection = bool(re.match(r"(?iu)^\s*(?:нет|неа|не\s+согласен)", str(reply.get("text") or "")))
            synthetic = {
                "record_id": f"{record.get('record_id')}:ACK:{reply_id}",
                "source_fact_id": record.get("source_fact_id") or record.get("record_id"),
                "origin_id": record.get("origin_id") or record.get("record_id"),
                "kind": "observation", "content_kind": "observation",
                "statement": str(reply.get("text") or "").strip(),
                "speech_act": "reject" if is_rejection else "answer" if record.get("kind") == "question" else "accept",
                "modality": "asserted", "evidence_ids": [reply_id],
                "source_word_ids": list(reply.get("source_word_ids", [])),
                "attributed_speakers": [reply.get("speaker")],
                "start": float(reply.get("start", 0)), "end": float(reply.get("end", reply.get("start", 0))),
                "dialogue_evidence": [reply], "verification_status": record.get("verification_status", "supported"),
                "synthetic_from_adjacency": True,
            }
            if not is_rejection and record.get("kind") != "question":
                # Resolve this exact adjacency pair before any separately
                # extracted fact at the same timestamp can steal the reply.
                synthetic["accepts_record_ids"] = [record.get("record_id")]
            expanded.append(synthetic)
            covered_evidence.add(reply_id)
    return expanded


def _event(record, proposition):
    raw_act = str(record.get("speech_act") or "assert").casefold()
    act = ACT_MAP.get(raw_act, "ask" if record.get("kind") == "question" else "propose" if record.get("kind") in {"proposal", "hypothesis"} else "commit" if record.get("kind") == "action" and record.get("modality") == "committed" else "decide" if record.get("kind") == "decision" else "assert")
    evidence = list(record.get("evidence_ids", []))
    semantic_event = {
        "record_id": record.get("record_id"), "proposition_id": proposition["proposition_id"],
        "speech_act": act, "evidence_ids": evidence,
        "speaker_refs": list(record.get("attributed_speakers", [])),
        "actor": record.get("grammatical_actor") or record.get("commitment_actor"),
        "recipient": record.get("recipient") or record.get("beneficiary"),
        "reporter": record.get("reporter"), "modality": epistemic_modality(record),
        "verification_status": record.get("verification_status", "supported"),
    }
    raw = json.dumps(semantic_event, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    speakers = list(record.get("attributed_speakers", []))
    confidence = record.get("confidence") or {}
    return {"event_id": "EV" + hashlib.sha256(raw.encode()).hexdigest()[:16], "proposition_id": proposition["proposition_id"], "source_record_id": record.get("record_id"), "source_fact_id": record.get("source_fact_id") or record.get("record_id"), "statement": record.get("statement") or "", "speaker": speakers[0] if len(speakers) == 1 else None, "speaker_candidates": speakers, "reporter": record.get("reporter") or (speakers[0] if len(speakers) == 1 else None), "grammatical_actor": record.get("grammatical_actor") or record.get("commitment_actor"), "recipient": record.get("recipient") or record.get("beneficiary"), "speech_act": act, "epistemic_modality": epistemic_modality(record), "verification_status": record.get("verification_status", "supported"), "timestamp": float(record.get("start", 0)), "primary_evidence_start": float(record.get("primary_evidence_start", record.get("start", 0))), "end": float(record.get("end", record.get("start", 0))), "evidence_ids": evidence, "context_ids": list(record.get("context_ids", [])), "dialogue_evidence": list(record.get("dialogue_evidence", [])), "source_word_ids": list(record.get("source_word_ids", [])), "thread_hint": record.get("thread_id") or record.get("episode_id"), "confidence": {"recognition": confidence.get("recognition", record.get("asr_confidence")), "speaker": confidence.get("speaker_identity", record.get("speaker_confidence")), "number": confidence.get("quantity", record.get("number_confidence")), "semantic": confidence.get("semantic_support", record.get("confidence_score"))}, "reference_resolution": record.get("reference_resolution")}


def build_meeting_graph(records, provenance=None, meeting_id=None):
    """No legacy state is accepted: records/evidence are the only semantic input."""
    records = sorted(_expand_embedded_replies(_expand_action_records(records)), key=lambda x: (float(x.get("start", 0)), x.get("record_id", "")))
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
    decisions = reduce_decisions(propositions, events, relations, records)
    tasks = reduce_tasks(propositions, events, relations, records)
    questions = reduce_questions(propositions, events, relations, records)
    rules, experiments = reduce_rules_and_experiments(propositions, events, decisions, relations)
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
        task_status = task_by_prop.get(prop["proposition_id"], {}).get("status")
        lifecycle = "superseded" if social == "superseded" or prop["proposition_id"] in superseded_props else "rejected" if social == "rejected" or prop["proposition_id"] in rejected_props or task_status == "cancelled" else "active"
        confidences = [x.get("confidence", {}) for x in prop_events]
        def risk(name, default=0.0):
            values = [x.get(name) for x in confidences if isinstance(x.get(name), (int, float))]
            return 1.0 - min(values) if values else default
        verification_states = {x.get("verification_status", "supported") for x in prop_events}
        modalities = {x.get("epistemic_modality", "unknown") for x in prop_events}
        source_fact_ids = list(dict.fromkeys(x.get("source_fact_id") or x.get("source_record_id") for x in prop_events))
        claim = {
            "claim_id": "C" + prop["proposition_id"][1:],
            "proposition_id": prop["proposition_id"],
            "event_ids": [x["event_id"] for x in prop_events],
            "kind": prop["claim_kind"], "content_kind": prop["content_kind"],
            "speech_act": event.get("speech_act", "assert"), "statement": prop["statement"],
            "semantic_signature": prop["semantic_signature"], "entities": prop["entities"],
            "quantities": prop["quantities"], "conditions": prop["conditions"],
            "polarity": prop["polarity"], "modality": event.get("epistemic_modality", "unknown"),
            "epistemic_modality": event.get("epistemic_modality", "unknown"),
            "social_state": social, "lifecycle": lifecycle, "evidence_ids": prop["evidence_ids"],
            "source_word_ids": list(dict.fromkeys(w for x in prop_events for w in x.get("source_word_ids", []))),
            "speaker_refs": sorted({s for x in prop_events for s in x.get("speaker_candidates", [])}),
            "source_record_id": source_fact_ids[0], "source_record_ids": source_fact_ids,
            "semantic_record_ids": prop["source_record_ids"],
            "start": min([x.get("timestamp", 0) for x in prop_events] or [0]),
            "end": max([x.get("end", x.get("timestamp", 0)) for x in prop_events] or [0]),
            "risk": {"recognition": risk("recognition", .5), "speaker": risk("speaker", .5),
                     "number": risk("number", .5 if prop["quantities"] else 0),
                     "modality": risk("semantic", .3)},
        }
        claim["primary_evidence_start"] = min([x.get("primary_evidence_start", x.get("timestamp", 0)) for x in prop_events] or [0])
        claim["verification_status"] = next(iter(verification_states)) if len(verification_states) == 1 else "insufficient_evidence"
        claim["epistemic_modality"] = claim["modality"] = next(iter(modalities)) if len(modalities) == 1 else "unknown"
        claim["field_support"] = {
            "statement": [{"event_id": x["event_id"], "status": x.get("verification_status", "supported"), "evidence_ids": x.get("evidence_ids", [])} for x in prop_events],
            "modality": [{"event_id": x["event_id"], "value": x.get("epistemic_modality", "unknown"), "evidence_ids": x.get("evidence_ids", [])} for x in prop_events],
        }
        source_records = [record for record in records if record.get("record_id") in prop.get("source_record_ids", [])]
        for field in ("actor", "predicate", "object", "recipient"):
            support = [
                {"record_id": record.get("record_id"), "evidence_ids": list(record.get("field_evidence", {}).get(field, []))}
                for record in source_records if record.get("field_evidence", {}).get(field)
            ]
            if support:
                claim["field_support"][field] = support
        claim["origin_ids"] = list(dict.fromkeys(record.get("origin_id") or record.get("record_id") for record in source_records))
        claim["topic_entities"] = list(dict.fromkeys(
            value for value in (
                *[entity.get("canonical_name") for entity in prop.get("entities", []) if isinstance(entity, dict)],
                *[record.get("subject") for record in source_records],
                *[record.get("object") for record in source_records],
                *[record.get("topic") for record in source_records],
            )
            if value and str(value).strip().casefold() not in {"прочее", "тема встречи", "unknown"}
            and len(str(value).strip()) <= 100
        ))
        claim["temporal_state"] = next((record.get("temporal_state") for record in reversed(source_records) if record.get("temporal_state")), "unknown")
        claim["commitment_state"] = next((record.get("commitment_state") for record in reversed(source_records) if record.get("commitment_state")), "unknown")
        claim["time_contract"] = next((record.get("time_contract") for record in reversed(source_records) if record.get("time_contract")), {})
        claim["unresolved_terms"] = list(dict.fromkeys(term for record in source_records for term in record.get("unresolved_terms", [])))
        claim["term_resolution_status"] = "requires_clarification" if claim["unresolved_terms"] else "resolved_or_not_applicable"
        claim["aspect_risks"] = {
            field: {
                "evidence_ids": list(dict.fromkeys(evidence for item in support for evidence in item.get("evidence_ids", []))),
                "risk": claim["risk"],
                "source": "field_evidence",
            }
            for field, support in claim["field_support"].items()
        }
        claim["dialogue_evidence"] = [item for x in prop_events for item in x.get("dialogue_evidence", [])]
        claim["context_ids"] = list(dict.fromkeys(value for x in prop_events for value in x.get("context_ids", [])))
        claim["protected_outcome"] = any(bool(record.get("protected_outcome")) for record in source_records)
        claim["parallel"] = any(bool(record.get("parallel")) for record in source_records)
        if prop["proposition_id"] in decision_by_prop:
            decision = decision_by_prop[prop["proposition_id"]]
            claim.update(
                decision_status=decision["status"],
                decision_evidence_ids=decision.get("decision_evidence_ids", []),
                acceptance_check=decision.get("acceptance_check"),
                acceptance_relation_ids=decision.get("acceptance_relation_ids", []),
                acceptance_evidence_ids=decision.get("acceptance_evidence_ids", []),
                accepted_by=decision.get("accepted_by", []),
            )
        if prop["proposition_id"] in task_by_prop:
            task = task_by_prop[prop["proposition_id"]]
            claim.update(task_status=task["status"], assignee=task.get("assignee"), automation_eligible=task.get("automation_eligible"), automation_status=task.get("automation_status", "unknown"), scope_state=task.get("scope_state"), time_scope=task.get("current_scope"), canonical_task_state_id=task["task_id"], canonical_task_anchor=prop["proposition_id"] == task["proposition_id"], commitment_strength=task.get("commitment_strength"), commitment_actor=task.get("commitment_actor"), due_raw=task.get("due_raw"), due_normalized=task.get("due_normalized"), due_resolution_status=task.get("due_resolution_status"))
            if task.get("status") == "self_committed" and task.get("source_commitment_evidence_ids"):
                claim["verification_status"] = "supported"
                claim["evidence_ids"] = list(dict.fromkeys(claim["evidence_ids"] + task["source_commitment_evidence_ids"]))
        if prop["proposition_id"] in question_by_prop:
            question = question_by_prop[prop["proposition_id"]]
            claim.update(question_status=question["status"], question_slots=question.get("missing_slots", []))
        generic_reply = bool(re.fullmatch(r"(?iu)\s*(?:(?:да[\s,!.—-]*)+|(?:нет|неа|ага|угу|ну\s+ладно)[\s,!.—-]*)", str(claim.get("statement") or "")))
        claim["dialogue_only"] = generic_reply and bool(prop_events) and all(x.get("speech_act") in {"accept", "reject", "answer", "assert"} for x in prop_events)
        claims.append(claim)
    claim_relations = [{**x, "source_claim_id": "C" + x["source_proposition_id"][1:], "target_claim_id": "C" + x["target_proposition_id"][1:]} for x in relations]
    episodes = build_episodes([claim for claim in claims if not claim.get("dialogue_only")])
    threads = build_threads(episodes, claims, claim_relations)
    dialogue_bundles = build_dialogue_bundles(episodes, threads, events, claims, claim_relations)
    stable_recording = (provenance or {}).get("recording_id") or (provenance or {}).get("audio_sha256")
    identity_payload = stable_recording or json.dumps([[x["proposition_id"], x["evidence_ids"]] for x in propositions], sort_keys=True)
    identity = meeting_id or "MG" + hashlib.sha256(str(identity_payload).encode()).hexdigest()[:16]
    generation_payload = json.dumps({
        "events": [{key: item.get(key) for key in ("event_id", "proposition_id", "speech_act", "epistemic_modality", "verification_status", "speaker", "grammatical_actor", "recipient", "evidence_ids")} for item in events],
        "relations": [{key: item.get(key) for key in ("relation_id", "type", "source_claim_id", "target_claim_id", "evidence_ids")} for item in claim_relations],
        "states": [{key: item.get(key) for key in ("claim_id", "lifecycle", "social_state", "modality", "verification_status")} for item in claims],
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    generation_id = (provenance or {}).get("generation_id") or "GEN" + hashlib.sha256((identity + generation_payload).encode()).hexdigest()[:16]
    graph = {"schema": "MeetingGraphSchema", "schema_version": 7, "meeting_id": identity, "recording_id": stable_recording, "generation_id": generation_id, "authoritative": True, "provenance": provenance or {}, "propositions": propositions, "dialogue_events": events, "events_by_proposition": events_by_prop, "relations": claim_relations, "decision_states": decisions, "task_states": tasks, "question_states": questions, "rule_states": rules, "experiment_states": experiments, "conflict_sets": conflict_sets(propositions, relations), "episodes": episodes, "threads": threads, "claims": claims, "tasks": tasks, "questions": questions, "decisions": decisions, "active_rules": rules, "experimental_results": experiments, "open_threads": [x for x in threads if x["state"] == "open"], "uncertainty": {"abstentions": [], "conflicts": sum(x["resolution"] == "unresolved" for x in conflict_sets(propositions, relations))}}
    graph["schema_version"] = 7
    graph["dialogue_bundles"] = dialogue_bundles
    graph["entity_registry"] = registry.snapshot()
    return graph


def compatibility_state(graph):
    """Read-only adapter for legacy renderers; never used to derive graph state."""
    events = [{
        "event_id": x["event_id"], "claim_id": "C" + x["proposition_id"][1:],
        "source_record_id": x.get("source_fact_id") or x["source_record_id"],
        "semantic_record_id": x["source_record_id"], "act": x["speech_act"],
        "content_kind": next(c["content_kind"] for c in graph["claims"] if c["proposition_id"] == x["proposition_id"]),
        "speech_act": x["speech_act"], "modality": x["epistemic_modality"],
        "presentation": next(c["statement"] for c in graph["claims"] if c["proposition_id"] == x["proposition_id"]),
        "evidence_ids": x["evidence_ids"], "speaker_ids": x["speaker_candidates"],
        "start": x["timestamp"], "end": x.get("end", x["timestamp"]),
        "lifecycle": next(c["lifecycle"] for c in graph["claims"] if c["proposition_id"] == x["proposition_id"]),
        "provenance": {"audio_sha256": graph.get("provenance", {}).get("audio_sha256"),
                       "source_word_ids": x.get("source_word_ids", [])},
    } for x in graph["dialogue_events"]]
    active = [x for x in events if x["lifecycle"] == "active"]
    return {"schema": "LegacyMeetingStateAdapter", "schema_version": 3, "state_id": graph["meeting_id"], "authoritative_source": "meeting_graph.json", "events": events, "relations": [{"relation_id": x["relation_id"], "relation": x["type"], "source_event": next((e["event_id"] for e in events if e["claim_id"] == x["source_claim_id"]), None), "target_event": next((e["event_id"] for e in events if e["claim_id"] == x["target_claim_id"]), None), "evidence_ids": x["evidence_ids"]} for x in graph["relations"]], "active_event_ids": [x["event_id"] for x in active], "views": {"summary": active, "timeline": active, "decisions": graph["decision_states"], "tasks": graph["task_states"], "atomic_tasks": graph["task_states"], "questions": graph["question_states"]}}
