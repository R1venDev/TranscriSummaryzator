"""Evidence-complete dialogue bundles used by reducers, editors and verifiers."""
from __future__ import annotations


def build_dialogue_bundles(episodes, threads, events, claims, relations):
    """Build connected, ordered source packets without rewriting source events."""
    event_by_id = {item["event_id"]: item for item in events}
    claim_by_id = {item["claim_id"]: item for item in claims}
    thread_by_episode = {
        episode_id: thread for thread in threads for episode_id in thread.get("episode_ids", [])
    }
    result = []
    for episode in episodes:
        episode_claims = [claim_by_id[value] for value in episode.get("claim_ids", []) if value in claim_by_id]
        event_ids = [value for claim in episode_claims for value in claim.get("event_ids", [])]
        utterances = sorted(
            (event_by_id[value] for value in dict.fromkeys(event_ids) if value in event_by_id),
            key=lambda item: (float(item.get("timestamp", 0)), item.get("event_id", "")),
        )
        claim_ids = set(episode.get("claim_ids", []))
        bundle_relations = [
            item for item in relations
            if item.get("source_claim_id") in claim_ids or item.get("target_claim_id") in claim_ids
        ]
        evidence = list(dict.fromkeys(
            value for item in utterances for value in item.get("evidence_ids", [])
        ))
        context = list(dict.fromkeys(
            value for item in utterances for value in item.get("context_ids", [])
            if value not in evidence
        ))
        thread = thread_by_episode.get(episode.get("episode_id"), {})
        continuation_ids = [value for value in thread.get("episode_ids", []) if value != episode.get("episode_id")]
        result.append({
            "bundle_id": "DB" + str(episode.get("episode_id", ""))[1:],
            "episode_id": episode.get("episode_id"),
            "thread_id": thread.get("thread_id"),
            "topic": episode.get("topic") or "Тема встречи",
            "ranges": [{"start": episode.get("start", 0), "end": episode.get("end", episode.get("start", 0))}],
            "utterances": utterances,
            "claim_ids": list(episode.get("claim_ids", [])),
            "question_ids": list(episode.get("question_ids", [])),
            "task_ids": list(episode.get("task_ids", [])),
            "relations": bundle_relations,
            "evidence_ids": evidence,
            "context_ids": context,
            "continuation_episode_ids": continuation_ids,
            "participants": list(episode.get("participants", [])),
        })
    return result
