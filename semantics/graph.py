"""Claim graph validation, correction lifecycle and latest-state reconciliation."""
from __future__ import annotations
from semantics.ontology import RELATION_KINDS


SUPERSEDING = {"corrects", "supersedes", "rejects"}


def normalize_relations(relations, claims):
    ids = {x.get("claim_id") for x in claims}
    result = []
    for index, raw in enumerate(relations or [], 1):
        kind = raw.get("type") or raw.get("relation")
        source = raw.get("source_claim_id") or raw.get("source_event")
        target = raw.get("target_claim_id") or raw.get("target_event")
        evidence = list(dict.fromkeys(raw.get("evidence_ids", [])))
        if kind not in RELATION_KINDS or source not in ids or target not in ids or source == target or not evidence:
            continue
        result.append({"relation_id": raw.get("relation_id") or f"R{index:05d}", "type": kind, "source_claim_id": source, "target_claim_id": target, "evidence_ids": evidence, "confidence": raw.get("confidence") or {"semantic_support": raw.get("semantic_support")}})
    return result


def reconcile_latest_state(claims, relations):
    by_id = {x.get("claim_id"): x for x in claims}
    for claim in claims:
        claim.setdefault("lifecycle", "active")
    for relation in sorted(relations, key=lambda x: x.get("relation_id", "")):
        if relation["type"] not in SUPERSEDING:
            continue
        target = by_id.get(relation["target_claim_id"])
        source = by_id.get(relation["source_claim_id"])
        if not target or not source:
            continue
        target["lifecycle"] = "rejected" if relation["type"] == "rejects" else "superseded"
        source["lifecycle"] = "active"
        target.setdefault("lifecycle_basis_relation_ids", []).append(relation["relation_id"])
    return claims


def cross_episode_allowed(claim_ids, relation_ids, claims, relations):
    by_claim = {x.get("claim_id"): x for x in claims}
    episodes = {by_claim[x].get("episode_id") for x in claim_ids if x in by_claim and by_claim[x].get("episode_id")}
    if len(episodes) <= 1:
        return True
    chosen = {x.get("relation_id") for x in relations if x.get("relation_id") in set(relation_ids)}
    connected = set()
    for relation in relations:
        if relation.get("relation_id") in chosen:
            connected.update((relation.get("source_claim_id"), relation.get("target_claim_id")))
    return set(claim_ids).issubset(connected)
