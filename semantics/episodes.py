"""Deterministic first-pass dialogue episodes and long-range topic threads."""
from __future__ import annotations
import re


TOPIC_SHIFT_RE = re.compile(r"(?iu)\b(?:теперь|дальше|следующ(?:ий|ая)|верн[её]мся|что касается|отдельно)\b")


def _tokens(value):
    return {x for x in re.findall(r"(?iu)[a-zа-яё0-9]+", str(value or "").casefold()) if len(x) > 2}


def _similarity(a, b):
    left, right = _tokens(a), _tokens(b)
    return len(left & right) / max(1, len(left | right))


def _entity_ids(claim):
    return {x.get("entity_id") for x in claim.get("entities", []) if isinstance(x, dict) and x.get("entity_id")}


def boundary_score(claim, prior, recent):
    """Hybrid discourse boundary signal; embedding scores may be supplied upstream."""
    gap = max(0.0, float(claim.get("start", 0)) - float(prior.get("end", prior.get("start", 0))))
    lexical_shift = 1.0 - _similarity(claim.get("statement"), " ".join(x.get("statement", "") for x in recent))
    entity_shift = 0.0 if _entity_ids(claim) & set().union(*(_entity_ids(x) for x in recent)) else .35
    act_shift = .2 if claim.get("kind") == "question" else 0.0
    explicit = .45 if TOPIC_SHIFT_RE.search(str(claim.get("statement") or "")) else 0.0
    embedding_shift = float(claim.get("discourse_features", {}).get("embedding_shift", 0))
    return min(1.0, gap / 300.0 + lexical_shift * .25 + entity_shift + act_shift + explicit + embedding_shift * .5)


def build_episodes(claims, max_gap=150.0, topic_threshold=0.72):
    ordered = sorted(claims, key=lambda x: (float(x.get("start", 0)), x.get("claim_id", "")))
    groups = []
    for claim in ordered:
        if not groups:
            groups.append([claim])
            continue
        prior = groups[-1][-1]
        gap = float(claim.get("start", 0)) - float(prior.get("end", prior.get("start", 0)))
        topic = claim.get("topic") or claim.get("statement") or ""
        prior_topic = " ".join(str(x.get("topic") or x.get("statement") or "") for x in groups[-1][-3:])
        score = boundary_score(claim, prior, groups[-1][-4:])
        if gap > max_gap or score >= topic_threshold:
            groups.append([claim])
        else:
            groups[-1].append(claim)
    result = []
    for index, group in enumerate(groups, 1):
        topics = [str(x.get("topic") or "").strip() for x in group if str(x.get("topic") or "").strip()]
        topic = max(topics, key=topics.count) if topics else str(group[0].get("statement") or "Тема")[:100]
        episode_id = f"E{index:04d}"
        for claim in group:
            claim["episode_id"] = episode_id
        result.append({
            "episode_id": episode_id,
            "start": min(float(x.get("start", 0)) for x in group),
            "end": max(float(x.get("end", x.get("start", 0))) for x in group),
            "topic": topic,
            "initiating_event_ids": [group[0].get("claim_id")],
            "claim_ids": [x.get("claim_id") for x in group],
            "question_ids": [x.get("claim_id") for x in group if x.get("content_kind") == "question"],
            "decision_ids": [x.get("claim_id") for x in group if x.get("content_kind") == "decision" and x.get("decision_status") == "accepted"],
            "task_ids": [x.get("claim_id") for x in group if x.get("content_kind") in {"action", "follow_up"}],
            "participants": sorted({s for x in group for s in x.get("speaker_refs", [])}),
            "outcome_claim_ids": [x.get("claim_id") for x in group if (x.get("content_kind") == "decision" and x.get("decision_status") == "accepted") or (x.get("content_kind") in {"action", "follow_up"} and x.get("task_status") in {"accepted", "self_committed", "in_progress", "blocked", "completed"}) or x.get("content_kind") == "experimental_result"],
            "open_threads": [],
        })
    return result


def build_threads(episodes, claims, relations=None):
    by_id = {x.get("claim_id"): x for x in claims}
    threads = []
    for episode in episodes:
        matched = None
        for thread in threads:
            episode_claims = [by_id[x] for x in episode.get("claim_ids", []) if x in by_id]
            thread_claims = [by_id[x] for x in thread.get("active_claim_ids", []) if x in by_id]
            entity_overlap = set().union(*(_entity_ids(x) for x in episode_claims)) & set().union(*(_entity_ids(x) for x in thread_claims)) if thread_claims else set()
            linked = any(r.get("source_claim_id") in episode.get("claim_ids", []) and r.get("target_claim_id") in thread.get("active_claim_ids", []) or r.get("target_claim_id") in episode.get("claim_ids", []) and r.get("source_claim_id") in thread.get("active_claim_ids", []) for r in relations or [])
            if _similarity(episode.get("topic"), thread.get("topic")) >= 0.18 or entity_overlap or linked:
                matched = thread
                break
        if matched is None:
            matched = {"thread_id": f"TH{len(threads)+1:04d}", "topic": episode.get("topic"), "episode_ids": [], "state": "open", "open_question_ids": [], "active_claim_ids": [], "latest_update": 0.0}
            threads.append(matched)
        matched["episode_ids"].append(episode["episode_id"])
        matched["latest_update"] = max(matched["latest_update"], episode["end"])
        episode["open_threads"] = [matched["thread_id"]]
        for claim_id in episode["claim_ids"]:
            claim = by_id.get(claim_id, {})
            claim["thread_id"] = matched["thread_id"]
            if claim.get("lifecycle", "active") == "active":
                matched["active_claim_ids"].append(claim_id)
            if claim.get("content_kind") == "question" and claim.get("question_status") not in {"answered", "rhetorical", "superseded"}:
                matched["open_question_ids"].append(claim_id)
    for thread in threads:
        thread["state"] = "open" if thread["open_question_ids"] else "resolved"
    return threads
