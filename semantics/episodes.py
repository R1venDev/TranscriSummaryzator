"""Deterministic first-pass dialogue episodes and long-range topic threads."""
from __future__ import annotations
import re


TOPIC_SHIFT_RE = re.compile(r"(?iu)\b(?:теперь|дальше|следующ(?:ий|ая)|верн[её]мся|что касается|отдельно)\b")


def _tokens(value):
    return {x for x in re.findall(r"(?iu)[a-zа-яё0-9]+", str(value or "").casefold()) if len(x) > 2}


def _similarity(a, b):
    left, right = _tokens(a), _tokens(b)
    return len(left & right) / max(1, len(left | right))


def build_episodes(claims, max_gap=150.0, topic_threshold=0.08):
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
        explicit_shift = bool(TOPIC_SHIFT_RE.search(str(claim.get("statement") or "")))
        if gap > max_gap or (explicit_shift and _similarity(topic, prior_topic) < topic_threshold):
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
            "question_ids": [x.get("claim_id") for x in group if x.get("kind") == "question"],
            "decision_ids": [x.get("claim_id") for x in group if x.get("kind") == "decision"],
            "task_ids": [x.get("claim_id") for x in group if x.get("kind") == "action"],
            "participants": sorted({s for x in group for s in x.get("speaker_refs", [])}),
            "outcome_claim_ids": [x.get("claim_id") for x in group if x.get("kind") in {"decision", "action", "experimental_result"}],
            "open_threads": [],
        })
    return result


def build_threads(episodes, claims):
    by_id = {x.get("claim_id"): x for x in claims}
    threads = []
    for episode in episodes:
        matched = None
        for thread in threads:
            if _similarity(episode.get("topic"), thread.get("topic")) >= 0.18:
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
            if claim.get("kind") == "question" and claim.get("question_status") not in {"answered", "rhetorical", "superseded"}:
                matched["open_question_ids"].append(claim_id)
    for thread in threads:
        thread["state"] = "open" if thread["open_question_ids"] else "resolved"
    return threads
