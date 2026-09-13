"""Retrieve state → claims → episodes → evidence, never lossy Markdown."""
from __future__ import annotations
import re


def _tokens(value):
    return {x for x in re.findall(r"(?iu)[a-zа-яё0-9]+", str(value or "").casefold()) if len(x) > 2}


def retrieve(question, project_state, meeting_states, limit=12):
    query = _tokens(question)
    candidates = []
    for meeting in meeting_states:
        episode_by_id = {x.get("episode_id"): x for x in meeting.get("episodes", [])}
        for claim in meeting.get("claims", []):
            overlap = len(query & _tokens(claim.get("statement")))
            if overlap:
                candidates.append((overlap, claim, episode_by_id.get(claim.get("episode_id"))))
    candidates.sort(key=lambda x: (-x[0], -float(x[1].get("start", 0))))
    return [{"claim": claim, "episode": episode, "evidence_ids": claim.get("evidence_ids", [])} for _, claim, episode in candidates[:limit]]
