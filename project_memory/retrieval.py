"""Hybrid lexical/entity/dense-hook/graph retrieval with lineage expansion."""
from __future__ import annotations
import math
import re

def _tokens(value):
    return {x for x in re.findall(r"(?iu)[a-zа-яё0-9]+", str(value or "").casefold()) if len(x) > 2}

def _cosine(left, right):
    if not left or not right or len(left) != len(right): return 0.0
    return sum(x*y for x, y in zip(left, right)) / max(1e-9, math.sqrt(sum(x*x for x in left))*math.sqrt(sum(y*y for y in right)))

def retrieve(question, project_state, meeting_states=(), limit=12, query_embedding=None):
    query = _tokens(question); candidates = []
    claims = list((project_state or {}).get("propositions", {}).values())
    if not claims: claims = [claim for meeting in meeting_states for claim in meeting.get("claims", [])]
    revision = (project_state or {}).get("revision", 0)
    for claim in claims:
        lexical = len(query & _tokens(claim.get("statement"))) / max(1, len(query))
        names = {str(x.get("canonical_name", "")).casefold() for x in claim.get("entities", [])}
        entity = sum(any(q in name for name in names) for q in query) / max(1, len(query)) if names else 0
        dense = _cosine(query_embedding, claim.get("embedding"))
        state = 1.0 if claim.get("lifecycle", "active") == "active" else .2
        recency = 1/(1+max(0, revision-claim.get("project_revision", 0)))
        score = 3*lexical + 2*entity + 2*dense + state + recency
        if score > .25: candidates.append((score, claim))
    candidates.sort(key=lambda x: (-x[0], -float(x[1].get("start", 0))))
    result, seen = [], set()
    lineage = (project_state or {}).get("lineage", {})
    for score, claim in candidates:
        if claim.get("proposition_id") in seen: continue
        family = claim.get("lineage_id"); family_ids = lineage.get(family, []) if family else []
        result.append({"claim": claim, "score": score, "lineage_ids": family_ids, "evidence_ids": claim.get("evidence_ids", [])}); seen.add(claim.get("proposition_id"))
        if len(result) >= limit: break
    return result
