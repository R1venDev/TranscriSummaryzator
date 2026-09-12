#!/usr/bin/env python3
"""Pure post-processing for cluster-level ReDimNet identification.

Model inference deliberately lives in redimnet_worker.py.  Keeping the
decision logic pure makes it cheap to benchmark and retune without rerunning
either diarizer or the embedding model.
"""
from __future__ import annotations

import math


def cosine(left, right):
    if not left or not right or len(left) != len(right):
        return -1.0
    return sum(float(a) * float(b) for a, b in zip(left, right))


def normalize(values):
    norm = math.sqrt(sum(float(x) ** 2 for x in values))
    return [float(x) / norm for x in values] if norm > 1e-9 else list(values)


def robust_centroid(vectors):
    vectors = [normalize(v) for v in vectors if v]
    if not vectors:
        return None
    if len(vectors) >= 3:
        medians = []
        for left in vectors:
            scores = sorted(cosine(left, right) for right in vectors)
            medians.append(scores[len(scores) // 2])
        medoid = vectors[max(range(len(vectors)), key=lambda i: medians[i])]
        scores = sorted((cosine(medoid, value), value) for value in vectors)
        keep = [value for score, value in scores[max(0, len(scores) // 5):]]
    else:
        keep = vectors
    return normalize([sum(v[i] for v in keep) / len(keep) for i in range(len(keep[0]))])


def match_clusters(cluster_groups, profiles, threshold=0.55, margin=0.08):
    """Identify whole session clusters; many clusters may map to one person."""
    matches, report = {}, {}
    for cluster, data in cluster_groups.items():
        embedding = data.get("embedding")
        scores = []
        for profile in profiles:
            prototype_score = cosine(embedding, profile.get("embedding"))
            reference_scores = sorted(
                (cosine(embedding, ref) for ref in profile.get("references", [])),
                reverse=True,
            )
            reference_score = sum(reference_scores[:3]) / min(3, len(reference_scores)) if reference_scores else prototype_score
            score = 0.75 * prototype_score + 0.25 * reference_score
            scores.append({
                "profile_id": profile["id"], "name": profile["name"],
                "score": round(score, 4), "prototype_score": round(prototype_score, 4),
            })
        scores.sort(key=lambda item: item["score"], reverse=True)
        best = scores[0] if scores else None
        runner = scores[1]["score"] if len(scores) > 1 else -1.0
        difference = best["score"] - runner if best else 0.0
        accepted = bool(best and best["score"] >= threshold and difference >= margin)
        report[cluster] = {
            "scores": scores, "best_similarity": best["score"] if best else -1.0,
            "margin": round(difference, 4), "accepted": accepted,
            "chunks": data.get("chunks", 0),
        }
        if accepted:
            matches[cluster] = {**best, "margin": round(difference, 4)}
    # High-confidence session anchors compensate for microphone/codec shift.
    # Only already accepted, well-separated clusters may influence them.
    session = {}
    for cluster, match in matches.items():
        if match["score"] >= threshold + 0.08 and match["margin"] >= margin + 0.04:
            session.setdefault(match["profile_id"], []).append(cluster_groups[cluster]["embedding"])
    session_prototypes = {}
    for profile in profiles:
        vectors = session.get(profile["id"], [])
        if vectors:
            # Keep the enrollment voice as an anchor; session audio receives
            # equal total weight and is never persisted into the global DB.
            session_prototypes[profile["id"]] = robust_centroid([profile["embedding"], robust_centroid(vectors)])
    for cluster, data in cluster_groups.items():
        if cluster in matches or not session_prototypes:
            continue
        scores = sorted((cosine(data["embedding"], vector), profile_id) for profile_id, vector in session_prototypes.items())
        best_score, best_id = scores[-1]
        runner = scores[-2][0] if len(scores) > 1 else -1.0
        difference = best_score - runner
        if best_score >= threshold and difference >= margin:
            profile = next(value for value in profiles if value["id"] == best_id)
            matches[cluster] = {"profile_id": best_id, "name": profile["name"], "score": round(best_score, 4), "prototype_score": round(best_score, 4), "margin": round(difference, 4), "session_prototype": True}
            report[cluster]["accepted"] = True
            report[cluster]["session_prototype"] = True
    return matches, report


def _identity(cluster, matches, unknown_ids):
    if cluster in matches:
        match = matches[cluster]
        return "profile:" + match["profile_id"], match["name"], match["score"], match["margin"], True
    if cluster not in unknown_ids:
        unknown_ids[cluster] = "UNKNOWN_{}".format(len(unknown_ids) + 1)
    return unknown_ids[cluster], unknown_ids[cluster], 0.0, 0.0, False


def resolve_timeline(consensus, matches, short_seconds=1.5, boundary_tolerance=0.3, local_decisions=None):
    """Resolve local model conflicts and preserve true overlap automatically."""
    unknown_ids, atomic, debug = {}, [], []
    local_decisions = local_decisions or {}
    for item_index, item in enumerate(consensus):
        primary = list(item.get("primary", []))
        verifier = list(item.get("verifier_mapped", []))
        agreed = set(primary) & set(verifier)
        candidates = list(dict.fromkeys(item.get("candidates", item.get("clusters", []))))
        identities = {cluster: _identity(cluster, matches, unknown_ids) for cluster in candidates}

        # Voice identity acts as arbiter. Different anonymous tracks that map
        # confidently to the same enrolled person collapse to one identity.
        known_primary = {identities[c][0] for c in primary if c in identities and identities[c][4]}
        known_verifier = {identities[c][0] for c in verifier if c in identities and identities[c][4]}
        local_match = local_decisions.get(item_index)
        if local_match:
            profile_id = local_match["profile_id"]
            selected_cluster = next((c for c in candidates if identities[c][4] and identities[c][0] == "profile:" + profile_id), None)
            if selected_cluster is None:
                selected_cluster = "local:" + profile_id
                identities[selected_cluster] = (
                    "profile:" + profile_id, local_match["name"], local_match["score"],
                    local_match["margin"], True,
                )
            chosen = [selected_cluster]
            reason = "redimnet_second_pass"
        elif agreed:
            chosen = list(agreed)
            reason = "diarizers_agree"
        elif known_primary and known_verifier and known_primary == known_verifier:
            chosen = [next(c for c in candidates if identities[c][0] in known_primary)]
            reason = "voice_id_resolved_same_person"
        elif known_primary and not known_verifier:
            chosen, reason = primary, "voice_id_primary"
        elif known_verifier and not known_primary:
            chosen, reason = verifier, "voice_id_verifier"
        elif (float(item["end"]) - float(item["start"]) <= short_seconds
              and known_verifier and known_primary and known_verifier != known_primary):
            chosen, reason = verifier, "short_turn_verifier"
        else:
            chosen = primary or verifier or list(item.get("clusters", []))
            reason = "primary_low_confidence"

        selected = []
        for cluster in chosen:
            speaker_id, name, voice_score, voice_margin, known = identities.get(cluster, _identity(cluster, matches, unknown_ids))
            if speaker_id in {x["speaker_id"] for x in selected}:
                continue
            agreement = float(item.get("agreement", 0))
            confidence = 0.45
            if reason == "primary_low_confidence":
                confidence = 0.58
            elif known and agreement >= 0.99:
                confidence = min(0.99, 0.70 + 0.20 * max(0.0, voice_score) + 0.09 * min(1.0, voice_margin / 0.2))
            elif known:
                confidence = min(0.89, 0.58 + 0.22 * max(0.0, voice_score) + 0.07 * min(1.0, voice_margin / 0.2))
            elif agreement >= 0.99:
                confidence = 0.72
            level = "HIGH" if confidence >= 0.9 else ("MEDIUM" if confidence >= 0.7 else "LOW")
            selected.append({
                "start": float(item["start"]), "end": float(item["end"]),
                "speaker_id": speaker_id, "speaker_name": name,
                "anonymous_cluster": cluster, "known_speaker": known,
                "confidence": round(confidence, 4), "confidence_level": level,
                "overlap": bool(item.get("overlap", False)), "decision": reason,
                "primary_tracks": list(item.get("primary", [])),
                "verifier_tracks": list(item.get("verifier_mapped", [])),
            })
        atomic.extend(selected)
        debug.append({**item, "resolved": selected, "decision": reason})

    # Short unknown islands inherit identity only when both neighbours agree
    # and neither diarizer claims an overlap/speaker change there.
    ordered = sorted(atomic, key=lambda x: (x["start"], x["end"], x["speaker_id"]))
    for index, item in enumerate(ordered):
        if item["overlap"] or item["end"] - item["start"] > short_seconds:
            continue
        if item["known_speaker"] and item["confidence_level"] != "LOW":
            continue
        previous = next((ordered[i] for i in range(index - 1, -1, -1) if ordered[i]["end"] <= item["start"] + 1e-6), None)
        following = next((ordered[i] for i in range(index + 1, len(ordered)) if ordered[i]["start"] >= item["end"] - 1e-6), None)
        track_neighbours = []
        for neighbour in (previous, following):
            if not neighbour or not neighbour["known_speaker"] or neighbour["overlap"]:
                continue
            same_primary = set(item.get("primary_tracks", [])) & set(neighbour.get("primary_tracks", []))
            same_verifier = set(item.get("verifier_tracks", [])) & set(neighbour.get("verifier_tracks", []))
            if same_primary or same_verifier:
                track_neighbours.append(neighbour)
        contextual = None
        if track_neighbours and len({value["speaker_id"] for value in track_neighbours}) == 1:
            contextual = track_neighbours[0]
        elif previous and following and previous["known_speaker"] and previous["speaker_id"] == following["speaker_id"] and not previous["overlap"] and not following["overlap"]:
            contextual = previous
        if contextual:
            for key in ("speaker_id", "speaker_name", "known_speaker"):
                item[key] = contextual[key]
            item["confidence"], item["confidence_level"], item["decision"] = 0.74, "MEDIUM", "short_turn_context"

    # Collapse artificial boundaries and split anonymous clusters that ReDimNet
    # identified as the same real person, but never collapse overlapping tracks.
    merged = []
    for item in ordered:
        if (merged and item["speaker_id"] == merged[-1]["speaker_id"]
                and item["overlap"] == merged[-1]["overlap"]
                and item["start"] - merged[-1]["end"] <= boundary_tolerance):
            merged[-1]["end"] = max(merged[-1]["end"], item["end"])
            merged[-1]["confidence"] = round(min(merged[-1]["confidence"], item["confidence"]), 4)
            merged[-1]["confidence_level"] = "HIGH" if merged[-1]["confidence"] >= 0.9 else ("MEDIUM" if merged[-1]["confidence"] >= 0.7 else "LOW")
            if item["anonymous_cluster"] != merged[-1]["anonymous_cluster"]:
                merged[-1]["anonymous_cluster"] = sorted(set(str(merged[-1]["anonymous_cluster"]).split("+") + [str(item["anonymous_cluster"])]))
        else:
            merged.append(dict(item))
    return merged, debug


def rttm_lines(segments, session="recording"):
    return [
        "SPEAKER {} 1 {:.3f} {:.3f} <NA> <NA> {} <NA> <NA>".format(
            session, item["start"], item["end"] - item["start"], item["speaker_id"]
        ) for item in segments
    ]
