"""Selective critical-span speech refinement and uncertainty lattices."""
from __future__ import annotations
import hashlib

def expected_value(priority):
    return float(priority.get("risk", 0))*float(priority.get("semantic_importance", 0))*float(priority.get("publication_probability", 1))

def schedule_repairs(spans, budget):
    return sorted(spans, key=lambda x: (-expected_value(x), x.get("start", 0)))[:budget]

def target_speaker_windows(primary, secondary, known_speakers, min_disagreement=.25):
    """Schedule TS-VAD-like refinement only for overlap/disagreement windows."""
    result = []
    for segment in primary:
        overlaps = [x for x in secondary if x.get("start", 0) < segment.get("end", 0) and x.get("end", 0) > segment.get("start", 0)]
        disagreement = 1.0 if overlaps and all(x.get("speaker") != segment.get("speaker") for x in overlaps) else 0.0
        if segment.get("overlap") or disagreement >= min_disagreement:
            result.append({"window_id": "TS" + hashlib.sha256(f"{segment.get('start')}|{segment.get('end')}".encode()).hexdigest()[:12], "start": segment.get("start"), "end": segment.get("end"), "speaker_embeddings": list(known_speakers), "reason": "overlap" if segment.get("overlap") else "diarizer_disagreement", "method": "target_speaker_activity"})
    return result

def asr_lattice(evidence_id, alternatives):
    normalized = [{"text": x.get("text", ""), "model": x.get("model"), "score": float(x.get("score", 0)), "word_alignment": x.get("word_alignment", [])} for x in alternatives]
    conflict = len({x["text"].casefold() for x in normalized}) > 1
    return {"evidence_id": evidence_id, "alternatives": sorted(normalized, key=lambda x: -x["score"]), "resolution": "unresolved" if conflict else "consensus", "publication_allowed": not conflict}

def calibrate_speaker_probability(features, coefficients=None):
    """Logistic calibrator output, distinct from raw similarity confidence."""
    import math
    coefficients = coefficients or {"bias": -1.5, "agreement": 2.2, "similarity": 2.0, "margin": 1.4, "duration": .08, "overlap": -1.2, "prototype_stability": 1.0}
    z = coefficients.get("bias", 0) + sum(coefficients.get(k, 0)*float(features.get(k, 0)) for k in coefficients if k != "bias")
    return 1/(1+math.exp(-max(-30, min(30, z))))
