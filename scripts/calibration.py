"""Condition-aware probability calibration for diarization and Voice ID."""
from __future__ import annotations
import json, math
from pathlib import Path


def sigmoid(value):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


def load_calibrator(path):
    if not path or not Path(path).is_file():
        return None
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("method") not in {"logistic", "isotonic"}:
        raise ValueError("unsupported calibration method")
    return value


def predict(calibrator, features, bucket="default"):
    if not calibrator:
        return None
    model = calibrator.get("buckets", {}).get(bucket) or calibrator.get("buckets", {}).get("default", {})
    if calibrator["method"] == "logistic":
        score = float(model.get("intercept", 0)) + sum(float(model.get("weights", {}).get(key, 0)) * float(value or 0) for key, value in features.items())
        return sigmoid(score)
    points = sorted((float(x), float(y)) for x, y in model.get("points", []))
    raw = float(features.get("score", 0))
    return next((y for x, y in points if raw <= x), points[-1][1] if points else None)


def voice_bucket(duration, overlap=False, low_snr=False, session_prototype=False, conflict=False):
    if conflict:
        return "conflict_second_pass"
    if overlap:
        return "overlap"
    if low_snr:
        return "low_snr"
    if session_prototype:
        return "session_prototype"
    return "clean_short" if duration < 1.5 else "clean_long"
