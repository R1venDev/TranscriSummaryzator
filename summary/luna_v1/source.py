"""Canonical, read-only projection of a completed transcript for Luna.

The on-disk transcript is never rewritten. Word IDs, audio paths, uncertainty
ledgers and voice metadata stay local; the model receives only utterances.
"""

from __future__ import annotations

from datetime import date
import hashlib
import json
import math
from pathlib import Path
import re


UNKNOWN_SPEAKER = "Участник не определён"
_SOURCE_DATE = re.compile(r"^(\d{2})\.(\d{2})\.(\d{4})(?:\b|\s|[—-])")


def _meeting_date(source_name: str) -> str | None:
    match = _SOURCE_DATE.search(source_name)
    if not match:
        return None
    day, month, year = (int(part) for part in match.groups())
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _milliseconds(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: timestamp is not numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name}: timestamp is negative or non-finite")
    return round(number * 1000)


def load_source(transcript_path: str | Path) -> tuple[str, dict, str]:
    """Return canonical JSON text, local source index, SHA-256 of original bytes.

    IDs match the historical 1-based U00001 convention, including the place
    of any empty utterance. The index holds exact source times and navigation
    anchors. It is not a semantic assertion that a cited claim is entailed.
    """
    raw = Path(transcript_path).read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    document = json.loads(raw)
    if not isinstance(document, dict) or not isinstance(document.get("utterances"), list):
        raise ValueError("transcript.json has no utterance list")
    utterances = document["utterances"]
    if not utterances:
        raise ValueError("transcript.json has no utterances")
    labels = document.get("speakers") or {}
    if not isinstance(labels, dict):
        raise ValueError("transcript.json has invalid speaker map")

    projected = []
    by_id = {}
    participants = []
    unattributed = False
    previous_start = -1
    for number, item in enumerate(utterances, 1):
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise ValueError(f"utterance {number} has no text")
        start_ms = _milliseconds(item.get("start"), f"utterance {number} start")
        end_ms = _milliseconds(item.get("end"), f"utterance {number} end")
        if start_ms < previous_start or end_ms < start_ms:
            raise ValueError(f"utterance {number} has invalid chronological range")
        previous_start = start_ms
        speaker_id = item.get("speaker")
        mapped = labels.get(speaker_id) if isinstance(speaker_id, str) else None
        speaker = str(mapped).strip() if isinstance(mapped, str) and mapped.strip() else UNKNOWN_SPEAKER
        if speaker == UNKNOWN_SPEAKER:
            unattributed = True
        elif speaker not in participants:
            participants.append(speaker)
        source_id = f"U{number:05d}"
        text = " ".join(item["text"].split())
        record = {
            "id": source_id,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "speaker": speaker,
            "text": text,
        }
        if bool((item.get("uncertainty") or {}).get("needs_review")):
            record["needs_review"] = True
        projected.append(record)
        by_id[source_id] = record

    source_name = str(document.get("source") or "")
    meeting_date = _meeting_date(source_name)
    duration_ms = _milliseconds(document.get("duration_seconds"), "meeting duration")
    if duration_ms < projected[-1]["end_ms"]:
        raise ValueError("meeting duration ends before the last utterance")
    payload = {
        "source_kind": "TRANSCRIPT_SOURCE",
        "source_name": source_name,
        "meeting_date": meeting_date,
        "duration_ms": duration_ms,
        "participants_by_transcript": participants,
        "unattributed_speech": unattributed,
        "utterances": projected,
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    source_index = {
        "source_sha256": sha256,
        "source_name": source_name,
        "meeting_date": meeting_date,
        "duration_ms": duration_ms,
        "participants": participants,
        "unattributed_speech": unattributed,
        "by_id": by_id,
    }
    return serialized, source_index, sha256
