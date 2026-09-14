#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from quality_schema import utterance_uncertainty, word_uncertainty
from config_schema import load_config
from evidence_ledger import attach_word_ids, ledger_document, record_resolution
from diagnostics import configure as configure_diagnostics, decision as diagnostic_decision
from diagnostics import event as diagnostic_event, publish as publish_diagnostics
from diagnostics import system_snapshot


ROOT = Path(__file__).resolve().parent
INBOX = ROOT / "inbox"
STATE = ROOT / "state"
JOBS = ROOT / "work" / "jobs"
GLOBAL_STAGE_CACHE = ROOT / "work" / "stage-cache"
OUTPUTS = ROOT / "outputs"
VOICE_PROFILES = ROOT / "voice_profiles"
DB_PATH = STATE / "queue.sqlite3"
CONFIG_PATH = ROOT / "config.json"
VOCABULARY_PATH = ROOT / "vocabulary.json"
DASHBOARD_PATH = ROOT / "dashboard.html"
PROFILES_PATH = ROOT / "profiles.html"
SUMMARY_BENCHMARK = Path("/mnt/shared-data/MeetingTranscript/summary-benchmark")
SUMMARY_STATUS_PATH = SUMMARY_BENCHMARK / "current.json"
MEDIA_EXTENSIONS = {".mkv", ".mp4", ".mov", ".m4v", ".webm", ".wav", ".mp3", ".m4a", ".flac", ".ogg"}
PROFILE_LOCK = threading.Lock()


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def config():
    return load_config(CONFIG_PATH, STATE / "config.resolved.json")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def diagnostic_environment(job_id, job_dir, component):
    return {
        "TRANSCRISUMMARY_DIAGNOSTICS": str(Path(job_dir) / "diagnostics.jsonl"),
        "TRANSCRISUMMARY_DIAGNOSTICS_TRACE": str(Path(job_dir) / "diagnostics.trace.jsonl"),
        "TRANSCRISUMMARY_COMPONENT": component,
        "TRANSCRISUMMARY_JOB_ID": str(job_id),
        "TRANSCRISUMMARY_RUN_ID": str(getattr(diagnostic_environment, "run_id", "")),
    }


STAGE_DEPENDENCIES = {
    "audio": ["pipeline.py", "scripts/diagnostics.py"],
    "diarizen": ["pipeline.py", "scripts/diarize_worker.py", "scripts/diagnostics.py"],
    "ultra": ["pipeline.py", "scripts/ultra_worker.py", "scripts/diagnostics.py"],
    "consensus": ["pipeline.py", "scripts/consensus.py", "scripts/diagnostics.py"],
    "asr": ["pipeline.py", "scripts/asr_worker.py", "scripts/model_common.py", "scripts/diagnostics.py"],
    "export": ["pipeline.py", "scripts/quality_schema.py", "scripts/evidence_ledger.py", "scripts/diagnostics.py"],
}


def stage_cache_key(stage, inputs):
    family = stage.split("-", 1)[0]
    paths = [ROOT / value for value in STAGE_DEPENDENCIES.get(family, ["pipeline.py"])]
    return _json_hash({"stage": stage, "inputs": inputs, "code": {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths if path.is_file()
    }})


def stage_cache_valid(job_dir, stage, key, artifacts):
    path = Path(job_dir) / "stage_cache.json"
    metadata = load_json(path) if path.is_file() else {}
    if metadata.get(stage, {}).get("key") == key and all(Path(job_dir, name).is_file() for name in artifacts):
        return True
    shared = GLOBAL_STAGE_CACHE / key[:2] / key
    manifest_path = shared / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = load_json(manifest_path)
    if manifest.get("key") != key or sorted(manifest.get("artifacts", {})) != sorted(artifacts):
        return False
    for name in artifacts:
        source = shared / name
        expected = manifest["artifacts"].get(name)
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            return False
    for name in artifacts:
        target = Path(job_dir, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".cache-tmp")
        try:
            os.link(shared / name, temporary)
        except OSError:
            shutil.copy2(shared / name, temporary)
        os.replace(temporary, target)
    metadata[stage] = {"key": key, "artifacts": list(artifacts), "completed_at": now(), "source": "global_content_addressed"}
    write_json(path, metadata)
    return True


def mark_stage_cached(job_dir, stage, key, artifacts):
    path = Path(job_dir) / "stage_cache.json"
    metadata = load_json(path) if path.is_file() else {}
    metadata[stage] = {"key": key, "artifacts": list(artifacts), "completed_at": now()}
    write_json(path, metadata)
    shared = GLOBAL_STAGE_CACHE / key[:2] / key
    shared.parent.mkdir(parents=True, exist_ok=True)
    lock_path = shared.parent / (key + ".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        manifest_path = shared / "manifest.json"
        if not manifest_path.is_file():
            temporary = shared.parent / (key + ".tmp-" + uuid.uuid4().hex)
            temporary.mkdir(parents=True)
            hashes = {}
            for name in artifacts:
                source = Path(job_dir, name)
                destination = temporary / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                hashes[name] = hashlib.sha256(destination.read_bytes()).hexdigest()
            write_json(temporary / "manifest.json", {"schema_version": 1, "key": key, "stage": stage, "artifacts": hashes, "created_at": now()})
            try:
                os.replace(temporary, shared)
            except OSError:
                shutil.rmtree(temporary, ignore_errors=True)


def artifact_provenance(path, producer, inputs=None, model=None):
    path = Path(path)
    return {
        "artifact": str(path.name),
        "producer": {"component": producer, "stage_version": 1},
        "model": model or {},
        "inputs": inputs or {},
        "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
        "created_at": now(),
    }


def connect():
    STATE.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY,
            fingerprint TEXT NOT NULL UNIQUE,
            source_path TEXT NOT NULL,
            original_name TEXT NOT NULL,
            status TEXT NOT NULL,
            stage TEXT NOT NULL,
            job_dir TEXT NOT NULL,
            output_dir TEXT,
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    columns = {row[1] for row in db.execute("PRAGMA table_info(jobs)")}
    migrations = {
        "progress": "ALTER TABLE jobs ADD COLUMN progress REAL NOT NULL DEFAULT 0",
        "detail": "ALTER TABLE jobs ADD COLUMN detail TEXT",
        "started_at": "ALTER TABLE jobs ADD COLUMN started_at TEXT",
        "finished_at": "ALTER TABLE jobs ADD COLUMN finished_at TEXT",
        "speaker_count": "ALTER TABLE jobs ADD COLUMN speaker_count INTEGER",
        "summary_status": "ALTER TABLE jobs ADD COLUMN summary_status TEXT NOT NULL DEFAULT 'not_started'",
        "summary_stage": "ALTER TABLE jobs ADD COLUMN summary_stage TEXT",
        "summary_progress": "ALTER TABLE jobs ADD COLUMN summary_progress REAL NOT NULL DEFAULT 0",
        "summary_detail": "ALTER TABLE jobs ADD COLUMN summary_detail TEXT",
        "summary_error": "ALTER TABLE jobs ADD COLUMN summary_error TEXT",
        "summary_started_at": "ALTER TABLE jobs ADD COLUMN summary_started_at TEXT",
        "summary_finished_at": "ALTER TABLE jobs ADD COLUMN summary_finished_at TEXT",
        "worker_id": "ALTER TABLE jobs ADD COLUMN worker_id TEXT",
        "lease_until": "ALTER TABLE jobs ADD COLUMN lease_until TEXT",
        "attempt_id": "ALTER TABLE jobs ADD COLUMN attempt_id TEXT",
        "content_sha256": "ALTER TABLE jobs ADD COLUMN content_sha256 TEXT",
    }
    for column, statement in migrations.items():
        if column not in columns:
            db.execute(statement)
    # Legacy fingerprints were raw content hashes. Preserve them for stable
    # output paths while backfilling the new explicit content digest.
    db.execute("UPDATE jobs SET content_sha256 = fingerprint WHERE content_sha256 IS NULL")
    db.execute("UPDATE jobs SET progress = 100 WHERE status = 'done' AND progress < 100")
    db.execute("UPDATE jobs SET detail = 'Готово' WHERE status = 'done' AND detail IS NULL")
    db.execute("UPDATE jobs SET started_at = created_at WHERE started_at IS NULL AND status IN ('running', 'done', 'failed')")
    db.execute("UPDATE jobs SET finished_at = updated_at WHERE finished_at IS NULL AND status IN ('done', 'failed')")
    db.commit()
    return db


def fingerprint(path):
    """Return the immutable content digest used by cache and provenance."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def submission_fingerprint(content_sha256, original_name):
    """Deduplicate only an identical recording submitted under the same name."""
    payload = str(content_sha256) + "\0" + Path(original_name).name
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def find_existing_job(db, content_sha256, original_name):
    submission = submission_fingerprint(content_sha256, original_name)
    return db.execute(
        "SELECT * FROM jobs WHERE fingerprint = ? OR (content_sha256 = ? AND original_name = ?) ORDER BY id LIMIT 1",
        (submission, content_sha256, Path(original_name).name),
    ).fetchone()


def safe_name(path):
    cleaned = "".join(c if c.isalnum() or c in "-_." else "_" for c in Path(path).stem).strip("._")
    return cleaned[:100] or "recording"


def normalize_speaker_count(value):
    if value in (None, "", "auto"):
        return None
    try:
        count = int(value)
    except (TypeError, ValueError):
        raise ValueError("Количество участников должно быть от 1 до 8")
    if count < 1 or count > 8:
        raise ValueError("Количество участников должно быть от 1 до 8")
    return count


def enqueue(path, known_fingerprint=None, speaker_count=None, original_name=None):
    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.suffix.casefold() not in MEDIA_EXTENSIONS:
        raise ValueError("Это не поддерживаемый медиафайл: {}".format(path))
    original_name = Path(original_name or path.name).name
    content_sha256 = known_fingerprint or fingerprint(path)
    fp = submission_fingerprint(content_sha256, original_name)
    db = connect()
    existing = find_existing_job(db, content_sha256, original_name)
    if existing:
        print("Уже в очереди: job {} ({})".format(existing["id"], existing["status"]))
        return existing["id"]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_dir = JOBS / ("{}-{}".format(stamp, fp[:10]))
    job_dir.mkdir(parents=True, exist_ok=False)
    cursor = db.execute(
        """INSERT INTO jobs
        (fingerprint, content_sha256, source_path, original_name, status, stage, job_dir, speaker_count, created_at, updated_at)
        VALUES (?, ?, ?, ?, 'queued', 'queued', ?, ?, ?, ?)""",
        (fp, content_sha256, str(path), original_name, str(job_dir), normalize_speaker_count(speaker_count), now(), now()),
    )
    db.commit()
    write_status_snapshot(db)
    job_id = cursor.lastrowid
    write_json(job_dir / "job.json", {"id": job_id, "fingerprint": fp, "content_sha256": content_sha256, "source": str(path), "created_at": now()})
    print("Добавлено в очередь: job {} — {}".format(job_id, original_name))
    if config().get("open_dashboard_on_job", True):
        open_dashboard()
    return job_id


def update_job(db, job_id, **values):
    if values.get("status") == "running" or "progress" in values:
        values.setdefault("lease_until", (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat(timespec="seconds"))
    values["updated_at"] = now()
    query = "UPDATE jobs SET {} WHERE id = ?".format(", ".join("{} = ?".format(key) for key in values))
    db.execute(query, tuple(values.values()) + (job_id,))
    db.commit()
    write_status_snapshot(db)


def status_rows(db):
    return db.execute(
        "SELECT id, original_name, status, stage, progress, detail, error, output_dir, speaker_count, created_at, started_at, finished_at, updated_at, summary_status, summary_stage, summary_progress, summary_detail, summary_error, summary_started_at, summary_finished_at FROM jobs ORDER BY id DESC LIMIT 50"
    ).fetchall()


def write_status_snapshot(db):
    payload = {"jobs": [dict(row) for row in status_rows(db)], "inbox": str(INBOX), "outputs": str(OUTPUTS), "updated_at": now()}
    write_json(STATE / "progress.json", payload)


def run_command(command, log_path, env=None, progress_callback=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    diagnostic_event("subprocess", category="stage", outcome="started", inputs={"command": list(map(str, command))}, refs={"log": str(log_path)})
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n[{}] {}\n".format(now(), " ".join(map(str, command))))
        log.flush()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            if progress_callback:
                progress_callback(line.strip())
        process.wait()
    diagnostic_event(
        "subprocess", category="stage", outcome="completed" if process.returncode == 0 else "failed",
        metrics={"return_code": process.returncode}, refs={"log": str(log_path)},
        duration_ms=round((time.monotonic() - started) * 1000, 3),
        severity="INFO" if process.returncode == 0 else "ERROR",
    )
    if process.returncode:
        raise RuntimeError("Команда завершилась с кодом {}. См. {}".format(process.returncode, log_path))


def validate_media(source):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(source)],
        capture_output=True,
        text=True,
    )
    if result.returncode or not result.stdout.strip():
        raise RuntimeError("FFprobe не смог прочитать медиафайл: {}".format(result.stderr.strip()))
    return float(result.stdout.strip())


def extract_audio(source, destination, track, log):
    run_command(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-y", "-i", str(source),
            "-map", "0:a:{}".format(track), "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", str(destination),
        ],
        log,
    )


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def profile_path(profile_id):
    if not profile_id or any(character not in "0123456789abcdef" for character in profile_id) or len(profile_id) != 32:
        raise ValueError("Профиль не найден")
    return VOICE_PROFILES / profile_id / "profile.json"


def load_voice_profiles():
    profiles = []
    if not VOICE_PROFILES.exists():
        return profiles
    for path in VOICE_PROFILES.glob("*/profile.json"):
        try:
            profile = load_json(path)
            samples = profile.get("samples", [])
            profile["sample_count"] = len(samples)
            profile["duration_seconds"] = round(sum(float(sample.get("duration_seconds", 0)) for sample in samples), 1)
            profile["ready"] = profile["sample_count"] >= 2 and profile["duration_seconds"] >= 30
            profiles.append(profile)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    return sorted(profiles, key=lambda item: (item.get("name", "").casefold(), item.get("created_at", "")))


def public_profile(profile):
    samples = profile.get("samples", [])
    sample_count = profile.get("sample_count", len(samples))
    duration_seconds = profile.get("duration_seconds", round(sum(float(sample.get("duration_seconds", 0)) for sample in samples), 1))
    return {
        "id": profile["id"],
        "name": profile["name"],
        "sample_count": sample_count,
        "duration_seconds": duration_seconds,
        "ready": profile.get("ready", sample_count >= 2 and duration_seconds >= 30),
        "samples": [
            {
                "id": sample["id"],
                "name": sample.get("name", "Образец"),
                "duration_seconds": sample.get("duration_seconds", 0),
                "created_at": sample.get("created_at"),
            }
            for sample in profile.get("samples", [])
        ],
    }


def propagate_profile_name(profile_id, name):
    """Keep labels assigned from this voice profile in finished exports in sync."""
    db = connect()
    jobs = db.execute("SELECT id, output_dir FROM jobs WHERE status = 'done' AND output_dir IS NOT NULL").fetchall()
    updated = 0
    for job in jobs:
        speakers_path = Path(job["output_dir"]) / "speakers.json"
        if not speakers_path.is_file():
            continue
        try:
            speaker_data = load_json(speakers_path)
            labels = speaker_data.get("labels", {})
            sources = speaker_data.get("label_sources", {})
            matches = speaker_data.get("voice_matches", {})
            changed = False
            for speaker, match in matches.items():
                if match.get("profile_id") != profile_id:
                    continue
                match["name"] = name
                if sources.get(speaker) == "voice_profile":
                    labels[speaker] = name
                changed = True
            profile_speaker = "profile:" + profile_id
            if sources.get(profile_speaker) == "voice_profile" and profile_speaker in labels:
                labels[profile_speaker] = name
                changed = True
            voice_segments_path = Path(job["output_dir"]) / "voice_segments.json"
            if voice_segments_path.is_file():
                voice_segments = load_json(voice_segments_path)
                segment_changed = False
                for segment in voice_segments.get("segments", []):
                    if segment.get("profile_id") == profile_id:
                        segment["name"] = name
                        segment_changed = True
                if segment_changed:
                    write_json(voice_segments_path, voice_segments)
                    changed = True
            if changed:
                write_json(speakers_path, speaker_data)
                rename_export(job["id"])
                updated += 1
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    return updated


def voice_embedding_model_path():
    matches = sorted((ROOT / "work" / "cache" / "huggingface" / "hub" / "models--pyannote--wespeaker-voxceleb-resnet34-LM" / "snapshots").glob("*/pytorch_model.bin"))
    if not matches:
        raise FileNotFoundError("Модель голосовых отпечатков не установлена")
    return matches[-1]


def normalized_mean(vectors):
    if not vectors:
        return None
    dimension = len(vectors[0])
    if not dimension or any(len(vector) != dimension for vector in vectors):
        return None
    mean = [sum(float(vector[index]) for vector in vectors) / len(vectors) for index in range(dimension)]
    length = sum(value * value for value in mean) ** 0.5
    return [value / length for value in mean] if length > 1e-8 else None


def cosine_similarity(left, right):
    if not left or not right or len(left) != len(right):
        return -1.0
    return sum(float(a) * float(b) for a, b in zip(left, right))


def extract_voice_embeddings(groups, destination, log, segments=None):
    manifest = destination.with_suffix(".manifest.json")
    write_json(manifest, {
        "groups": {str(key): [str(path) for path in paths] for key, paths in groups.items()},
        "segments": {
            str(key): [dict(item, path=str(item["path"])) for item in items]
            for key, items in (segments or {}).items()
        },
    })
    try:
        run_command([
            str(ROOT / ".venv-diarizen" / "bin" / "python"),
            str(ROOT / "scripts" / "voice_embedding_worker.py"),
            "--manifest", str(manifest),
            "--output", str(destination),
            "--model", str(voice_embedding_model_path()),
            "--device", str(config().get("voice_embedding_device", "auto")),
        ], log)
    finally:
        manifest.unlink(missing_ok=True)
    return load_json(destination).get("groups", {})


def overlap(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def timestamp(seconds, srt=False):
    millis = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, ms = divmod(remainder, 1000)
    separator = "," if srt else "."
    return "{:02d}:{:02d}:{:02d}{}{:03d}".format(hours, minutes, secs, separator, ms)


def assign_speakers(words, intervals, cfg):
    assigned = []
    for word in words:
        start, end = float(word["start"]), float(word["end"])
        duration = max(0.04, end - start)
        scores = {}
        active = set()
        evidence_confidence = []
        for interval in intervals:
            amount = overlap(start, end, interval["start"], interval["end"])
            if amount > 0:
                scores[interval["speaker"]] = scores.get(interval["speaker"], 0.0) + amount
                active.add(interval["speaker"])
                if interval.get("confidence") is not None:
                    evidence_confidence.append(float(interval["confidence"]))
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best = ranked[0] if ranked else (None, 0.0)
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        confidence = best[1] / duration
        clear = confidence >= cfg["speaker_match_min_overlap"] and (best[1] - second) / duration >= cfg["speaker_match_margin"]
        speaker = best[0] if clear else None
        flags = list(word.get("flags", []))
        if len(active) > 1:
            flags.append("overlap")
        if not clear:
            flags.append("ambiguous" if ranked else "no_diarization")
        if evidence_confidence and max(evidence_confidence) < 0.7:
            flags.append("low_confidence")
        evidence = [{"speaker": value, "overlap_seconds": round(amount, 4), "source": "diarization"} for value, amount in ranked]
        item = dict(word, speaker=speaker, speaker_confidence=round(confidence, 3), flags=sorted(set(flags)), speaker_evidence=evidence)
        item["resolution"] = {"selected": speaker, "method": "acoustic_overlap" if clear else "unresolved", "risk": "low" if clear else "high"}
        item["speaker_resolution_history"] = [{"speaker": speaker, "reason": item["resolution"]["method"]}]
        diagnostic_decision(
            "word_speaker_overlap", speaker if speaker is not None else "unresolved",
            candidates=[value for value, _ in ranked],
            metrics={"start": start, "end": end, "duration": duration, "best_overlap_ratio": confidence, "margin_ratio": (best[1] - second) / duration, "active_speakers": len(active), "diarization_confidence_max": max(evidence_confidence) if evidence_confidence else None, "candidate_overlaps": evidence},
            thresholds={"minimum_overlap": cfg["speaker_match_min_overlap"], "minimum_margin": cfg["speaker_match_margin"], "low_diarization_confidence": 0.7},
            reasons=item["flags"] or ["overlap_and_margin_passed"], refs={"word_id": item.get("word_id"), "source_word_ids": item.get("source_word_ids", []), "text": item.get("text")},
        )
        assigned.append(item)

    # ASR often starts very short words a few milliseconds before diarization.
    # Recover them from the immediately adjacent, unambiguous speaker context.
    for index, word in enumerate(assigned):
        if word["speaker"] is not None or float(word["end"]) - float(word["start"]) > 0.35:
            continue
        previous = next((assigned[i] for i in range(index - 1, -1, -1) if assigned[i]["speaker"] is not None), None)
        following = next((assigned[i] for i in range(index + 1, len(assigned)) if assigned[i]["speaker"] is not None), None)
        candidates = []
        if previous is not None:
            candidates.append((max(0.0, float(word["start"]) - float(previous["end"])), previous["speaker"]))
        if following is not None:
            candidates.append((max(0.0, float(following["start"]) - float(word["end"])), following["speaker"]))
        candidates.sort()
        if candidates and candidates[0][0] <= float(cfg.get("word_boundary_context_seconds", 0.4)):
            nearest = candidates[0][1]
            if len(candidates) == 1 or candidates[1][1] == nearest or candidates[0][0] + 0.12 < candidates[1][0]:
                record_resolution(word, nearest, "context_nearest", 0.75, "medium")
                word["flags"] = sorted(set(word["flags"] + ["speaker_context"]))
                word["speaker_inferred_from_context"] = True

    # Bridge short ambiguous sequences when the same confidently assigned
    # speaker continues on both sides. Diarization and ASR boundaries often
    # disagree slightly, especially where voices overlap.
    index = 0
    while index < len(assigned):
        if assigned[index]["speaker"] is not None:
            index += 1
            continue
        start_index = index
        while index < len(assigned) and assigned[index]["speaker"] is None:
            index += 1
        end_index = index
        previous = assigned[start_index - 1] if start_index > 0 else None
        following = assigned[end_index] if end_index < len(assigned) else None
        if previous is None or following is None or previous["speaker"] != following["speaker"]:
            continue
        sequence = assigned[start_index:end_index]
        sequence_duration = float(sequence[-1]["end"]) - float(sequence[0]["start"])
        left_gap = max(0.0, float(sequence[0]["start"]) - float(previous["end"]))
        right_gap = max(0.0, float(following["start"]) - float(sequence[-1]["end"]))
        bridge_gap = min(0.8, float(cfg.get("utterance_gap_seconds", 1.2)))
        if len(sequence) <= 5 and sequence_duration <= 2.0 and left_gap <= bridge_gap and right_gap <= bridge_gap:
            for word in sequence:
                score = max(0.65, min(0.8, float(word["speaker_confidence"])))
                record_resolution(word, previous["speaker"], "context_bridge", score, "medium")
                word["flags"] = sorted(set(word["flags"] + ["speaker_context"]))
                word["speaker_inferred_from_context"] = True
    return assigned


def phrase_units(words, cfg):
    """Split ASR words into short linguistic units suitable for voice ID."""
    units = []
    current = []
    max_duration = float(cfg.get("voice_identity_max_phrase_seconds", 10.0))
    pause = float(cfg.get("voice_identity_phrase_gap_seconds", 0.65))
    terminal = (".", "?", "!", "…")
    for index, word in enumerate(words):
        if current:
            previous = current[-1][1]
            gap = max(0.0, float(word["start"]) - float(previous["end"]))
            previous_text = str(previous.get("text", "")).rstrip('"\'»”)]}')
            too_long = float(word["end"]) - float(current[0][1]["start"]) > max_duration
            if gap >= pause or previous_text.endswith(terminal) or too_long:
                units.append(current)
                current = []
        current.append((index, word))
    if current:
        units.append(current)
    return units


def clause_units(words, cfg):
    """Join pause-split ASR units while punctuation says the clause continues."""
    clauses = []
    continuation_gap = float(cfg.get("clause_coherence_gap_seconds", 2.5))
    for unit in phrase_units(words, cfg):
        if clauses:
            previous_word = clauses[-1][-1][1]
            current_word = unit[0][1]
            gap = float(current_word["start"]) - float(previous_word["end"])
            previous_text = str(previous_word.get("text", "")).rstrip()
            if gap <= continuation_gap and not previous_text.endswith((".", "?", "!", "…")):
                clauses[-1].extend(unit)
                continue
        clauses.append(list(unit))
    return clauses


def apply_domain_vocabulary(words, cfg):
    """Apply explicit, punctuation-preserving word and phrase corrections."""
    vocabulary = load_json(VOCABULARY_PATH) if VOCABULARY_PATH.is_file() else {}
    replacements = {
        str(source).casefold(): str(target)
        for source, target in {
            **vocabulary.get("replacements", {}),
            **cfg.get("domain_vocabulary", {}),
        }.items()
    }
    phrases = {
        tuple(str(source).casefold().split()): str(target)
        for source, target in vocabulary.get("phrases", {}).items()
    }
    contextual_rules = list(vocabulary.get("contextual_rules", []))
    corrected = []
    token_pattern = re.compile(r"^([^\w@]*)(.*?)([^\w]*)$", re.UNICODE)
    parsed = []
    for word in words:
        match = token_pattern.match(str(word.get("text", "")))
        parsed.append(match.groups() if match else ("", str(word.get("text", "")), ""))
    phrase_lengths = sorted({len(key) for key in phrases}, reverse=True)
    index = 0
    while index < len(words):
        phrase_match = None
        for length in phrase_lengths:
            if index + length > len(words):
                continue
            key = tuple(parsed[position][1].casefold() for position in range(index, index + length))
            if key in phrases:
                phrase_match = (length, phrases[key])
                break
        if phrase_match:
            length, canonical = phrase_match
            originals = [str(words[position].get("text", "")) for position in range(index, index + length)]
            item = dict(words[index])
            item["end"] = words[index + length - 1]["end"]
            item["text"] = parsed[index][0] + canonical + parsed[index + length - 1][2]
            item["asr_text"] = " ".join(originals)
            item["raw_text"] = item["asr_text"]
            item["normalization_operations"] = [{"source": item["asr_text"], "target": item["text"], "policy": "SAFE_EXACT"}]
            item["source_word_ids"] = [words[position]["word_id"] for position in range(index, index + length)]
            item["vocabulary_corrected"] = True
            corrected.append(item)
            index += length
            continue
        word = words[index]
        item = dict(word)
        text = str(item.get("text", ""))
        item.setdefault("raw_text", text)
        item.setdefault("normalization_operations", [])
        leading, token, trailing = parsed[index]
        canonical = replacements.get(token.casefold())
        applied_policy = "SAFE_EXACT"
        if canonical is None:
            context = " ".join(str(words[pos].get("text", "")) for pos in range(max(0, index - 5), min(len(words), index + 6))).casefold()
            uncertain = ((item.get("asr_confidence") is not None and float(item.get("asr_confidence")) < 0.8) or bool(set(item.get("flags", [])) & {"asr_boundary", "asr_alternative"}))
            for rule in contextual_rules:
                if token.casefold() != str(rule.get("source", "")).casefold():
                    continue
                required = [str(value).casefold() for value in rule.get("required_context", [])]
                if rule.get("policy") == "CONTEXT_REQUIRED" and any(value in context for value in required) and (uncertain or not rule.get("require_asr_uncertainty", False)):
                    canonical, applied_policy = str(rule.get("target")), "CONTEXT_REQUIRED"
                    break
        if canonical is not None:
            item["text"] = leading + canonical + trailing
            item["asr_text"] = text
            item["raw_text"] = text
            item["vocabulary_corrected"] = True
            item["normalization_operations"] = list(item.get("normalization_operations", [])) + [{"source": token, "target": canonical, "policy": applied_policy}]
        corrected.append(item)
        index += 1
    return corrected


def identity_units(asr_words, diarization, cfg):
    """Build 6–12 second voice-ID windows without crossing a diarized speaker change."""
    assigned = assign_speakers(asr_words, diarization.get("intervals", []), cfg)
    base = []
    for unit in phrase_units(assigned, cfg):
        weights = {}
        for _, word in unit:
            speaker = word.get("speaker")
            if speaker is None:
                continue
            weights[speaker] = weights.get(speaker, 0.0) + max(0.04, float(word["end"]) - float(word["start"]))
        speaker = max(weights, key=weights.get) if weights else None
        base.append({"speaker": speaker, "words": unit})

    merged = []
    max_duration = float(cfg.get("voice_identity_max_window_seconds", 12.0))
    merge_gap = float(cfg.get("voice_identity_merge_gap_seconds", 1.5))
    for item in base:
        if merged:
            previous = merged[-1]
            start = float(previous["words"][0][1]["start"])
            gap = max(0.0, float(item["words"][0][1]["start"]) - float(previous["words"][-1][1]["end"]))
            combined = float(item["words"][-1][1]["end"]) - start
            if item["speaker"] is not None and item["speaker"] == previous["speaker"] and gap <= merge_gap and combined <= max_duration:
                previous["words"].extend(item["words"])
                continue
        merged.append(item)
    return [item["words"] for item in merged]


def apply_voice_segments(words, voice_segments):
    if not voice_segments:
        return words
    for word in words:
        start, end = float(word["start"]), float(word["end"])
        duration = max(0.04, end - start)
        ranked = sorted(
            ((overlap(start, end, segment["start"], segment["end"]), segment) for segment in voice_segments),
            key=lambda item: item[0],
            reverse=True,
        )
        if not ranked or ranked[0][0] / duration < 0.55:
            continue
        segment = ranked[0][1]
        record_resolution(word, "profile:" + segment["profile_id"], "redimnet_phrase", segment["score"], "low")
        word["flags"] = list(word.get("flags", []))
        if "voice_profile" not in word["flags"]:
            word["flags"].append("voice_profile")
    return words


def smooth_phrase_speakers(words, cfg):
    """Remove short speaker islands inside one ASR phrase without hiding real turns."""
    max_island = float(cfg.get("speaker_island_max_seconds", 2.2))
    dominance = float(cfg.get("speaker_phrase_dominance", 0.68))
    for unit in phrase_units(words, cfg):
        if len(unit) < 2:
            continue
        weights = {}
        for _, word in unit:
            if word.get("speaker") is None:
                continue
            duration = max(0.04, float(word["end"]) - float(word["start"]))
            weights[word["speaker"]] = weights.get(word["speaker"], 0.0) + duration
        if not weights:
            continue
        ranked = sorted(weights.items(), key=lambda item: item[1], reverse=True)
        dominant, dominant_weight = ranked[0]
        dominant_ratio = dominant_weight / max(0.04, sum(weights.values()))
        if dominant_ratio < float(cfg.get("speaker_phrase_low_confidence_dominance", 0.55)):
            continue
        position = 0
        while position < len(unit):
            if unit[position][1].get("speaker") == dominant:
                position += 1
                continue
            start = position
            while position < len(unit) and unit[position][1].get("speaker") != dominant:
                position += 1
            island = unit[start:position]
            island_duration = float(island[-1][1]["end"]) - float(island[0][1]["start"])
            low_evidence = all("low_confidence" in word.get("flags", []) for _, word in island)
            # A direct ReDimNet phrase identity is stronger evidence than the
            # duration-majority heuristic used by this cleanup pass.
            if any("voice_profile" in word.get("flags", []) for _, word in island):
                continue
            if island_duration > max_island or (dominant_ratio < dominance and not low_evidence):
                continue
            for _, word in island:
                record_resolution(word, dominant, "phrase_smoothing", max(0.7, float(word.get("speaker_confidence", 0))), "medium")
                word["flags"] = list(word.get("flags", []))
                if "speaker_smoothed" not in word["flags"]:
                    word["flags"].append("speaker_smoothed")
    return words


def cohere_clause_speakers(words, cfg):
    """Remove only weak speaker islands inside a continuous ASR clause.

    Punctuation is not acoustic evidence.  In particular an unfinished phrase
    may be interrupted by another person, so this pass must never overwrite a
    supported identity merely because the text looks syntactically continuous.
    """
    if not words:
        return words
    gap_limit = float(cfg.get("clause_coherence_gap_seconds", 2.5))
    short_limit = float(cfg.get("clause_coherence_short_seconds", 3.0))
    for clause in clause_units(words, cfg):
        runs = []
        for index, word in clause:
            if not runs or runs[-1]["speaker"] != word.get("speaker"):
                runs.append({"speaker": word.get("speaker"), "items": [(index, word)]})
            else:
                runs[-1]["items"].append((index, word))
        for left, right in zip(runs, runs[1:]):
            if left["speaker"] is None or right["speaker"] is None or left["speaker"] == right["speaker"]:
                continue
            left_first, left_last = left["items"][0][1], left["items"][-1][1]
            right_first, right_last = right["items"][0][1], right["items"][-1][1]
            if float(right_first["start"]) - float(left_last["end"]) > gap_limit:
                continue
            left_duration = float(left_last["end"]) - float(left_first["start"])
            right_duration = float(right_last["end"]) - float(right_first["start"])
            if min(left_duration, right_duration) > short_limit:
                continue
            dominant, weaker = (left, right) if left_duration >= right_duration else (right, left)
            weak_words = [word for _, word in weaker["items"]]
            weak_evidence = all(
                "low_confidence" in word.get("flags", [])
                or "ambiguous" in word.get("flags", [])
                or word.get("speaker") is None
                for word in weak_words
            )
            boundary_gap = max(0.0, float(right_first["start"]) - float(left_last["end"]))
            continuation_text = next((
                str(word.get("text", "")).lstrip("—–- «\"'(")
                for _, word in dominant["items"]
                if str(word.get("text", "")).lstrip("—–- «\"'(")
            ), "")
            continuation_without_voice_anchor = (
                weaker is left
                and left_duration <= float(cfg.get("clause_unanchored_prefix_max_seconds", 1.5))
                and not any("voice_profile" in word.get("flags", []) for word in weak_words)
                and any("voice_profile" in word.get("flags", []) for _, word in dominant["items"])
                and continuation_text[:1].islower()
                and boundary_gap <= float(cfg.get("clause_unanchored_prefix_gap_seconds", 2.0))
            )
            # A known voice on each side is a real acoustic boundary unless the
            # shorter side was explicitly marked weak and the boundary is tight.
            both_known = all(str(run.get("speaker", "")).startswith("profile:") for run in (left, right))
            if (both_known and not continuation_without_voice_anchor
                    and (not weak_evidence or boundary_gap > float(cfg.get("clause_coherence_acoustic_gap_seconds", 0.35)))):
                continue
            for _, word in weaker["items"]:
                record_resolution(word, dominant["speaker"], "clause_coherence", max(0.72, float(word.get("speaker_confidence", 0))), "high")
                word["flags"] = list(word.get("flags", []))
                if "clause_coherence" not in word["flags"]:
                    word["flags"].append("clause_coherence")
    return words


def bridge_short_profile_phrases(words, cfg):
    """Attach a very short sentence to an adjacent, verified voice identity."""
    maximum = float(cfg.get("voice_identity_context_max_seconds", 2.0))
    gap_limit = float(cfg.get("voice_identity_context_gap_seconds", 0.3))
    # First repair single boundary tokens even when the surrounding ASR phrase
    # contains a real speaker change and therefore cannot be smoothed whole.
    for index, word in enumerate(words):
        if word.get("speaker") is not None or float(word["end"]) - float(word["start"]) > 1.0:
            continue
        previous = words[index - 1] if index else None
        following = words[index + 1] if index + 1 < len(words) else None
        chosen = None
        if previous and following and previous.get("speaker") == following.get("speaker") and str(previous.get("speaker", "")).startswith("profile:"):
            if float(word["start"]) - float(previous["end"]) <= 4.0 and float(following["start"]) - float(word["end"]) <= 4.0:
                chosen = previous["speaker"]
        if chosen is None and previous and str(previous.get("speaker", "")).startswith("profile:") and str(word.get("text", "")).rstrip().endswith((".", ",", ":", ";", "?", "!")):
            if float(word["start"]) - float(previous["end"]) <= 2.0:
                chosen = previous["speaker"]
        if chosen is None and following and str(following.get("speaker", "")).startswith("profile:") and str(following.get("text", ""))[:1].islower():
            if float(following["start"]) - float(word["end"]) <= 0.5:
                chosen = following["speaker"]
        if chosen:
            record_resolution(word, chosen, "profile_context", max(0.7, float(word.get("speaker_confidence", 0))), "high")
            word["flags"] = sorted(set(word.get("flags", []) + ["speaker_context"]))
    for unit in phrase_units(words, cfg):
        if any(str(word.get("speaker", "")).startswith("profile:") for _, word in unit):
            continue
        start_index = unit[0][0]
        end_index = unit[-1][0]
        duration = float(unit[-1][1]["end"]) - float(unit[0][1]["start"])
        if duration > maximum:
            continue
        previous = words[start_index - 1] if start_index > 0 else None
        following = words[end_index + 1] if end_index + 1 < len(words) else None
        candidates = []
        if previous is not None and str(previous.get("speaker", "")).startswith("profile:"):
            candidates.append((max(0.0, float(unit[0][1]["start"]) - float(previous["end"])), previous["speaker"]))
        if following is not None and str(following.get("speaker", "")).startswith("profile:"):
            candidates.append((max(0.0, float(following["start"]) - float(unit[-1][1]["end"])), following["speaker"]))
        candidates.sort()
        if not candidates:
            continue
        chosen = None
        if len(candidates) > 1 and candidates[0][1] == candidates[1][1] and max(candidates[0][0], candidates[1][0]) <= float(cfg.get("voice_identity_same_context_gap_seconds", 4.0)):
            chosen = candidates[0][1]
        elif previous is not None and str(previous.get("speaker", "")).startswith("profile:"):
            previous_gap = max(0.0, float(unit[0][1]["start"]) - float(previous["end"]))
            final_text = str(unit[-1][1].get("text", "")).rstrip()
            if previous_gap <= float(cfg.get("voice_identity_punctuation_gap_seconds", 2.0)) and final_text.endswith((".", ",", ":", ";", "?", "!")):
                chosen = previous["speaker"]
        if chosen is None and following is not None and str(following.get("speaker", "")).startswith("profile:"):
            following_gap = max(0.0, float(following["start"]) - float(unit[-1][1]["end"]))
            following_text = str(following.get("text", ""))
            if following_gap <= max(gap_limit, 0.5) and following_text[:1].islower():
                chosen = following["speaker"]
        if chosen is None and candidates[0][0] <= gap_limit and (len(candidates) == 1 or candidates[0][1] == candidates[1][1] or candidates[0][0] + 0.12 < candidates[1][0]):
            chosen = candidates[0][1]
        if chosen is None:
            continue
        for _, word in unit:
            record_resolution(word, chosen, "profile_phrase_context", max(0.72, float(word.get("speaker_confidence", 0))), "high")
            word["flags"] = list(word.get("flags", []))
            if "speaker_context" not in word["flags"]:
                word["flags"].append("speaker_context")
    return words


def speaker_display(speaker, labels):
    if speaker is None:
        return "Участник не определён"
    return labels.get(speaker) or "Спикер {}".format(speaker)


def turn_needs_speaker_review(turn, labels):
    """Show a UI warning only while the displayed identity is unresolved.

    Acoustic boundary flags remain available in transcript.json and review.csv,
    but they must not keep a turn red after a voice profile or a manual label has
    resolved the speaker.
    """
    speaker = turn.get("speaker")
    return speaker is None or not str(labels.get(speaker, "")).strip()


def utterances(words, gap):
    result = []
    for word in words:
        previous_text = str(result[-1]["words"][-1].get("text", "")).rstrip() if result else ""
        adaptive_gap = max(gap, float(config().get("utterance_continuation_gap_seconds", 3.0))) if result and not previous_text.endswith((".", "?", "!", "…")) else gap
        if not result or word["speaker"] != result[-1]["speaker"] or word["start"] - result[-1]["end"] > adaptive_gap:
            result.append({"start": word["start"], "end": word["end"], "speaker": word["speaker"], "words": [word]})
        else:
            result[-1]["end"] = word["end"]
            result[-1]["words"].append(word)
    for item in result:
        item["text"] = " ".join(word["text"] for word in item["words"]).strip()
        item["flags"] = sorted({flag for word in item["words"] for flag in word["flags"]})
        item["uncertainty"] = utterance_uncertainty(item["words"])
    return result


def split_srt(items, max_duration=8.0, max_chars=92):
    result = []
    for item in items:
        current = None
        for word in item["words"]:
            if current is None or word["end"] - current["start"] > max_duration or len(current["text"]) + len(word["text"]) + 1 > max_chars:
                current = {"start": word["start"], "end": word["end"], "speaker": item["speaker"], "text": word["text"]}
                result.append(current)
            else:
                current["end"] = word["end"]
                current["text"] += " " + word["text"]
    return result


def export_results(job, duration, diarization, asr, output_dir, cfg):
    output_dir.mkdir(parents=True, exist_ok=True)
    speakers = sorted({item["speaker"] for item in diarization["intervals"]})
    speaker_path = output_dir / "speakers.json"
    if speaker_path.exists():
        speaker_data = load_json(speaker_path)
        labels = speaker_data.get("labels", {})
    else:
        speaker_data = {}
        labels = {}
    speaker_data.update({"labels": labels, "available_speakers": speakers, "instructions": "Задайте имя в labels и выполните: ./pipeline rename <job-id>"})
    write_json(speaker_path, speaker_data)

    raw_words = attach_word_ids(asr["words"])
    normalized_words = apply_domain_vocabulary(raw_words, cfg)
    words = assign_speakers(normalized_words, diarization["intervals"], cfg)
    voice_segments_path = output_dir / "voice_segments.json"
    voice_segments = load_json(voice_segments_path).get("segments", []) if voice_segments_path.exists() else []
    words = apply_voice_segments(words, voice_segments)
    words = bridge_short_profile_phrases(words, cfg)
    words = smooth_phrase_speakers(words, cfg)
    words = cohere_clause_speakers(words, cfg)
    turns = utterances(words, cfg["utterance_gap_seconds"])
    for word in words:
        word["uncertainty"] = word_uncertainty(word)
    flag_counts = {}
    resolution_counts = {}
    for word in words:
        for flag in word.get("flags", []):
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
        method = word.get("resolution", {}).get("method", "unknown")
        resolution_counts[method] = resolution_counts.get(method, 0) + 1
    diagnostic_event(
        "speaker_word_assignment", category="decision", outcome="completed",
        inputs={"words": len(words), "diarization_intervals": len(diarization.get("intervals", []))},
        metrics={"assigned_words": sum(word.get("speaker") is not None for word in words), "unassigned_words": sum(word.get("speaker") is None for word in words), "flag_counts": flag_counts, "resolution_method_counts": resolution_counts, "speaker_confidence_min": min((float(word.get("speaker_confidence", 0)) for word in words), default=None), "speaker_confidence_mean": sum(float(word.get("speaker_confidence", 0)) for word in words) / max(1, len(words)), "utterances": len(turns)},
        thresholds={"minimum_overlap": cfg.get("speaker_match_min_overlap"), "minimum_margin": cfg.get("speaker_match_margin"), "word_boundary_context_seconds": cfg.get("word_boundary_context_seconds"), "utterance_gap_seconds": cfg.get("utterance_gap_seconds")},
        refs={"word_level_evidence": "transcript.json#words", "review": "review.csv"},
    )
    payload = {
        "schema_version": 3,
        "source": job["original_name"],
        "duration_seconds": round(duration, 3),
        "models": {"diarization": diarization.get("model"), "asr": asr.get("model"), "vad": asr.get("vad")},
        "speakers": labels,
        "diarization": diarization["intervals"],
        "words": words,
        "utterances": [{k: v for k, v in item.items() if k != "words"} for item in turns],
        "uncertainty_schema": {
            "version": 1,
            "recognition_confidence": "model value when available; otherwise null",
            "speaker_confidence": "pipeline heuristic, not a calibrated probability",
            "needs_review": "true when the source contains an explicit risk flag or inferred speaker",
        },
    }
    write_json(output_dir / "transcript.json", payload)
    ledger = ledger_document(raw_words, words, turns)
    write_json(output_dir / "transcript" / "words.json", {"schema_version": 1, "words": ledger["words"]})
    write_json(output_dir / "transcript" / "normalization.json", {"schema_version": 1, "tokens": ledger["normalized_tokens"]})
    write_json(output_dir / "semantics" / "evidence_spans.json", {"schema_version": 1, "spans": ledger["evidence_spans"]})

    md = ["# {}".format(Path(job["original_name"]).stem), ""]
    txt = []
    for item in turns:
        label = speaker_display(item["speaker"], labels)
        marker = " ⚠" if turn_needs_speaker_review(item, labels) else ""
        md.append("**[{}] {}{}:** {}".format(timestamp(item["start"]), label, marker, item["text"]))
        md.append("")
        txt.append("[{}] {}{}: {}".format(timestamp(item["start"]), label, marker, item["text"]))
    (output_dir / "transcript.md").write_text("\n".join(md).rstrip() + "\n", encoding="utf-8")
    (output_dir / "transcript.txt").write_text("\n".join(txt) + "\n", encoding="utf-8")

    srt_lines = []
    for index, item in enumerate(split_srt(turns), 1):
        srt_lines.extend([
            str(index),
            "{} --> {}".format(timestamp(item["start"], True), timestamp(item["end"], True)),
            "{}: {}".format(speaker_display(item["speaker"], labels), item["text"]),
            "",
        ])
    (output_dir / "subtitles.srt").write_text("\n".join(srt_lines), encoding="utf-8")

    with (output_dir / "review.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["start", "end", "text", "speaker", "flags", "speaker_confidence"])
        for word in words:
            if word["flags"]:
                writer.writerow([timestamp(word["start"]), timestamp(word["end"]), word["text"], word["speaker"] or "", ";".join(word["flags"]), word["speaker_confidence"]])


def create_speaker_samples(audio, intervals, output_dir, log):
    samples_dir = output_dir / "speaker_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    for interval in intervals:
        start, end = float(interval["start"]), float(interval["end"])
        duration = end - start
        if duration < 1.8:
            continue
        contaminated = any(
            other is not interval
            and other["speaker"] != interval["speaker"]
            and overlap(start, end, float(other["start"]), float(other["end"])) > 0.08
            for other in intervals
        )
        if not contaminated:
            candidates.append((min(duration, 8.0), start, interval["speaker"]))
    per_speaker = {}
    sample_limit = max(3, int(config().get("voice_meeting_samples_per_speaker", 8)))
    for duration, start, speaker in sorted(candidates, reverse=True):
        count = per_speaker.get(speaker, 0)
        if count >= sample_limit:
            continue
        destination = samples_dir / ("speaker_{}_{}.wav".format(speaker, count + 1))
        run_command([
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(start), "-t", str(duration), "-i", str(audio), "-c:a", "pcm_s16le", str(destination),
        ], log)
        per_speaker[speaker] = count + 1


def identify_voice_phrases(audio, asr, diarization, enrolled, output_dir, cfg, log):
    units = identity_units(asr.get("words", []), diarization, cfg)
    minimum = float(cfg.get("voice_identity_min_phrase_seconds", 2.0))
    segment_groups = {}
    metadata = {}
    for index, unit in enumerate(units):
        start = float(unit[0][1]["start"])
        end = float(unit[-1][1]["end"])
        if end - start < minimum:
            continue
        key = "phrase_{:05d}".format(index)
        segment_groups[key] = [{"path": Path(audio), "start": start, "end": end}]
        metadata[key] = {
            "start": round(start, 3),
            "end": round(end, 3),
            "text": " ".join(str(word.get("text", "")) for _, word in unit).strip(),
        }
    if not segment_groups:
        write_json(Path(output_dir) / "voice_segments.json", {"segments": [], "message": "Нет достаточно длинных фраз"})
        return [], {}, float(cfg.get("voice_identity_threshold", 0.56))

    embedding_path = Path(output_dir) / "voice_phrase_embeddings.json"
    embeddings = extract_voice_embeddings({}, embedding_path, log, segments=segment_groups)
    cross_profile = max(
        [cosine_similarity(left["embedding"], right["embedding"]) for index, left in enumerate(enrolled) for right in enrolled[index + 1:]]
        or [-1.0]
    )
    base_threshold = float(cfg.get("voice_identity_threshold", 0.56))
    threshold = max(base_threshold, cross_profile + float(cfg.get("voice_identity_profile_separation", 0.02)))
    margin = float(cfg.get("voice_identity_margin", 0.12))
    strong_threshold = float(cfg.get("voice_identity_strong_threshold", 0.72))
    strong_margin = float(cfg.get("voice_identity_strong_margin", 0.08))
    accepted = []
    scores_report = {}
    for key, data in embeddings.items():
        scores = sorted(
            [
                {"profile_id": profile["id"], "name": profile["name"], "score": round(cosine_similarity(data.get("embedding"), profile["embedding"]), 4)}
                for profile in enrolled
            ],
            key=lambda item: item["score"],
            reverse=True,
        )
        scores_report[key] = scores
        if not scores:
            continue
        runner_up = scores[1]["score"] if len(scores) > 1 else -1.0
        difference = scores[0]["score"] - runner_up
        duration = metadata[key]["end"] - metadata[key]["start"]
        short_window = duration < float(cfg.get("voice_identity_short_window_seconds", 4.0))
        short_threshold = float(cfg.get("voice_identity_short_threshold", 0.72))
        short_margin = float(cfg.get("voice_identity_short_margin", 0.14))
        accepted_match = (
            scores[0]["score"] >= short_threshold and difference >= short_margin
            if short_window
            else (
                (scores[0]["score"] >= threshold and difference >= margin)
                or (scores[0]["score"] >= strong_threshold and difference >= strong_margin)
            )
        )
        if not accepted_match:
            continue
        accepted.append({
            **metadata[key],
            "profile_id": scores[0]["profile_id"],
            "name": scores[0]["name"],
            "score": scores[0]["score"],
            "margin": round(difference, 4),
        })
    payload = {
        "model": "pyannote/wespeaker-voxceleb-resnet34-LM",
        "threshold": round(threshold, 4),
        "margin": margin,
        "segments": accepted,
        "scores": scores_report,
    }
    write_json(Path(output_dir) / "voice_segments.json", payload)
    return accepted, scores_report, threshold


REDIMNET_MODEL = "PalabraAI/ReDimNet2-B6-vb2+vox2+cnc2_v0-lm"


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def extract_redimnet_embeddings(groups, destination, log, cfg):
    """Run ReDimNet in its isolated environment and cache by exact manifest."""
    serializable = {
        str(group): [dict(item, path=str(item["path"])) if isinstance(item, dict) else str(item) for item in items]
        for group, items in groups.items()
    }
    key = _json_hash({
        "model": REDIMNET_MODEL, "repository": cfg["redimnet_repository"],
        "revision": cfg["redimnet_revision"], "groups": serializable,
    })
    key_path = Path(str(destination) + ".key")
    if Path(destination).is_file() and key_path.is_file() and key_path.read_text().strip() == key:
        return load_json(destination).get("groups", {})
    manifest = Path(str(destination) + ".manifest.json")
    write_json(manifest, {"groups": serializable})
    try:
        run_command([
            str(ROOT / ".venv-fusion" / "bin" / "python"),
            str(ROOT / "scripts" / "redimnet_worker.py"),
            "--manifest", str(manifest), "--output", str(destination),
            "--cache", str(ROOT / "work" / "cache"),
            "--device", str(cfg.get("redimnet_device", "auto")),
            "--repository", cfg["redimnet_repository"],
            "--revision", cfg["redimnet_revision"],
        ], log, env=dict(os.environ, TORCH_HOME=str(ROOT / "work" / "cache" / "torch"), PYTHONUNBUFFERED="1"))
        key_path.write_text(key + "\n", encoding="utf-8")
    finally:
        manifest.unlink(missing_ok=True)
    return load_json(destination).get("groups", {})


def enrollment_signature(profiles):
    values = []
    for profile in profiles:
        if not profile.get("ready"):
            continue
        samples = []
        for sample in profile.get("samples", []):
            path = VOICE_PROFILES / profile["id"] / sample.get("audio", "missing")
            if path.is_file():
                stat = path.stat()
                samples.append([str(path), stat.st_size, stat.st_mtime_ns])
        values.append([profile["id"], samples])
    return _json_hash({"model": REDIMNET_MODEL, "profiles": values})


def ensure_redimnet_enrollment(cfg, log):
    profiles = [profile for profile in load_voice_profiles() if profile.get("ready")]
    if not profiles:
        return []
    destination = VOICE_PROFILES / "_redimnet2_enrollment.json"
    signature = enrollment_signature(profiles)
    if destination.is_file():
        cached = load_json(destination)
        if cached.get("signature") == signature:
            names = {profile["id"]: profile["name"] for profile in profiles}
            result = cached.get("profiles", [])
            for item in result:
                item["name"] = names.get(item["id"], item.get("name", item["id"]))
            return result
    groups = {
        profile["id"]: [VOICE_PROFILES / profile["id"] / sample["audio"] for sample in profile.get("samples", [])]
        for profile in profiles
    }
    raw_path = VOICE_PROFILES / "_redimnet2_enrollment.raw.json"
    groups_result = extract_redimnet_embeddings(groups, raw_path, log, cfg)
    result = []
    by_id = {profile["id"]: profile for profile in profiles}
    for profile_id, data in groups_result.items():
        if profile_id in by_id and data.get("embedding"):
            result.append({"id": profile_id, "name": by_id[profile_id]["name"], **data})
    write_json(destination, {"model": REDIMNET_MODEL, "signature": signature, "profiles": result})
    return result


def cluster_anchor_groups(audio, intervals, cfg, consensus=None):
    """Collect long, clean, cross-diarizer-agreed speech per anonymous cluster."""
    if consensus:
        # Include verifier-only candidates from disputed regions.  They are
        # precisely the tracks that can reveal an under-clustered primary
        # result.  ReDimNet will reject them unless global evidence is strong.
        clean = []
        for item in consensus:
            if item.get("overlap"):
                continue
            for speaker in item.get("candidates", item.get("clusters", [])):
                clean.append({"speaker": speaker, "start": item["start"], "end": item["end"]})
    else:
        clean = [item for item in intervals if not item.get("overlap") and float(item.get("consensus_confidence", 0)) >= 0.7]
    merged = []
    for item in sorted(clean, key=lambda value: (str(value["speaker"]), float(value["start"]))):
        current = {"speaker": str(item["speaker"]), "start": float(item["start"]), "end": float(item["end"])}
        if merged and merged[-1]["speaker"] == current["speaker"] and current["start"] - merged[-1]["end"] <= 0.35:
            merged[-1]["end"] = max(merged[-1]["end"], current["end"])
        else:
            merged.append(current)
    groups = {}
    minimum = float(cfg.get("redimnet_anchor_min_seconds", 3.0))
    target = float(cfg.get("redimnet_anchor_target_seconds", 30.0))
    by_speaker = {}
    for item in merged:
        if item["end"] - item["start"] >= minimum:
            by_speaker.setdefault(item["speaker"], []).append(item)
    for speaker, items in by_speaker.items():
        total = 0.0
        for item in sorted(items, key=lambda value: value["end"] - value["start"], reverse=True):
            if total >= target:
                break
            duration = min(item["end"] - item["start"], target - total, 12.0)
            if duration >= minimum:
                groups.setdefault(speaker, []).append({"path": Path(audio), "start": item["start"] + 0.1, "end": item["start"] + duration - 0.1})
                total += duration
    return groups


def _profile_support(start, end, timeline, matches, source="verifier_mapped"):
    """Return duration support per known profile for one diarizer source."""
    support = {}
    for item in timeline or []:
        duration = overlap(start, end, float(item["start"]), float(item["end"]))
        if duration <= 0 or item.get("overlap"):
            continue
        for cluster in item.get(source, []):
            match = matches.get(str(cluster)) or matches.get(cluster)
            if match:
                profile_id = match["profile_id"]
                support[profile_id] = support.get(profile_id, 0.0) + duration
    return support


def _track_support(start, end, timeline, source="verifier_mapped"):
    """Return raw diarizer-track support without requiring known identity."""
    support = {}
    for item in timeline or []:
        duration = overlap(start, end, float(item["start"]), float(item["end"]))
        if duration <= 0:
            continue
        for track in item.get(source, []):
            support[str(track)] = support.get(str(track), 0.0) + duration
    return support


def diarization_aware_clause_units(words, cfg, timeline=None, matches=None):
    """Split textual clauses at independently verified acoustic identities."""
    if not timeline or not matches:
        return clause_units(words, cfg)
    result = []
    minimum_boundary_gap = float(cfg.get("voice_identity_boundary_gap_seconds", 0.25))
    for clause in clause_units(words, cfg):
        current, previous_profile, previous_tracks = [], None, set()
        for pair in clause:
            _, word = pair
            start, end = float(word["start"]), float(word["end"])
            support = _profile_support(start, end, timeline, matches, "verifier_mapped")
            profile = max(support, key=support.get) if support else None
            raw_support = _track_support(start, end, timeline, "verifier_mapped")
            tracks = {track for track, amount in raw_support.items() if amount > 0}
            identity_changed = profile and previous_profile and profile != previous_profile
            track_changed = tracks and previous_tracks and tracks != previous_tracks
            if current and (identity_changed or track_changed):
                gap = max(0.0, start - float(current[-1][1]["end"]))
                # Even a short silence is meaningful when independent tracks
                # switch identities.  Zero-gap word boundaries remain together
                # to avoid splitting on frame-level verifier jitter.
                if gap >= minimum_boundary_gap:
                    result.append(current)
                    current = []
            current.append(pair)
            if profile:
                previous_profile = profile
            if tracks:
                previous_tracks = tracks
        if current:
            result.append(current)
    return result


def phrase_identity_groups(audio, asr, cfg, timeline=None, matches=None):
    groups, metadata = {}, {}
    minimum = float(cfg.get("redimnet_phrase_min_seconds", 1.8))
    maximum = float(cfg.get("redimnet_phrase_max_seconds", 24.0))
    for number, unit in enumerate(diarization_aware_clause_units(asr.get("words", []), cfg, timeline, matches)):
        start, end = float(unit[0][1]["start"]), float(unit[-1][1]["end"])
        if end - start < minimum:
            continue
        key = "phrase_{:05d}".format(number)
        groups[key] = [{"path": Path(audio), "start": start, "end": min(end, start + maximum)}]
        verifier_support = _profile_support(start, end, timeline, matches, "verifier_mapped")
        primary_support = _profile_support(start, end, timeline, matches, "primary")
        metadata[key] = {
            "start": start, "end": end,
            "text": " ".join(str(word.get("text", "")) for _, word in unit),
            "verifier_support": verifier_support,
            "primary_support": primary_support,
        }
    return groups, metadata


def conflict_embedding_groups(audio, consensus, matches, cfg):
    """Prepare only substantial known-vs-known disagreements for pass two."""
    def known_ids(clusters):
        return {matches[value]["profile_id"] for value in clusters if value in matches}

    runs = []
    for index, item in enumerate(consensus):
        if item.get("overlap"):
            continue
        primary = list(item.get("primary", []))
        verifier = list(item.get("verifier_mapped", []))
        if not known_ids(primary) or not known_ids(verifier) or known_ids(primary) == known_ids(verifier):
            continue
        signature = (tuple(primary), tuple(verifier))
        if runs and runs[-1]["signature"] == signature and float(item["start"]) - runs[-1]["end"] <= 0.35:
            runs[-1]["end"] = float(item["end"])
            runs[-1]["indices"].append(index)
        else:
            runs.append({"signature": signature, "start": float(item["start"]), "end": float(item["end"]), "indices": [index]})
    groups, index_map = {}, {}
    minimum = float(cfg.get("redimnet_second_pass_min_seconds", 1.8))
    context = float(cfg.get("redimnet_second_pass_context_seconds", 0.35))
    for number, run in enumerate(runs):
        if run["end"] - run["start"] < minimum:
            continue
        key = "conflict_{:05d}".format(number)
        groups[key] = [{"path": Path(audio), "start": max(0.0, run["start"] - context), "end": run["end"] + context}]
        index_map[key] = run["indices"]
    return groups, index_map


def identify_speakers(output_dir, cfg, log, audio=None, asr=None, diarization=None, cache_dir=None, consensus=None):
    """Cluster-level ReDimNet ID followed by automatic per-region arbitration."""
    from scripts.speaker_identity import match_clusters, resolve_timeline, rttm_lines
    from scripts.calibration import load_calibrator

    enrolled = ensure_redimnet_enrollment(cfg, log)
    if not enrolled:
        diagnostic_decision("speaker_identity_fusion", "skipped", metrics={"profiles": 0}, reasons=["no_enrolled_voice_profiles"])
        return {"matches": {}, "message": "Добавьте голосовые образцы", "diarization": diarization}
    groups = cluster_anchor_groups(audio, diarization.get("intervals", []), cfg, consensus)
    if not groups:
        diagnostic_decision("speaker_identity_fusion", "skipped", metrics={"profiles": len(enrolled), "cluster_groups": 0}, reasons=["no_clean_anchor_segments"])
        return {"matches": {}, "message": "Нет чистых опорных фрагментов", "diarization": diarization}
    embedding_path = Path(cache_dir or output_dir) / "redimnet_cluster_embeddings.json"
    cluster_embeddings = extract_redimnet_embeddings(groups, embedding_path, log, cfg)
    threshold = float(cfg.get("redimnet_known_threshold", 0.55))
    margin = float(cfg.get("redimnet_known_margin", 0.08))
    matches, scores = match_clusters(cluster_embeddings, enrolled, threshold, margin)
    for cluster, score_report in scores.items():
        match = matches.get(cluster)
        diagnostic_decision(
            "voice_id_cluster", match.get("name") if match else "unresolved",
            candidates=[item.get("name") for item in score_report.get("scores", [])],
            metrics={"scores": score_report.get("scores", []), "best_similarity": score_report.get("best_similarity"), "margin": score_report.get("margin"), "chunks": score_report.get("chunks"), "session_prototype": score_report.get("session_prototype", False)},
            thresholds={"minimum_similarity": threshold, "minimum_margin": margin},
            reasons=["score_and_margin_passed" if match else "score_or_margin_below_threshold"], refs={"cluster": cluster, "profile_id": match.get("profile_id") if match else None},
        )
    timeline = consensus or []
    if not timeline:
        timeline = [{
            "start": item["start"], "end": item["end"], "primary": [item["speaker"]],
            "verifier_mapped": [item["speaker"]], "candidates": [item["speaker"]],
            "agreement": 1.0, "overlap": item.get("overlap", False),
        } for item in diarization.get("intervals", [])]
    phrase_groups, phrase_metadata = phrase_identity_groups(audio, asr or {"words": []}, cfg, timeline, matches)
    phrase_segments, phrase_scores = [], {}
    if phrase_groups:
        phrase_path = Path(cache_dir or output_dir) / "redimnet_phrase_embeddings.json"
        phrase_embeddings = extract_redimnet_embeddings(phrase_groups, phrase_path, log, cfg)
        phrase_matches, phrase_scores = match_clusters(
            phrase_embeddings, enrolled,
            float(cfg.get("redimnet_phrase_threshold", 0.78)),
            float(cfg.get("redimnet_phrase_margin", 0.20)),
        )
        # A short/codec-damaged phrase can have a lower absolute cosine score.
        # Accept it only when the margin is decisive and the independent Ultra
        # track supports the same enrolled identity over most of the phrase.
        corroborated_threshold = float(cfg.get("redimnet_corroborated_phrase_threshold", 0.58))
        corroborated_margin = float(cfg.get("redimnet_corroborated_phrase_margin", 0.20))
        corroborated_coverage = float(cfg.get("redimnet_corroborated_phrase_coverage", 0.72))
        for key, report_item in phrase_scores.items():
            if key in phrase_matches or not report_item.get("scores"):
                continue
            best = report_item["scores"][0]
            meta = phrase_metadata.get(key, {})
            duration = max(0.04, float(meta.get("end", 0)) - float(meta.get("start", 0)))
            coverage = float(meta.get("verifier_support", {}).get(best["profile_id"], 0.0)) / duration
            if (best["score"] >= corroborated_threshold
                    and float(report_item.get("margin", 0)) >= corroborated_margin
                    and coverage >= corroborated_coverage):
                phrase_matches[key] = {**best, "margin": report_item["margin"], "corroborated": True}
                report_item["accepted"] = True
                report_item["corroborated"] = True
                report_item["verifier_coverage"] = round(coverage, 4)
        for key, match in phrase_matches.items():
            phrase_segments.append({
                **{k: v for k, v in phrase_metadata[key].items() if not k.endswith("_support")}, "profile_id": match["profile_id"],
                "name": match["name"], "score": match["score"],
                "margin": match["margin"],
                "source": "redimnet_phrase_corroborated" if match.get("corroborated") else "redimnet_phrase",
            })
        for key, score_report in phrase_scores.items():
            match = phrase_matches.get(key)
            diagnostic_decision(
                "voice_id_phrase", match.get("name") if match else "unresolved",
                candidates=[item.get("name") for item in score_report.get("scores", [])],
                metrics={"scores": score_report.get("scores", []), "best_similarity": score_report.get("best_similarity"), "margin": score_report.get("margin"), "corroborated": score_report.get("corroborated", False), "verifier_coverage": score_report.get("verifier_coverage")},
                thresholds={"minimum_similarity": cfg.get("redimnet_phrase_threshold"), "minimum_margin": cfg.get("redimnet_phrase_margin"), "corroborated_similarity": corroborated_threshold, "corroborated_margin": corroborated_margin, "corroborated_coverage": corroborated_coverage},
                reasons=["direct_or_corroborated_threshold_passed" if match else "voice_evidence_insufficient"], refs={"phrase": key, **phrase_metadata.get(key, {})},
            )
    conflict_groups, conflict_indices = conflict_embedding_groups(audio, timeline, matches, cfg)
    local_decisions = {}
    if conflict_groups:
        conflict_path = Path(cache_dir or output_dir) / "redimnet_conflict_embeddings.json"
        conflict_embeddings = extract_redimnet_embeddings(conflict_groups, conflict_path, log, cfg)
        local_matches, local_scores = match_clusters(
            conflict_embeddings, enrolled,
            float(cfg.get("redimnet_second_pass_threshold", threshold)),
            float(cfg.get("redimnet_second_pass_margin", margin)),
        )
        for key, match in local_matches.items():
            for index in conflict_indices.get(key, []):
                item = timeline[index]
                duration = float(item["end"]) - float(item["start"])
                candidate_profiles = {
                    matches[cluster]["profile_id"]
                    for cluster in item.get("candidates", item.get("clusters", []))
                    if cluster in matches
                }
                # Embeddings are not reliable enough to overwrite a genuine
                # sub-1.5-second interjection. The exception is strong evidence
                # for a third identity while both diarizers are active: that
                # indicates a contaminated anonymous cluster, not a short turn.
                third_identity = (
                    match["profile_id"] not in candidate_profiles
                    and bool(item.get("primary")) and bool(item.get("verifier_mapped"))
                )
                if duration >= float(cfg.get("short_turn_seconds", 1.5)) or third_identity:
                    local_decisions[index] = match
    else:
        local_scores = {}
    for key, score_report in local_scores.items():
        match = local_matches.get(key) if conflict_groups else None
        diagnostic_decision(
            "voice_id_conflict_second_pass", match.get("name") if match else "unresolved",
            candidates=[item.get("name") for item in score_report.get("scores", [])],
            metrics={"scores": score_report.get("scores", []), "best_similarity": score_report.get("best_similarity"), "margin": score_report.get("margin"), "affected_intervals": conflict_indices.get(key, [])},
            thresholds={"minimum_similarity": cfg.get("redimnet_second_pass_threshold", threshold), "minimum_margin": cfg.get("redimnet_second_pass_margin", margin), "short_turn_seconds": cfg.get("short_turn_seconds", 1.5)},
            reasons=["second_pass_match" if match else "second_pass_insufficient"], refs={"conflict_region": key},
        )
    final_segments, debug = resolve_timeline(
        timeline, matches,
        short_seconds=float(cfg.get("short_turn_seconds", 1.5)),
        boundary_tolerance=float(cfg.get("boundary_tolerance_ms", 300)) / 1000.0,
        local_decisions=local_decisions,
        calibrator=load_calibrator(cfg.get("speaker_calibration_file")),
    )
    final_diarization = {
        "model": "DiariZen+Ultra+ReDimNet2 automatic fusion",
        "intervals": [{
            "start": item["start"], "end": item["end"], "speaker": item["speaker_id"],
            "confidence": item["confidence"], "confidence_level": item["confidence_level"],
            "confidence_source": item.get("confidence_source"), "calibration_bucket": item.get("calibration_bucket"),
            "overlap": item["overlap"], "known_speaker": item["known_speaker"],
            "decision": item["decision"],
        } for item in final_segments],
    }
    labels = {"profile:" + profile["id"]: profile["name"] for profile in enrolled}
    speaker_path = Path(output_dir) / "speakers.json"
    speaker_data = load_json(speaker_path) if speaker_path.exists() else {"labels": {}}
    speaker_data.setdefault("labels", {}).update(labels)
    speaker_data["label_sources"] = {key: "redimnet_profile" for key in labels}
    speaker_data["voice_matches"] = matches
    write_json(speaker_path, speaker_data)
    write_json(Path(output_dir) / "voice_segments.json", {
        "model": REDIMNET_MODEL, "segments": phrase_segments, "scores": phrase_scores,
        "threshold": float(cfg.get("redimnet_phrase_threshold", 0.78)),
        "margin": float(cfg.get("redimnet_phrase_margin", 0.20)),
    })
    report = {"model": REDIMNET_MODEL, "threshold": threshold, "margin": margin, "matches": matches, "scores": scores, "identified_phrases": len(phrase_segments), "phrase_scores": phrase_scores, "second_pass": {"regions": len(conflict_groups), "resolved_intervals": len(local_decisions), "scores": local_scores}}
    write_json(Path(output_dir) / "voice_matches.json", report)
    write_json(Path(output_dir) / "result.json", final_segments)
    write_json(Path(output_dir) / "debug.json", {"matches": report, "timeline": debug})
    (Path(output_dir) / "result.rttm").write_text("\n".join(rttm_lines(final_segments, Path(audio).stem)) + "\n", encoding="utf-8")
    report["diarization"] = final_diarization
    decision_counts = {}
    confidence_levels = {}
    for item in final_segments:
        decision_counts[item.get("decision", "unknown")] = decision_counts.get(item.get("decision", "unknown"), 0) + 1
        confidence_levels[item.get("confidence_level", "unknown")] = confidence_levels.get(item.get("confidence_level", "unknown"), 0) + 1
    diagnostic_event(
        "speaker_identity_fusion", category="decision", outcome="completed",
        inputs={"profiles": len(enrolled), "cluster_groups": len(groups), "consensus_intervals": len(timeline)},
        metrics={"cluster_matches": len(matches), "cluster_score_reports": len(scores), "identified_phrases": len(phrase_segments), "conflict_regions": len(conflict_groups), "second_pass_resolutions": len(local_decisions), "final_segments": len(final_segments), "decision_counts": decision_counts, "confidence_levels": confidence_levels},
        thresholds={"cluster_score": threshold, "cluster_margin": margin, "phrase_score": cfg.get("redimnet_phrase_threshold"), "phrase_margin": cfg.get("redimnet_phrase_margin"), "corroborated_score": cfg.get("redimnet_corroborated_phrase_threshold"), "corroborated_margin": cfg.get("redimnet_corroborated_phrase_margin"), "corroborated_coverage": cfg.get("redimnet_corroborated_phrase_coverage"), "second_pass_score": cfg.get("redimnet_second_pass_threshold", threshold), "second_pass_margin": cfg.get("redimnet_second_pass_margin", margin)},
        refs={"complete_scores": "voice_matches.json", "timeline_decisions": "debug.json", "voice_segments": "voice_segments.json"},
    )
    return report


def copy_artifacts(job_dir, output_dir):
    for name in ("diarization.rttm", "ultra.rttm", "ultra.json", "track_mapping.json", "consensus.json", "redimnet_cluster_embeddings.json"):
        source = job_dir / name
        if source.exists():
            shutil.copy2(source, output_dir / name)
    shutil.copy2(job_dir / "processing.log", output_dir / "processing.log")
    publish_diagnostics(Path(job_dir) / "diagnostics.jsonl", output_dir)


def consensus_diarization(job_dir, diarization):
    path = Path(job_dir) / "consensus.json"
    if not path.exists():
        return diarization
    intervals = []
    for segment in load_json(path).get("intervals", []):
        for speaker in segment.get("clusters", []):
            intervals.append({"start": segment["start"], "end": segment["end"], "speaker": speaker, "consensus_confidence": segment.get("confidence"), "overlap": segment.get("overlap", False)})
    return {**diarization, "model": "DiariZen+Ultra consensus", "intervals": intervals}


def process_job(job_id):
    cfg = config()
    db = connect()
    job = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not job:
        raise ValueError("Нет задания {}".format(job_id))
    source = Path(job["source_path"])
    if not source.exists():
        raise FileNotFoundError("Исходный файл больше не существует: {}".format(source))
    job_dir = Path(job["job_dir"])
    run_id = job["attempt_id"] or uuid.uuid4().hex
    diagnostic_environment.run_id = run_id
    configure_diagnostics(job_dir / "diagnostics.jsonl", component="pipeline", run_id=run_id, job_id=job_id)
    log = job_dir / "processing.log"
    audio = job_dir / "audio.wav"
    diar_json = job_dir / "diarization.json"
    rttm = job_dir / "diarization.rttm"
    ultra_json = job_dir / "ultra.json"
    ultra_rttm = job_dir / "ultra.rttm"
    mapping_json = job_dir / "track_mapping.json"
    consensus_json = job_dir / "consensus.json"
    asr_json = job_dir / "asr.json"
    duration_path = job_dir / "duration.json"
    content_sha256 = job["content_sha256"] or fingerprint(source)
    final_dir = OUTPUTS / ("{}-{}".format(safe_name(source), job["fingerprint"][:8]))
    publishing = final_dir.with_name(final_dir.name + ".publishing")
    audio_key = stage_cache_key("audio-v1", {"source": content_sha256, "track": cfg["audio_track"], "rate": 16000, "channels": 1})
    diar_key = stage_cache_key("diarizen-v2", {"audio": audio_key, "model": cfg["diarization_model"], "revision": cfg["diarization_model_revision"], "embedding_model": cfg["diarization_embedding_model"], "embedding_revision": cfg["diarization_embedding_revision"], "batch": cfg["diarization_batch_size"], "min": cfg["diarization_min_speakers"], "max": cfg["diarization_max_speakers"], "exact": job["speaker_count"]})
    ultra_key = stage_cache_key("ultra-v1", {"audio": audio_key, "model": cfg.get("ultra_model"), "revision": cfg["ultra_model_revision"], "streaming": [340, 40, 40, 300]})
    consensus_key = stage_cache_key("consensus-v2", {"diarizen": diar_key, "ultra": ultra_key, "boundary_ms": cfg.get("boundary_tolerance_ms", 300)})
    asr_key = stage_cache_key("gigaam-v1", {"audio": audio_key, "model": cfg["gigaam_model"], "language": cfg.get("language", "ru")})
    diagnostic_event(
        "job", category="stage", outcome="started",
        inputs={"original_name": job["original_name"], "content_sha256": content_sha256, "speaker_count": job["speaker_count"]},
        refs={"job_dir": str(job_dir)}, metrics={"config_sha256": _json_hash(cfg), "system": system_snapshot(job_dir)},
    )

    try:
        update_job(db, job_id, status="running", stage="validate", progress=2, detail="Проверяю запись", error=None, started_at=job["started_at"] or now(), finished_at=None)
        if not duration_path.exists():
            duration = validate_media(source)
            write_json(duration_path, {"seconds": duration})
        else:
            duration = load_json(duration_path)["seconds"]
        diagnostic_event("media_validation", category="observation", outcome="accepted", metrics={"duration_seconds": duration}, refs={"source": str(source)})
        update_job(db, job_id, progress=5)
        audio_cached = stage_cache_valid(job_dir, "audio", audio_key, ["audio.wav"])
        diagnostic_decision("stage_cache.audio", "hit" if audio_cached else "miss", metrics={"cache_key": audio_key}, refs={"artifacts": ["audio.wav"]})
        if not audio_cached:
            update_job(db, job_id, stage="extract_audio", progress=7, detail="Извлекаю звуковую дорожку")
            extract_audio(source, audio, cfg["audio_track"], log)
            mark_stage_cached(job_dir, "audio", audio_key, ["audio.wav"])
        update_job(db, job_id, progress=12)
        diar_cached = stage_cache_valid(job_dir, "diarizen", diar_key, ["diarization.json", "diarization.rttm"])
        diagnostic_decision("stage_cache.diarizen", "hit" if diar_cached else "miss", metrics={"cache_key": diar_key}, refs={"artifacts": ["diarization.json", "diarization.rttm"]})
        if not diar_cached:
            update_job(db, job_id, stage="diarization", progress=15, detail="DiariZen: определяю участников")
            env = dict(os.environ, **diagnostic_environment(job_id, job_dir, "diarizen"), PYTHONPATH=str(ROOT / "scripts"), PYTHONUNBUFFERED="1", PYTORCH_ENABLE_MPS_FALLBACK="1", PYTORCH_ALLOC_CONF="expandable_segments:True", HF_HOME=str(ROOT / "work" / "cache" / "huggingface"), MPLCONFIGDIR=str(ROOT / "work" / "cache" / "matplotlib"), PYTHONPYCACHEPREFIX=str(ROOT / "work" / "pycache"))
            def diarization_progress(line):
                markers = {
                    "Extracting segmentations.": (20, "DiariZen: анализирую участки речи"),
                    "Extracting Embeddings.": (40, "DiariZen: сравниваю голоса"),
                    "Clustering.": (52, "DiariZen: объединяю голоса по участникам"),
                }
                if line in markers:
                    progress, detail = markers[line]
                    update_job(db, job_id, progress=progress, detail=detail)
                    return
                prefix = "DIARIZEN_PROGRESS "
                if not line.startswith(prefix):
                    return
                payload = json.loads(line[len(prefix):])
                current = int(payload.get("current", 0))
                total = max(1, int(payload.get("total", 1)))
                if payload.get("step") == "segmentation":
                    progress = 20 + 20 * current / total
                    detail = "DiariZen: анализ речи, пакет {} из {}".format(current, total)
                elif payload.get("step") == "embeddings":
                    progress = 40 + 12 * current / total
                    detail = "DiariZen: сравнение голосов, пакет {} из {}".format(current, total)
                else:
                    return
                update_job(db, job_id, progress=round(progress, 1), detail=detail)
            diarization_command = [
                str(ROOT / ".venv-diarizen" / "bin" / "python"), str(ROOT / "scripts" / "diarize_worker.py"),
                "--audio", str(audio), "--output", str(diar_json), "--rttm", str(rttm),
                "--model", cfg["diarization_model"], "--cache", str(ROOT / "work" / "cache" / "huggingface" / "hub"),
                "--revision", cfg["diarization_model_revision"],
                "--embedding-model", cfg["diarization_embedding_model"],
                "--embedding-revision", cfg["diarization_embedding_revision"],
                "--device", cfg["diarization_device"],
                "--batch-size", str(cfg.get("diarization_batch_size", 8)),
                "--min-speakers", str(cfg.get("diarization_min_speakers", 1)),
                "--max-speakers", str(cfg["diarization_max_speakers"]),
            ]
            if job["speaker_count"] is not None:
                diarization_command.extend(["--num-speakers", str(job["speaker_count"])])
            run_command(diarization_command, log, env=env, progress_callback=diarization_progress)
            mark_stage_cached(job_dir, "diarizen", diar_key, ["diarization.json", "diarization.rttm"])
        update_job(db, job_id, progress=56)
        ultra_cached = stage_cache_valid(job_dir, "ultra", ultra_key, ["ultra.json", "ultra.rttm"])
        diagnostic_decision("stage_cache.ultra", "hit" if ultra_cached else "miss", metrics={"cache_key": ultra_key}, refs={"artifacts": ["ultra.json", "ultra.rttm"]})
        if not ultra_cached:
            update_job(db, job_id, stage="ultra_diarization", progress=57, detail="Ultra Sortformer: независимо проверяю участников")
            env = dict(os.environ, **diagnostic_environment(job_id, job_dir, "ultra"), PYTHONUNBUFFERED="1", HF_HOME=str(ROOT / "work" / "cache" / "ultra"), TORCH_HOME=str(ROOT / "work" / "cache" / "torch"))
            run_command([
                str(ROOT / ".venv-fusion" / "bin" / "python"), str(ROOT / "scripts" / "ultra_worker.py"),
                "--audio", str(audio), "--output", str(ultra_json), "--rttm", str(ultra_rttm),
                "--model", cfg.get("ultra_model", "mago-ai/ultra_diar_streaming_sortformer_8spk_v1"),
                "--cache", str(ROOT / "work" / "cache" / "ultra"), "--device", cfg.get("ultra_device", "auto"),
                "--revision", cfg["ultra_model_revision"],
            ], log, env=env)
            mark_stage_cached(job_dir, "ultra", ultra_key, ["ultra.json", "ultra.rttm"])
        consensus_cached = stage_cache_valid(job_dir, "consensus", consensus_key, ["consensus.json", "track_mapping.json"])
        diagnostic_decision("stage_cache.consensus", "hit" if consensus_cached else "miss", metrics={"cache_key": consensus_key}, refs={"artifacts": ["consensus.json", "track_mapping.json"]})
        if not consensus_cached:
            update_job(db, job_id, stage="consensus", progress=66, detail="Сопоставляю дорожки DiariZen и Ultra")
            run_command([
                sys.executable, str(ROOT / "scripts" / "consensus.py"),
                "--primary", str(diar_json), "--verifier", str(ultra_json),
                "--mapping", str(mapping_json), "--output", str(consensus_json),
                "--boundary-tolerance", str(cfg.get("boundary_tolerance_ms", 300) / 1000.0),
            ], log, env=dict(os.environ, **diagnostic_environment(job_id, job_dir, "consensus")))
            mark_stage_cached(job_dir, "consensus", consensus_key, ["consensus.json", "track_mapping.json"])
        asr_cached = stage_cache_valid(job_dir, "asr", asr_key, ["asr.json"])
        diagnostic_decision("stage_cache.asr", "hit" if asr_cached else "miss", metrics={"cache_key": asr_key}, refs={"artifacts": ["asr.json"]})
        if not asr_cached:
            update_job(db, job_id, stage="transcription", progress=68, detail="GigaAM: готовлю распознавание речи")
            env = dict(os.environ, **diagnostic_environment(job_id, job_dir, "asr"), PYTHONPATH=str(ROOT / "scripts"), PYTORCH_ENABLE_MPS_FALLBACK="1", HF_HOME=str(ROOT / "work" / "cache" / "huggingface"), MPLCONFIGDIR=str(ROOT / "work" / "cache" / "matplotlib"), PYTHONPYCACHEPREFIX=str(ROOT / "work" / "pycache"))
            def transcription_progress(line):
                prefix = "PIPELINE_PROGRESS "
                if not line.startswith(prefix):
                    return
                payload = json.loads(line[len(prefix):])
                current = int(payload.get("current", 0))
                total = max(1, int(payload.get("total", 1)))
                progress = 68 + 22 * current / total
                update_job(db, job_id, progress=round(progress, 1), detail="GigaAM: фрагмент {} из {}".format(current, total))
            run_command([
                str(ROOT / ".venv-gigaam" / "bin" / "python"), str(ROOT / "scripts" / "asr_worker.py"),
                "--audio", str(audio), "--output", str(asr_json), "--model", cfg["gigaam_model"],
                "--cache", str(ROOT / "work" / "cache" / "gigaam"), "--device", cfg["asr_device"],
                "--vad-threshold", str(cfg["vad_threshold"]),
                "--vad-min-speech-ms", str(cfg["vad_min_speech_ms"]),
                "--vad-min-silence-ms", str(cfg["vad_min_silence_ms"]),
                "--vad-speech-pad-ms", str(cfg["vad_speech_pad_ms"]),
                "--chunk-seconds", str(cfg["asr_chunk_seconds"]),
                "--overlap-seconds", str(cfg["asr_overlap_seconds"]),
            ], log, env=env, progress_callback=transcription_progress)
            mark_stage_cached(job_dir, "asr", asr_key, ["asr.json"])
        update_job(db, job_id, stage="export", progress=92, detail="Совмещаю текст с участниками и создаю файлы")
        if publishing.exists():
            shutil.rmtree(publishing)
        publishing.mkdir(parents=True)
        if final_dir.exists() and (final_dir / "speakers.json").exists():
            shutil.copy2(final_dir / "speakers.json", publishing / "speakers.json")
        raw_diarization = load_json(diar_json)
        fused_diarization = consensus_diarization(job_dir, raw_diarization)
        export_results(job, duration, fused_diarization, load_json(asr_json), publishing, cfg)
        if load_voice_profiles():
            update_job(db, job_id, stage="speaker_identification", progress=96, detail="ReDimNet2: определяю личности и разрешаю спорные участки")
            identity_report = identify_speakers(
                publishing,
                cfg,
                log,
                audio=audio,
                asr=load_json(asr_json),
                diarization=fused_diarization,
                cache_dir=job_dir,
                consensus=load_json(consensus_json).get("intervals", []),
            )
            fused_diarization = identity_report.get("diarization", fused_diarization)
            export_results(job, duration, fused_diarization, load_json(asr_json), publishing, cfg)
        copy_artifacts(job_dir, publishing)
        manifests = []
        for relative, producer in (("transcript.json", "export"), ("semantics/evidence_spans.json", "evidence_ledger"), ("consensus.json", "consensus")):
            artifact = publishing / relative
            if artifact.is_file():
                manifests.append(artifact_provenance(artifact, producer, {"audio_sha256": content_sha256}))
        write_json(publishing / "source.manifest.json", {
            "schema_version": 1, "source": job["original_name"], "audio_sha256": content_sha256,
            "config_sha256": _json_hash(cfg), "artifacts": manifests,
        })
        if final_dir.exists():
            archive = ROOT / "backups" / "transcript_exports"
            archive.mkdir(parents=True, exist_ok=True)
            os.replace(final_dir, archive / (final_dir.name + "-" + uuid.uuid4().hex))
        os.replace(publishing, final_dir)
        summary_values = {}
        if cfg.get("summary_enabled", True):
            summary_values = {
                "summary_status": "queued",
                "summary_stage": "summary_queued",
                "summary_progress": 0,
                "summary_detail": "Саммари ожидает запуска",
                "summary_error": None,
                "summary_started_at": None,
                "summary_finished_at": None,
            }
        update_job(db, job_id, status="done", stage="done", progress=100, detail="Готово", output_dir=str(final_dir), error=None, finished_at=now(), **summary_values)
        diagnostic_event("job", category="stage", outcome="completed", metrics={"duration_seconds": duration}, refs={"output_dir": str(final_dir)})
        publish_diagnostics(job_dir / "diagnostics.jsonl", final_dir)
        if cfg.get("notify") and sys.platform == "darwin":
            subprocess.run(["osascript", "-e", 'display notification "Расшифровка готова" with title "Meeting Transcript"'], check=False)
        print("Готово: {}".format(final_dir))
    except Exception as exc:
        diagnostic_event("job", category="stage", outcome="failed", severity="ERROR", error=exc, refs={"processing_log": str(log)})
        with log.open("a", encoding="utf-8") as stream:
            traceback.print_exc(file=stream)
        update_job(db, job_id, status="failed", detail="Ошибка обработки", error=str(exc), finished_at=now())
        raise


def run_next():
    if not config().get("processing_enabled", True):
        return False
    db = connect()
    worker_id = "{}:{}".format(os.uname().nodename, os.getpid())
    attempt_id = uuid.uuid4().hex
    lease_until = (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat(timespec="seconds")
    db.execute("BEGIN IMMEDIATE")
    job = db.execute("SELECT id FROM jobs WHERE status = 'queued' ORDER BY id LIMIT 1").fetchone()
    if not job:
        db.commit()
        return False
    claimed = db.execute(
        "UPDATE jobs SET status='running', stage='claimed', worker_id=?, attempt_id=?, lease_until=?, updated_at=? WHERE id=? AND status='queued'",
        (worker_id, attempt_id, lease_until, now(), job["id"]),
    )
    db.commit()
    if claimed.rowcount != 1:
        return False
    process_job(job["id"])
    return True


def process_summary(job_id, force=False):
    cfg = config()
    db = connect()
    job = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not job or job["status"] != "done" or not job["output_dir"]:
        raise ValueError("Сначала должна завершиться расшифровка")
    output_dir = Path(job["output_dir"])
    transcript = output_dir / "transcript.json"
    if not transcript.is_file():
        raise FileNotFoundError("Не найден transcript.json")
    job_dir = Path(job["job_dir"])
    summary_run_id = uuid.uuid4().hex
    diagnostic_environment.run_id = summary_run_id
    configure_diagnostics(job_dir / "diagnostics.jsonl", component="pipeline.summary", run_id=summary_run_id, job_id=job_id)
    log = job_dir / "summary-processing.log"
    diagnostic_event(
        "summary_job", category="stage", outcome="started",
        inputs={"force": bool(force)}, refs={"transcript": str(transcript), "output_dir": str(output_dir)},
    )
    update_job(
        db, job_id,
        summary_status="running", summary_stage="summary_prepare", summary_progress=1,
        summary_detail="Подготавливаю стенограмму", summary_error=None,
        summary_started_at=job["summary_started_at"] or now(), summary_finished_at=None,
    )

    worker_failure_detail = None

    def summary_progress(line):
        nonlocal worker_failure_detail
        prefix = "SUMMARY_PROGRESS "
        if not line.startswith(prefix):
            return
        try:
            payload = json.loads(line[len(prefix):])
        except json.JSONDecodeError:
            return
        if payload.get("stage") == "summary_failed":
            worker_failure_detail = str(payload.get("detail") or "").strip() or None
            return
        update_job(
            db, job_id,
            summary_stage=payload.get("stage", "summary_running"),
            summary_progress=float(payload.get("progress", 0)),
            summary_detail=payload.get("detail", "Создаю саммари"),
        )

    command = [
        sys.executable, str(ROOT / "scripts" / "summary_worker.py"),
        "--transcript", str(transcript),
        "--output", str(output_dir),
        "--cache", str(job_dir / "summary_cache"),
        "--config", str(CONFIG_PATH),
    ]
    if force:
        command.append("--force")
    try:
        run_command(
            command, log,
            env=dict(os.environ, **diagnostic_environment(job_id, job_dir, "summary"), PYTHONUNBUFFERED="1", PYTHONPATH=str(ROOT / "scripts")),
            progress_callback=summary_progress,
        )
        update_job(
            db, job_id,
            summary_status="done", summary_stage="summary_done", summary_progress=100,
            summary_detail="Саммари готово", summary_error=None, summary_finished_at=now(),
        )
        diagnostic_event("summary_job", category="stage", outcome="completed", refs={"summary": str(output_dir / "summary.json")})
        publish_diagnostics(job_dir / "diagnostics.jsonl", output_dir)
        shutil.copy2(log, output_dir / "summary-processing.log")
        return True
    except Exception as exc:
        diagnostic_event("summary_job", category="stage", outcome="failed", severity="ERROR", error=exc, refs={"summary_log": str(log)})
        with log.open("a", encoding="utf-8") as stream:
            traceback.print_exc(file=stream)
        update_job(
            db, job_id,
            summary_status="failed", summary_stage="summary_failed",
            summary_detail="Не удалось создать саммари",
            summary_error=worker_failure_detail or str(exc),
            summary_finished_at=now(),
        )
        publish_diagnostics(job_dir / "diagnostics.jsonl", output_dir)
        if log.is_file():
            shutil.copy2(log, output_dir / "summary-processing.log")
        return False


def run_next_summary():
    if not config().get("summary_enabled", True):
        return False
    db = connect()
    job = db.execute(
        "SELECT id, summary_status FROM jobs WHERE status = 'done' AND summary_status IN ('queued', 'queued_force') ORDER BY id LIMIT 1"
    ).fetchone()
    if not job:
        return False
    process_summary(job["id"], force=job["summary_status"] == "queued_force")
    return True


def scan_inbox(seen, cfg):
    current = time.time()
    present = set()
    for path in sorted(INBOX.iterdir()):
        if not path.is_file() or path.suffix.casefold() not in MEDIA_EXTENSIONS or path.name.endswith(".partial"):
            continue
        resolved = str(path.resolve())
        present.add(resolved)
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        previous = seen.get(resolved)
        if previous is None or previous[0] != signature:
            seen[resolved] = (signature, current, False)
            continue
        if not previous[2] and current - previous[1] >= cfg["stable_seconds"] and current - stat.st_mtime >= cfg["stable_seconds"]:
            enqueue(path)
            seen[resolved] = (signature, previous[1], True)
    for missing in set(seen) - present:
        del seen[missing]


def watch():
    cfg = config()
    seen = {}
    lock_stream = (STATE / "watcher.lock").open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Обработчик уже запущен.", flush=True)
        return
    dashboard = start_dashboard(background=True)
    db = connect()
    db.execute("UPDATE jobs SET status = 'queued', detail = 'Возобновляю после истечения lease', worker_id = NULL, attempt_id = NULL, lease_until = NULL, updated_at = ? WHERE status = 'running' AND (lease_until IS NULL OR lease_until < ?)", (now(), now()))
    db.execute("UPDATE jobs SET summary_status = 'queued', summary_detail = 'Возобновляю после перезапуска', updated_at = ? WHERE summary_status = 'running'", (now(),))
    db.commit()
    write_status_snapshot(db)
    print("Наблюдаю за {}. Для остановки нажмите Ctrl-C.".format(INBOX), flush=True)
    try:
        while True:
            try:
                scan_inbox(seen, cfg)
                if not run_next():
                    run_next_summary()
            except Exception as exc:
                print("Ошибка: {}".format(exc), file=sys.stderr, flush=True)
            time.sleep(cfg["poll_seconds"])
    finally:
        if dashboard:
            dashboard.shutdown()


def show_status():
    rows = connect().execute("SELECT id, original_name, status, stage, progress, error, output_dir FROM jobs ORDER BY id DESC LIMIT 50").fetchall()
    if not rows:
        print("Очередь пуста.")
        return
    for row in rows:
        line = "#{:<4} {:<9} {:<16} {:>5.1f}% {}".format(row["id"], row["status"], row["stage"], row["progress"], row["original_name"])
        if row["error"]:
            line += " — " + row["error"]
        print(line)


def dashboard_payload():
    rows = status_rows(connect())
    uploads = []
    for metadata_path in INBOX.glob(".*.upload.json"):
        try:
            metadata = load_json(metadata_path)
            total = max(1, int(metadata["size"]))
            partial = INBOX / metadata.get("partial_name", ("." + metadata["name"] + ".partial"))
            received = partial.stat().st_size if partial.exists() else 0
            uploads.append({
                "name": metadata["name"],
                "received": received,
                "size": total,
                "progress": round(min(99.9, 100 * received / total), 1),
                "started_at": metadata.get("started_at"),
            })
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    return {"jobs": [dict(row) for row in rows], "uploads": uploads, "inbox": str(INBOX), "outputs": str(OUTPUTS), "processing_enabled": config().get("processing_enabled", True), "updated_at": now()}


class DashboardHandler(BaseHTTPRequestHandler):
    def send_bytes(self, body, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, value, status=200):
        self.send_bytes(
            json.dumps(value, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/profiles/create":
            name = parse_qs(parsed.query).get("name", [""])[0].strip()
            if not name or len(name) > 80 or any(ord(character) < 32 for character in name):
                self.send_json({"error": "Введите имя длиной до 80 символов"}, 400)
                return
            with PROFILE_LOCK:
                if any(profile.get("name", "").casefold() == name.casefold() for profile in load_voice_profiles()):
                    self.send_json({"error": "Профиль с таким именем уже существует"}, 409)
                    return
                profile_id = uuid.uuid4().hex
                path = profile_path(profile_id)
                path.parent.mkdir(parents=True, exist_ok=False)
                profile = {"id": profile_id, "name": name, "samples": [], "created_at": now(), "updated_at": now()}
                write_json(path, profile)
            self.send_json({"ok": True, "profile": public_profile(profile)}, 201)
            return

        if parsed.path == "/api/profiles/rename":
            params = parse_qs(parsed.query)
            profile_id = params.get("id", [""])[0]
            name = params.get("name", [""])[0].strip()
            if not name or len(name) > 80 or any(ord(character) < 32 for character in name):
                self.send_json({"error": "Введите имя длиной до 80 символов"}, 400)
                return
            try:
                path = profile_path(profile_id)
            except ValueError as exc:
                self.send_json({"error": str(exc)}, 404)
                return
            with PROFILE_LOCK:
                if not path.is_file():
                    self.send_json({"error": "Профиль не найден"}, 404)
                    return
                if any(
                    profile.get("id") != profile_id and profile.get("name", "").casefold() == name.casefold()
                    for profile in load_voice_profiles()
                ):
                    self.send_json({"error": "Профиль с таким именем уже существует"}, 409)
                    return
                profile = load_json(path)
                profile["name"] = name
                profile["updated_at"] = now()
                write_json(path, profile)
            updated_exports = propagate_profile_name(profile_id, name)
            self.send_json({"ok": True, "profile": public_profile(profile), "updated_exports": updated_exports})
            return

        if parsed.path == "/api/profiles/delete":
            profile_id = parse_qs(parsed.query).get("id", [""])[0]
            try:
                path = profile_path(profile_id)
            except ValueError as exc:
                self.send_json({"error": str(exc)}, 404)
                return
            with PROFILE_LOCK:
                if not path.is_file():
                    self.send_json({"error": "Профиль не найден"}, 404)
                    return
                trash = VOICE_PROFILES / ".trash"
                trash.mkdir(parents=True, exist_ok=True)
                destination = trash / (profile_id + "-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
                if destination.exists():
                    destination = trash / (profile_id + "-" + uuid.uuid4().hex[:8])
                os.replace(path.parent, destination)
            self.send_json({"ok": True, "recoverable": True})
            return

        if parsed.path == "/api/profiles/sample":
            params = parse_qs(parsed.query)
            profile_id = params.get("id", [""])[0]
            raw_name = params.get("name", [""])[0].strip()
            try:
                path = profile_path(profile_id)
            except ValueError as exc:
                self.send_json({"error": str(exc)}, 404)
                return
            if not path.is_file():
                self.send_json({"error": "Профиль не найден"}, 404)
                return
            name = Path(raw_name).name
            if not name or name != raw_name or Path(name).suffix.casefold() not in MEDIA_EXTENSIONS:
                self.send_json({"error": "Выберите поддерживаемый видео- или аудиофайл"}, 400)
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                length = 0
            if length <= 0 or length > int(config().get("max_profile_upload_bytes", 4 * 1024**3)):
                self.send_json({"error": "Некорректный размер файла"}, 400)
                return

            sample_id = uuid.uuid4().hex
            directory = path.parent
            partial = directory / ("." + sample_id + Path(name).suffix + ".partial")
            audio = directory / (sample_id + ".wav")
            embedding_path = directory / ("." + sample_id + ".redimnet.json")
            log = directory / "processing.log"
            remaining = length
            try:
                with partial.open("xb") as stream:
                    while remaining:
                        block = self.rfile.read(min(8 * 1024 * 1024, remaining))
                        if not block:
                            raise ConnectionError("Передача файла прервалась")
                        stream.write(block)
                        remaining -= len(block)
                    stream.flush()
                    os.fsync(stream.fileno())
                duration = validate_media(partial)
                if duration < 3:
                    raise ValueError("Образец должен быть длиннее 3 секунд")
                max_duration = float(config().get("max_profile_sample_seconds", 600))
                if duration > max_duration:
                    raise ValueError("Один образец должен быть короче {} минут".format(int(max_duration // 60)))
                extract_audio(partial, audio, 0, log)
                validate_media(audio)
                groups = extract_redimnet_embeddings({"sample": [audio]}, embedding_path, log, config())
                result = groups.get("sample")
                if not result or not result.get("embedding"):
                    raise ValueError("В образце не найдено достаточно голоса")
                sample = {
                    "id": sample_id,
                    "name": name,
                    "audio": audio.name,
                    "duration_seconds": round(duration, 1),
                    "redimnet_embedding": result["embedding"],
                    "redimnet_references": result.get("references", []),
                    "embedding_model": REDIMNET_MODEL,
                    "embedding_chunks": result.get("chunks", 0),
                    "created_at": now(),
                }
                with PROFILE_LOCK:
                    profile = load_json(path)
                    profile.setdefault("samples", []).append(sample)
                    profile["updated_at"] = now()
                    write_json(path, profile)
                self.send_json({"ok": True, "profile": public_profile(profile)}, 201)
            except Exception as exc:
                audio.unlink(missing_ok=True)
                self.send_json({"error": str(exc) or "Не удалось обработать образец"}, 500)
            finally:
                partial.unlink(missing_ok=True)
                embedding_path.unlink(missing_ok=True)
            return

        if parsed.path == "/api/profiles/sample/delete":
            params = parse_qs(parsed.query)
            profile_id = params.get("id", [""])[0]
            sample_id = params.get("sample", [""])[0]
            try:
                path = profile_path(profile_id)
            except ValueError as exc:
                self.send_json({"error": str(exc)}, 404)
                return
            if any(character not in "0123456789abcdef" for character in sample_id) or len(sample_id) != 32:
                self.send_json({"error": "Образец не найден"}, 404)
                return
            with PROFILE_LOCK:
                if not path.is_file():
                    self.send_json({"error": "Профиль не найден"}, 404)
                    return
                profile = load_json(path)
                sample = next((item for item in profile.get("samples", []) if item.get("id") == sample_id), None)
                if not sample:
                    self.send_json({"error": "Образец не найден"}, 404)
                    return
                profile["samples"] = [item for item in profile.get("samples", []) if item.get("id") != sample_id]
                profile["updated_at"] = now()
                write_json(path, profile)
                (path.parent / sample.get("audio", "missing")).unlink(missing_ok=True)
            self.send_json({"ok": True, "profile": public_profile(profile)})
            return

        if parsed.path == "/api/apply-profiles":
            try:
                job_id = int(parse_qs(parsed.query).get("id", [""])[0])
            except ValueError:
                self.send_json({"error": "Неверный номер записи"}, 400)
                return
            db = connect()
            job = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if not job or job["status"] != "done" or not job["output_dir"]:
                self.send_json({"error": "Сначала дождитесь окончания расшифровки"}, 409)
                return
            try:
                job_dir = Path(job["job_dir"])
                report = identify_speakers(
                    Path(job["output_dir"]),
                    config(),
                    job_dir / "processing.log",
                    audio=job_dir / "audio.wav",
                    asr=load_json(job_dir / "asr.json"),
                    diarization=consensus_diarization(job_dir, load_json(job_dir / "diarization.json")),
                    cache_dir=job_dir,
                    consensus=load_json(job_dir / "consensus.json").get("intervals", []) if (job_dir / "consensus.json").exists() else None,
                )
                rename_export(job_id, report.get("diarization"))
                update_job(
                    db, job_id,
                    summary_status="queued_force", summary_stage="summary_queued", summary_progress=0,
                    summary_detail="Имена обновлены — пересоберу саммари", summary_error=None,
                    summary_started_at=None, summary_finished_at=None,
                )
                self.send_json({"ok": True, **report})
            except Exception as exc:
                self.send_json({"error": str(exc) or "Не удалось сопоставить голоса"}, 500)
            return

        if parsed.path == "/api/speakers":
            params = parse_qs(parsed.query)
            try:
                job_id = int(params.get("id", [""])[0])
                speaker_count = normalize_speaker_count(params.get("count", ["auto"])[0])
            except (TypeError, ValueError) as exc:
                self.send_json({"error": str(exc) or "Неверные параметры"}, 400)
                return
            db = connect()
            job = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if not job:
                self.send_json({"error": "Запись не найдена"}, 404)
                return
            if job["status"] == "running":
                self.send_json({"error": "Дождитесь завершения текущей обработки"}, 409)
                return
            job_dir = Path(job["job_dir"])
            for name in ("diarization.json", "diarization.rttm"):
                (job_dir / name).unlink(missing_ok=True)
            detail = (
                "В очереди: определю {} участников".format(speaker_count)
                if speaker_count is not None
                else "В очереди: определю число участников автоматически"
            )
            update_job(
                db,
                job_id,
                speaker_count=speaker_count,
                status="queued",
                stage="queued",
                progress=12,
                detail=detail,
                error=None,
                started_at=None,
                finished_at=None,
                summary_status="waiting",
                summary_stage="summary_waiting",
                summary_progress=0,
                summary_detail="Ожидает новой расшифровки",
                summary_error=None,
                summary_started_at=None,
                summary_finished_at=None,
            )
            self.send_json({"ok": True, "job_id": job_id, "speaker_count": speaker_count})
            return

        if parsed.path == "/api/summary":
            params = parse_qs(parsed.query)
            try:
                job_id = int(params.get("id", [""])[0])
            except ValueError:
                self.send_json({"error": "Неверный номер записи"}, 400)
                return
            db = connect()
            job = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if not job or job["status"] != "done" or not job["output_dir"]:
                self.send_json({"error": "Сначала дождитесь окончания расшифровки"}, 409)
                return
            if job["summary_status"] == "running":
                self.send_json({"error": "Саммари уже создаётся"}, 409)
                return
            update_job(
                db, job_id,
                summary_status="queued_force", summary_stage="summary_queued", summary_progress=0,
                summary_detail="Саммари поставлено в очередь", summary_error=None,
                summary_started_at=None, summary_finished_at=None,
            )
            self.send_json({"ok": True, "job_id": job_id, "summary_status": "queued_force"})
            return

        if parsed.path != "/api/upload":
            self.send_json({"error": "Адрес не найден"}, 404)
            return

        params = parse_qs(parsed.query)
        raw_name = params.get("name", [""])[0].strip()
        name = Path(raw_name).name
        try:
            speaker_count = normalize_speaker_count(params.get("speakers", ["auto"])[0])
        except ValueError as exc:
            self.send_json({"error": str(exc)}, 400)
            return
        if (
            not name
            or name != raw_name
            or any(ord(character) < 32 for character in name)
            or Path(name).suffix.casefold() not in MEDIA_EXTENSIONS
        ):
            self.send_json({"error": "Выберите поддерживаемый видео- или аудиофайл"}, 400)
            return

        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            length = 0
        max_upload_bytes = int(config().get("max_upload_bytes", 200 * 1024**3))
        if length <= 0:
            self.send_json({"error": "Браузер не передал размер файла"}, 411)
            return
        if length > max_upload_bytes:
            self.send_json({"error": "Файл превышает разрешённый размер"}, 413)
            return

        INBOX.mkdir(parents=True, exist_ok=True)
        destination = INBOX / name
        if destination.exists():
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            destination = INBOX / ("{}-{}{}".format(destination.stem, stamp, destination.suffix))
            counter = 2
            while destination.exists():
                destination = INBOX / ("{}-{}-{}{}".format(Path(name).stem, stamp, counter, Path(name).suffix))
                counter += 1

        token = uuid.uuid4().hex
        partial = INBOX / (".web-upload-{}.partial".format(token))
        metadata_path = INBOX / (".web-upload-{}.upload.json".format(token))
        write_json(
            metadata_path,
            {
                "name": destination.name,
                "partial_name": partial.name,
                "size": length,
                "started_at": now(),
                "source": "web",
            },
        )

        remaining = length
        try:
            with partial.open("xb") as stream:
                while remaining:
                    block = self.rfile.read(min(8 * 1024 * 1024, remaining))
                    if not block:
                        raise ConnectionError("Передача файла прервалась")
                    stream.write(block)
                    remaining -= len(block)
                stream.flush()
                os.fsync(stream.fileno())
            fp = fingerprint(partial)
            db = connect()
            existing = find_existing_job(db, fp, name)
            if existing:
                duplicates = INBOX / "duplicates"
                duplicates.mkdir(parents=True, exist_ok=True)
                archived = duplicates / destination.name
                if archived.exists():
                    archived = duplicates / ("{}-{}{}".format(
                        archived.stem, token[:8], archived.suffix
                    ))
                os.replace(partial, archived)
                metadata_path.unlink(missing_ok=True)
                result_url = "/result?id={}".format(existing["id"]) if existing["status"] == "done" and existing["output_dir"] else None
                self.send_json(
                    {
                        "ok": True,
                        "duplicate": True,
                        "job_id": existing["id"],
                        "status": existing["status"],
                        "result_url": result_url,
                    },
                    200,
                )
                return

            os.replace(partial, destination)
            job_id = enqueue(destination, known_fingerprint=fp, speaker_count=speaker_count, original_name=name)
            metadata_path.unlink(missing_ok=True)
            self.send_json(
                {"ok": True, "duplicate": False, "name": destination.name, "job_id": job_id},
                201,
            )
        except Exception as exc:
            partial.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            self.send_json({"error": str(exc) or "Не удалось сохранить файл"}, 500)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self.send_bytes(b'{"status":"ok"}', "application/json; charset=utf-8")
            return
        if parsed.path == "/api/status":
            self.send_json(dashboard_payload())
            return
        if parsed.path == "/api/summary-test":
            if SUMMARY_STATUS_PATH.is_file():
                try:
                    self.send_json(load_json(SUMMARY_STATUS_PATH))
                except Exception as exc:
                    self.send_json({"status": "failed", "detail": str(exc)}, 500)
            else:
                self.send_json({"status": "idle", "progress": 0, "phase": "Ожидание", "detail": "Тест ещё не запущен"})
            return
        if parsed.path == "/summary-download":
            run = parse_qs(parsed.query).get("run", [""])[0]
            if run not in {"qwen35", "qwen38"}:
                self.send_bytes(b"Not found", "text/plain", 404)
                return
            target = SUMMARY_BENCHMARK / run / "summary.md"
            if not target.is_file():
                self.send_bytes(b"Not ready", "text/plain", 404)
                return
            body = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/markdown; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="summary-{}.md"'.format(run))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path in ("/summary-test", "/summary-test.html"):
            page = r'''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Тест саммари</title><style>
:root{color-scheme:dark;font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display",sans-serif}*{box-sizing:border-box}body{margin:0;min-height:100vh;background:#0c0e13;color:#f5f7fb}main{width:min(820px,calc(100% - 32px));margin:40px auto}.back{color:#b8d9ff;text-decoration:none;font-weight:650}h1{font-size:30px;margin:26px 0 8px}.lead{color:#929db0;line-height:1.5;margin-bottom:24px}.card{background:#171a22;border:1px solid #292e3b;border-radius:20px;padding:26px;box-shadow:0 16px 50px #0005}.top{display:flex;align-items:flex-start;justify-content:space-between;gap:18px}.model{font-size:21px;font-weight:700}.phase{margin-top:6px;color:#cbd2df}.percent{font-size:34px;font-weight:780;color:#a9d2ff;font-variant-numeric:tabular-nums}.track{height:18px;margin:24px 0 16px;border-radius:99px;overflow:hidden;background:#292e3b}.bar{height:100%;width:0;border-radius:inherit;background:linear-gradient(90deg,#377dff,#72d5ff);transition:width .5s ease;position:relative}.bar.running:after{content:"";position:absolute;inset:0;background:linear-gradient(100deg,transparent 25%,#fff5 50%,transparent 75%);animation:shine 1.5s infinite}@keyframes shine{from{transform:translateX(-100%)}to{transform:translateX(100%)}}.detail{font-size:16px;line-height:1.45}.meta{margin-top:10px;color:#858fa2;font-size:14px}.ready{color:#62d69a}.failed{color:#ff7d86}.actions{display:flex;gap:10px;flex-wrap:wrap;margin-top:22px}.button{display:inline-flex;padding:10px 14px;border-radius:10px;background:#2d75e8;color:white;text-decoration:none;font-weight:650}.button.secondary{background:#252b37;color:#dce8f8;border:1px solid #394354}.hidden{display:none}@media(max-width:520px){main{margin:24px auto}.card{padding:20px}.percent{font-size:28px}}</style></head><body><main><a class="back" href="/">← К расшифровкам</a><h1>Тест саммари</h1><p class="lead">Обе модели получают одну и ту же проверенную базу фактов из новой Linux-расшифровки.</p><section class="card"><div class="top"><div><div class="model" id="model">Подготовка…</div><div class="phase" id="phase">Загружаю состояние</div></div><div class="percent" id="percent">0%</div></div><div class="track"><div class="bar running" id="bar"></div></div><div class="detail" id="detail">Соединяюсь с сервером</div><div class="meta" id="meta"></div><div class="actions"><a class="button secondary hidden" id="q35" href="/summary-download?run=qwen35">Скачать Qwen 3.5</a><a class="button hidden" id="q38" href="/summary-download?run=qwen38">Скачать Qwen 3.8</a></div></section></main><script>
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));function duration(n){n=Math.max(0,Number(n)||0);return `${Math.floor(n/60)} мин ${Math.round(n%60)} сек`}async function refresh(){try{const d=await fetch('/api/summary-test',{cache:'no-store'}).then(r=>r.json()),p=Math.max(0,Math.min(100,Number(d.progress)||0));document.querySelector('#model').textContent=d.model||'Тест саммари';document.querySelector('#phase').textContent=d.phase||d.status;document.querySelector('#percent').textContent=Math.round(p)+'%';const b=document.querySelector('#bar');b.style.width=p+'%';b.classList.toggle('running',d.status==='running');document.querySelector('#detail').textContent=d.detail||'';document.querySelector('#meta').textContent=`Прошло: ${duration(d.elapsed_seconds)}${d.generated_tokens?' · получено фрагментов: '+d.generated_tokens:''}`;document.querySelector('#percent').className='percent '+(d.status==='done'?'ready':d.status==='failed'?'failed':'');document.querySelector('#q35').classList.toggle('hidden',!(d.run==='qwen35'&&d.status==='done')&&d.run!=='qwen38');document.querySelector('#q38').classList.toggle('hidden',!(d.run==='qwen38'&&d.status==='done'))}catch(e){document.querySelector('#detail').textContent='Нет связи с сервером'}}refresh();setInterval(refresh,2000)</script></body></html>'''
            self.send_bytes(page.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/profiles":
            self.send_json({"profiles": [public_profile(profile) for profile in load_voice_profiles()]})
            return
        if parsed.path == "/api/profiles/audio":
            params = parse_qs(parsed.query)
            try:
                path = profile_path(params.get("id", [""])[0])
            except ValueError:
                self.send_bytes(b"Not found", "text/plain", 404)
                return
            sample_id = params.get("sample", [""])[0]
            profile = load_json(path) if path.is_file() else {}
            sample = next((item for item in profile.get("samples", []) if item.get("id") == sample_id), None)
            audio = path.parent / sample.get("audio", "missing") if sample else None
            if not audio or not audio.is_file():
                self.send_bytes(b"Not found", "text/plain", 404)
                return
            self.send_bytes(audio.read_bytes(), "audio/wav")
            return
        if parsed.path == "/open-output":
            job_id = parse_qs(parsed.query).get("id", [""])[0]
            row = connect().execute("SELECT output_dir FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if sys.platform == "darwin" and row and row["output_dir"] and Path(row["output_dir"]).is_dir():
                subprocess.Popen(["/usr/bin/open", row["output_dir"]])
                self.send_bytes(b"OK", "text/plain")
            else:
                self.send_bytes(b"Not found", "text/plain", 404)
            return
        if parsed.path == "/download":
            job_id = parse_qs(parsed.query).get("id", [""])[0]
            name = parse_qs(parsed.query).get("file", [""])[0]
            allowed = {"transcript.md", "transcript.txt", "subtitles.srt", "transcript.json", "diarization.rttm", "result.json", "result.rttm", "debug.json", "review.csv", "speakers.json", "processing.log", "summary-processing.log", "summary.md", "summary.json", "summary_audit.json", "semantic_records.json", "tasks.json", "run_manifest.json", "diagnostics.jsonl", "diagnostics.trace.jsonl", "diagnostics_summary.json", "public_items.json", "publication_audit.json", "summary_plan.json"}
            row = connect().execute("SELECT output_dir FROM jobs WHERE id = ?", (job_id,)).fetchone()
            target = Path(row["output_dir"]) / name if row and row["output_dir"] and name in allowed else None
            if not target or not target.is_file():
                self.send_bytes(b"Not found", "text/plain", 404)
                return
            body = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
            self.send_header("Content-Disposition", 'attachment; filename="{}"'.format(target.name))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/summary":
            job_id = parse_qs(parsed.query).get("id", [""])[0]
            row = connect().execute(
                "SELECT original_name, output_dir, summary_status, summary_progress, summary_detail, summary_error FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            if not row:
                self.send_bytes(b"Not found", "text/plain", 404)
                return
            body_path = Path(row["output_dir"]) / "summary.html" if row["output_dir"] else None
            ready = row["summary_status"] == "done" and body_path and body_path.is_file()
            content = body_path.read_text(encoding="utf-8") if ready else ""
            title = html.escape(row["original_name"])
            status = html.escape(row["summary_detail"] or "Саммари ещё не создано")
            error = html.escape(row["summary_error"] or "")
            download_specs = [
                ("summary.md", "Markdown", True),
                ("summary.json", "JSON", False),
                ("tasks.json", "Задачи", False),
                ("semantic_records.json", "Тезисы", False),
                ("summary_audit.json", "Аудит", False),
                ("publication_audit.json", "Аудит публикации", False),
                ("public_items.json", "Public items", False),
                ("diagnostics.jsonl", "Подробная диагностика", False),
                ("diagnostics.trace.jsonl", "Per-item trace", False),
                ("diagnostics_summary.json", "Сводка диагностики", False),
                ("run_manifest.json", "Manifest", False),
            ]
            downloads = ""
            if ready:
                downloads = "".join(
                    '<a{} href="/download?id={}&file={}">{}</a>'.format(
                        ' class="primary"' if primary else "", job_id, name, label
                    )
                    for name, label, primary in download_specs
                    if Path(row["output_dir"], name).is_file()
                )
            page = """<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Саммари — {title}</title><style>:root{{color-scheme:dark;font-family:-apple-system,BlinkMacSystemFont,"SF Pro Text",sans-serif}}*{{box-sizing:border-box}}body{{margin:0;background:#0c0e13;color:#eef1f7}}main{{width:min(980px,calc(100% - 32px));margin:32px auto 64px}}nav{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:18px}}a,button{{display:inline-flex;align-items:center;padding:10px 13px;border:0;border-radius:10px;background:#252b37;color:#d8e9ff;text-decoration:none;font:650 14px/1.2 -apple-system,BlinkMacSystemFont,sans-serif;cursor:pointer}}button.primary,a.primary{{background:#2d75e8;color:white}}article,.status{{background:#171a22;border:1px solid #292e3b;border-radius:20px;padding:24px;box-shadow:0 16px 50px #0005}}h1{{font-size:28px}}h2{{margin-top:34px;font-size:21px}}h3{{margin-top:25px;font-size:17px}}p,li{{font-size:17px;line-height:1.6}}li{{margin:8px 0}}.track{{height:16px;background:#292e3b;border-radius:99px;overflow:hidden;margin:18px 0}}.bar{{height:100%;background:linear-gradient(90deg,#377dff,#72d5ff);transition:width .4s}}.muted{{color:#8f99aa}}.error{{color:#ff8f98}}button:disabled{{opacity:.5;cursor:wait}}</style></head><body><main><nav><a href="/">← К записям</a><a href="/result?id={job_id}">Расшифровка</a>{downloads}<button class="primary" id="rerun" type="button">Создать заново</button></nav><div id="state" class="status" style="display:{state_display}"><strong id="statusText">{status}</strong><div class="track"><div class="bar" id="bar" style="width:{progress}%"></div></div><div class="muted" id="percent">{progress}%</div><div class="error">{error}</div></div><article id="content" style="display:{content_display}">{content}</article></main><script>const id={job_id};const button=document.querySelector('#rerun');button.addEventListener('click',async()=>{{button.disabled=true;const r=await fetch('/api/summary?id='+id,{{method:'POST'}}),d=await r.json();if(!r.ok){{button.disabled=false;alert(d.error||'Ошибка')}}else location.reload()}});async function refresh(){{const d=await fetch('/api/status',{{cache:'no-store'}}).then(r=>r.json()),j=(d.jobs||[]).find(x=>x.id===id);if(!j)return;const p=Math.round(Number(j.summary_progress)||0);document.querySelector('#statusText').textContent=j.summary_detail||j.summary_status;document.querySelector('#bar').style.width=p+'%';document.querySelector('#percent').textContent=p+'%';if(j.summary_status==='done'&&document.querySelector('#content').style.display==='none')location.reload()}}setInterval(refresh,2000)</script></body></html>""".format(
                title=title, job_id=int(job_id), content=content,
                status=status, error=error, progress=round(float(row["summary_progress"] or 0)),
                state_display="none" if ready else "block", content_display="block" if ready else "none",
                downloads=downloads,
            )
            self.send_bytes(page.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/result":
            job_id = parse_qs(parsed.query).get("id", [""])[0]
            row = connect().execute("SELECT original_name, output_dir, summary_status FROM jobs WHERE id = ?", (job_id,)).fetchone()
            transcript = Path(row["output_dir"]) / "transcript.json" if row and row["output_dir"] else None
            if not transcript or not transcript.is_file():
                self.send_bytes(b"Not found", "text/plain", 404)
                return
            title = html.escape(row["original_name"])
            transcript_data = load_json(transcript)
            labels = transcript_data.get("speakers", {})
            rendered_turns = []
            for item in transcript_data.get("utterances", []):
                speaker = item.get("speaker")
                label = html.escape(speaker_display(speaker, labels))
                text = html.escape(item.get("text", ""))
                timecode = timestamp(float(item.get("start", 0)))
                needs_review = turn_needs_speaker_review(item, labels)
                speaker_class = "unknown" if speaker is None else "speaker-{}".format(sum(ord(character) for character in str(speaker)) % 6)
                warning = '<span class="warning">Проверить</span>' if needs_review else ""
                start_seconds = float(item.get("start", 0))
                anchor = "t-{}".format(round(start_seconds * 1000))
                rendered_turns.append(
                    '<section id="{}" data-start="{:.3f}" class="turn{}"><time>{}</time><div class="speech"><div class="who"><span class="speaker {}">{}</span>{}</div><div class="text">{}</div></div></section>'.format(
                        anchor,
                        start_seconds,
                        " needs-review" if needs_review else "",
                        timecode,
                        speaker_class,
                        label,
                        warning,
                        text,
                    )
                )
            content = "".join(rendered_turns) or '<div class="empty">В расшифровке пока нет реплик.</div>'
            links = " ".join('<a href="/download?id={}&file={}">{}</a>'.format(job_id, name, label) for name, label in (("transcript.md", "Markdown"), ("transcript.txt", "TXT"), ("subtitles.srt", "SRT"), ("transcript.json", "JSON")))
            diagnostics_path = Path(row["output_dir"]) / "diagnostics.jsonl"
            if diagnostics_path.is_file():
                links += ' <a href="/download?id={}&file=diagnostics.jsonl">Диагностика</a>'.format(job_id)
            links += ' <a href="/summary?id={}">{}</a>'.format(job_id, "Саммари" if row["summary_status"] == "done" else "Прогресс саммари")
            page = """<!doctype html><html lang=\"ru\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>{title}</title><style>:root{{color-scheme:dark;font-family:-apple-system,BlinkMacSystemFont,\"SF Pro Text\",sans-serif}}*{{box-sizing:border-box}}body{{margin:0;background:#0c0e13;color:#f5f7fb}}main{{width:min(980px,calc(100% - 32px));margin:32px auto 64px}}nav{{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:20px}}a,button{{display:inline-flex;align-items:center;padding:9px 12px;border:0;border-radius:10px;background:#202530;color:#b8d9ff;text-decoration:none;font:600 14px/1.2 -apple-system,BlinkMacSystemFont,\"SF Pro Text\",sans-serif;cursor:pointer}}a:hover,button:hover{{background:#293140;color:white}}button.primary{{margin-left:auto;background:#2d75e8;color:white}}button:disabled{{opacity:.55;cursor:wait}}article{{background:#171a22;border:1px solid #292e3b;border-radius:20px;padding:8px 24px 18px;box-shadow:0 16px 50px #0005}}h1{{font-size:24px;line-height:1.25;margin:20px 0 22px;overflow-wrap:anywhere}}.turn{{display:grid;grid-template-columns:112px minmax(0,1fr);gap:16px;padding:18px 0;border-top:1px solid #292e3b}}.turn:first-of-type{{border-top:0}}time{{color:#8490a4;font:14px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;font-variant-numeric:tabular-nums}}.speech{{min-width:0}}.who{{display:flex;align-items:center;gap:8px;min-height:26px;margin-bottom:7px}}.speaker{{display:inline-flex;padding:4px 9px;border-radius:999px;font-size:13px;font-weight:700;color:#eaf3ff;background:#285b91}}.speaker-1{{background:#6650a4}}.speaker-2{{background:#7a4c22}}.speaker-3{{background:#246558}}.speaker-4{{background:#75435e}}.speaker-5{{background:#4f5f79}}.speaker.unknown{{background:#713b42;color:#ffe9ec}}.warning{{font-size:12px;color:#ffc1c7}}.text{{font-size:18px;line-height:1.58;color:#eef1f7;overflow-wrap:anywhere;word-break:normal}}.needs-review{{background:linear-gradient(90deg,#ff65720d,transparent 70%)}}.empty{{padding:20px;color:#8993a5}}@media(max-width:650px){{main{{margin-top:18px}}article{{padding:6px 18px 14px}}button.primary{{margin-left:0}}.turn{{grid-template-columns:1fr;gap:5px;padding:16px 0}}time{{font-size:13px}}.text{{font-size:17px}}}}</style></head><body><main><nav><a href=\"/\">← К списку записей</a><a href=\"/profiles\">Голосовые профили</a>{links}<button class=\"primary\" id=\"applyProfiles\" type=\"button\">Определить имена</button></nav><article><h1>{title}</h1>{content}</article></main><script>const b=document.querySelector('#applyProfiles');b.addEventListener('click',async()=>{{b.disabled=true;b.textContent='Сопоставляю голоса…';try{{const r=await fetch('/api/apply-profiles?id={job_id}',{{method:'POST'}}),d=await r.json();if(!r.ok)throw new Error(d.error||'Не удалось определить имена');const count=Object.keys(d.matches||{{}}).length;if(count)location.reload();else{{b.textContent=d.message||'Совпадений нет';setTimeout(()=>{{b.disabled=false;b.textContent='Определить имена'}},3000)}}}}catch(e){{b.textContent=e.message;setTimeout(()=>{{b.disabled=false;b.textContent='Определить имена'}},3500)}}}});</script></body></html>""".format(title=title, content=content, links=links, job_id=job_id)
            self.send_bytes(page.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path in ("/", "/index.html"):
            self.send_bytes(DASHBOARD_PATH.read_bytes(), "text/html; charset=utf-8")
            return
        if parsed.path in ("/profiles", "/profiles.html"):
            self.send_bytes(PROFILES_PATH.read_bytes(), "text/html; charset=utf-8")
            return
        self.send_bytes(b"Not found", "text/plain", 404)

    def log_message(self, format, *args):
        return


def start_dashboard(background=False):
    port = int(config().get("dashboard_port", 8765))
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), DashboardHandler)
    except OSError:
        if background:
            return None
        raise
    if background:
        threading.Thread(target=server.serve_forever, name="meeting-dashboard", daemon=True).start()
    else:
        print("Прогресс: http://127.0.0.1:{}".format(port), flush=True)
        server.serve_forever()
    return server


def open_dashboard():
    port = int(config().get("dashboard_port", 8765))
    subprocess.Popen(["/usr/bin/open", "http://127.0.0.1:{}".format(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def rename_export(job_id, diarization=None):
    db = connect()
    job = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not job or not job["output_dir"]:
        raise ValueError("У задания ещё нет готового результата")
    job_dir = Path(job["job_dir"])
    output_dir = Path(job["output_dir"])
    if diarization is None:
        result_path = output_dir / "result.json"
        if result_path.is_file():
            diarization = {
                "model": "DiariZen+Ultra+ReDimNet2 automatic fusion",
                "intervals": [
                    {"start": item["start"], "end": item["end"], "speaker": item["speaker_id"],
                     "confidence": item.get("confidence"), "overlap": item.get("overlap", False)}
                    for item in load_json(result_path)
                ],
            }
        else:
            diarization = consensus_diarization(job_dir, load_json(job_dir / "diarization.json"))
    export_results(job, load_json(job_dir / "duration.json")["seconds"], diarization, load_json(job_dir / "asr.json"), output_dir, config())
    if config().get("summary_enabled", True):
        update_job(db, job_id, summary_status="queued", summary_stage="summary_queued",
                   summary_progress=0, summary_detail="Стенограмма изменена; требуется новое саммари",
                   summary_error=None, summary_started_at=None, summary_finished_at=None)
    print("Имена применены: {}".format(output_dir))


def doctor():
    checks = [
        ("FFmpeg", shutil.which("ffmpeg")),
        ("FFprobe", shutil.which("ffprobe")),
        ("Python DiariZen", ROOT / ".venv-diarizen" / "bin" / "python"),
        ("Python GigaAM", ROOT / ".venv-gigaam" / "bin" / "python"),
        ("Python Ultra/ReDimNet", ROOT / ".venv-fusion" / "bin" / "python"),
    ]
    okay = True
    for label, value in checks:
        exists = bool(value) and Path(value).exists()
        okay &= exists
        print("{} {}: {}".format("✓" if exists else "✗", label, value or "не найден"))
    print("✓ inbox: {}".format(INBOX))
    print("✓ outputs: {}".format(OUTPUTS))
    return 0 if okay else 1


def main():
    for directory in (INBOX, STATE, JOBS, OUTPUTS):
        directory.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(description="Локальная расшифровка встреч")
    sub = parser.add_subparsers(dest="command", required=True)
    process_parser = sub.add_parser("process", help="добавить файл и сразу обработать")
    process_parser.add_argument("file")
    retry_parser = sub.add_parser("retry", help="повторить задание с незавершённого этапа")
    retry_parser.add_argument("job_id", type=int)
    rename_parser = sub.add_parser("rename", help="переэкспортировать после изменения speakers.json")
    rename_parser.add_argument("job_id", type=int)
    sub.add_parser("watch", help="наблюдать за inbox")
    sub.add_parser("once", help="обработать следующее задание")
    sub.add_parser("status", help="показать очередь")
    sub.add_parser("dashboard", help="показать страницу прогресса")
    sub.add_parser("doctor", help="проверить установку")
    args = parser.parse_args()

    if args.command == "process":
        process_job(enqueue(args.file))
    elif args.command == "retry":
        process_job(args.job_id)
    elif args.command == "rename":
        rename_export(args.job_id)
    elif args.command == "watch":
        watch()
    elif args.command == "once":
        run_next()
    elif args.command == "status":
        show_status()
    elif args.command == "dashboard":
        start_dashboard()
    elif args.command == "doctor":
        return doctor()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
