#!/usr/bin/env python3
"""Structured, append-only diagnostics shared by every pipeline process.

The JSONL is intended to be attached to a bug report.  It records why a
branch was selected, the measured values and thresholds, without leaking
environment secrets or requiring the original processing log to be parsed.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import sys
import threading
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = 1
_path = None
_trace_path = None
_component = os.environ.get("TRANSCRISUMMARY_COMPONENT", "unknown")
_run_id = os.environ.get("TRANSCRISUMMARY_RUN_ID")
_job_id = os.environ.get("TRANSCRISUMMARY_JOB_ID")
_lock = threading.Lock()
_SENSITIVE = ("password", "passwd", "secret", "token", "authorization", "cookie", "api_key")


def configure(path=None, *, component=None, run_id=None, job_id=None, trace_path=None):
    global _path, _trace_path, _component, _run_id, _job_id
    value = path or os.environ.get("TRANSCRISUMMARY_DIAGNOSTICS")
    _path = Path(value) if value else None
    trace_value = trace_path or os.environ.get("TRANSCRISUMMARY_DIAGNOSTICS_TRACE")
    _trace_path = Path(trace_value) if trace_value else (_path.with_name("diagnostics.trace.jsonl") if _path else None)
    if component:
        _component = str(component)
    if run_id:
        _run_id = str(run_id)
    if job_id is not None:
        _job_id = str(job_id)
    if _path:
        os.environ["TRANSCRISUMMARY_DIAGNOSTICS"] = str(_path)
    if _trace_path:
        os.environ["TRANSCRISUMMARY_DIAGNOSTICS_TRACE"] = str(_trace_path)
    os.environ["TRANSCRISUMMARY_COMPONENT"] = _component
    if _run_id:
        os.environ["TRANSCRISUMMARY_RUN_ID"] = _run_id
    if _job_id:
        os.environ["TRANSCRISUMMARY_JOB_ID"] = _job_id
    return _path


def _safe(value, key="", depth=0):
    lowered = key.casefold()
    telemetry_token_key = lowered in {"prompt_tokens", "output_tokens", "prompt_eval_count", "eval_count", "tokens"}
    if not telemetry_token_key and any(marker in lowered for marker in _SENSITIVE):
        return "[REDACTED]"
    if depth > 8:
        return "[MAX_DEPTH]"
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return round(value, 8) if math.isfinite(value) else str(value)
    if isinstance(value, (str, Path)):
        text = str(value)
        return text if len(text) <= 12000 else text[:12000] + "…[TRUNCATED]"
    if isinstance(value, dict):
        return {str(k): _safe(v, str(k), depth + 1) for k, v in list(value.items())[:500]}
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        if key.casefold() == "command":
            redacted, hide_next = [], False
            for item in items[:1000]:
                text = str(item)
                if hide_next:
                    redacted.append("[REDACTED]")
                    hide_next = False
                    continue
                redacted.append(text)
                hide_next = text.startswith("-") and any(marker in text.casefold() for marker in _SENSITIVE)
            return redacted
        safe = [_safe(item, key, depth + 1) for item in items[:1000]]
        if len(items) > 1000:
            safe.append({"truncated_items": len(items) - 1000})
        return safe
    return _safe(str(value), key, depth + 1)


def event(name, *, category="observation", outcome=None, inputs=None, metrics=None,
          thresholds=None, reasons=None, refs=None, severity="INFO", duration_ms=None,
          component=None, error=None):
    path = _path or configure()
    item_ref_keys = {"fact_id", "claim_id", "event_id", "relation_id", "source_event", "target_event"}
    if category in {"trace", "word", "candidate", "voice_id_phrase", "low_level_arbitration"} or (category == "decision" and item_ref_keys & set((refs or {}).keys())):
        path = _trace_path
    if not path:
        return None
    payload = {
        "schema_version": SCHEMA_VERSION,
        "event_id": uuid.uuid4().hex,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "run_id": _run_id,
        "job_id": _job_id,
        "component": component or _component,
        "category": category,
        "name": str(name),
        "severity": severity,
        "outcome": outcome,
        "inputs": inputs or {},
        "metrics": metrics or {},
        "thresholds": thresholds or {},
        "reasons": reasons or [],
        "refs": refs or {},
        "duration_ms": duration_ms,
    }
    if error is not None:
        payload["error"] = {"type": type(error).__name__, "message": str(error)} if isinstance(error, BaseException) else error
    raw = (json.dumps(_safe(payload), ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(descriptor, raw)
        finally:
            os.close(descriptor)
    return payload["event_id"]


def decision(name, selected, *, candidates=None, metrics=None, thresholds=None, reasons=None, refs=None, component=None):
    return event(
        name, category="decision", outcome=selected, inputs={"candidates": candidates or []},
        metrics=metrics, thresholds=thresholds, reasons=reasons, refs=refs, component=component,
    )


def system_snapshot(path=None):
    target = Path(path or ".")
    try:
        disk = shutil.disk_usage(target)
        disk_metrics = {"total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free}
    except OSError:
        disk_metrics = {}
    try:
        load = list(os.getloadavg())
    except (AttributeError, OSError):
        load = []
    return {
        "hostname": platform.node(), "platform": platform.platform(), "python": sys.version,
        "pid": os.getpid(), "cpu_count": os.cpu_count(), "load_average": load,
        "disk": disk_metrics,
    }


@contextmanager
def stage(name, *, inputs=None, refs=None, component=None):
    started = time.monotonic()
    event(name, category="stage", outcome="started", inputs=inputs, refs=refs, component=component)
    try:
        yield
    except Exception as exc:
        event(name, category="stage", outcome="failed", refs=refs, component=component,
              severity="ERROR", duration_ms=round((time.monotonic() - started) * 1000, 3), error=exc)
        raise
    else:
        event(name, category="stage", outcome="completed", refs=refs, component=component,
              duration_ms=round((time.monotonic() - started) * 1000, 3))


def summarize(path):
    path = Path(path)
    counters = {name: Counter() for name in ("components", "categories", "severities", "outcomes")}
    first = last = last_error = None
    total = malformed = 0
    durations, slow, tokens, calls, retries, cache_hits, cache_total = [], [], Counter(), 0, 0, 0, 0
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            total += 1
            first = first or item.get("timestamp")
            last = item.get("timestamp") or last
            for field, bucket in (("component", "components"), ("category", "categories"), ("severity", "severities"), ("outcome", "outcomes")):
                if item.get(field) is not None:
                    counters[bucket][str(item[field])] += 1
            if item.get("severity") == "ERROR" or item.get("error"):
                last_error = {key: item.get(key) for key in ("timestamp", "component", "name", "outcome", "error", "refs")}
            duration = item.get("duration_ms")
            if isinstance(duration, (int, float)):
                durations.append(float(duration)); slow.append({"name": item.get("name"), "component": item.get("component"), "duration_ms": duration, "outcome": item.get("outcome")})
            if item.get("name") == "llm_request":
                calls += item.get("outcome") == "completed"
                retries += item.get("outcome") in {"retryable_output_limit", "failed"}
                for key in ("prompt_eval_count", "eval_count", "prompt_tokens", "output_tokens"):
                    value = item.get("metrics", {}).get(key)
                    if isinstance(value, (int, float)): tokens[key] += value
            if item.get("name") in {"llm_cache", "stage_cache.audio", "stage_cache.diarizen", "stage_cache.ultra", "stage_cache.consensus", "stage_cache.asr"}:
                cache_total += 1; cache_hits += item.get("outcome") == "hit"
    durations.sort()
    def percentile(p):
        return durations[min(len(durations)-1, int((len(durations)-1)*p))] if durations else None
    return {
        "schema_version": SCHEMA_VERSION, "events": total, "malformed_lines": malformed,
        "first_timestamp": first, "last_timestamp": last,
        **{name: dict(value) for name, value in counters.items()},
        "last_error": last_error,
        "jsonl_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
        "jsonl_bytes": path.stat().st_size if path.is_file() else 0,
        "calls": calls, "tokens": dict(tokens), "retries": retries,
        "cache_hit_ratio": cache_hits / cache_total if cache_total else None,
        "latency_ms": {"p50": percentile(.50), "p95": percentile(.95), "max": max(durations) if durations else None},
        "top_slow_requests": sorted(slow, key=lambda x: x["duration_ms"], reverse=True)[:10],
    }


def publish(source, output_dir):
    source, output_dir = Path(source), Path(output_dir)
    if not source.is_file():
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "diagnostics.jsonl"
    shutil.copy2(source, target)
    summary = summarize(source)
    temporary = output_dir / "diagnostics_summary.json.tmp"
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_dir / "diagnostics_summary.json")
    trace = source.with_name("diagnostics.trace.jsonl")
    if trace.is_file():
        shutil.copy2(trace, output_dir / "diagnostics.trace.jsonl")
        summary["trace_sha256"] = hashlib.sha256(trace.read_bytes()).hexdigest()
        summary["trace_bytes"] = trace.stat().st_size
        temporary = output_dir / "diagnostics_summary.json.tmp"
        temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, output_dir / "diagnostics_summary.json")
    return summary


configure()
