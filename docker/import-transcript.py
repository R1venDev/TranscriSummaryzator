#!/usr/bin/env python3
"""Import an existing transcript. No ASR, model loading, or paid dispatch."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pipeline
from summary.luna_v1.source import load_source

MAX_BYTES = 64 * 1024 * 1024


def import_transcript(raw: bytes, name: str) -> dict:
    if not name or len(name) > 200 or name != Path(name).name or any(ord(c) < 32 for c in name):
        raise ValueError("Use a plain display name, up to 200 characters")
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError("Transcript must be nonempty and no larger than 64 MiB")
    pipeline.STATE.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=pipeline.STATE, suffix=".json") as temporary:
        temporary.write(raw)
        temporary.flush()
        _, index, content_sha = load_source(temporary.name)
    identity = hashlib.sha256(b"imported-transcript\0" + name.encode("utf-8") + b"\0" + raw).hexdigest()
    if pipeline.CURRENT.get() != pipeline.DEFAULT:
        identity = hashlib.sha256((identity+"\0"+pipeline.CURRENT.get()).encode()).hexdigest()
    db = pipeline.connect()
    try:
        db.execute("BEGIN IMMEDIATE")
        existing = db.execute("SELECT id,summary_status FROM jobs WHERE fingerprint=?", (identity,)).fetchone()
        if existing:
            db.commit()
            return {"job_id": existing["id"], "imported": False, "summary_status": existing["summary_status"]}
        output = pipeline.OUTPUTS / ("imported-" + identity)
        job_dir = pipeline.JOBS / ("imported-" + identity)
        for path in (output, job_dir):
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
        transcript = output / "transcript.json"
        if transcript.exists() and transcript.read_bytes() != raw:
            raise ValueError("An existing import has different bytes; refusing to overwrite")
        if not transcript.exists():
            with transcript.open("xb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
        stamp = pipeline.now()
        cursor = db.execute(
            """INSERT INTO jobs
            (fingerprint,content_sha256,source_path,original_name,status,stage,job_dir,
             output_dir,progress,detail,created_at,updated_at,finished_at,
             summary_status,summary_detail,project_id)
            VALUES (?, ?, ?, ?, 'done', 'imported_transcript', ?, ?, 100, ?, ?, ?, ?,
                    'not_started', ?, ?)""",
            (identity, content_sha, str(transcript), name, str(job_dir), str(output),
             "Импортирована готовая стенограмма; аудио не обрабатывалось", stamp, stamp, stamp,
             "Готово к запуску саммари вручную", pipeline.CURRENT.get()),
        )
        job_id = cursor.lastrowid
        pipeline.write_json(job_dir / "job.json", {
            "id": job_id, "source_kind": "imported_transcript", "content_sha256": content_sha,
            "source": str(transcript), "created_at": stamp,
            "utterances": len(index["by_id"]), "speech_processing_performed": False,
        })
        db.commit()
        pipeline.write_status_snapshot(db)
        return {"job_id": job_id, "imported": True, "summary_status": "not_started"}
    finally:
        db.close()


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Name displayed in the dashboard")
    parser.add_argument("--project-id", default="default")
    args = parser.parse_args()
    pipeline.Projects(pipeline.STATE).require(args.project_id)
    raw = sys.stdin.buffer.read(MAX_BYTES + 1)
    with pipeline.scope(args.project_id):
        print(json.dumps(import_transcript(raw, args.name), ensure_ascii=False))


if __name__ == "__main__":
    main()
