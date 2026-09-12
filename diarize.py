#!/usr/bin/env python3
"""One-command entry point: audio + enrollment -> identified diarization."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pipeline


def prepare_enrollment(source):
    source = Path(source).expanduser().resolve()
    people = [path for path in source.iterdir() if path.is_dir()]
    signature = hashlib.sha256()
    for person in sorted(people):
        for sample in sorted(path for path in person.iterdir() if path.is_file() and path.suffix.casefold() in pipeline.MEDIA_EXTENSIONS):
            stat = sample.stat()
            signature.update(f"{person.name}/{sample.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    target = pipeline.ROOT / "work" / "cache" / "external-enrollment" / signature.hexdigest()[:20]
    target.mkdir(parents=True, exist_ok=True)
    for person in people:
        samples = [path for path in sorted(person.iterdir()) if path.is_file() and path.suffix.casefold() in pipeline.MEDIA_EXTENSIONS]
        if not samples:
            continue
        profile_id = hashlib.md5(person.name.encode("utf-8")).hexdigest()
        directory = target / profile_id
        directory.mkdir(parents=True, exist_ok=True)
        records = []
        for index, sample in enumerate(samples):
            audio = directory / f"sample_{index + 1:03d}.wav"
            if not audio.is_file():
                subprocess.run([
                    "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(sample),
                    "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(audio),
                ], check=True)
            records.append({"id": hashlib.md5(str(sample).encode()).hexdigest(), "name": sample.name, "audio": audio.name, "duration_seconds": pipeline.validate_media(audio)})
        pipeline.write_json(directory / "profile.json", {"id": profile_id, "name": person.name, "samples": records})
    return target


def main():
    parser = argparse.ArgumentParser(description="Automatic Russian speaker diarization")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--enrollment", default=str(pipeline.VOICE_PROFILES))
    parser.add_argument("--output")
    args = parser.parse_args()
    if Path(args.enrollment).expanduser().resolve() != pipeline.VOICE_PROFILES.resolve():
        pipeline.VOICE_PROFILES = prepare_enrollment(args.enrollment)
    job_id = pipeline.enqueue(args.audio)
    pipeline.process_job(job_id)
    row = pipeline.connect().execute("SELECT output_dir FROM jobs WHERE id = ?", (job_id,)).fetchone()
    result = Path(row["output_dir"])
    if args.output:
        destination = Path(args.output).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("result.json", "result.rttm", "debug.json"):
            shutil.copy2(result / name, destination / name)
        result = destination
    print(result)


if __name__ == "__main__":
    main()
