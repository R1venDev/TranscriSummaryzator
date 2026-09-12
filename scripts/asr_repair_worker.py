#!/usr/bin/env python3
"""Batch second-pass ASR for high-risk evidence windows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from model_common import choose_torch_device, write_json
from asr_worker import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="v3_e2e_rnnt")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import soundfile as sf
    audio, rate = sf.read(args.audio, dtype="float32")
    if rate != 16000:
        raise ValueError(f"Expected 16000 Hz audio, got {rate}")
    if getattr(audio, "ndim", 1) != 1:
        audio = audio.mean(axis=1)
    requests = json.loads(Path(args.manifest).read_text(encoding="utf-8"))["requests"]
    device = choose_torch_device(args.device)
    model = load_model(args.model, device, args.cache)
    repairs = []
    for index, request in enumerate(requests, 1):
        start = max(0.0, float(request["start"])); end = min(len(audio) / rate, float(request["end"]))
        clip = audio[int(start * rate):int(end * rate)]
        clip_hash = hashlib.sha256(clip.tobytes()).hexdigest()
        with __import__("tempfile").TemporaryDirectory(prefix="evidence-repair-") as directory:
            path = Path(directory) / "window.wav"
            sf.write(path, clip, rate, subtype="PCM_16")
            result = model.transcribe(str(path), word_timestamps=True)
        repairs.append({
            **request, "text": str(result.text or "").strip(), "audio_clip_sha256": clip_hash,
            "model": args.model, "device": device, "method": "second_pass_no_vad_full_window",
        })
        print("REPAIR_PROGRESS " + json.dumps({"current": index, "total": len(requests)}), flush=True)
    write_json(args.output, {"schema_version": 1, "repairs": repairs})


if __name__ == "__main__":
    main()
