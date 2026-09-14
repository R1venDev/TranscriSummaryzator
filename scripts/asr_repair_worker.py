#!/usr/bin/env python3
"""Batch second-pass ASR for high-risk evidence windows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from model_common import choose_torch_device, write_json
from asr_worker import load_model
from evidence_repair import audio_chunk_ranges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="v3_e2e_rnnt")
    parser.add_argument("--secondary-model")
    parser.add_argument("--secondary-revision")
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
        texts, words = [], []
        ranges = audio_chunk_ranges(len(clip), rate)
        with __import__("tempfile").TemporaryDirectory(prefix="evidence-repair-") as directory:
            for part_index, (left, right) in enumerate(ranges, 1):
                path = Path(directory) / f"window-{part_index:03d}.wav"
                sf.write(path, clip[left:right], rate, subtype="PCM_16")
                result = model.transcribe(str(path), word_timestamps=True)
                if str(result.text or "").strip():
                    texts.append(str(result.text).strip())
                for word in getattr(result, "words", None) or []:
                    words.append({
                        "text": str(word.text),
                        "start": round(start + left / rate + float(word.start), 3),
                        "end": round(start + left / rate + float(word.end), 3),
                    })
        repairs.append({
            **request, "text": " ".join(texts), "words": words, "audio_clip_sha256": clip_hash,
            "model": args.model, "device": device,
            "method": "second_pass_no_vad_fixed_chunks", "chunk_count": len(ranges),
        })
        print("REPAIR_PROGRESS " + json.dumps({"current": index, "total": len(requests)}), flush=True)
    if args.secondary_model:
        # Load a genuinely independent model family only after GigaAM is released.
        del model
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        if not args.secondary_revision:
            raise ValueError("--secondary-revision is required for immutable ASR provenance")
        from huggingface_hub import snapshot_download
        from faster_whisper import WhisperModel
        repository = args.secondary_model if "/" in args.secondary_model else f"mobiuslabsgmbh/faster-whisper-{args.secondary_model}"
        snapshot = snapshot_download(repo_id=repository, revision=args.secondary_revision, cache_dir=args.cache, allow_patterns=["config.json", "model.bin", "preprocessor_config.json", "tokenizer.json", "vocabulary.json"])
        secondary = WhisperModel(snapshot, device="cuda" if device != "cpu" else "cpu", compute_type="float16" if device != "cpu" else "int8")
        for repair in repairs:
            start, end = float(repair["start"]), float(repair["end"])
            clip = audio[int(start * rate):int(end * rate)]
            with __import__("tempfile").NamedTemporaryFile(suffix=".wav") as temporary:
                sf.write(temporary.name, clip, rate, subtype="PCM_16")
                segments, _ = secondary.transcribe(temporary.name, language="ru", word_timestamps=True, vad_filter=False)
                alternative = " ".join(segment.text.strip() for segment in segments).strip()
            repair["alternatives"] = [{"text": alternative, "source": "faster_whisper", "model": args.secondary_model, "revision": args.secondary_revision, "resolved_path": snapshot}] if alternative else []
            repair["status"] = "disputed" if alternative and alternative.casefold() != str(repair.get("text", "")).casefold() else "agreed"
    write_json(args.output, {"schema_version": 2, "repairs": repairs})


if __name__ == "__main__":
    main()
