#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import re
import tempfile
from pathlib import Path

from model_common import choose_torch_device, write_json


def make_chunks(regions: list[dict], max_seconds: float = 22.0, overlap: float = 0.45) -> list[tuple[float, float]]:
    if not regions:
        return []
    merged: list[list[float]] = []
    for region in regions:
        start, end = float(region["start"]), float(region["end"])
        if not merged or start - merged[-1][1] > 0.45 or end - merged[-1][0] > max_seconds:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    chunks: list[tuple[float, float]] = []
    for start, end in merged:
        cursor = start
        while end - cursor > max_seconds:
            chunks.append((cursor, cursor + max_seconds))
            cursor += max_seconds - overlap
        if end - cursor >= 0.20:
            chunks.append((cursor, end))
    return chunks


def same_word(a: dict, b: dict) -> bool:
    normalize = lambda text: re.sub(r"[^\w]+", "", text.casefold())
    return (
        normalize(a["text"]) == normalize(b["text"])
        and abs(a["start"] - b["start"]) < 0.25
        and abs(a["end"] - b["end"]) < 0.35
    )


def merge_chunk_words(existing, incoming, chunk_start, previous_end=None):
    """Deduplicate only repeated hypotheses in the overlap of two chunks."""
    if previous_end is None or previous_end <= chunk_start:
        return existing + list(incoming)
    candidates = tuple(word for word in existing
                       if word["start"] >= chunk_start - 0.1
                       and word["end"] <= previous_end + 0.1)
    for word in candidates:
        flags = word.setdefault("flags", [])
        if "asr_boundary" not in flags:
            flags.append("asr_boundary")
    accepted = []
    for word in incoming:
        in_overlap = word["start"] >= chunk_start - 0.1 and word["end"] <= previous_end + 0.1
        if in_overlap and any(same_word(word, old) for old in candidates):
            continue
        if in_overlap:
            word = dict(word, flags=list(dict.fromkeys(word.get("flags", []) + ["asr_boundary"])))
        accepted.append(word)
    return existing + accepted


def load_model(model_name: str, device: str, cache: str):
    import gigaam

    return gigaam.load_model(
        model_name,
        device=device,
        fp16_encoder=device != "cpu",
        download_root=cache,
    )


def transcribe(args, device: str) -> dict:
    import soundfile as sf
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    import gigaam

    audio, sample_rate = sf.read(args.audio, dtype="float32")
    if sample_rate != 16000:
        raise ValueError(f"Expected 16000 Hz audio, got {sample_rate}")
    if getattr(audio, "ndim", 1) != 1:
        audio = audio.mean(axis=1)

    vad = load_silero_vad(onnx=True)
    stamps = get_speech_timestamps(
        torch.from_numpy(audio),
        vad,
        sampling_rate=sample_rate,
        threshold=args.vad_threshold,
        min_speech_duration_ms=args.vad_min_speech_ms,
        min_silence_duration_ms=args.vad_min_silence_ms,
        speech_pad_ms=args.vad_speech_pad_ms,
        return_seconds=True,
    )
    chunks = make_chunks(stamps, args.chunk_seconds, args.overlap_seconds)
    print("PIPELINE_PROGRESS " + json.dumps({"current": 0, "total": len(chunks)}), flush=True)
    model = load_model(args.model, device, args.cache)
    words: list[dict] = []
    segments: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="gigaam-chunks-") as temp_dir:
        for index, (start, end) in enumerate(chunks):
            chunk_path = Path(temp_dir) / f"{index:06d}.wav"
            sf.write(chunk_path, audio[int(start * sample_rate): int(end * sample_rate)], sample_rate, subtype="PCM_16")
            result = model.transcribe(str(chunk_path), word_timestamps=True)
            chunk_words = [
                {
                    "text": word.text,
                    "start": round(float(word.start) + start, 3),
                    "end": round(float(word.end) + start, 3),
                    "asr_chunk_index": index,
                    "asr_confidence": float(word.confidence) if getattr(word, "confidence", None) is not None else None,
                }
                for word in (result.words or [])
            ]
            words = merge_chunk_words(words, chunk_words, start, segments[-1]["end"] if segments else None)
            segments.append(
                {
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "text": result.text,
                }
            )
            print("PIPELINE_PROGRESS " + json.dumps({"current": index + 1, "total": len(chunks)}), flush=True)
    words.sort(key=lambda item: (item["start"], item["end"]))
    return {
        "model": args.model,
        "device": device,
        "vad": {
            "model": "silero-vad-v6-onnx", "threshold": args.vad_threshold,
            "min_speech_duration_ms": args.vad_min_speech_ms,
            "min_silence_duration_ms": args.vad_min_silence_ms,
            "speech_pad_ms": args.vad_speech_pad_ms,
        },
        "segments": segments,
        "words": words,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="v3_e2e_rnnt")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--vad-threshold", type=float, default=0.42)
    parser.add_argument("--vad-min-speech-ms", type=int, default=180)
    parser.add_argument("--vad-min-silence-ms", type=int, default=320)
    parser.add_argument("--vad-speech-pad-ms", type=int, default=220)
    parser.add_argument("--chunk-seconds", type=float, default=22.0)
    parser.add_argument("--overlap-seconds", type=float, default=0.45)
    args = parser.parse_args()

    device = choose_torch_device(args.device)
    try:
        result = transcribe(args, device)
    except Exception as exc:
        if device != "mps":
            raise
        print(f"Metal inference failed ({exc}); retrying GigaAM on CPU.", flush=True)
        gc.collect()
        result = transcribe(args, "cpu")
    write_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
