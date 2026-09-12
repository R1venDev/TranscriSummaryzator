#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from model_common import choose_torch_device, write_json


def normalized(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length < 1e-8:
        raise ValueError("Модель не смогла построить голосовой отпечаток")
    return vector / length


def speech_chunks(audio: np.ndarray, sample_rate: int) -> list[np.ndarray]:
    minimum = int(2.0 * sample_rate)
    window = int(6.0 * sample_rate)
    step = int(5.0 * sample_rate)
    chunks = []
    for start in range(0, len(audio), step):
        chunk = audio[start:start + window]
        if len(chunk) < minimum:
            break
        # Skip silence and almost-silent room tone. Diarization samples are
        # already selected from non-overlapping speech regions.
        rms = float(np.sqrt(np.mean(np.square(chunk), dtype=np.float64)))
        if rms < 0.0005:
            continue
        # Keep quiet microphones useful without letting a loud sample dominate.
        gain = min(10.0, 0.05 / max(rms, 1e-8))
        chunk = np.clip(chunk * gain, -1.0, 1.0)
        if len(chunk) < window:
            chunk = np.pad(chunk, (0, window - len(chunk)))
        chunks.append(chunk.astype(np.float32, copy=False))
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    import soundfile as sf
    import torch
    from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

    device_name = choose_torch_device(args.device)
    device = torch.device(device_name)
    extractor = PretrainedSpeakerEmbedding(args.model, device=device)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    results = {}

    all_groups = set(manifest.get("groups", {})) | set(manifest.get("segments", {}))
    audio_cache = {}
    for group in all_groups:
        embeddings = []
        speech_seconds = 0.0
        items = [(path, None, None) for path in manifest.get("groups", {}).get(group, [])]
        items.extend(
            (item["path"], float(item["start"]), float(item["end"]))
            for item in manifest.get("segments", {}).get(group, [])
        )
        for path, start, end in items:
            if path not in audio_cache:
                audio_cache[path] = sf.read(path, dtype="float32")
            audio, sample_rate = audio_cache[path]
            if sample_rate != 16000:
                raise ValueError(f"Ожидалось аудио 16 кГц, получено {sample_rate}")
            if getattr(audio, "ndim", 1) != 1:
                audio = audio.mean(axis=1)
            selected = audio if start is None else audio[max(0, int(start * sample_rate)):min(len(audio), int(end * sample_rate))]
            chunks = speech_chunks(selected, sample_rate)
            for chunk in chunks:
                tensor = torch.from_numpy(chunk)[None, None]
                embeddings.append(normalized(extractor(tensor)[0]))
                speech_seconds += min(6.0, len(selected) / sample_rate)
        if embeddings:
            centroid = normalized(np.mean(np.stack(embeddings), axis=0))
            results[str(group)] = {
                "embedding": [round(float(value), 8) for value in centroid],
                "chunks": len(embeddings),
                "speech_seconds": round(speech_seconds, 2),
            }

    write_json(args.output, {"model": "pyannote/wespeaker-voxceleb-resnet34-LM", "device": device_name, "groups": results})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
