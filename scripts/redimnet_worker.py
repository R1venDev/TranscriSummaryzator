#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def normalize(vector):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    return vector / max(1e-9, float(np.linalg.norm(vector)))


def chunks(audio, sample_rate=16000, seconds=6.0, step=5.0):
    window, stride = int(seconds * sample_rate), int(step * sample_rate)
    # ReDimNet is less reliable on short speech, but discarding it entirely
    # leaves the diarizers' hardest conflicts without acoustic evidence.  Keep
    # >=1.2 s phrases, zero-pad them, and let the stricter corroborated matcher
    # decide whether the result is safe to use.
    if len(audio) < int(1.2 * sample_rate):
        return []
    result = []
    for start in range(0, len(audio), stride):
        item = audio[start:start + window]
        if len(item) < int(1.2 * sample_rate):
            break
        if float(np.sqrt(np.mean(item * item))) < 0.0005:
            continue
        if len(item) < window:
            item = np.pad(item, (0, window - len(item)))
        result.append(item.astype(np.float32, copy=False))
    return result


def robust_centroid(vectors):
    values = np.stack([normalize(v) for v in vectors])
    if len(values) >= 3:
        similarities = values @ values.T
        medoid = int(np.argmax(np.median(similarities, axis=1)))
        keep = similarities[medoid] >= np.quantile(similarities[medoid], 0.2)
        values = values[keep]
    return normalize(np.mean(values, axis=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    os.environ["TORCH_HOME"] = str(Path(args.cache) / "torch")

    import soundfile as sf
    import torch

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load("PalabraAI/redimnet2", "redimnet2", model_name="b6", train_type="lm", dataset="vb2+vox2+cnc2_v0", pretrained=True, trust_repo=True)
    model.eval().to(device)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    results = {}
    errors = []
    for group, inputs in manifest["groups"].items():
        embeddings = []
        for item in inputs:
            path = item if isinstance(item, str) else item["path"]
            try:
                audio, rate = sf.read(path, dtype="float32")
            except Exception as exc:
                errors.append({"group": group, "path": str(path), "error": str(exc)})
                continue
            if getattr(audio, "ndim", 1) != 1:
                audio = audio.mean(axis=1)
            if rate != 16000:
                raise ValueError(f"Ожидалось 16 кГц: {path}")
            if isinstance(item, dict):
                audio = audio[int(float(item["start"]) * rate):int(float(item["end"]) * rate)]
            for chunk in chunks(audio, rate):
                tensor = torch.from_numpy(chunk)[None].to(device)
                context = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else __import__("contextlib").nullcontext()
                with torch.inference_mode(), context:
                    embeddings.append(normalize(model(tensor).float().cpu().numpy()[0]))
        if embeddings:
            results[group] = {"embedding": robust_centroid(embeddings).tolist(), "references": [v.tolist() for v in embeddings], "chunks": len(embeddings)}
    Path(args.output).write_text(json.dumps({"model": "PalabraAI/ReDimNet2-B6-vb2+vox2+cnc2_v0-lm", "device": device, "groups": results, "errors": errors}, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
