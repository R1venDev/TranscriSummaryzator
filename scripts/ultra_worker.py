#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from diagnostics import decision as diagnostic_decision, event as diagnostic_event


def parse_segment(value):
    if isinstance(value, str):
        start, end, speaker = value.split(maxsplit=2)
        return {"start": round(float(start), 3), "end": round(float(end), 3), "speaker": speaker}
    if isinstance(value, dict):
        return {"start": round(float(value["start"]), 3), "end": round(float(value["end"]), 3), "speaker": str(value.get("speaker", value.get("label")))}
    raise ValueError(f"Неизвестный формат Ultra segment: {value!r}")


def write_rttm(path, session, intervals):
    lines = [f'SPEAKER {session} 1 {x["start"]:.3f} {x["end"] - x["start"]:.3f} <NA> <NA> {x["speaker"]} <NA> <NA>' for x in intervals]
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rttm", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()

    import torch
    from huggingface_hub import hf_hub_download
    from nemo.collections.asr.models import SortformerEncLabelModel

    model_path = hf_hub_download(args.model, "ultra_diar_streaming_sortformer_8spk_v1.nemo", revision=args.revision, cache_dir=args.cache)
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    diagnostic_decision("ultra_device", device, candidates=[args.device, "cuda", "cpu"], reasons=["runtime_device_selection"])
    model = SortformerEncLabelModel.restore_from(model_path, map_location=device, strict=False)
    model.eval()
    model.sortformer_modules.chunk_len = 340
    model.sortformer_modules.chunk_right_context = 40
    model.sortformer_modules.fifo_len = 40
    model.sortformer_modules.spkcache_update_period = 300
    context = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else __import__("contextlib").nullcontext()
    with torch.inference_mode(), context:
        raw = model.diarize(audio=[args.audio], batch_size=1, verbose=True)[0]
    intervals = sorted((parse_segment(item) for item in raw), key=lambda x: (x["start"], x["end"], x["speaker"]))
    diagnostic_event(
        "ultra_result", outcome="completed", inputs={"model": args.model, "revision": args.revision, "device": device},
        metrics={"intervals": len(intervals), "speakers": len({item["speaker"] for item in intervals}), "speech_seconds_sum": sum(item["end"] - item["start"] for item in intervals)},
        thresholds={"chunk_len": 340, "right_context": 40, "fifo_len": 40, "speaker_cache_update_period": 300}, refs={"output": args.output, "rttm": args.rttm},
    )
    payload = {"model": args.model, "revision": args.revision, "device": device, "model_file": Path(model_path).name, "intervals": intervals}
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_rttm(args.rttm, Path(args.rttm).stem, intervals)


if __name__ == "__main__":
    main()
