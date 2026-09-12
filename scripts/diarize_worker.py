#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from model_common import choose_torch_device, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rttm", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-speakers", type=int, default=1)
    parser.add_argument("--max-speakers", type=int, default=5)
    parser.add_argument("--num-speakers", type=int)
    args = parser.parse_args()

    import torch
    from diarizen.pipelines.inference import DiariZenPipeline

    device = choose_torch_device(args.device)
    rttm_path = Path(args.rttm)
    rttm_path.parent.mkdir(parents=True, exist_ok=True)
    session = rttm_path.stem

    pipeline = DiariZenPipeline.from_pretrained(
        args.model,
        cache_dir=None,
        rttm_out_dir=str(rttm_path.parent),
    )
    pipeline.segmentation_batch_size = args.batch_size
    pipeline.embedding_batch_size = args.batch_size
    pipeline.min_speakers = args.min_speakers
    pipeline.max_speakers = args.max_speakers
    pipeline.num_speakers = args.num_speakers
    if device != "cpu":
        try:
            pipeline.to(torch.device(device))
        except Exception as exc:
            print(f"Requested accelerator unavailable for DiariZen ({exc}); using CPU.", flush=True)
            device = "cpu"
            pipeline.to(torch.device("cpu"))

    try:
        result = pipeline(args.audio, sess_name=session)
    except Exception as exc:
        if device != "mps":
            raise
        print(f"Metal inference failed ({exc}); retrying on CPU.", flush=True)
        pipeline.to(torch.device("cpu"))
        result = pipeline(args.audio, sess_name=session)
        device = "cpu"

    intervals = []
    for turn, _, speaker in result.itertracks(yield_label=True):
        intervals.append(
            {
                "start": round(float(turn.start), 3),
                "end": round(float(turn.end), 3),
                "speaker": str(speaker),
            }
        )
    intervals.sort(key=lambda item: (item["start"], item["end"], item["speaker"]))
    write_json(
        args.output,
        {
            "model": args.model,
            "device": device,
            "min_speakers": args.min_speakers,
            "max_speakers": args.max_speakers,
            "num_speakers": args.num_speakers,
            "intervals": intervals,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
