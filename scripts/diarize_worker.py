#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from model_common import choose_torch_device, write_json
from diagnostics import decision as diagnostic_decision, event as diagnostic_event


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rttm", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--embedding-revision", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-speakers", type=int, default=1)
    parser.add_argument("--max-speakers", type=int, default=5)
    parser.add_argument("--num-speakers", type=int)
    args = parser.parse_args()

    import torch
    from huggingface_hub import hf_hub_download, snapshot_download
    from diarizen.pipelines.inference import DiariZenPipeline

    device = choose_torch_device(args.device)
    diagnostic_decision("diarization_device", device, candidates=[args.device, "cuda", "mps", "cpu"], reasons=["runtime_device_selection"])
    diagnostic_event("diarization_configuration", outcome="accepted", inputs={"model": args.model}, thresholds={"min_speakers": args.min_speakers, "max_speakers": args.max_speakers, "exact_speakers": args.num_speakers, "batch_size": args.batch_size})
    rttm_path = Path(args.rttm)
    rttm_path.parent.mkdir(parents=True, exist_ok=True)
    session = rttm_path.stem

    model_root = snapshot_download(repo_id=args.model, revision=args.revision, cache_dir=args.cache)
    embedding_model = hf_hub_download(repo_id=args.embedding_model, filename="pytorch_model.bin", revision=args.embedding_revision, cache_dir=args.cache)
    pipeline = DiariZenPipeline(
        diarizen_hub=Path(model_root).expanduser().absolute(),
        embedding_model=embedding_model,
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
            diagnostic_event("diarization_device_fallback", category="decision", outcome="cpu", reasons=["accelerator_unavailable"], error=exc)
            device = "cpu"
            pipeline.to(torch.device("cpu"))

    try:
        result = pipeline(args.audio, sess_name=session)
    except Exception as exc:
        if device != "mps":
            raise
        print(f"Metal inference failed ({exc}); retrying on CPU.", flush=True)
        diagnostic_event("diarization_device_fallback", category="decision", outcome="cpu", reasons=["mps_inference_failed"], error=exc)
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
    speakers = sorted({item["speaker"] for item in intervals})
    diagnostic_event(
        "diarization_result", outcome="completed",
        metrics={"intervals": len(intervals), "speakers": len(speakers), "speech_seconds_sum": sum(item["end"] - item["start"] for item in intervals)},
        inputs={"speaker_ids": speakers, "device": device}, refs={"output": args.output, "rttm": args.rttm},
    )
    write_json(
        args.output,
        {
            "model": args.model,
            "revision": args.revision,
            "embedding_model": args.embedding_model,
            "embedding_revision": args.embedding_revision,
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
