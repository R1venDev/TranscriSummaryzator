#!/usr/bin/env python3
"""Add uncertainty and structured records to an already audited summary."""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from quality_schema import evidence_uncertainty
from summary_worker import (
    PIPELINE_VERSION,
    Ollama,
    atomic_json,
    atomic_text,
    build_semantic_registry,
    load_json,
    render_html,
    render_markdown,
    semantic_metrics,
    stable_hash,
    transcript_utterances,
)


def enrich_facts(facts, transcript):
    utterance_by_id = {item["id"]: item for item in transcript_utterances(transcript)}
    result = []
    for original in facts:
        fact = dict(original)
        evidence = []
        for old in fact.get("evidence", []):
            source = utterance_by_id.get(old.get("id"), {})
            item = dict(old)
            item["flags"] = list(source.get("flags", item.get("flags", [])))
            item["uncertainty"] = source.get("uncertainty", item.get("uncertainty", {}))
            evidence.append(item)
        fact["evidence"] = evidence
        fact["uncertainty"] = evidence_uncertainty(evidence)
        result.append(fact)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    summary_path = args.output / "summary.json"
    transcript_path = args.output / "transcript.json"
    summary = load_json(summary_path)
    transcript = load_json(transcript_path)
    facts = enrich_facts(summary.get("facts", []), transcript)
    if not facts:
        raise ValueError("В summary.json нет проверенных фактов")

    cfg = load_json(args.config)
    run_dir = args.cache / ("structured-backfill-" + stable_hash({
        "facts": facts,
        "model": cfg.get("summary_auditor_model"),
        "version": PIPELINE_VERSION,
    })[:12])
    registry = build_semantic_registry(
        Ollama(cfg.get("ollama_url", "http://127.0.0.1:11434")),
        cfg.get("summary_auditor_model", "qwen3.5:9b-q4_K_M"),
        facts,
        run_dir,
    )
    counts = semantic_metrics(registry, facts)
    coverage = dict(summary.get("coverage", {}), semantic=counts)
    markdown = render_markdown(
        summary["document"], facts, coverage,
        metadata={
            "source": transcript.get("source"),
            "duration_seconds": transcript.get("duration_seconds"),
            "project": cfg.get("summary_project_name", "Project"),
        },
        semantic_registry=registry,
    )

    history = args.output / "summary_history"
    history.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for name in ("summary.md", "summary.json", "summary_audit.json", "semantic_records.json", "tasks.json"):
        source = args.output / name
        if source.is_file():
            shutil.copy2(source, history / f"{stamp}-{name}")

    summary.update({
        "pipeline_version": PIPELINE_VERSION,
        "facts": facts,
        "coverage": coverage,
        "semantic_records_file": "semantic_records.json",
        "tasks_file": "tasks.json",
    })
    audit_path = args.output / "summary_audit.json"
    audit = load_json(audit_path) if audit_path.is_file() else {}
    audit.update({"pipeline_version": PIPELINE_VERSION, "coverage": coverage, "semantic": counts})
    atomic_text(args.output / "summary.md", markdown)
    atomic_text(args.output / "summary.html", render_html(markdown))
    atomic_json(summary_path, summary)
    atomic_json(args.output / "semantic_records.json", registry)
    atomic_json(args.output / "tasks.json", {"schema_version": registry["schema_version"], "tasks": registry["tasks"]})
    atomic_json(audit_path, audit)
    print({"pipeline_version": PIPELINE_VERSION, **counts})


if __name__ == "__main__":
    main()
