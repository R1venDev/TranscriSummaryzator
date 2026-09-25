"""Versioned Luna summary generation, sealed before pointer replacement."""
from __future__ import annotations

import hashlib
import fcntl
import json
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from . import render_document, validate_document
from .ledger import write_private_json


CONTRACT_VERSION = "luna_summary_v1"
REQUIRED_FILES = frozenset({
    "summary.md", "summary.html", "summary.fragment.html", "summary.json",
    "tasks.json", "transcript.html", "run_manifest.json", "model_document.json",
})


def _bytes(name: str, value) -> bytes:
    if name.endswith(".json"):
        return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    if not isinstance(value, str):
        raise ValueError(f"{name} is not text")
    return value.encode("utf-8")


def _verify_staged_target(target: Path, *, generation_id: str, document: dict,
                          source_sha: str, semantic_key: str, job_id: str,
                          remote_batch_id: str, credential_id: str,
                          prompt_sha256: str, schema_sha256: str) -> dict:
    """A prior crash may have sealed the target before switching the pointer."""
    manifest = json.loads((target / "generation_manifest.json").read_text(encoding="utf-8"))
    digests = manifest.get("artifact_sha256")
    if (manifest.get("contract_version") != CONTRACT_VERSION
            or manifest.get("generation_id") != generation_id
            or manifest.get("source_sha256") != source_sha
            or not isinstance(digests, dict) or set(digests) != REQUIRED_FILES
            or manifest.get("verified_artifact_sha256") != digests.get("summary.md")):
        raise ValueError("staged_generation_manifest_mismatch")
    for name in REQUIRED_FILES:
        expected = digests[name]
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("staged_generation_digest_invalid")
        if hashlib.sha256((target / name).read_bytes()).hexdigest() != expected:
            raise ValueError("staged_generation_artifact_changed")
    run = json.loads((target / "run_manifest.json").read_text(encoding="utf-8"))
    expected_run = {
        "contract_version": CONTRACT_VERSION, "semantic_key": semantic_key,
        "source_sha256": source_sha, "job_id": job_id,
        "remote_batch_id": remote_batch_id, "credential_id": credential_id,
        "prompt_sha256": prompt_sha256, "schema_sha256": schema_sha256,
    }
    if any(run.get(key) != value for key, value in expected_run.items()):
        raise ValueError("staged_generation_identity_mismatch")
    if hashlib.sha256(_bytes("model_document.json", document)).hexdigest() != digests["model_document.json"]:
        raise ValueError("staged_generation_document_mismatch")
    return manifest


def _switch_pointer(output_dir: Path, manifest: dict, before_pointer: Callable[[], None] | None) -> None:
    pointer = output_dir / "summary_current.json"
    if before_pointer is not None:
        before_pointer()
    temporary = pointer.with_name("." + pointer.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump({"generation_id": manifest["generation_id"],
                       "verified_artifact_sha256": manifest["artifact_sha256"]["summary.md"]},
                      handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, pointer)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_locked(*, document: dict, source_index: dict, transcript_path: Path,
                     output_dir: Path, semantic_key: str, job_id: str,
                     remote_batch_id: str, credential_id: str,
                     prompt_sha256: str, schema_sha256: str,
                     effective_tasks: list[dict] | None = None,
                     generation_id: str | None = None,
                     before_pointer: Callable[[], None] | None = None) -> tuple[str, Path]:
    """Commit a complete generation or leave the old pointer untouched."""
    validate_document(document, source_index)
    original_sha = source_index["source_sha256"]
    if hashlib.sha256(Path(transcript_path).read_bytes()).hexdigest() != original_sha:
        raise ValueError("source_changed_before_publication")
    rendered = render_document(document, source_index, effective_tasks)
    target_parent = Path(output_dir) / "summary_generations"
    target_parent.mkdir(parents=True, exist_ok=True)
    generation_id = generation_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:12]
    if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[0-9a-f]{12}", generation_id):
        raise AssertionError("bad generation id")
    candidate = target_parent / (".candidate-" + generation_id)
    target = target_parent / generation_id
    if target.exists():
        staged = _verify_staged_target(target, generation_id=generation_id, document=document,
            source_sha=original_sha, semantic_key=semantic_key, job_id=job_id,
            remote_batch_id=remote_batch_id, credential_id=credential_id,
            prompt_sha256=prompt_sha256, schema_sha256=schema_sha256)
        pointer_path = Path(output_dir) / "summary_current.json"
        if pointer_path.exists():
            current = json.loads(pointer_path.read_text(encoding="utf-8"))
            if current.get("generation_id") == generation_id:
                if current.get("verified_artifact_sha256") != staged["artifact_sha256"]["summary.md"]:
                    raise ValueError("staged_generation_pointer_digest_mismatch")
                return generation_id, target
            current_id = current.get("generation_id")
            if not isinstance(current_id, str) or not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[0-9a-f]{12}", current_id):
                raise ValueError("current_generation_pointer_invalid")
            current_manifest = json.loads((target_parent / current_id / "generation_manifest.json").read_text(encoding="utf-8"))
            if not isinstance(current_manifest.get("created_at"), str) or current_manifest["created_at"] >= staged["created_at"]:
                raise ValueError("newer_generation_already_selected")
        # A human edit can land after the task preview but before its commit.
        # That aborted publication leaves a sealed, unpublished target with
        # the old task projection. Preserve it for diagnosis and render the
        # newly previewed effective state from the saved model document.
        current_tasks_sha = hashlib.sha256(_bytes("tasks.json", rendered["tasks.json"])).hexdigest()
        if current_tasks_sha != staged["artifact_sha256"]["tasks.json"]:
            superseded = target_parent / (".superseded-" + generation_id + "-" + uuid.uuid4().hex[:8])
            os.replace(target, superseded)
        else:
            _switch_pointer(Path(output_dir), staged, before_pointer)
            return generation_id, target
    candidate.mkdir(mode=0o700)
    try:
        run_manifest = {
            "contract_version": CONTRACT_VERSION,
            "semantic_key": semantic_key,
            "source_sha256": original_sha,
            "model": "openai/gpt-6-luna:batch",
            "provider": "openai",
            "privacy_mode": "batch_gateway_retention_up_to_30d_provider_zdr_off_user_authorized",
            "job_id": job_id,
            "remote_batch_id": remote_batch_id,
            "credential_id": credential_id,
            "prompt_sha256": prompt_sha256,
            "schema_sha256": schema_sha256,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        rendered["run_manifest.json"] = run_manifest
        rendered["model_document.json"] = document
        digests = {}
        for name, value in rendered.items():
            if name not in REQUIRED_FILES:
                raise ValueError("unexpected render file")
            payload = _bytes(name, value)
            path = candidate / name
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            digests[name] = hashlib.sha256(payload).hexdigest()
        if set(digests) != REQUIRED_FILES:
            raise ValueError("generation file set incomplete")
        manifest = {
            "contract_version": CONTRACT_VERSION,
            "generation_id": generation_id,
            "created_at": run_manifest["created_at"],
            "source_sha256": original_sha,
            "artifact_sha256": digests,
            "verified_artifact_sha256": digests["summary.md"],
        }
        write_private_json(candidate / "generation_manifest.json", manifest)
        if hashlib.sha256(Path(transcript_path).read_bytes()).hexdigest() != original_sha:
            raise ValueError("source_changed_during_publication")
        os.replace(candidate, target)
        _switch_pointer(Path(output_dir), manifest, before_pointer)
        return generation_id, target
    finally:
        if candidate.exists():
            shutil.rmtree(candidate)


def publish_document(**kwargs) -> tuple[str, Path]:
    """Serialize generation and task-state publication for one meeting."""
    output_dir = Path(kwargs["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".summary_publication.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            return _publish_locked(**kwargs)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def is_luna_generation(path: Path) -> bool:
    try:
        manifest = json.loads((Path(path) / "generation_manifest.json").read_text())
        return manifest.get("contract_version") == CONTRACT_VERSION
    except (OSError, ValueError, json.JSONDecodeError):
        return False
