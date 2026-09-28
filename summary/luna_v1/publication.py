"""Versioned Luna summary generation, sealed before pointer replacement."""
from __future__ import annotations

import hashlib
import fcntl
import json
import os
import re
import shutil
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from . import load_source, render_document, validate_document
from .ledger import write_private_json


CONTRACT_VERSION = "luna_summary_v1"
REQUIRED_FILES = frozenset({
    "summary.md", "summary.html", "summary.fragment.html", "summary.json",
    "tasks.json", "transcript.html", "run_manifest.json", "model_document.json",
})
REVIEW_SIDECAR_FILE = "review_sidecar.json"


def _sidecar_for_generation(template: dict, *, generation_id: str,
                            source_sha: str, artifact_sha256: dict[str, str]) -> dict:
    """Bind a local review report to the exact sealed source and artifacts.

    The sidecar hashes canonical generation files, excluding itself and the
    generation manifest, so its own digest can be placed in that manifest
    without a circular hash dependency.
    """
    if not isinstance(template, dict):
        raise ValueError("review_sidecar_invalid")
    sidecar = deepcopy(template)
    for field, expected in (("generation_id", generation_id),
                            ("source_revision", source_sha)):
        supplied = sidecar.get(field)
        if supplied not in (None, "", expected):
            raise ValueError(f"review_sidecar_{field}_mismatch")
        sidecar[field] = expected
    hashes = [{"name": name, "sha256": artifact_sha256[name]}
              for name in sorted(REQUIRED_FILES)]
    supplied_hashes = sidecar.get("artifact_hashes")
    if supplied_hashes not in (None, []) and supplied_hashes != hashes:
        raise ValueError("review_sidecar_artifact_hashes_mismatch")
    sidecar["artifact_hashes"] = hashes
    from .batch_stage_contracts_v1 import PublicationReviewSidecar
    return PublicationReviewSidecar.model_validate(sidecar).model_dump()


def _bytes(name: str, value) -> bytes:
    if name.endswith(".json"):
        return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    if not isinstance(value, str):
        raise ValueError(f"{name} is not text")
    return value.encode("utf-8")


def _verify_staged_target(target: Path, *, generation_id: str, document: dict,
                          source_sha: str, semantic_key: str, job_id: str,
                          remote_batch_id: str, credential_id: str,
                          prompt_sha256: str, schema_sha256: str,
                          quality_review: dict | None = None,
                          review_sidecar: dict | None = None) -> dict:
    """A prior crash may have sealed the target before switching the pointer."""
    manifest = json.loads((target / "generation_manifest.json").read_text(encoding="utf-8"))
    digests = manifest.get("artifact_sha256")
    if (manifest.get("contract_version") != CONTRACT_VERSION
            or manifest.get("generation_id") != generation_id
            or manifest.get("source_sha256") != source_sha
            or not isinstance(digests, dict)
            or set(digests) != (REQUIRED_FILES | ({REVIEW_SIDECAR_FILE} if review_sidecar is not None else set()))
            or manifest.get("verified_artifact_sha256") != digests.get("summary.md")):
        raise ValueError("staged_generation_manifest_mismatch")
    for name in digests:
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
    if run.get("quality_review") != quality_review:
        raise ValueError("staged_generation_quality_mismatch")
    if hashlib.sha256(_bytes("model_document.json", document)).hexdigest() != digests["model_document.json"]:
        raise ValueError("staged_generation_document_mismatch")
    if review_sidecar is not None:
        expected_sidecar = _sidecar_for_generation(review_sidecar,
            generation_id=generation_id, source_sha=source_sha,
            artifact_sha256=digests)
        if hashlib.sha256(_bytes(REVIEW_SIDECAR_FILE, expected_sidecar)).hexdigest() != digests[REVIEW_SIDECAR_FILE]:
            raise ValueError("staged_generation_review_sidecar_mismatch")
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
                     quality_review: dict | None = None,
                     review_sidecar: dict | None = None,
                     before_pointer: Callable[[], None] | None = None) -> tuple[str, Path]:
    """Commit a complete generation or leave the old pointer untouched."""
    validate_document(document, source_index)
    original_sha = source_index["source_sha256"]
    if hashlib.sha256(Path(transcript_path).read_bytes()).hexdigest() != original_sha:
        raise ValueError("source_changed_before_publication")
    if review_sidecar is not None:
        if not isinstance(quality_review, dict):
            raise ValueError("review_sidecar_requires_quality_review")
        expected_status = {
            "review_completed": "source_first_checked",
            "reviewed_with_uncertainties": "unresolved",
            "review_incomplete": "review_incomplete",
        }.get(review_sidecar.get("review_status"))
        if expected_status is None or quality_review.get("status") != expected_status:
            raise ValueError("review_sidecar_quality_status_mismatch")
    rendered = render_document(document, source_index, effective_tasks, quality_review)
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
            prompt_sha256=prompt_sha256, schema_sha256=schema_sha256,
            quality_review=quality_review, review_sidecar=review_sidecar)
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
            "quality_review": quality_review,
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
        if review_sidecar is not None:
            sidecar = _sidecar_for_generation(review_sidecar,
                generation_id=generation_id, source_sha=original_sha,
                artifact_sha256=digests)
            payload = _bytes(REVIEW_SIDECAR_FILE, sidecar)
            path = candidate / REVIEW_SIDECAR_FILE
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            digests[REVIEW_SIDECAR_FILE] = hashlib.sha256(payload).hexdigest()
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


def adopt_sealed_generation(*, source_generation: Path, source_output_dir: Path,
                            output_dir: Path, transcript_path: Path,
                            task_db_path: Path, source_sha256: str,
                            semantic_key: str, job_id: str,
                            credential_id: str, remote_batch_id: str,
                            prompt_sha256: str, schema_sha256: str,
                            accepted_document_sha256: str) -> tuple[str, Path]:
    """Publish an accepted result for the same source in another local output.

    This is a file and pointer transaction, never a new inference or task
    reconciliation. The original generation and human task edits stay intact.
    """
    from .tasks import TaskStore

    def fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    source_generation = Path(source_generation)
    source_output_dir = Path(source_output_dir)
    output_dir = Path(output_dir)
    transcript_path = Path(transcript_path)
    if (source_generation.is_symlink()
            or source_generation.parent.resolve() !=
            (source_output_dir / "summary_generations").resolve()
            or not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[0-9a-f]{12}",
                                source_generation.name)):
        raise ValueError("adoption_source_generation_invalid")
    if transcript_path.resolve() != (output_dir / "transcript.json").resolve():
        raise ValueError("adoption_transcript_path_invalid")
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".summary_publication.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            _, source_index, actual_sha = load_source(transcript_path)
            if actual_sha != source_sha256:
                raise ValueError("adoption_source_revision_changed")
            manifest_path = source_generation / "generation_manifest.json"
            run_path = source_generation / "run_manifest.json"
            document_path = source_generation / "model_document.json"
            sidecar_path = source_generation / REVIEW_SIDECAR_FILE
            for path in (manifest_path, run_path, document_path, sidecar_path):
                if path.is_symlink() or not path.is_file():
                    raise ValueError("adoption_source_artifact_invalid")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            run = json.loads(run_path.read_text(encoding="utf-8"))
            document_bytes = document_path.read_bytes()
            if hashlib.sha256(document_bytes).hexdigest() != accepted_document_sha256:
                raise ValueError("adoption_accepted_document_changed")
            document = json.loads(document_bytes)
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if not all(isinstance(value, dict) for value in
                       (manifest, run, document, sidecar)):
                raise ValueError("adoption_source_artifact_invalid")
            if (manifest.get("source_sha256") != source_sha256
                    or manifest.get("generation_id") != source_generation.name
                    or not isinstance(manifest.get("created_at"), str)):
                raise ValueError("adoption_source_manifest_mismatch")
            if (run.get("job_id") != job_id
                    or run.get("semantic_key") != semantic_key
                    or run.get("source_sha256") != source_sha256):
                raise ValueError("adoption_source_run_mismatch")
            _verify_staged_target(source_generation,
                generation_id=source_generation.name, document=document,
                source_sha=source_sha256, semantic_key=semantic_key,
                job_id=job_id, remote_batch_id=remote_batch_id,
                credential_id=credential_id,
                prompt_sha256=prompt_sha256, schema_sha256=schema_sha256,
                quality_review=run.get("quality_review"),
                review_sidecar=sidecar)
            digests = manifest["artifact_sha256"]
            for name in digests:
                path = source_generation / name
                if path.is_symlink() or not path.is_file():
                    raise ValueError("adoption_source_artifact_invalid")
            validate_document(document, source_index)
            sealed_tasks = json.loads((source_generation / "tasks.json").read_text(encoding="utf-8"))
            if not isinstance(sealed_tasks, list):
                raise ValueError("adoption_tasks_invalid")
            action_ids = [task.get("action_id") if isinstance(task, dict) else None
                          for task in sealed_tasks]
            effective = TaskStore(task_db_path).effective_for_sealed(
                source_sha256, document["tasks"], action_ids)
            render_document(document, source_index, effective, run.get("quality_review"))

            target_parent = output_dir / "summary_generations"
            target_parent.mkdir(parents=True, exist_ok=True)
            target = target_parent / source_generation.name
            pointer = output_dir / "summary_current.json"
            if pointer.exists():
                current = json.loads(pointer.read_text(encoding="utf-8"))
                current_id = current.get("generation_id")
                if not isinstance(current_id, str) or not re.fullmatch(
                        r"[0-9]{8}-[0-9]{6}-[0-9a-f]{12}", current_id):
                    raise ValueError("adoption_current_pointer_invalid")
                current_manifest = json.loads((target_parent / current_id /
                    "generation_manifest.json").read_text(encoding="utf-8"))
                if (not isinstance(current_manifest, dict)
                        or not isinstance(current_manifest.get("artifact_sha256"), dict)
                        or current_manifest.get("generation_id") != current_id
                        or current_manifest.get("source_sha256") != source_sha256
                        or current.get("verified_artifact_sha256") !=
                        current_manifest.get("artifact_sha256", {}).get("summary.md")):
                    raise ValueError("adoption_current_generation_invalid")
                if current_id == source_generation.name:
                    if (target / "generation_manifest.json").read_bytes() != manifest_path.read_bytes():
                        raise ValueError("adoption_existing_generation_conflict")
                    _verify_staged_target(target, generation_id=target.name,
                        document=document, source_sha=source_sha256,
                        semantic_key=semantic_key, job_id=job_id,
                        remote_batch_id=remote_batch_id,
                        credential_id=credential_id,
                        prompt_sha256=prompt_sha256, schema_sha256=schema_sha256,
                        quality_review=run.get("quality_review"), review_sidecar=sidecar)
                    return target.name, target
                if (not isinstance(current_manifest.get("created_at"), str)
                        or current_manifest["created_at"] >= manifest["created_at"]):
                    raise ValueError("newer_generation_already_selected")

            if target.is_symlink():
                raise ValueError("adoption_existing_generation_conflict")
            if target.exists():
                if (target / "generation_manifest.json").read_bytes() != manifest_path.read_bytes():
                    raise ValueError("adoption_existing_generation_conflict")
                _verify_staged_target(target, generation_id=target.name,
                    document=document, source_sha=source_sha256,
                    semantic_key=semantic_key, job_id=job_id,
                    remote_batch_id=remote_batch_id,
                    credential_id=credential_id,
                    prompt_sha256=prompt_sha256, schema_sha256=schema_sha256,
                    quality_review=run.get("quality_review"), review_sidecar=sidecar)
            else:
                candidate = target_parent / (".candidate-adopt-" +
                                             source_generation.name + "-" + uuid.uuid4().hex[:8])
                candidate.mkdir(mode=0o700)
                try:
                    for name in sorted(digests) + ["generation_manifest.json"]:
                        payload = (source_generation / name).read_bytes()
                        with (candidate / name).open("xb") as handle:
                            handle.write(payload)
                            handle.flush()
                            os.fsync(handle.fileno())
                    _verify_staged_target(candidate,
                        generation_id=source_generation.name, document=document,
                        source_sha=source_sha256, semantic_key=semantic_key,
                        job_id=job_id, remote_batch_id=remote_batch_id,
                        credential_id=credential_id,
                        prompt_sha256=prompt_sha256, schema_sha256=schema_sha256,
                        quality_review=run.get("quality_review"), review_sidecar=sidecar)
                    fsync_directory(candidate)
                    os.replace(candidate, target)
                    fsync_directory(target_parent)
                finally:
                    if candidate.exists():
                        shutil.rmtree(candidate)
            if hashlib.sha256(transcript_path.read_bytes()).hexdigest() != source_sha256:
                raise ValueError("adoption_source_revision_changed")
            _switch_pointer(output_dir, manifest, None)
            fsync_directory(output_dir)
            return source_generation.name, target
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def is_luna_generation(path: Path) -> bool:
    try:
        manifest = json.loads((Path(path) / "generation_manifest.json").read_text())
        return manifest.get("contract_version") == CONTRACT_VERSION
    except (OSError, ValueError, json.JSONDecodeError):
        return False
