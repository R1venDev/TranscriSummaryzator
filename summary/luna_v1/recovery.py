"""One bounded, offline recovery of a failed v3 Luna publication.

This path reads already saved writer/audit responses.  It never creates an API
client, dispatches a job, or changes the sealed source and raw responses.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import PROMPT_PATH, SCHEMA, load_source, validate_document
from .audit import AUDIT_PROMPT_PATH, AUDIT_SCHEMA, apply_audit, coverage_warnings, validate_audit
from .ledger import Ledger, write_private_json
from .publication import publish_document
from .tasks import TaskStore


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CODE_FILES = (
    "__init__.py", "audit.py", "contract.py", "engine.py", "ledger.py",
    "publication.py", "recovery.py", "render.py", "source.py", "tasks.py", "prompt_v1.md",
    "prompt_audit_v3.md", "output_schema_v1.json", "output_schema_audit_v1.json",
)
_RECOVERY_STATUS = "audit_recovered_unverified"


@dataclass(frozen=True)
class RecoveryExpected:
    source_sha256: str
    root_manifest_sha256: str
    draft_sha256: str
    audit_sha256: str
    code_sha256: str
    error_code: str

    def validate(self) -> None:
        for name in ("source_sha256", "root_manifest_sha256", "draft_sha256",
                     "audit_sha256", "code_sha256"):
            if not _SHA256.fullmatch(getattr(self, name)):
                raise ValueError(f"invalid expected {name}")
        if "due and its source references disagree" not in self.error_code:
            raise ValueError("recovery only supports the saved null-due evidence failure")


def recovery_code_sha256() -> str:
    """Pin the local code, prompts and schemas used by the offline operation."""
    directory = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in _CODE_FILES:
        digest.update(name.encode("utf-8") + b"\0")
        digest.update((directory / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _canonical_sha(value: object) -> str:
    return _sha(json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":")).encode("utf-8"))


def _read_exact(path: Path, expected_sha256: str, name: str) -> bytes:
    data = path.read_bytes()
    if _sha(data) != expected_sha256:
        raise ValueError(f"{name} SHA-256 changed")
    return data


def _write_once(path: Path, data: bytes) -> None:
    """Install an immutable evidence file without exposing partial writes."""
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"immutable recovery file changed: {path.name}")
        return
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f"immutable recovery file changed: {path.name}")
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _recovery_lock(artifacts: Path):
    lock = artifacts / ".offline_recovery.lock"
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def recover_saved_v3(*, private_root: Path, root_job_id: str,
                     audit_job_id: str, expected: RecoveryExpected,
                     apply: bool = False) -> dict:
    """Review or publish a saved v3 audit after a local contract-only failure.

    ``apply=False`` computes the candidate without writing publication state.
    ``apply=True`` is idempotent across a crash after sealing or pointer swap.
    The caller must inspect the source meaning of the saved audit separately;
    structural validation alone is not a semantic PASS.
    """
    expected.validate()
    if recovery_code_sha256() != expected.code_sha256:
        raise ValueError("recovery code SHA-256 changed")
    private_root = Path(private_root)
    ledger = Ledger(private_root)
    try:
        root = ledger.get(root_job_id)
        audit = ledger.stage(root_job_id, "audit")
        verify = ledger.stage(root_job_id, "verify")
        if (root is None or root["kind"] != "summary" or audit is None
                or audit["id"] != audit_job_id or audit["status"] != "stage_complete"
                or verify is not None or not root["remote_id"]):
            raise ValueError("recovery job graph changed")
        artifacts = Path(root["artifact_dir"])
        audit_artifacts = Path(audit["artifact_dir"])
        if (artifacts.resolve() != (private_root / "jobs" / root_job_id).resolve()
                or audit_artifacts.resolve() != (private_root / "jobs" / audit_job_id).resolve()
                or Path(audit["accepted_document_path"]) != audit_artifacts / "audit_report.json"):
            raise ValueError("recovery artifact location changed")

        with _recovery_lock(artifacts) if apply else _null_context():
            # Re-read after taking the lock; the scheduler cannot resubmit a
            # terminal root, and the conditional ledger transition catches
            # any other state change before acceptance.
            root = ledger.get(root_job_id)
            if root["status"] not in {"failed_validation", "accepted"}:
                raise ValueError("recovery root is not terminal")
            output_dir = Path(root["output_dir"])
            transcript_path = output_dir / "transcript.json"
            _, source_index, source_sha = load_source(transcript_path)
            if source_sha != expected.source_sha256 or root["source_sha256"] != source_sha:
                raise ValueError("recovery source identity changed")

            root_manifest_bytes = _read_exact(artifacts / "manifest.json",
                                               expected.root_manifest_sha256, "root manifest")
            root_manifest = json.loads(root_manifest_bytes)
            if (root_manifest.get("job_id") != root_job_id
                    or root_manifest.get("source_sha256") != source_sha
                    or root_manifest.get("quality_policy_version") != "luna_auto_audit_v3"
                    or root_manifest.get("prompt_sha256") != _sha(PROMPT_PATH.read_bytes())
                    or root_manifest.get("schema_sha256") != _canonical_sha(SCHEMA)
                    or root_manifest.get("audit_prompt_sha256") != _sha(AUDIT_PROMPT_PATH.read_bytes())
                    or root_manifest.get("audit_schema_sha256") != _canonical_sha(AUDIT_SCHEMA)):
                raise ValueError("recovery request identity changed")
            draft = json.loads(_read_exact(artifacts / "draft_document.json",
                                           expected.draft_sha256, "draft"))
            report = json.loads(_read_exact(audit_artifacts / "audit_report.json",
                                            expected.audit_sha256, "audit"))
            audit_manifest = json.loads((audit_artifacts / "manifest.json").read_text(encoding="utf-8"))
            if (audit_manifest.get("root_job_id") != root_job_id
                    or audit_manifest.get("source_sha256") != source_sha
                    or audit_manifest.get("target_document_sha256") != _canonical_sha(draft)
                    or audit_manifest.get("prompt_sha256") != root_manifest["audit_prompt_sha256"]
                    or audit_manifest.get("schema_sha256") != root_manifest["audit_schema_sha256"]):
                raise ValueError("recovery audit target changed")
            if json.loads((artifacts / "candidate_document.json").read_text(encoding="utf-8")) != draft:
                raise ValueError("original failed candidate changed")
            writer_raw = json.loads((artifacts / "native_response.json").read_text(encoding="utf-8"))
            if json.loads(writer_raw["text"]) != draft:
                raise ValueError("writer raw differs from saved draft")
            if json.loads(json.loads((audit_artifacts / "native_response.json").read_text(encoding="utf-8"))["text"]) != report:
                raise ValueError("audit raw differs from saved report")
            for stage_row, stage_artifacts in ((root, artifacts), (audit, audit_artifacts)):
                terminal = json.loads((stage_artifacts / "batch_terminal.json").read_text(encoding="utf-8"))
                if terminal.get("id") != stage_row["remote_id"] or terminal.get("status") != "completed":
                    raise ValueError("saved Batch terminal identity changed")

            validate_document(draft, source_index)
            validate_audit(report, draft, source_index)
            recovered, unresolved = apply_audit(draft, report, source_index)
            validate_document(recovered, source_index)
            warnings = coverage_warnings(report, source_index)
            warning_path = audit_artifacts / "coverage_warnings.json"
            if warning_path.exists() and json.loads(warning_path.read_text(encoding="utf-8"))["warnings"] != warnings:
                raise ValueError("saved coverage warnings changed")
            quality_review = {
                "status": _RECOVERY_STATUS,
                "unresolved_count": len(unresolved),
                "coverage_warning_count": len(warnings),
                "reason": "saved_audit_applied_after_contract_fix_without_verify",
                "audit_job_id": audit_job_id,
                "verify_job_id": None,
            }
            candidate_bytes = _json_bytes(recovered)
            generation_id = datetime.fromtimestamp(root["created_at"], timezone.utc).strftime(
                "%Y%m%d-%H%M%S") + "-" + root_job_id[:12]
            candidate_path = artifacts / "recovered_candidate_document.json"
            quality_file = artifacts / "quality_decision.json"
            original_quality_path = artifacts / "quality_decision_before_recovery.json"
            original_quality = (original_quality_path.read_bytes() if original_quality_path.exists()
                                else quality_file.read_bytes())
            original_decision = json.loads(original_quality)
            if (original_decision.get("quality_review", {}).get("status") != "audit_unavailable"
                    or original_decision["quality_review"].get("reason") != "invalid_audit_patch"):
                raise ValueError("original failed quality decision changed")
            current_quality = quality_file.read_bytes()
            updated_quality = _json_bytes({"quality_review": quality_review})
            if current_quality not in {original_quality, updated_quality}:
                raise ValueError("current quality decision changed")
            intent = {
                "recovery_version": 1,
                "root_job_id": root_job_id,
                "audit_job_id": audit_job_id,
                "source_sha256": source_sha,
                "root_manifest_sha256": expected.root_manifest_sha256,
                "draft_sha256": expected.draft_sha256,
                "audit_sha256": expected.audit_sha256,
                "code_sha256": expected.code_sha256,
                "original_quality_decision_sha256": _sha(original_quality),
                "original_error_code": expected.error_code,
                "candidate_sha256": _sha(candidate_bytes),
                "generation_id": generation_id,
                "quality_review": quality_review,
            }
            intent_bytes = _json_bytes(intent)
            if root["status"] == "accepted":
                pointer = json.loads((output_dir / "summary_current.json").read_text(encoding="utf-8"))
                generation = output_dir / "summary_generations" / generation_id
                sealed_manifest = json.loads((generation / "generation_manifest.json").read_text(encoding="utf-8"))
                if (root["accepted_document_path"] != str(candidate_path)
                        or root["generation_id"] != generation_id
                        or pointer.get("generation_id") != generation_id
                        or pointer.get("verified_artifact_sha256") != sealed_manifest["artifact_sha256"]["summary.md"]
                        or _sha((generation / "summary.md").read_bytes()) != sealed_manifest["artifact_sha256"]["summary.md"]
                        or candidate_path.read_bytes() != candidate_bytes
                        or (artifacts / "recovery_intent.json").read_bytes() != intent_bytes
                        or current_quality != updated_quality):
                    raise ValueError("accepted recovery is inconsistent")
                if apply:
                    _write_once(artifacts / "recovery_receipt.json", _json_bytes({
                        "status": "accepted_recovered", "root_job_id": root_job_id,
                        "audit_job_id": audit_job_id, "generation_id": generation_id,
                        "source_sha256": source_sha, "candidate_sha256": _sha(candidate_bytes),
                        "new_api_dispatches": 0,
                    }))
                return {"status": "already_recovered", "job_id": root_job_id,
                        "generation_id": generation_id, "new_generations": 0}
            if root["error_code"] != expected.error_code or root["generation_id"] is not None:
                raise ValueError("original failure identity changed")
            pointer_path = output_dir / "summary_current.json"
            already_selected = pointer_path.exists()
            if pointer_path.exists():
                pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
                if pointer.get("generation_id") != generation_id:
                    raise ValueError("a different generation is selected")
            result = {"status": "ready_offline" if not apply else "accepted_recovered",
                      "job_id": root_job_id, "audit_job_id": audit_job_id,
                      "generation_id": generation_id, "source_sha256": source_sha,
                      "tasks": len(recovered["tasks"]), "patches": len(report["patches"]),
                      "unresolved": len(unresolved), "coverage_warnings": len(warnings),
                      "new_generations": 0 if not apply or already_selected else 1,
                      "new_api_dispatches": 0}
            if not apply:
                return result

            _write_once(original_quality_path, original_quality)
            _write_once(candidate_path, candidate_bytes)
            _write_once(artifacts / "recovery_intent.json", intent_bytes)
            task_store = TaskStore(private_root / "tasks.sqlite3")
            plan = task_store.preview_reconcile(source_sha, recovered["tasks"])

            def commit_if_pointer_unchanged() -> None:
                # publish_document invokes this while holding its per-output
                # publication lock, immediately before the pointer swap.
                # The earlier preflight alone cannot close this race.
                if pointer_path.exists():
                    selected = json.loads(pointer_path.read_text(encoding="utf-8"))
                    if selected.get("generation_id") != generation_id:
                        raise ValueError("a different generation was selected before recovery commit")
                task_store.commit_reconcile(plan)

            published_id, generation_path = publish_document(
                document=recovered, source_index=source_index,
                transcript_path=transcript_path, output_dir=output_dir,
                semantic_key=root["semantic_key"], job_id=root_job_id,
                remote_batch_id=root["remote_id"], credential_id=root["credential_id"],
                prompt_sha256=root_manifest["prompt_sha256"],
                schema_sha256=root_manifest["schema_sha256"],
                effective_tasks=plan.effective_tasks, generation_id=generation_id,
                quality_review=quality_review,
                before_pointer=commit_if_pointer_unchanged,
            )
            manifest = json.loads((generation_path / "generation_manifest.json").read_text(encoding="utf-8"))
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            if (published_id != generation_id or pointer.get("generation_id") != generation_id
                    or pointer.get("verified_artifact_sha256") != manifest["artifact_sha256"]["summary.md"]):
                raise ValueError("recovered publication pointer mismatch")
            write_private_json(quality_file, {"quality_review": quality_review})
            ledger.accept_recovered_failed(
                root_job_id, audit_job_id=audit_job_id,
                expected_error_code=expected.error_code,
                document_path=candidate_path, generation_id=generation_id,
            )
            _write_once(artifacts / "recovery_receipt.json", _json_bytes({
                "status": "accepted_recovered", "root_job_id": root_job_id,
                "audit_job_id": audit_job_id, "generation_id": generation_id,
                "source_sha256": source_sha, "candidate_sha256": _sha(candidate_bytes),
                "new_api_dispatches": 0,
            }))
            return result
    finally:
        ledger.close()


@contextmanager
def _null_context():
    yield
