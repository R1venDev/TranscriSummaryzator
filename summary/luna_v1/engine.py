"""Summary-only Luna Batch dispatch, recovery and local publication.

The watcher calls submit once and a low-frequency scheduler calls poll_once.
Neither path touches audio, ASR, diarization, Ollama, or Plane.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_UP
from email.utils import parsedate_to_datetime
from pathlib import Path

from scripts.summary_credentials import CredentialError, CredentialStore, credential_dispatch_guard

from . import PROMPT_PATH, SCHEMA, SCHEMA_ID, load_source, validate_document
from .audit import (AUDIT_PROMPT_PATH, AUDIT_SCHEMA, AUDIT_SCHEMA_ID,
                    apply_audit, build_audit_input, coverage_warnings,
                    validate_audit)
from .batch import BatchClient, BatchError, MODEL, TERMINAL, extract_one_completed, valid_batch_id
from .ledger import Ledger, usd_micros, write_private_json
from .publication import publish_document
from .route import MAX_COMPLETION_TOKENS, RouteBlocked, verify_batch_route
from .tasks import RevisionConflict, TaskStore


PRIVACY_MODE = "batch_gateway_retention_up_to_30d_provider_zdr_off_user_authorized"
REASONING_EFFORT = "medium"
QUALITY_POLICY_VERSION = "luna_auto_audit_v2"
QUALITY_CREDENTIAL_WAIT_SECONDS = 30 * 60
QUALITY_BATCH_WAIT_SECONDS = 26 * 60 * 60


def _json_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _request_body(source_text: str) -> dict:
    instruction = PROMPT_PATH.read_text(encoding="utf-8")
    return {
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": source_text},
        ],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": SCHEMA_ID, "strict": True, "schema": SCHEMA,
        }},
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "reasoning": {"effort": REASONING_EFFORT},
    }


def _semantic_identity(source_sha256: str, source_text: str, workspace_scope: str,
                       prompt_sha256: str, schema_sha256: str, force_nonce: str | None) -> str:
    material = {
        "source_sha256": source_sha256,
        "projected_source_sha256": _sha(source_text.encode("utf-8")),
        "workspace_scope": workspace_scope,
        "prompt_sha256": prompt_sha256,
        "schema_sha256": schema_sha256,
        "model": MODEL,
        "privacy_mode": PRIVACY_MODE,
        "reasoning_effort": REASONING_EFFORT,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "quality_policy_version": QUALITY_POLICY_VERSION,
        "audit_prompt_sha256": _sha(AUDIT_PROMPT_PATH.read_bytes()),
        "audit_schema_sha256": _sha(_json_bytes(AUDIT_SCHEMA)),
        "force_nonce": force_nonce,
    }
    return _sha(_json_bytes(material))


def _credential_store(private_root: Path) -> CredentialStore:
    return CredentialStore(Path(private_root) / "credentials.sqlite3")


def _publish_verified_document(*, document: dict, job: dict, request_manifest: dict,
                               source_index: dict, transcript_path: Path,
                               output_dir: Path, private_root: Path) -> tuple[str, Path]:
    """Seal one model document, then commit matching task edits before pointer swap."""
    validate_document(document, source_index)
    task_store = TaskStore(Path(private_root) / "tasks.sqlite3")
    plan = task_store.preview_reconcile(source_index["source_sha256"], document["tasks"])
    generation_id = datetime.fromtimestamp(job["created_at"], timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + job["id"][:12]
    quality_file = Path(job["artifact_dir"]) / "quality_decision.json"
    quality_review = json.loads(quality_file.read_text(encoding="utf-8"))["quality_review"] if quality_file.exists() else None
    return publish_document(
        document=document, source_index=source_index, transcript_path=transcript_path,
        output_dir=output_dir, semantic_key=job["semantic_key"],
        job_id=job["id"], remote_batch_id=job["remote_id"],
        credential_id=job["credential_id"],
        prompt_sha256=request_manifest["prompt_sha256"],
        schema_sha256=request_manifest["schema_sha256"],
        effective_tasks=plan.effective_tasks, generation_id=generation_id,
        quality_review=quality_review,
        before_pointer=lambda: task_store.commit_reconcile(plan),
    )


def _reuse_accepted(*, job: dict, output_dir: Path, transcript_path: Path,
                    source_index: dict, private_root: Path, ledger: Ledger) -> dict:
    """Publish a previously accepted model document for another local consumer."""
    artifacts = Path(job["artifact_dir"])
    request_manifest = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
    if request_manifest["source_sha256"] != source_index["source_sha256"]:
        raise ValueError("accepted_cache_source_mismatch")
    document = json.loads(Path(job["accepted_document_path"]).read_text(encoding="utf-8"))
    generation_id, _ = _publish_verified_document(
        document=document, job=job, request_manifest=request_manifest,
        source_index=source_index, transcript_path=transcript_path,
        output_dir=output_dir, private_root=private_root,
    )
    ledger.mark_consumer(job["semantic_key"], output_dir, generation_id=generation_id)
    return {"status": "accepted_cache_hit", "job_id": job["id"],
            "semantic_key": job["semantic_key"], "generation_id": generation_id,
            "new_generations": 0}


def submit(*, transcript_path: Path, output_dir: Path, private_root: Path,
           force_nonce: str | None = None, client_factory=BatchClient) -> dict:
    """Persist intent and upper-bound reserve before one physical POST."""
    transcript_path = Path(transcript_path)
    output_dir = Path(output_dir)
    source_text, source_index, source_sha = load_source(transcript_path)
    prompt_sha = _sha(PROMPT_PATH.read_bytes())
    schema_sha = _sha(_json_bytes(SCHEMA))
    request_body = _request_body(source_text)
    serialized_request = _json_bytes(request_body)
    ledger = Ledger(private_root)
    guard = None
    try:
        store = _credential_store(private_root)
        guard_context = credential_dispatch_guard(getattr(store, "path", Path(private_root) / "credentials.sqlite3"))
        guard_context.__enter__()
        guard = guard_context
        candidates = store.dispatch_candidates()
        if not candidates:
            return {"status": "credential_required", "source_sha256": source_sha}
        selected = candidates[0]  # ordered primary; no automatic account hopping
        scope = selected.get("workspace_id") or ("unverified-key-version:" + selected["id"] + ":" + str(selected["version"]))
        semantic_key = _semantic_identity(source_sha, source_text, scope, prompt_sha, schema_sha, force_nonce)
        prior = ledger.attach_consumer(semantic_key, source_sha, output_dir)
        if prior:
            if prior["status"] == "accepted":
                return _reuse_accepted(job=prior, output_dir=output_dir,
                                       transcript_path=transcript_path,
                                       source_index=source_index, private_root=private_root,
                                       ledger=ledger)
            return {"status": prior["status"], "job_id": prior["id"], "semantic_key": semantic_key}
        token = store.reveal_for_dispatch(selected["id"], selected["version"])
        client = client_factory(token)
        route = verify_batch_route(client)  # free GET /key and catalog checks
        if route.workspace_id != selected.get("workspace_id"):
            raise RouteBlocked("credential_workspace_changed_since_check")
        reserve_micros = route.reserve_microusd(serialized_request)
        decision = ledger.reserve(
            semantic_key=semantic_key, source_sha256=source_sha, output_dir=output_dir,
            credential_id=selected["id"], credential_version=selected["version"],
            workspace_id=selected.get("workspace_id"), max_cost_microusd=reserve_micros,
        )
        if decision.kind != "new":
            return {"status": decision.kind, "reason": decision.reason, "job_id": decision.job_id}
        job = ledger.get(decision.job_id)
        artifacts = Path(job["artifact_dir"])
        write_private_json(artifacts / "manifest.json", {
            "job_id": job["id"], "semantic_key": semantic_key,
            "source_sha256": source_sha, "output_dir": str(output_dir),
            "credential_id": selected["id"], "credential_version": selected["version"],
            "workspace_id": selected.get("workspace_id"), "model": MODEL,
            "provider": "openai", "privacy_mode": PRIVACY_MODE,
            "prompt_sha256": prompt_sha, "schema_sha256": schema_sha,
            "quality_policy_version": QUALITY_POLICY_VERSION,
            "audit_prompt_sha256": _sha(AUDIT_PROMPT_PATH.read_bytes()),
            "audit_schema_sha256": _sha(_json_bytes(AUDIT_SCHEMA)),
            "request_sha256": _sha(serialized_request),
            "reserve_microusd": reserve_micros,
            "pricing": {
                "prompt_usd_per_token": str(route.prompt_usd_per_token),
                "completion_usd_per_token": str(route.completion_usd_per_token),
                "cache_write_usd_per_token": str(route.cache_write_usd_per_token),
                "request_usd": str(route.request_usd),
            },
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "reasoning_effort": REASONING_EFFORT,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        write_private_json(artifacts / "request.json", request_body)
        # An admin may rotate after the free preflight. The durable reserved
        # row now blocks replacement; re-read the exact version before POST.
        try:
            token = store.reveal_for_dispatch(selected["id"], selected["version"])
        except CredentialError:
            ledger.cancel_before_submit(job["id"], "credential_changed_before_post")
            return {"status": "cancelled_before_submit", "job_id": job["id"]}
        if not ledger.mark_submitting(job["id"]):
            return {"status": "submission_state_conflict", "job_id": job["id"]}
        client = client_factory(token)
        try:
            reply = client.submit(job["custom_id"], request_body)
            write_private_json(artifacts / "submit_response.json", reply.body)
            remote_id = reply.body.get("id")
            if reply.status_code != 202 or not valid_batch_id(remote_id):
                ledger.submission_result(job["id"], remote_id=None, error_code="unexpected_submit_response")
                return {"status": "submission_unknown", "job_id": job["id"]}
            ledger.submission_result(job["id"], remote_id=remote_id)
            return {"status": "submitted", "job_id": job["id"], "remote_id": remote_id,
                    "reserved_usd": reserve_micros / 1_000_000}
        except BatchError as exc:
            definite = exc.code in {400, 401, 402, 403, 404, 422, 429}
            ledger.submission_result(job["id"], remote_id=None, error_code=exc.reason,
                                     definite_rejection=definite)
            write_private_json(artifacts / "submit_error.json", {
                "code": exc.code, "reason": exc.reason, "retry_after": exc.retry_after,
                "definite_rejection": definite,
            })
            return {"status": "rejected_before_submit" if definite else "submission_unknown",
                    "job_id": job["id"], "error_code": exc.reason}
    except (CredentialError, RouteBlocked, BatchError) as exc:
        return {"status": "preflight_blocked", "reason": str(exc)}
    finally:
        try:
            if guard is not None:
                guard.__exit__(None, None, None)
        finally:
            ledger.close()


def _cost_micros(usage: dict) -> int | None:
    value = usage.get("cost")
    if value is None:
        return None
    try:
        amount = Decimal(str(value))
    except InvalidOperation:
        return None
    if not amount.is_finite() or amount < 0:
        return None
    return int((amount * 1_000_000).to_integral_value(rounding=ROUND_UP))


def _append_poll_event(artifacts: Path, value: dict) -> None:
    """Retain each physical GET/DELETE attempt without copying a secret to logs."""
    artifacts.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = _json_bytes({"at": datetime.now(timezone.utc).isoformat(), **value}) + b"\n"
    fd = os.open(artifacts / "batch_poll_events.jsonl", os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "ab") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        raise


def _poll_backoff(retry_after: str | None) -> int:
    if retry_after:
        try:
            return max(30, min(3600, int(float(retry_after))))
        except ValueError:
            try:
                seconds = (parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds()
                return max(30, min(3600, math.ceil(seconds)))
            except (TypeError, ValueError, OverflowError):
                pass
    return 300


def _finish_raw(ledger: Ledger, job: dict) -> dict:
    artifacts = Path(job["artifact_dir"])
    batch = json.loads((artifacts / "batch_terminal.json").read_text(encoding="utf-8"))
    submission = json.loads((artifacts / "submit_response.json").read_text(encoding="utf-8"))
    if submission.get("id") != job["remote_id"] or batch.get("model") != submission.get("model"):
        raise ValueError("batch_submission_identity_mismatch")
    request_manifest = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
    body, usage = extract_one_completed(batch, job["custom_id"])
    resolved_model = submission.get("model")
    allowed_models = {MODEL, "openai/gpt-6-luna"}
    if isinstance(resolved_model, str) and resolved_model.startswith("openai/gpt-6-luna-"):
        allowed_models.add(resolved_model)
    if body.get("model") not in allowed_models:
        raise ValueError("response_model_mismatch")
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("response_choices_invalid")
    choice = choices[0]
    if choice.get("finish_reason") != "stop":
        raise ValueError("response_not_terminal_stop")
    message = choice.get("message") or {}
    if message.get("refusal") or not isinstance(message.get("content"), str):
        raise ValueError("response_refusal_or_content_invalid")
    raw_text = message["content"]
    write_private_json(artifacts / "native_response.json", {"text": raw_text, "usage": usage})
    document = json.loads(raw_text)
    transcript_path = Path(job["output_dir"]) / "transcript.json"
    _, source_index, source_sha = load_source(transcript_path)
    if source_sha != job["source_sha256"]:
        raise ValueError("source_revision_changed")
    if job["kind"] == "summary":
        if (_sha(PROMPT_PATH.read_bytes()) != request_manifest["prompt_sha256"]
                or _sha(_json_bytes(SCHEMA)) != request_manifest["schema_sha256"]):
            raise ValueError("code_or_prompt_changed_while_pending")
        if "quality_policy_version" not in request_manifest:
            # Complete a pre-migration Batch with its original one-call
            # contract; do not change an already submitted request's policy.
            validate_document(document, source_index)
            write_private_json(artifacts / "candidate_document.json", document)
            generation_id, generation_path = _publish_verified_document(
                document=document, job=job, request_manifest=request_manifest,
                source_index=source_index, transcript_path=transcript_path,
                output_dir=Path(job["output_dir"]), private_root=ledger.root,
            )
            ledger.accepted(job["id"], artifacts / "candidate_document.json", generation_id)
            ledger.mark_consumer(job["semantic_key"], Path(job["output_dir"]), generation_id=generation_id)
            return {"status": "accepted_legacy", "job_id": job["id"],
                    "generation_id": generation_id, "generation_path": str(generation_path),
                    "consumer_results": _publish_accepted_consumers(ledger, ledger.get(job["id"]))}
        if (request_manifest.get("quality_policy_version") != QUALITY_POLICY_VERSION
                or request_manifest.get("audit_prompt_sha256") != _sha(AUDIT_PROMPT_PATH.read_bytes())
                or request_manifest.get("audit_schema_sha256") != _sha(_json_bytes(AUDIT_SCHEMA))):
            raise ValueError("code_or_prompt_changed_while_pending")
        if not isinstance(document, dict):
            raise ValueError("draft_document_not_object")
        try:
            validate_document(document, source_index)
        except (ValueError, KeyError, TypeError) as exc:
            # A parseable draft can still be repaired by the separate audit.
            # Preserve the exact failure and never treat it as an accepted hit.
            write_private_json(artifacts / "draft_validation.json", {
                "status": "invalid", "reason": type(exc).__name__ + ":" + str(exc)[:200]})
        write_private_json(artifacts / "draft_document.json", document)
        ledger.mark_quality_pending(job["id"])
        return {"status": "draft_ready", "job_id": job["id"],
                "reserved_usd": job["reserved_microusd"] / 1_000_000}
    if job["kind"] not in {"audit", "verify"}:
        raise ValueError("unknown_quality_stage")
    if (_sha(AUDIT_PROMPT_PATH.read_bytes()) != request_manifest["prompt_sha256"]
            or _sha(_json_bytes(AUDIT_SCHEMA)) != request_manifest["schema_sha256"]):
        raise ValueError("audit_code_or_prompt_changed_while_pending")
    root = ledger.get(job["root_job_id"])
    if root is None or root["source_sha256"] != source_sha:
        raise ValueError("audit_root_source_mismatch")
    target_file = Path(root["artifact_dir"]) / ("draft_document.json" if job["kind"] == "audit" else "revised_document.json")
    target_bytes = target_file.read_bytes()
    target = json.loads(target_bytes)
    if _sha(_json_bytes(target)) != request_manifest["target_document_sha256"]:
        raise ValueError("audit_target_changed")
    validate_audit(document, target, source_index, mode=job["kind"])
    warnings = coverage_warnings(document, source_index)
    if warnings:
        write_private_json(artifacts / "coverage_warnings.json", {"warnings": warnings})
    report_file = artifacts / "audit_report.json"
    write_private_json(report_file, document)
    ledger.stage_completed(job["id"], report_file)
    return {"status": "stage_complete", "stage": job["kind"], "job_id": job["id"],
            "root_job_id": root["id"], "findings": len(document["findings"])}


def _quality_request_body(source_text: str, target: dict, *, kind: str,
                          prior_findings: list[dict]) -> dict:
    if kind not in {"audit", "verify"}:
        raise ValueError("invalid quality request kind")
    return {
        "messages": [
            {"role": "system", "content": AUDIT_PROMPT_PATH.read_text(encoding="utf-8")},
            {"role": "user", "content": build_audit_input(
                source_text, target, mode=kind, prior_findings=prior_findings)},
        ],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": AUDIT_SCHEMA_ID, "strict": True, "schema": AUDIT_SCHEMA,
        }},
        "max_completion_tokens": 12_000 if kind == "audit" else 6_000,
        "reasoning": {"effort": REASONING_EFFORT},
    }


def _quality_stage_key(root: dict, kind: str, request_body: dict) -> str:
    return _sha(_json_bytes({
        "root_semantic_key": root["semantic_key"], "kind": kind,
        "request_sha256": _sha(_json_bytes(request_body)),
        "prompt_sha256": _sha(AUDIT_PROMPT_PATH.read_bytes()),
        "schema_sha256": _sha(_json_bytes(AUDIT_SCHEMA)),
        "quality_policy_version": QUALITY_POLICY_VERSION,
    }))


def _start_quality_stage(ledger: Ledger, root: dict, kind: str, target: dict,
                         prior_findings: list[dict], client_factory) -> dict:
    """Reserve, record and submit one dependent call, never duplicating a POST."""
    transcript_path = Path(root["output_dir"]) / "transcript.json"
    source_text, _, source_sha = load_source(transcript_path)
    if source_sha != root["source_sha256"]:
        raise ValueError("source_revision_changed_before_audit")
    request_body = _quality_request_body(source_text, target, kind=kind,
                                         prior_findings=prior_findings)
    request_bytes = _json_bytes(request_body)
    target_sha = _sha(_json_bytes(target))
    semantic_key = _quality_stage_key(root, kind, request_body)
    existing = ledger.stage(root["id"], kind)
    if existing is not None and existing["semantic_key"] != semantic_key:
        raise ValueError("quality_stage_identity_changed")
    guard = None
    stage = existing

    def cancel_reserved(code: str) -> None:
        current = ledger.get(stage["id"]) if stage is not None else None
        if current is not None and current["status"] == "reserved":
            ledger.cancel_before_submit(current["id"], code)

    try:
        store = _credential_store(ledger.root)
        guard_context = credential_dispatch_guard(getattr(store, "path", ledger.root / "credentials.sqlite3"))
        guard_context.__enter__()
        guard = guard_context
        if existing is None:
            candidates = store.dispatch_candidates()
            if not candidates:
                return {"status": "unavailable", "reason": "credential_required"}
            selected = candidates[0]
        else:
            selected = {"id": existing["credential_id"], "version": existing["credential_version"],
                        "workspace_id": existing["workspace_id"]}
        if selected.get("workspace_id") != root["workspace_id"]:
            cancel_reserved("audit_workspace_changed")
            return {"status": "unavailable", "reason": "audit_workspace_changed"}
        token = store.reveal_for_dispatch(selected["id"], selected["version"])
        route = verify_batch_route(client_factory(token))
        if route.workspace_id != root["workspace_id"]:
            cancel_reserved("audit_route_workspace_changed")
            return {"status": "unavailable", "reason": "audit_route_workspace_changed"}
        reserve = route.reserve_microusd(request_bytes,
                                         max_completion_tokens=request_body["max_completion_tokens"])
        decision = ledger.reserve(
            semantic_key=semantic_key, source_sha256=source_sha,
            output_dir=Path(root["output_dir"]), credential_id=selected["id"],
            credential_version=selected["version"], workspace_id=route.workspace_id,
            max_cost_microusd=reserve, kind=kind, root_job_id=root["id"],
        )
        if decision.kind == "blocked":
            return {"status": "unavailable", "reason": decision.reason}
        stage = ledger.get(decision.job_id)
        artifacts = Path(stage["artifact_dir"])
        manifest = {
            "job_id": stage["id"], "root_job_id": root["id"], "stage": kind,
            "semantic_key": semantic_key, "source_sha256": source_sha,
            "target_document_sha256": target_sha,
            "request_sha256": _sha(request_bytes),
            "prompt_sha256": _sha(AUDIT_PROMPT_PATH.read_bytes()),
            "schema_sha256": _sha(_json_bytes(AUDIT_SCHEMA)),
            "credential_id": stage["credential_id"],
            "credential_version": stage["credential_version"],
            "workspace_id": stage["workspace_id"], "model": MODEL,
            "privacy_mode": PRIVACY_MODE,
            "reserve_microusd": stage["reserved_microusd"],
            "max_completion_tokens": request_body["max_completion_tokens"],
        }
        manifest_path = artifacts / "manifest.json"
        if manifest_path.exists():
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            if saved != manifest:
                raise ValueError("saved_quality_manifest_changed")
        else:
            write_private_json(manifest_path, manifest)
            write_private_json(artifacts / "request.json", request_body)
        if stage["status"] != "reserved":
            return {"status": stage["status"], "job_id": stage["id"]}
        try:
            token = store.reveal_for_dispatch(stage["credential_id"], stage["credential_version"])
        except CredentialError:
            ledger.cancel_before_submit(stage["id"], "credential_changed_before_post")
            return {"status": "unavailable", "reason": "credential_changed_before_post", "job_id": stage["id"]}
        if not ledger.mark_submitting(stage["id"]):
            return {"status": "submitting", "job_id": stage["id"]}
        client = client_factory(token)
        try:
            reply = client.submit(stage["custom_id"], request_body)
            write_private_json(artifacts / "submit_response.json", reply.body)
            remote_id = reply.body.get("id")
            if reply.status_code != 202 or not valid_batch_id(remote_id):
                ledger.submission_result(stage["id"], remote_id=None,
                                         error_code="unexpected_submit_response")
                return {"status": "submission_unknown", "job_id": stage["id"]}
            ledger.submission_result(stage["id"], remote_id=remote_id)
            return {"status": "submitted", "job_id": stage["id"], "remote_id": remote_id}
        except BatchError as exc:
            definite = exc.code in {400, 401, 402, 403, 404, 422, 429}
            ledger.submission_result(stage["id"], remote_id=None, error_code=exc.reason,
                                     definite_rejection=definite)
            write_private_json(artifacts / "submit_error.json", {
                "code": exc.code, "reason": exc.reason, "retry_after": exc.retry_after,
                "definite_rejection": definite,
            })
            return {"status": "rejected_before_submit" if definite else "submission_unknown",
                    "job_id": stage["id"], "reason": exc.reason}
    except (CredentialError, RouteBlocked, BatchError) as exc:
        cancel_reserved("stage_preflight_unavailable")
        return {"status": "unavailable", "reason": str(exc)[:120]}
    finally:
        if guard is not None:
            guard.__exit__(None, None, None)


def _finalize_quality(ledger: Ledger, root: dict, document: dict, source_index: dict,
                      *, status: str, unresolved_count: int = 0,
                      reason: str | None = None) -> dict:
    artifacts = Path(root["artifact_dir"])
    transcript_path = Path(root["output_dir"]) / "transcript.json"
    request_manifest = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
    audit = ledger.stage(root["id"], "audit")
    verify = ledger.stage(root["id"], "verify")
    coverage_warning_count = 0
    for stage in (audit, verify):
        if stage is None:
            continue
        warning_file = Path(stage["artifact_dir"]) / "coverage_warnings.json"
        if warning_file.exists():
            coverage_warning_count += len(json.loads(warning_file.read_text(encoding="utf-8"))["warnings"])
    if coverage_warning_count and status == "checked":
        status = "coverage_incomplete"
        reason = "coverage_report_inconsistent"
    quality_review = {
        "status": status, "unresolved_count": unresolved_count,
        "coverage_warning_count": coverage_warning_count,
        "reason": reason, "audit_job_id": audit["id"] if audit else None,
        "verify_job_id": verify["id"] if verify else None,
    }
    write_private_json(artifacts / "candidate_document.json", document)
    write_private_json(artifacts / "quality_decision.json", {"quality_review": quality_review})
    generation_id, generation_path = _publish_verified_document(
        document=document, job=root, request_manifest=request_manifest,
        source_index=source_index, transcript_path=transcript_path,
        output_dir=Path(root["output_dir"]), private_root=ledger.root,
    )
    ledger.accepted(root["id"], artifacts / "candidate_document.json", generation_id)
    ledger.mark_consumer(root["semantic_key"], Path(root["output_dir"]), generation_id=generation_id)
    consumer_results = _publish_accepted_consumers(ledger, ledger.get(root["id"]))
    return {"status": "accepted", "job_id": root["id"], "generation_id": generation_id,
            "generation_path": str(generation_path), "quality_review": quality_review,
            "consumer_results": consumer_results}


def _stage_timeout_reason(stage: dict) -> str | None:
    age = time.time() - stage["created_at"]
    if stage["status"] == "submitting" and time.time() - stage["updated_at"] > 300:
        return "submission_outcome_unknown"
    if stage["status"] == "credential_required" and age > QUALITY_CREDENTIAL_WAIT_SECONDS:
        return "credential_required_timeout"
    if stage["status"] in {"submitted", "polling", "credential_required"} and age > QUALITY_BATCH_WAIT_SECONDS:
        return "batch_result_timeout"
    return None


def _advance_quality(ledger: Ledger, root: dict, client_factory) -> dict:
    """Advance one saved workflow; semantic uncertainty is published visibly."""
    artifacts = Path(root["artifact_dir"])
    transcript_path = Path(root["output_dir"]) / "transcript.json"
    source_text, source_index, source_sha = load_source(transcript_path)
    if source_sha != root["source_sha256"]:
        raise ValueError("source_revision_changed_during_quality_review")
    draft = json.loads((artifacts / "draft_document.json").read_text(encoding="utf-8"))
    audit = ledger.stage(root["id"], "audit")
    if audit is None or audit["status"] == "reserved":
        started = _start_quality_stage(ledger, root, "audit", draft, [], client_factory)
        if started["status"] == "unavailable":
            return _finalize_quality(ledger, root, draft, source_index,
                status="audit_unavailable", reason=started.get("reason"))
        return {"status": "audit_" + started["status"], "job_id": root["id"],
                "stage_job_id": started.get("job_id")}
    if audit["status"] != "stage_complete":
        if audit["status"] in {"submission_unknown", "rejected_before_submit",
                               "cancelled_before_submit", "failed_validation",
                               "remote_failed", "remote_expired", "remote_cancelled"}:
            return _finalize_quality(ledger, root, draft, source_index,
                status="audit_unavailable", reason=audit["status"])
        timeout = _stage_timeout_reason(audit)
        if timeout:
            return _finalize_quality(ledger, root, draft, source_index,
                status="audit_unavailable", reason=timeout)
        return {"status": "audit_pending", "job_id": root["id"], "stage_job_id": audit["id"]}
    report = json.loads(Path(audit["accepted_document_path"]).read_text(encoding="utf-8"))
    try:
        revised, unresolved = apply_audit(draft, report, source_index, mode="audit")
    except (ValueError, json.JSONDecodeError) as exc:
        write_private_json(artifacts / "audit_apply_error.json", {"reason": str(exc)[:200]})
        return _finalize_quality(ledger, root, draft, source_index,
            status="audit_unavailable", reason="invalid_audit_patch")
    write_private_json(artifacts / "revised_document.json", revised)
    if not report["patches"]:
        return _finalize_quality(ledger, root, revised, source_index,
            status="unresolved" if unresolved else "checked", unresolved_count=len(unresolved))
    verify = ledger.stage(root["id"], "verify")
    if verify is None or verify["status"] == "reserved":
        started = _start_quality_stage(ledger, root, "verify", revised,
                                       report["findings"], client_factory)
        if started["status"] == "unavailable":
            return _finalize_quality(ledger, root, revised, source_index,
                status="verify_unavailable", unresolved_count=len(unresolved),
                reason=started.get("reason"))
        return {"status": "verify_" + started["status"], "job_id": root["id"],
                "stage_job_id": started.get("job_id")}
    if verify["status"] != "stage_complete":
        if verify["status"] in {"submission_unknown", "rejected_before_submit",
                                "cancelled_before_submit", "failed_validation",
                                "remote_failed", "remote_expired", "remote_cancelled"}:
            return _finalize_quality(ledger, root, revised, source_index,
                status="verify_unavailable", unresolved_count=len(unresolved),
                reason=verify["status"])
        timeout = _stage_timeout_reason(verify)
        if timeout:
            return _finalize_quality(ledger, root, revised, source_index,
                status="verify_unavailable", unresolved_count=len(unresolved),
                reason=timeout)
        return {"status": "verify_pending", "job_id": root["id"], "stage_job_id": verify["id"]}
    verification = json.loads(Path(verify["accepted_document_path"]).read_text(encoding="utf-8"))
    try:
        checked, later_unresolved = apply_audit(revised, verification, source_index, mode="verify")
    except (ValueError, json.JSONDecodeError) as exc:
        write_private_json(artifacts / "verify_apply_error.json", {"reason": str(exc)[:200]})
        return _finalize_quality(ledger, root, revised, source_index,
            status="verify_unavailable", unresolved_count=len(unresolved),
            reason="invalid_verify_report")
    if verification["patches"]:
        write_private_json(artifacts / "postverify_document.json", checked)
    return _finalize_quality(ledger, root, checked, source_index,
        status=("postverify_corrected_unchecked" if verification["patches"] else
                "unresolved" if unresolved or later_unresolved else "checked"),
        unresolved_count=len(unresolved) + len(later_unresolved),
        reason="final_bounded_correction" if verification["patches"] else None)


def _publish_accepted_consumers(ledger: Ledger, job: dict) -> list[dict]:
    """Fan out one accepted raw result locally; never ask the model again."""
    manifest = json.loads((Path(job["artifact_dir"]) / "manifest.json").read_text(encoding="utf-8"))
    document = json.loads(Path(job["accepted_document_path"]).read_text(encoding="utf-8"))
    results = []
    for consumer in ledger.consumer_rows(job["semantic_key"], status="pending"):
        output_dir = Path(consumer["output_dir"])
        transcript_path = output_dir / "transcript.json"
        try:
            _, source_index, source_sha = load_source(transcript_path)
            if source_sha != job["source_sha256"]:
                raise ValueError("consumer_source_revision_changed")
            generation_id, _ = _publish_verified_document(
                document=document, job=job, request_manifest=manifest,
                source_index=source_index, transcript_path=transcript_path,
                output_dir=output_dir, private_root=ledger.root,
            )
            ledger.mark_consumer(job["semantic_key"], output_dir, generation_id=generation_id)
            results.append({"output_dir": str(output_dir), "status": "published", "generation_id": generation_id})
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            # An I/O interruption can be retried from the accepted document.
            # A changed source is a per-consumer failure, not a failed model.
            if isinstance(exc, (OSError, RevisionConflict)):
                status = "pending_retry"
            else:
                ledger.mark_consumer(job["semantic_key"], output_dir,
                                     error_code=type(exc).__name__ + ":" + str(exc)[:100])
                status = "failed_source_revision" if str(exc) == "consumer_source_revision_changed" else "failed_consumer"
            results.append({"output_dir": str(output_dir), "status": status,
                            "reason": type(exc).__name__ + ":" + str(exc)[:120]})
    return results


def poll_once(*, private_root: Path, client_factory=BatchClient) -> list[dict]:
    """One bounded scheduler tick; never sleeps or submits a second POST."""
    ledger = Ledger(private_root)
    outcomes = []
    try:
        for job in ledger.raw_ready():
            try:
                outcomes.append(_finish_raw(ledger, job))
            except (OSError, RevisionConflict) as exc:
                ledger.defer_raw(job["id"], delay_seconds=180, error_code=type(exc).__name__ + ":" + str(exc)[:100])
                outcomes.append({"status": "publication_pending_retry", "job_id": job["id"],
                                 "reason": type(exc).__name__})
            except (ValueError, json.JSONDecodeError) as exc:
                ledger.failed_validation(job["id"], type(exc).__name__ + ":" + str(exc)[:120])
                outcomes.append({"status": "failed_validation", "job_id": job["id"], "reason": str(exc)[:120]})
        for job in ledger.accepted_with_pending_consumers():
            outcomes.append({"status": "consumer_recovery", "job_id": job["id"],
                             "consumer_results": _publish_accepted_consumers(ledger, job)})
        for job in ledger.pending_remote()[:4]:
            artifacts = Path(job["artifact_dir"])
            try:
                store = _credential_store(private_root)
                token = store.reveal_for_existing_job(job["credential_id"], job["credential_version"])
            except CredentialError:
                ledger.credential_required(job["id"])
                _append_poll_event(artifacts, {"operation": "credential_lookup", "status": "credential_required",
                                               "remote_id": job["remote_id"]})
                outcomes.append({"status": "credential_required", "job_id": job["id"], "remote_id": job["remote_id"]})
                continue
            client = client_factory(token)
            try:
                reply = client.get(job["remote_id"])
            except BatchError as exc:
                delay = _poll_backoff(exc.retry_after)
                ledger.defer_poll(job["id"], delay_seconds=delay, error_code=exc.reason)
                _append_poll_event(artifacts, {"operation": "GET", "status": "error", "remote_id": job["remote_id"],
                                               "code": exc.code, "reason": exc.reason, "retry_after": exc.retry_after,
                                               "next_delay_seconds": delay})
                outcomes.append({"status": "poll_error", "job_id": job["id"], "reason": exc.reason})
                continue
            batch = reply.body
            if batch.get("id") != job["remote_id"]:
                ledger.defer_poll(job["id"], delay_seconds=600, error_code="remote_identity_mismatch")
                _append_poll_event(artifacts, {"operation": "GET", "status": "remote_identity_mismatch",
                                               "remote_id": job["remote_id"]})
                outcomes.append({"status": "remote_identity_mismatch", "job_id": job["id"]})
                continue
            remote_status = batch.get("status")
            _append_poll_event(artifacts, {"operation": "GET", "status": remote_status,
                                           "remote_id": job["remote_id"],
                                           "request_counts": batch.get("request_counts"), "usage": batch.get("usage")})
            if remote_status in TERMINAL:
                write_private_json(artifacts / "batch_terminal.json", batch)
            else:
                write_private_json(artifacts / "batch_progress.json", {
                    "id": batch.get("id"), "status": remote_status,
                    "request_counts": batch.get("request_counts"), "usage": batch.get("usage"),
                })
            usage = batch.get("usage") or {}
            ledger.poll_result(job["id"], remote_status, usage_cost_microusd=_cost_micros(usage),
                               error_code=None if remote_status == "completed" else str(batch.get("error") or "")[:120] or None)
            if remote_status == "completed":
                try:
                    accepted = _finish_raw(ledger, ledger.get(job["id"]))
                    outcomes.append(accepted)
                except (OSError, RevisionConflict) as exc:
                    ledger.defer_raw(job["id"], delay_seconds=180, error_code=type(exc).__name__ + ":" + str(exc)[:100])
                    outcomes.append({"status": "publication_pending_retry", "job_id": job["id"],
                                     "reason": type(exc).__name__})
                except (ValueError, json.JSONDecodeError) as exc:
                    ledger.failed_validation(job["id"], type(exc).__name__ + ":" + str(exc)[:120])
                    outcomes.append({"status": "failed_validation", "job_id": job["id"], "reason": str(exc)[:120]})
            else:
                outcomes.append({"status": remote_status, "job_id": job["id"], "remote_id": job["remote_id"]})
            if remote_status in TERMINAL:
                try:
                    deletion = client.delete(job["remote_id"])
                    write_private_json(artifacts / "batch_delete.json", deletion.body)
                    _append_poll_event(artifacts, {"operation": "DELETE", "status": "completed",
                                                   "remote_id": job["remote_id"], "code": deletion.status_code})
                except BatchError as exc:
                    write_private_json(artifacts / "batch_delete_error.json", {"code": exc.code, "reason": exc.reason})
                    _append_poll_event(artifacts, {"operation": "DELETE", "status": "error",
                                                   "remote_id": job["remote_id"], "code": exc.code, "reason": exc.reason})
        for root in ledger.quality_pending():
            try:
                outcomes.append(_advance_quality(ledger, root, client_factory))
            except (OSError, RevisionConflict) as exc:
                ledger.defer_quality(root["id"], delay_seconds=180,
                                     error_code=type(exc).__name__ + ":" + str(exc)[:100])
                outcomes.append({"status": "quality_pending_retry", "job_id": root["id"],
                                 "reason": type(exc).__name__})
            except (ValueError, json.JSONDecodeError) as exc:
                ledger.failed_quality(root["id"], type(exc).__name__ + ":" + str(exc)[:100])
                outcomes.append({"status": "failed_quality_integrity", "job_id": root["id"],
                                 "reason": str(exc)[:120]})
        return outcomes
    finally:
        ledger.close()
