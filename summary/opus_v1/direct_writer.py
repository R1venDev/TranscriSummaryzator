"""One isolated full-transcript Opus writer comparison; no publication path.

This is an administrator-triggered experiment. It shares the application's
credential store and monetary ledger, but never creates a summary consumer or
updates a generation pointer or task override database.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from scripts.summary_credentials import (CredentialError, CredentialStore,
                                         credential_dispatch_guard)
from summary.luna_v1.contract import SCHEMA, validate_document
from summary.luna_v1.ledger import (Ledger, OPUS_DIRECT_CALL_CAP_MICROUSD,
                                    OPUS_DIRECT_KIND, OPUS_DIRECT_POLICY,
                                    OPUS_DIRECT_WEEK_CAP_MICROUSD,
                                    write_private_json)
from summary.luna_v1.render import render_document
from summary.luna_v1.source import load_source

from .batch import (BATCH_MODEL_IDS, MODEL, PRIVACY_MODE, PROVIDER,
                    WORKSPACE_IO_LOGGING_ENABLED, BatchClient, BatchError,
                    batch_id_from_submit, extract_one_completed)
from .route import RouteBlocked, verify_batch_route


PROMPT_PATH = Path(__file__).with_name("prompt_direct_writer_v2.md")
OUTPUT_CAP = 24_000
REASONING_EFFORT = "low"
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{7,127}\Z")


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_request(source_text: str) -> dict:
    source = json.loads(source_text)
    if (source.get("source_kind") != "TRANSCRIPT_SOURCE"
            or _json_bytes(source) != source_text.encode("utf-8")
            or not isinstance(source.get("utterances"), list)
            or not source["utterances"]):
        raise ValueError("noncanonical_transcript_source")
    # The production schema remains the local acceptance contract. Passing it
    # as a provider grammar failed for the full document; here it is static
    # instruction text, and the entire response is validated locally.
    instructions = (PROMPT_PATH.read_text(encoding="utf-8")
                    + "\n\n<output_schema_json>\n"
                    + _json_bytes(SCHEMA).decode("utf-8")
                    + "\n</output_schema_json>")
    return {
        "messages": [
            {"role": "system", "content": instructions},
            {"role": "user", "content": source_text},
        ],
        "max_completion_tokens": OUTPUT_CAP,
        "reasoning": {"effort": REASONING_EFFORT},
    }


def _plan(*, transcript_path: Path, output_dir: Path, run_id: str,
          selected: dict) -> dict:
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise ValueError("invalid_direct_opus_run_id")
    scope = selected.get("workspace_id")
    if not isinstance(scope, str) or not scope:
        raise ValueError("judge_workspace_unverified")
    source_text, source_index, source_sha = load_source(transcript_path)
    request = build_request(source_text)
    prompt_sha, schema_sha = _sha(PROMPT_PATH.read_bytes()), _sha(_json_bytes(SCHEMA))
    request_sha = _sha(_json_bytes(request))
    semantic_key = _sha(_json_bytes({
        "kind": OPUS_DIRECT_KIND, "policy": OPUS_DIRECT_POLICY,
        "run_id": run_id, "source_sha256": source_sha,
        "projected_source_sha256": _sha(source_text.encode("utf-8")),
        "request_sha256": request_sha, "prompt_sha256": prompt_sha,
        "schema_sha256": schema_sha, "model": MODEL,
        "workspace_scope": scope, "output_dir": str(Path(output_dir)),
    }))
    return {"request": request, "source_sha256": source_sha,
            "source_index": source_index, "semantic_key": semantic_key,
            "prompt_sha256": prompt_sha, "schema_sha256": schema_sha,
            "request_sha256": request_sha}


def _store(private_root: Path) -> CredentialStore:
    return CredentialStore(Path(private_root) / "credentials.sqlite3")


def authorize(*, transcript_path: Path, output_dir: Path, private_root: Path,
              run_id: str, authorization_ref: str,
              store_factory=_store) -> dict:
    """Bind a separately approved $1.60 cap to an exact future run identity."""
    store = store_factory(private_root)
    candidates = store.dispatch_candidates(role="judge")
    if not candidates:
        raise CredentialError("judge_credential_required")
    selected = candidates[0]
    plan = _plan(transcript_path=Path(transcript_path), output_dir=Path(output_dir),
                 run_id=run_id, selected=selected)
    ledger = Ledger(private_root)
    try:
        created = ledger.authorize_direct_opus_run(
            semantic_key=plan["semantic_key"], source_sha256=plan["source_sha256"],
            output_dir=Path(output_dir), authorization_ref=authorization_ref)
    finally:
        ledger.close()
    return {"created": created, "run_id": run_id,
            "semantic_key": plan["semantic_key"],
            "source_sha256": plan["source_sha256"],
            "weekly_cap_microusd": OPUS_DIRECT_WEEK_CAP_MICROUSD,
            "per_call_cap_microusd": OPUS_DIRECT_CALL_CAP_MICROUSD}


def _pin(path: Path, value: dict) -> None:
    if path.exists():
        if path.is_symlink() or _sha(_json_bytes(json.loads(path.read_text(encoding="utf-8")))) != _sha(_json_bytes(value)):
            raise ValueError("direct_opus_saved_sidecar_changed")
    else:
        write_private_json(path, value)


def _seal_reserved(*, ledger: Ledger, job: dict, plan: dict,
                   transcript_path: Path, output_dir: Path, run_id: str,
                   route) -> None:
    """Recreate and pin pre-POST sidecars after a crash before sealing intent."""
    reserve = route.reserve_microusd(limit_microusd=OPUS_DIRECT_CALL_CAP_MICROUSD)
    if (job["status"] != "reserved" or job["kind"] != OPUS_DIRECT_KIND
            or job["semantic_key"] != plan["semantic_key"]
            or job["source_sha256"] != plan["source_sha256"]
            or job["workspace_id"] != route.workspace_id
            or job["output_dir"] != str(output_dir)
            or job["reserved_microusd"] != reserve):
        raise ValueError("direct_opus_reserved_intent_changed")
    authorization = ledger.weekly_authorization_for_root(job["id"])
    if (authorization is None or authorization["quality_policy_version"] != OPUS_DIRECT_POLICY
            or authorization["cap_microusd"] != OPUS_DIRECT_WEEK_CAP_MICROUSD):
        raise ValueError("direct_opus_authorization_changed")
    manifest = {
        "kind": OPUS_DIRECT_KIND, "policy_version": OPUS_DIRECT_POLICY,
        "run_id": run_id, "job_id": job["id"], "custom_id": job["custom_id"],
        "semantic_key": plan["semantic_key"],
        "source_sha256": plan["source_sha256"],
        "transcript_path": str(transcript_path), "output_dir": str(output_dir),
        "credential_id": job["credential_id"],
        "credential_version": job["credential_version"],
        "workspace_id": route.workspace_id, "provider": "openrouter_claude_opus",
        "model": MODEL, "provider_only": PROVIDER,
        "privacy_mode": PRIVACY_MODE,
        "workspace_io_logging_enabled": WORKSPACE_IO_LOGGING_ENABLED,
        "prompt_sha256": plan["prompt_sha256"],
        "schema_sha256": plan["schema_sha256"],
        "output_contract": "prompt_json_with_local_luna_summary_v1_validation",
        "request_sha256": plan["request_sha256"],
        "inline_request_count": 1,
        "max_completion_tokens": OUTPUT_CAP,
        "reasoning_effort": REASONING_EFFORT,
        "reserved_microusd": reserve,
        "weekly_cap_microusd": OPUS_DIRECT_WEEK_CAP_MICROUSD,
        "authorization_ref": authorization["authorization_ref"],
        "pricing": {
            "prompt_usd_per_token": str(route.prompt_usd_per_token),
            "completion_usd_per_token": str(route.completion_usd_per_token),
            "request_usd": str(route.request_usd)},
        "created_at": datetime.fromtimestamp(job["created_at"], timezone.utc).isoformat(),
    }
    artifacts = Path(job["artifact_dir"])
    _pin(artifacts / "request.json", plan["request"])
    _pin(artifacts / "manifest.json", manifest)


def _dispatch(ledger: Ledger, job: dict, *, store: CredentialStore,
              client_factory=BatchClient) -> dict:
    artifacts = Path(job["artifact_dir"])
    manifest = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
    request = json.loads((artifacts / "request.json").read_text(encoding="utf-8"))
    if (manifest.get("kind") != OPUS_DIRECT_KIND
            or manifest.get("semantic_key") != job["semantic_key"]
            or manifest.get("request_sha256") != _sha(_json_bytes(request))
            or manifest.get("credential_id") != job["credential_id"]
            or manifest.get("credential_version") != job["credential_version"]):
        raise ValueError("direct_opus_saved_request_mismatch")
    try:
        token = store.reveal_for_dispatch(job["credential_id"],
                                          job["credential_version"], role="judge")
    except CredentialError:
        ledger.cancel_before_submit(job["id"], "credential_changed_before_post")
        return {"status": "cancelled_before_submit", "job_id": job["id"]}
    if not ledger.mark_submitting(job["id"]):
        return {"status": "submission_state_conflict", "job_id": job["id"]}
    try:
        reply = client_factory(token).submit(job["custom_id"], request,
                                             allow_prompt_json=True)
        write_private_json(artifacts / "submit_response.json", reply.body)
        remote_id = batch_id_from_submit(reply.body)
        if reply.status_code != 202:
            raise ValueError("unexpected_submit_response")
        ledger.submission_result(job["id"], remote_id=remote_id)
        return {"status": "submitted", "job_id": job["id"],
                "remote_id": remote_id, "reserved_usd": job["reserved_microusd"] / 1_000_000}
    except BatchError as exc:
        definite = exc.code in {400, 401, 402, 403, 404, 422, 429}
        ledger.submission_result(job["id"], remote_id=None,
                                 error_code=exc.reason, definite_rejection=definite)
        write_private_json(artifacts / "submit_error.json", {
            "code": exc.code, "reason": exc.reason,
            "retry_after": exc.retry_after, "definite_rejection": definite})
        return {"status": "rejected_before_submit" if definite else "submission_unknown",
                "job_id": job["id"], "error_code": exc.reason}
    except ValueError as exc:
        # A POST may have succeeded even when its reply is malformed. Never
        # free the reserve or issue a second POST automatically.
        ledger.submission_result(job["id"], remote_id=None,
                                 error_code="submit_receipt_unverified")
        write_private_json(artifacts / "submit_error.json", {"reason": str(exc)[:120]})
        return {"status": "submission_unknown", "job_id": job["id"]}


def submit(*, transcript_path: Path, output_dir: Path, private_root: Path,
           run_id: str, client_factory=BatchClient,
           store_factory=_store, route_factory=verify_batch_route) -> dict:
    """One physical Batch POST after durable intent and shared budget reserve."""
    transcript_path, output_dir = Path(transcript_path), Path(output_dir)
    private_root = Path(private_root)
    store = store_factory(private_root)
    ledger = Ledger(private_root)
    try:
        with credential_dispatch_guard(getattr(store, "path", private_root / "credentials.sqlite3")):
            candidates = store.dispatch_candidates(role="judge")
            if not candidates:
                return {"status": "credential_required"}
            selected = candidates[0]
            plan = _plan(transcript_path=transcript_path, output_dir=output_dir,
                         run_id=run_id, selected=selected)
            prior = ledger.by_semantic_key(plan["semantic_key"])
            if prior is not None:
                if prior["kind"] != OPUS_DIRECT_KIND:
                    raise ValueError("direct_opus_semantic_collision")
                if prior["status"] == "reserved":
                    token = store.reveal_for_dispatch(selected["id"], selected["version"],
                                                      role="judge")
                    route = route_factory(client_factory(token), plan["request"],
                                          max_output_tokens=OUTPUT_CAP,
                                          allow_prompt_json=True)
                    _seal_reserved(ledger=ledger, job=prior, plan=plan,
                                   transcript_path=transcript_path, output_dir=output_dir,
                                   run_id=run_id, route=route)
                    return _dispatch(ledger, prior, store=store,
                                     client_factory=client_factory)
                return {"status": prior["status"], "job_id": prior["id"],
                        "remote_id": prior["remote_id"]}
            token = store.reveal_for_dispatch(selected["id"], selected["version"],
                                              role="judge")
            route = route_factory(client_factory(token), plan["request"],
                                  max_output_tokens=OUTPUT_CAP,
                                  allow_prompt_json=True)
            if route.workspace_id != selected["workspace_id"]:
                raise RouteBlocked("credential_workspace_changed_since_check")
            reserve = route.reserve_microusd(limit_microusd=OPUS_DIRECT_CALL_CAP_MICROUSD)
            decision = ledger.reserve_direct_opus(
                semantic_key=plan["semantic_key"], source_sha256=plan["source_sha256"],
                output_dir=output_dir, credential_id=selected["id"],
                credential_version=selected["version"], workspace_id=route.workspace_id,
                max_cost_microusd=reserve)
            if decision.kind != "new":
                return {"status": decision.kind, "reason": decision.reason,
                        "job_id": decision.job_id}
            job = ledger.get(decision.job_id)
            _seal_reserved(ledger=ledger, job=job, plan=plan,
                           transcript_path=transcript_path, output_dir=output_dir,
                           run_id=run_id, route=route)
            return _dispatch(ledger, job, store=store,
                             client_factory=client_factory)
    except (CredentialError, RouteBlocked, BatchError) as exc:
        return {"status": "preflight_blocked", "reason": str(exc)}
    finally:
        ledger.close()


def _private_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def finish_raw(ledger: Ledger, job: dict) -> dict:
    """Parse one terminal result and save local comparison artifacts only."""
    if job["kind"] != OPUS_DIRECT_KIND or job["status"] != "completed_raw":
        raise ValueError("not_direct_opus_raw")
    artifacts = Path(job["artifact_dir"])
    manifest = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
    request = json.loads((artifacts / "request.json").read_text(encoding="utf-8"))
    submission = json.loads((artifacts / "submit_response.json").read_text(encoding="utf-8"))
    batch = json.loads((artifacts / "batch_terminal.json").read_text(encoding="utf-8"))
    if (manifest.get("kind") != OPUS_DIRECT_KIND
            or manifest.get("policy_version") != OPUS_DIRECT_POLICY
            or manifest.get("job_id") != job["id"]
            or manifest.get("semantic_key") != job["semantic_key"]
            or manifest.get("source_sha256") != job["source_sha256"]
            or manifest.get("credential_id") != job["credential_id"]
            or manifest.get("credential_version") != job["credential_version"]
            or manifest.get("workspace_id") != job["workspace_id"]
            or manifest.get("request_sha256") != _sha(_json_bytes(request))
            or manifest.get("prompt_sha256") != _sha(PROMPT_PATH.read_bytes())
            or manifest.get("schema_sha256") != _sha(_json_bytes(SCHEMA))
            or submission.get("id") != job["remote_id"]
            or batch.get("id") != job["remote_id"]
            or submission.get("model") not in BATCH_MODEL_IDS
            or batch.get("model") != submission.get("model")):
        raise ValueError("direct_opus_saved_identity_mismatch")
    body, usage = extract_one_completed(
        batch, job["custom_id"], expected_batch_id=job["remote_id"],
        manifest=manifest, saved_request=request)
    if body.get("model") not in BATCH_MODEL_IDS:
        raise ValueError("direct_opus_response_model_mismatch")
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("direct_opus_response_choices_invalid")
    choice = choices[0]
    message = choice.get("message")
    if (not isinstance(message, dict) or choice.get("finish_reason") != "stop"
            or message.get("refusal")
            or not isinstance(message.get("content"), str)):
        raise ValueError("direct_opus_response_incomplete")
    raw_text = message["content"]
    write_private_json(artifacts / "native_response.json", {"text": raw_text, "usage": usage})
    document = json.loads(raw_text)
    transcript_path = Path(manifest["transcript_path"])
    _, source_index, source_sha = load_source(transcript_path)
    if source_sha != job["source_sha256"]:
        raise ValueError("direct_opus_source_changed")
    validate_document(document, source_index)
    rendered = render_document(document, source_index)
    write_private_json(artifacts / "candidate_document.json", document)
    write_private_json(artifacts / "summary.json", rendered["summary.json"])
    write_private_json(artifacts / "tasks.json", rendered["tasks.json"])
    _private_text(artifacts / "summary.md", rendered["summary.md"])
    _private_text(artifacts / "summary.html", rendered["summary.html"])
    ledger.direct_opus_completed(job["id"], artifacts / "candidate_document.json")
    return {"status": "direct_complete", "job_id": job["id"],
            "document_path": str(artifacts / "candidate_document.json"),
            "billed_microusd": job["billed_microusd"],
            "reserved_microusd": job["reserved_microusd"],
            "production_published": False}
