"""Durable, summary-only Luna source-first Batch orchestration.

The existing summary scheduler calls :func:`submit_source_first` for a queued
meeting and :func:`poll_source_first_once` on later ticks.  Both functions use
the existing credential store and Luna ledger.  This module never calls sync
Chat, changes speech inputs, or asks a model to decide publication/budget.

Native reports and terminal Batch envelopes are immutable private artifacts.
Ledger rows are the execution state; stage readiness is derived from their
recorded attempts and saved artifacts, not from another state store.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_UP
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
import uuid

from scripts.summary_credentials import (
    CredentialError, CredentialStore, credential_dispatch_guard,
)

from .batch import (
    BatchClient, BatchError, MODEL, PROVIDER, SUBMIT_MODEL, TERMINAL, build_batch_payload,
    parse_batch_items, valid_batch_id, validate_prepared_batch_payload,
)
from .batch_stage_contracts_v1 import (
    STAGE_PROMPT_VERSION, developer_message_for_stage,
    prompt_sha256_for_stage, response_format_for_stage, schema_for_stage,
)
from .contract import SCHEMA as WRITER_SCHEMA, validate_document
from .ledger import (
    BatchItemIntent, Ledger, source_first_job_cap_microusd,
    batch_item_intent_for_body,
)
from .render import HEADINGS
from .route import Route, RouteBlocked, verify_source_first_batch_route
from .capacity import CAPACITY_POLICY, allocate_outputs, input_cost
from .source_first_core import (
    POLICY_VERSION, SourceSnapshot, accepted_patch_document, build_surfaces,
    canonical_bytes, digest, load_snapshot, normalize_inventories,
    partition_surfaces, plan_dimensions, plan_packets, stage_patch_candidate,
    salvage_inventory, validate_audit, validate_global, validate_inventory, validate_patch_plan,
    validate_verification, safe_mark_document, remap_mark_targets,
)
from .review_salvage import salvage_review
from .patch_normalization import NORMALIZATION_VERSION, normalize_patch_report
from .tasks import ReconciliationConflict, RevisionConflict, TaskStore
from .publication import adopt_sealed_generation, publish_document


STAGES = ("writer", "extract", "audit", "global", "repair", "verify")
STAGE_EFFORT = {"writer": "medium", "extract": "medium", "audit": "high",
                "global": "high", "repair": "high", "verify": "high"}
_DEFINITE_REJECTION = frozenset({400, 401, 402, 403, 404, 422})
_ACTIVE_REMOTE = frozenset({"validating", "in_progress", "finalizing", "cancelling"})


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _private_file_once(path: Path, data: bytes) -> str:
    """Seal a private artifact once; repeated ticks may only reuse same bytes."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError("immutable_artifact_conflict")
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha(data)


def _json_file_once(path: Path, value: object) -> str:
    return _private_file_once(path, canonical_bytes(value) + b"\n")


def _batch_file_once(path: Path, envelope: dict) -> str:
    # The gateway consumes these top-level keys in this exact insertion order.
    return _private_file_once(path, json.dumps(envelope, ensure_ascii=False,
        separators=(",", ":")).encode("utf-8"))


def _read_sealed_bytes(path: Path, expected_sha256: str) -> bytes:
    raw = path.read_bytes()
    if _sha(raw) != expected_sha256:
        raise ValueError("sealed_artifact_hash_mismatch")
    return raw


def _read_sealed(path: Path, expected_sha256: str) -> dict:
    raw = _read_sealed_bytes(path, expected_sha256)
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("sealed_artifact_not_object")
    return parsed


def _snapshot_from_manifest(manifest: dict) -> SourceSnapshot:
    source = manifest["snapshot"]
    records = tuple(source["records"])
    snapshot = SourceSnapshot(Path(source["transcript_path"]),
        source["source_sha256"], source["source_text"],
        source["index"], records)
    if len(snapshot.ids) != len(set(snapshot.ids)):
        raise ValueError("saved_snapshot_duplicate_u_id")
    return snapshot


def _manifest(snapshot: SourceSnapshot, packets: tuple[dict, ...], dimensions: dict,
              *, output_dir: Path, workspace_id: str, force_nonce: str | None,
              route: Route, planned_capacity_microusd: int,
              forecast_microusd: int) -> dict:
    return {
        "policy_version": POLICY_VERSION,
        "prompt_version": STAGE_PROMPT_VERSION,
        "source_revision": snapshot.source_sha256,
        "output_dir": str(output_dir),
        "workspace_id": workspace_id,
        "model_submit_slug": route.model,
        "resolved_batch_endpoint": route.batch_endpoint_model,
        "provider_endpoint_tag": route.provider_endpoint_tag,
        "prompt_hashes": {stage: prompt_sha256_for_stage(stage) for stage in STAGES},
        "schema_hashes": {stage: digest(schema_for_stage(stage)) for stage in STAGES},
        "force_nonce": force_nonce,
        "dimensions": dimensions,
        "planned_capacity_microusd": planned_capacity_microusd,
        "capacity_basis": {
            "forecast_microusd": forecast_microusd,
            "source_bytes": len(snapshot.source_text.encode("utf-8")),
            "output_capacity_policy": CAPACITY_POLICY,
            "verified_endpoint_output_limit": route.max_completion_tokens,
            "prompt_usd_per_token": str(route.prompt_usd_per_token),
            "completion_usd_per_token": str(route.completion_usd_per_token),
            "cache_write_usd_per_token": str(route.cache_write_usd_per_token),
            "request_usd": str(route.request_usd),
        },
        "packets": list(packets),
        "snapshot": {
            "transcript_path": str(snapshot.transcript_path),
            "source_sha256": snapshot.source_sha256,
            "source_text": snapshot.source_text,
            "index": snapshot.index,
            "records": list(snapshot.records),
        },
    }


def _user_json(payload: dict) -> str:
    return canonical_bytes(payload).decode("utf-8")


def _chat_body(stage: str, payload: dict, *, output_cap: int | None = None) -> dict:
    """OpenRouter Chat item body; no tools, sampling, stream or sync endpoint."""
    body = {
        "messages": [
            {"role": "developer", "content": developer_message_for_stage(stage)},
            {"role": "user", "content": _user_json(payload)},
        ],
        "response_format": response_format_for_stage(stage),
        "reasoning": {"effort": STAGE_EFFORT[stage]},
    }
    if output_cap is not None:
        body["max_tokens"] = output_cap
    return body


def _writer_payload(snapshot: SourceSnapshot) -> dict:
    sections = ("main", "timecodes", "tasks", "questions", "technical",
                "ideas", "verification", "chapters")
    return {
        "SOURCE": json.loads(snapshot.source_text),
        "SECTION_SPEC": [{"field": field, "heading": heading}
                         for field, heading in zip(sections, HEADINGS, strict=True)],
        "WRITER_SCHEMA_DESCRIPTION": {
            "schema_version": WRITER_SCHEMA["properties"]["schema_version"]["enum"][0],
            "top_level_fields": WRITER_SCHEMA["required"],
            "task_fields": WRITER_SCHEMA["$defs"]["task"]["required"],
        },
    }


def _extraction_payload(packet: dict) -> dict:
    core = set(packet["core_ids"])
    rows = packet["records"]
    first = next(number for number, row in enumerate(rows) if row["id"] in core)
    last = max(number for number, row in enumerate(rows) if row["id"] in core)
    before, after = rows[:first], rows[last + 1:]
    speakers = []
    seen_speakers = set()
    for row in rows:
        pair = (row["speaker_id"], row["speaker_label"])
        if pair not in seen_speakers:
            speakers.append({"speaker_id": pair[0], "speaker_label": pair[1]})
            seen_speakers.add(pair)
    return {
        "CORE_SOURCE": [row for row in rows if row["id"] in core],
        "CONTEXT_BEFORE": before, "CONTEXT_AFTER": after,
        "SPEAKERS": speakers,
        "EXPECTED_CORE_IDS": packet["core_ids"],
    }


def _items_for_wave(workflow_id: str, stage: str,
                    payloads: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    items = []
    for packet_id, payload in payloads:
        custom_id = f"sf-{workflow_id[:12]}-{stage}-{packet_id}-a1"
        items.append((custom_id, _chat_body(stage, payload)))
    return items


def _batch_intent(*, ledger: Ledger, workflow: dict, stage: str,
                  items: list[tuple[str, dict]], route: Route) -> tuple[object, Path, str]:
    """Reserve actual item bodies before the only possible POST."""
    envelope = build_batch_payload(items)
    intent_key = digest({"workflow_id": workflow["id"], "stage": stage,
                         "requests": envelope["requests"]})
    payload_path = Path(ledger.root) / "source_first" / workflow["semantic_key"] / "batches" / (intent_key + ".json")
    payload_sha = _batch_file_once(payload_path, envelope)
    item_intents: list[BatchItemIntent] = []
    for custom_id, body in items:
        reserve = route.reserve_microusd(canonical_bytes(body),
                                         max_completion_tokens=body["max_tokens"],
                                         authorized_job_cap_microusd=workflow["planned_reserve_microusd"])
        item_intents.append(batch_item_intent_for_body(
            custom_id, body, reserve_microusd=reserve))
    decision = ledger.reserve_batch_intent(
        workflow_id=workflow["id"], intent_key=intent_key, stage=stage,
        credential_id=workflow["credential_id"],
        credential_version=workflow["credential_version"],
        workspace_id=workflow["workspace_id"],
        payload_path=payload_path, payload_sha256=payload_sha,
        items=item_intents,
    )
    return decision, payload_path, payload_sha


def _post_reserved(*, ledger: Ledger, attempt_id: str, payload_path: Path,
                   payload_sha: str, client: BatchClient) -> dict:
    # A damaged local payload has never been submitted. Check its seal before
    # changing the durable intent to the ambiguous submitting state.
    payload_bytes = _read_sealed_bytes(payload_path, payload_sha)
    validate_prepared_batch_payload(payload_bytes, expected_sha256=payload_sha)
    if not ledger.mark_batch_submitting(attempt_id, expected_payload_sha256=payload_sha):
        return {"status": "pending", "attempt_id": attempt_id}
    try:
        reply = client.submit_prepared(payload_bytes, expected_sha256=payload_sha)
        remote_id = reply.body.get("id")
        if reply.status_code != 202 or not valid_batch_id(remote_id):
            ledger.record_batch_submission(attempt_id, remote_id=None,
                                           error_code="unexpected_submit_response")
            return {"status": "submission_unknown", "attempt_id": attempt_id}
        ledger.record_batch_submission(attempt_id, remote_id=remote_id)
        # Persist the actual acknowledgement, not a reconstructed terminal.
        # The remote ID is durable first so a failed receipt write cannot
        # cause a replacement POST after restart.
        attempt = ledger.get_batch_attempt(attempt_id)
        workflow = ledger.get_batch_workflow(attempt["workflow_id"])
        receipt = Path(ledger.root) / "source_first" / workflow["semantic_key"] / "submissions" / (attempt_id + ".json")
        try:
            _json_file_once(receipt, {"http_status": reply.status_code,
                "payload_sha256": payload_sha, "batch_id": remote_id,
                "reply": reply.body})
        except (OSError, ValueError):
            # Dispatch is known. Capture failure is visible and never retryable.
            ledger.db.execute("UPDATE batch_attempts SET error_code=? WHERE id=?",
                              ("submit_receipt_capture_failed", attempt_id))
        return {"status": "submitted", "attempt_id": attempt_id,
                "remote_batch_id": remote_id}
    except (BatchError, ValueError) as exc:
        definite = isinstance(exc, BatchError) and exc.code in _DEFINITE_REJECTION
        reason = exc.reason if isinstance(exc, BatchError) else "post_outcome_unknown"
        ledger.record_batch_submission(attempt_id, remote_id=None,
                                       error_code=reason,
                                       definite_rejection=definite)
        return {"status": "rejected_before_submit" if definite else "submission_unknown",
                "attempt_id": attempt_id, "error_code": reason}


def _planned_capacity_hold(route: Route, snapshot: SourceSnapshot,
                           packets: tuple[dict, ...]) -> int:
    """Known source-input cost only; output is allocated from the saved cap.

    The UI must not present this floor as a quote for unknown future reports.
    Later exact bodies and reasoning/output capacity are reserved before POST.
    """
    source_bytes = len(snapshot.source_text.encode("utf-8"))
    packet_bytes = sum(len(canonical_bytes(packet["records"])) for packet in packets)
    input_upper = ((source_bytes * 4 + packet_bytes * 2) * 6 + 4) // 5
    return input_cost(route, input_upper)


def _capacity_items(*, ledger: Ledger, workflow: dict, route: Route,
                    snapshot: SourceSnapshot, packets: tuple[dict, ...],
                    stage: str, items: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    """Seal allocations before intents; restart reuses identical request bytes."""
    path = Path(ledger.root) / "source_first" / workflow["semantic_key"] / "capacity" / (stage + ".json")
    input_hash = digest(items)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["input_hash"] != input_hash or saved["policy"] != CAPACITY_POLICY:
            raise RouteBlocked("saved_capacity_identity_changed")
        if len(saved["items"]) != len(items):
            raise RouteBlocked("saved_capacity_identity_changed")
        for (identity, body), (expected, original) in zip(saved["items"], items, strict=True):
            cap = body.get("max_tokens")
            if (identity != expected or type(cap) is not int or not 1 <= cap <= route.max_completion_tokens
                    or {key: value for key, value in body.items() if key != "max_tokens"} != original):
                raise RouteBlocked("saved_capacity_identity_changed")
        return [(row[0], row[1]) for row in saved["items"]]
    attempts = ledger.list_batch_attempts(workflow["id"])
    spent_or_held = sum(row["billed_microusd"] if row["billed_microusd"] is not None
        else row["reserved_microusd"] for row in attempts
        if row["status"] not in {"rejected_no_charge", "cancelled_before_submit"})
    available = workflow["planned_reserve_microusd"] - spent_or_held
    source_bytes = len(snapshot.source_text.encode("utf-8"))
    # Reserve known future source input, plus one output share per future item.
    if stage == "wave1":
        future = len(packets) + 3
        future_bytes = sum(len(canonical_bytes(p["records"])) for p in packets) + source_bytes * 3
    else:
        future = {"audit": 3, "global": 2, "repair": 1, "verify": 0}[stage]
        future_bytes = source_bytes * future
    pricing = replace(route, key_limit_remaining_usd=None)
    bodies, basis = allocate_outputs(pricing, [body for _, body in items],
        available_microusd=available, future_items=future,
        future_input_tokens=(future_bytes * 6 + 4) // 5)
    prepared = [(identity, body) for (identity, _), body in zip(items, bodies, strict=True)]
    saved = {"policy": CAPACITY_POLICY, "input_hash": input_hash,
             "items": prepared, "basis": basis}
    _json_file_once(path, saved)
    # Canonical key order must be identical on the first send and a restart.
    return [(row[0], row[1]) for row in json.loads(canonical_bytes(saved))["items"]]


def _reserve_and_post_wave1(*, ledger: Ledger, workflow: dict,
                            snapshot: SourceSnapshot, packets: tuple[dict, ...],
                            route: Route, client: BatchClient) -> dict:
    writer = _items_for_wave(workflow["id"], "writer",
                             [("full", _writer_payload(snapshot))])
    extract = _items_for_wave(workflow["id"], "extract",
                              [(packet["packet_id"], _extraction_payload(packet))
                               for packet in packets])
    allocated = _capacity_items(ledger=ledger, workflow=workflow, route=route,
        snapshot=snapshot, packets=packets, stage="wave1", items=writer + extract)
    writer, extract = allocated[:len(writer)], allocated[len(writer):]
    prior = {attempt["stage"]: attempt for attempt in
             ledger.list_batch_attempts(workflow["id"])
             if attempt["stage"] in {"writer", "extract"}}
    if any(attempt["status"] in {"rejected_no_charge", "cancelled_before_submit"}
           for attempt in prior.values()):
        # Older interrupted runs may have left the peer reserved. A terminal
        # pre-submit failure can never authorize a lone first-wave POST.
        for attempt in prior.values():
            if attempt["status"] == "reserved" and attempt["post_count"] == 0:
                ledger.cancel_batch_before_submit(attempt["id"],
                                                  "wave1_peer_unavailable")
        return {"status": "wave1_incomplete", "workflow_id": workflow["id"],
                "reason": "wave1_peer_unavailable"}
    # Capacity validation for both independent Batch bodies precedes either
    # stage reservation, so a too-large extraction packet leaves no half-wave.
    # The key's remaining balance is shared by every first-wave item. Pricing
    # without the per-item key check lets us compare their sum once, before a
    # stage can be reserved or submitted. Already attempted POSTs are excluded
    # on a restart because their charge has consumed the key's current balance.
    pricing_route = replace(route, key_limit_remaining_usd=None)
    remaining_wave_reserve = 0
    for stage, items in (("writer", writer), ("extract", extract)):
        for _, body in items:
            reserve = pricing_route.reserve_microusd(canonical_bytes(body),
                                                     max_completion_tokens=body["max_tokens"],
                                                     authorized_job_cap_microusd=workflow["planned_reserve_microusd"])
            if not prior.get(stage) or prior[stage]["post_count"] == 0:
                remaining_wave_reserve += reserve
    if (route.key_limit_remaining_usd is not None
            and Decimal(remaining_wave_reserve)
            > route.key_limit_remaining_usd * 1_000_000):
        raise RouteBlocked("key_budget_insufficient")
    writer_decision, writer_path, writer_sha = _batch_intent(
        ledger=ledger, workflow=workflow, stage="writer", items=writer,
        route=pricing_route)
    extract_decision, extract_path, extract_sha = _batch_intent(
        ledger=ledger, workflow=workflow, stage="extract", items=extract,
        route=pricing_route)
    if writer_decision.kind == "blocked" or extract_decision.kind == "blocked":
        # Keep the successful reservation as a durable hold. The poller skips
        # a reserved first-wave attempt until its peer exists; a later tick
        # can retry the missing reservation without creating a new intent.
        return {"status": "wave_reservation_blocked", "workflow_id": workflow["id"],
                "writer": writer_decision.kind, "extract": extract_decision.kind}
    outcomes = []
    for stage, decision, path, payload_sha in (
            ("writer", writer_decision, writer_path, writer_sha),
            ("extract", extract_decision, extract_path, extract_sha)):
        if stage == "extract" and not (
                outcomes[0]["status"] in {"submitted", "polling",
                                           "credential_required", "completed"}
                and outcomes[0].get("remote_batch_id")):
            if outcomes[0]["status"] in {"rejected_before_submit", "rejected_no_charge",
                                         "cancelled_before_submit"}:
                peer = ledger.get_batch_attempt(decision.attempt_id)
                if peer["status"] == "reserved" and peer["post_count"] == 0:
                    ledger.cancel_batch_before_submit(decision.attempt_id,
                                                      "writer_rejected_before_submit")
            peer = ledger.get_batch_attempt(decision.attempt_id)
            outcomes.append({"status": peer["status"], "attempt_id": peer["id"],
                             "remote_batch_id": peer["remote_id"]})
            break
        current = ledger.get_batch_attempt(decision.attempt_id)
        if current["status"] == "reserved":
            outcomes.append(_post_reserved(ledger=ledger, attempt_id=decision.attempt_id,
                payload_path=path, payload_sha=payload_sha, client=client))
        else:
            outcomes.append({"status": current["status"], "attempt_id": current["id"],
                             "remote_batch_id": current["remote_id"]})
    return {"status": "submitted" if all(o["status"] in {"submitted", "completed"}
                                  for o in outcomes) else "wave1_incomplete",
            "workflow_id": workflow["id"], "batches": outcomes}


def _saved_policy_matches(row: dict, manifest: dict, snapshot: SourceSnapshot,
                          output_dir: Path, force_nonce: str | None) -> bool:
    """Require the saved job to match this source and current code policy."""
    expected_prompts = {stage: prompt_sha256_for_stage(stage) for stage in STAGES}
    expected_schemas = {stage: digest(schema_for_stage(stage)) for stage in STAGES}
    saved_snapshot = manifest.get("snapshot")
    return (row["source_sha256"] == snapshot.source_sha256
            and Path(row["output_dir"]) == output_dir
            and manifest.get("source_revision") == snapshot.source_sha256
            and manifest.get("output_dir") == str(output_dir)
            and isinstance(saved_snapshot, dict)
            and saved_snapshot.get("source_sha256") == snapshot.source_sha256
            and manifest.get("policy_version") == POLICY_VERSION
            and manifest.get("prompt_version") == STAGE_PROMPT_VERSION
            and manifest.get("prompt_hashes") == expected_prompts
            and manifest.get("schema_hashes") == expected_schemas
            and manifest.get("force_nonce") == force_nonce
            and manifest.get("workspace_id") == row["workspace_id"]
            and manifest.get("model_submit_slug") == SUBMIT_MODEL
            and manifest.get("resolved_batch_endpoint") == MODEL
            and manifest.get("provider_endpoint_tag") == PROVIDER)


def _existing_same_policy(ledger: Ledger, snapshot: SourceSnapshot,
                          output_dir: Path, force_nonce: str | None) -> dict | None:
    """Zero-call reuse of a sealed result from this exact code route policy."""
    for row in ledger.list_source_first_workflows(states=("active", "accepted")):
        if (row["source_sha256"] != snapshot.source_sha256
                or Path(row["output_dir"]) != output_dir):
            continue
        try:
            manifest = _read_sealed(Path(row["manifest_path"]), row["manifest_sha256"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if not _saved_policy_matches(row, manifest, snapshot,
                                     output_dir, force_nonce):
            continue
        if row["status"] == "accepted":
            try:
                _read_sealed(Path(row["accepted_document_path"]),
                             row["accepted_document_sha256"])
            except (OSError, ValueError, TypeError):
                continue
            return {"status": "accepted_cache_hit", "workflow_id": row["id"]}
        return {"status": "pending", "workflow_id": row["id"]}
    return None


def submit_source_first(*, transcript_path: Path, output_dir: Path,
                        private_root: Path, force_nonce: str | None = None,
                        client_factory=BatchClient) -> dict:
    """Plan and send wave 1 (one writer Batch and K extraction items Batch).

    A blocked preflight never sends private input.  A repeated call reuses the
    existing workflow/attempt intent and cannot submit another Batch for the
    same input.  Later waves are driven by the ordinary polling scheduler.
    """
    snapshot = load_snapshot(Path(transcript_path))
    packets = plan_packets(snapshot)
    dimensions = plan_dimensions(packets)
    if dimensions["status"] != "ready":
        return {"status": dimensions["status"], "reason": dimensions["reason"]}
    ledger = Ledger(private_root)
    try:
        cached = _existing_same_policy(ledger, snapshot, Path(output_dir), force_nonce)
        if cached is not None:
            return cached
        store = CredentialStore(Path(private_root) / "credentials.sqlite3")
        with credential_dispatch_guard(store.path):
            candidates = store.dispatch_candidates()
            if not candidates:
                return {"status": "credential_required"}
            selected = client = route = None
            for candidate in candidates:
                token = store.reveal_for_dispatch(candidate["id"], candidate["version"])
                candidate_client = client_factory(token)
                try:
                    candidate_route = verify_source_first_batch_route(candidate_client)
                except RouteBlocked as exc:
                    # A key restricted to another model is not a Luna key.
                    # Do not skip privacy, policy, balance or route failures.
                    if str(exc) == "model_not_allowed_for_key":
                        continue
                    raise
                selected, client, route = candidate, candidate_client, candidate_route
                break
            if selected is None:
                return {"status": "credential_required",
                        "reason": "no_luna_batch_credential"}
            if route.workspace_id != selected.get("workspace_id"):
                raise RouteBlocked("credential_workspace_changed_since_check")
            try:
                job_cap = source_first_job_cap_microusd()
            except ValueError as exc:
                return {"status": "preflight_blocked", "reason": str(exc)}
            semantic_key = digest({
                "source_sha256": snapshot.source_sha256,
                "policy_version": POLICY_VERSION,
                "prompt_hashes": {stage: prompt_sha256_for_stage(stage) for stage in STAGES},
                "schema_hashes": {stage: digest(schema_for_stage(stage)) for stage in STAGES},
                "workspace_id": route.workspace_id,
                "model_submit_slug": route.model,
                "resolved_batch_endpoint": route.batch_endpoint_model,
                "provider_endpoint_tag": route.provider_endpoint_tag,
                "force_nonce": force_nonce,
            })
            existing = next((row for row in ledger.list_source_first_workflows(
                states=("active", "accepted", "failed", "cancelled"))
                if row["semantic_key"] == semantic_key), None)
            if existing is not None:
                if Path(existing["output_dir"]) != Path(output_dir):
                    if existing["status"] != "accepted":
                        return {"status": "source_identity_output_conflict",
                                "workflow_id": existing["id"]}
                    try:
                        saved = _read_sealed(Path(existing["manifest_path"]),
                                             existing["manifest_sha256"])
                        if (not _saved_policy_matches(existing, saved, snapshot,
                                Path(existing["output_dir"]), force_nonce)
                                or existing["workspace_id"] != route.workspace_id):
                            raise ValueError("adoption_policy_mismatch")
                        _read_sealed(Path(existing["accepted_document_path"]),
                                     existing["accepted_document_sha256"])
                        writer = _stage_attempt(ledger, existing["id"], "writer")
                        if not writer or not valid_batch_id(writer["remote_id"]):
                            raise ValueError("adoption_writer_receipt_missing")
                        generation_id, _ = adopt_sealed_generation(
                            source_generation=Path(existing["accepted_document_path"]).parent,
                            source_output_dir=Path(existing["output_dir"]),
                            output_dir=Path(output_dir), transcript_path=Path(transcript_path),
                            task_db_path=Path(private_root) / "tasks.sqlite3",
                            source_sha256=snapshot.source_sha256,
                            semantic_key=semantic_key, job_id=existing["id"],
                            credential_id=existing["credential_id"],
                            remote_batch_id=writer["remote_id"],
                            prompt_sha256=saved["prompt_hashes"]["writer"],
                            schema_sha256=saved["schema_hashes"]["writer"],
                            accepted_document_sha256=existing["accepted_document_sha256"],
                        )
                    except (OSError, ValueError, KeyError, TypeError,
                            sqlite3.Error) as exc:
                        return {"status": "adoption_blocked", "workflow_id": existing["id"],
                                "reason": str(exc)[:120] if isinstance(exc, ValueError)
                                else type(exc).__name__}
                    return {"status": "accepted_cache_hit", "workflow_id": existing["id"],
                            "generation_id": generation_id}
                if (existing["status"] == "failed"
                        and existing["error_code"] == "rolling_week_budget_exceeded"):
                    try:
                        saved = _read_sealed(Path(existing["manifest_path"]),
                                             existing["manifest_sha256"])
                    except (OSError, ValueError, KeyError, TypeError):
                        saved = None
                    if (saved is not None
                            and _saved_policy_matches(existing, saved, snapshot,
                                                      Path(output_dir), force_nonce)
                            and isinstance(saved.get("planned_capacity_microusd"), int)
                            and 0 < saved["planned_capacity_microusd"] <= job_cap
                            and ledger.reactivate_source_first_budget_refusal(
                                existing["id"], semantic_key=semantic_key,
                                source_sha256=snapshot.source_sha256,
                                output_dir=Path(output_dir),
                                manifest_sha256=existing["manifest_sha256"])):
                        return {"status": "pending", "workflow_id": existing["id"],
                                "reason": "rolling_week_budget_retry_queued"}
                return {"status": ("accepted" if existing["status"] == "accepted"
                                   else "pending" if existing["status"] == "active"
                                   else existing["status"]),
                        "workflow_id": existing["id"]}
            forecast = _planned_capacity_hold(route, snapshot, packets)
            if forecast > job_cap:
                return {"status": "budget_blocked", "reason": "full_chain_capacity_exceeds_saved_cap",
                        "planned_capacity_microusd": forecast,
                        "authorized_job_cap_microusd": job_cap}
            # The future D0 and sparse findings cannot be priced exactly yet.
            # Hold only the saved authorized logical-job cap, not a maximum
            # context-window payload; unused dollars are released at terminal.
            planned_hold = job_cap
            manifest = _manifest(snapshot, packets, dimensions,
                output_dir=Path(output_dir), workspace_id=route.workspace_id,
                force_nonce=force_nonce, route=route,
                planned_capacity_microusd=planned_hold,
                forecast_microusd=forecast)
            root = Path(private_root) / "source_first" / semantic_key
            manifest_path = root / "manifest.json"
            manifest_sha = _json_file_once(manifest_path, manifest)
            decision = ledger.create_source_first_job(
                semantic_key=semantic_key, source_sha256=snapshot.source_sha256,
                output_dir=Path(output_dir), manifest_path=manifest_path,
                manifest_sha256=manifest_sha,
                credential_id=selected["id"], credential_version=selected["version"],
                workspace_id=route.workspace_id,
            )
            if decision.kind != "new":
                return {"status": decision.kind, "workflow_id": decision.job_id,
                        "reason": decision.reason}
            workflow = ledger.get_batch_workflow(decision.job_id)
            plan_reservation = ledger.reserve_source_first_plan(workflow["id"],
                    plan_sha256=manifest_sha,
                    reserve_microusd=planned_hold)
            if plan_reservation is False or getattr(plan_reservation, "kind", None) == "blocked":
                # Weekly capacity can become available after the rolling
                # window advances. Keep this zero-POST semantic job active so
                # the poller can reserve its saved plan on a later tick.
                return {"status": "budget_blocked", "workflow_id": workflow["id"],
                        "reason": "rolling_week_budget_exceeded"}
            workflow = ledger.get_batch_workflow(workflow["id"])
            return _reserve_and_post_wave1(ledger=ledger, workflow=workflow,
                snapshot=snapshot, packets=packets, route=route, client=client)
    except (CredentialError, RouteBlocked, BatchError) as exc:
        return {"status": "preflight_blocked", "reason": str(exc)}
    finally:
        ledger.close()


def _money_micros(value: object) -> int | None:
    if value is None:
        return None
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError):
        return None
    if not number.is_finite() or number < 0:
        return None
    return int((number * 1_000_000).to_integral_value(rounding=ROUND_UP))


def _token_count(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _credential_for_read(store: CredentialStore, attempt: dict,
                         workflow: dict) -> str:
    try:
        return store.reveal_for_existing_job(attempt["credential_id"],
                                             attempt["credential_version"])
    except CredentialError:
        # Rotation inside the same allowed workspace may resume GET only.
        for candidate in store.dispatch_candidates():
            if candidate.get("workspace_id") == workflow["workspace_id"]:
                try:
                    return store.reveal_for_dispatch(candidate["id"], candidate["version"])
                except CredentialError:
                    pass
        raise


def _expected_ids(ledger: Ledger, attempt: dict) -> list[str]:
    return [item["custom_id"] for item in ledger.batch_items(attempt["id"])]


def _recover_unknown(*, ledger: Ledger, attempt: dict, client: BatchClient,
                     private_root: Path) -> dict:
    """Search a bounded reservation window, then require exact GET evidence.

    The ledger has no immutable POST timestamp. Reservations proceed directly
    to POST, so a remote Batch outside the first hour after reservation cannot
    be identified automatically. An incomplete scan leaves the charge held.
    """
    class RecoveryDeferred(Exception):
        pass

    expected = set(_expected_ids(ledger, attempt))
    try:
        created_at = Decimal(str(attempt["created_at"]))
        if not created_at.is_finite() or created_at <= 0 or not expected:
            raise ValueError("invalid_recovery_window_or_items")
        # created_after/before are strict. Include the reservation second and
        # one minute of clock skew; cap the other end at one hour thereafter.
        window_start = int(created_at) - 60
        window_end = int(created_at) + 3601
        cursor = None
        seen_ids = set()
        candidates = []
        for _ in range(3):
            listing = client.list_batches(limit=100, after=cursor,
                                          created_after=window_start,
                                          created_before=window_end)
            body = listing.body
            if (not isinstance(body, dict) or not isinstance(body.get("data"), list)
                    or type(body.get("has_more")) is not bool):
                raise ValueError("invalid_recovery_listing")
            rows = body["data"]
            for row in rows:
                if not isinstance(row, dict) or not valid_batch_id(row.get("id")):
                    raise ValueError("invalid_recovery_row")
                batch_id = row["id"]
                if batch_id in seen_ids:
                    raise ValueError("repeated_recovery_batch")
                seen_ids.add(batch_id)
                if (row.get("model") != SUBMIT_MODEL
                        or row.get("endpoint") != "/v1/chat/completions"
                        or row.get("status") != "completed"):
                    continue
                remote_created = row.get("created_at")
                counts = row.get("request_counts")
                if (type(remote_created) is not int
                        or not window_start < remote_created < window_end
                        or not isinstance(counts, dict)
                        or type(counts.get("total")) is not int):
                    raise ValueError("invalid_recovery_candidate_metadata")
                if counts["total"] == len(expected):
                    candidates.append(row)
            if len(candidates) > 8:
                raise RecoveryDeferred("too_many_recovery_candidates")
            if not body["has_more"]:
                break
            next_cursor = body.get("last_id")
            if (not rows or not valid_batch_id(next_cursor)
                    or next_cursor != rows[-1]["id"]
                    or next_cursor == cursor):
                raise ValueError("invalid_recovery_cursor")
            cursor = next_cursor
        else:
            raise RecoveryDeferred("recovery_page_limit")
        matches = []
        for row in candidates:
            batch_id = row["id"]
            remote = client.get(batch_id).body
            if (not isinstance(remote, dict) or remote.get("id") != batch_id
                    or remote.get("created_at") != row["created_at"]
                    or remote.get("model") != SUBMIT_MODEL
                    or remote.get("endpoint") != "/v1/chat/completions"
                    or remote.get("status") != "completed"
                    or not isinstance(remote.get("request_counts"), dict)
                    or remote["request_counts"].get("total") != len(expected)):
                raise ValueError("recovery_batch_metadata_changed")
            results = remote.get("results")
            if (not isinstance(results, list)
                    or {item.get("custom_id") for item in results if isinstance(item, dict)} != expected
                    or len(results) != len(expected)):
                continue
            parsed = parse_batch_items(remote, sorted(expected), expected_batch_id=batch_id)
            if (parsed["duplicate_ids"] or parsed["extra_ids"] or parsed["missing_ids"]
                    or parsed["invalid_result_count"]):
                continue
            matches.append(remote)
            if len(matches) > 1:
                break
        if len(matches) == 1:
            remote = matches[0]
            evidence_path = (private_root / "source_first" / ledger.get_batch_workflow(
                attempt["workflow_id"])["semantic_key"] / "recovery" /
                (attempt["id"] + ".json"))
            evidence_sha = _json_file_once(evidence_path, remote)
            ledger.attach_recovered_batch_remote(attempt["id"],
                remote_id=remote["id"], evidence_path=evidence_path,
                evidence_sha256=evidence_sha)
            return {"attempt_id": attempt["id"], "status": "remote_recovered",
                    "remote_batch_id": remote["id"]}
        reason = "ambiguous_remote_candidates" if len(matches) > 1 else "remote_identity_unavailable"
    except RecoveryDeferred as exc:
        reason = str(exc)
    except (BatchError, ValueError):
        reason = "recovery_metadata_unavailable"
    ledger.defer_batch_unknown(attempt["id"], delay_seconds=600, error_code=reason)
    return {"attempt_id": attempt["id"], "status": "submission_unknown", "reason": reason}


def _terminal_route_attestation(client: BatchClient, intent: dict,
                                remote: dict) -> dict | None:
    """Resolve a Batch submit alias only through this route's exact catalog ID.

    The gateway may report its canonical dated model in a terminal envelope
    even though the sealed POST used the public submit alias. A family-prefix
    comparison would also accept a different model, so require the catalog's
    explicit canonical_slug and its single pinned OpenAI Batch endpoint.
    """
    if (intent.get("model") != SUBMIT_MODEL
            or intent.get("endpoint") != "/v1/chat/completions"
            or intent.get("provider") != {"only": [PROVIDER]}
            or remote.get("endpoint") != intent["endpoint"]):
        return None
    remote_model = remote.get("model")
    if remote_model == SUBMIT_MODEL:
        return {"resolution": "submitted_alias", "submitted_model": SUBMIT_MODEL,
                "reported_model": remote_model, "endpoint": intent["endpoint"],
                "provider_only": PROVIDER}
    if not isinstance(remote_model, str) or not remote_model:
        return None
    try:
        details = client.model_details().body.get("data")
        endpoint_data = client.model_endpoints().body.get("data")
    except (AttributeError, BatchError, TypeError, ValueError):
        return None
    if (not isinstance(details, dict) or details.get("id") != MODEL
            or details.get("canonical_slug") != remote_model
            or not isinstance(endpoint_data, dict)
            or endpoint_data.get("id") != MODEL
            or not isinstance(endpoint_data.get("endpoints"), list)):
        return None
    eligible = [entry for entry in endpoint_data["endpoints"]
                if isinstance(entry, dict) and entry.get("tag") == PROVIDER
                and entry.get("provider_name") == "OpenAI"
                and entry.get("model_id") == MODEL]
    if len(eligible) != 1:
        return None
    return {"resolution": "catalog_canonical_slug", "catalog_model_id": MODEL,
            "catalog_canonical_slug": remote_model,
            "submitted_model": SUBMIT_MODEL, "reported_model": remote_model,
            "endpoint": intent["endpoint"], "provider_only": PROVIDER,
            "catalog_provider_name": "OpenAI"}


def _missing_batch_observation(*, client, attempt: dict, workflow: dict,
                               ledger: Ledger, owner: str, private_root: Path) -> dict:
    # A stale/moved credential or a truncated workspace listing is not
    # evidence that a submitted job is unavailable. Never rotate or POST here.
    key = client.current_key().body.get("data", {})
    if (not isinstance(key, dict) or key.get("workspace_id") != workflow["workspace_id"]
            or attempt["workspace_id"] != workflow["workspace_id"]):
        raise ValueError("missing_batch_workspace_unverified")
    listing = client.list_batches(limit=100,
        created_after=int(attempt["created_at"]) - 60,
        created_before=int(attempt["created_at"]) + 3601).body
    rows = listing.get("data")
    if (not isinstance(rows, list) or listing.get("has_more") is not False
            or any(not isinstance(row, dict) or not valid_batch_id(row.get("id")) for row in rows)
            or any(row["id"] == attempt["remote_id"] for row in rows)):
        raise ValueError("missing_batch_list_unconfirmed")
    # Save only this application's identity; foreign workspace jobs are not
    # downloaded, recorded or touched.
    observation = {"kind": "workspace_confirmed_batch_missing",
        "batch_id": attempt["remote_id"], "workspace_id": workflow["workspace_id"],
        "http_status": 404, "list_complete": True, "listed": False,
        "observed_at": time.time()}
    path = private_root / "source_first" / workflow["semantic_key"] / "diagnostics" / (
        attempt["id"] + "-missing-" + owner + ".json")
    sha = _json_file_once(path, observation)
    status = ledger.record_batch_missing(attempt["id"], owner,
        evidence_path=path, evidence_sha256=sha)
    return {"attempt_id": attempt["id"], "status": status,
            "reason": "remote_batch_not_found", "charge": "unknown_hold_retained"}


def _poll_attempt(*, ledger: Ledger, attempt: dict, workflow: dict,
                  private_root: Path, client_factory) -> dict:
    owner = "source-first-" + uuid.uuid4().hex
    claimed = ledger.claim_batch_poll(attempt["id"], owner)
    if claimed is None:
        return {"attempt_id": attempt["id"], "status": "not_due"}
    try:
        store = CredentialStore(private_root / "credentials.sqlite3")
        token = _credential_for_read(store, attempt, workflow)
        client = client_factory(token)
        reply = client.get(attempt["remote_id"])
        remote = reply.body
        if remote.get("id") != attempt["remote_id"]:
            raise ValueError("batch_identity_mismatch")
        status = remote.get("status")
        if status in _ACTIVE_REMOTE:
            ledger.record_batch_poll(attempt["id"], owner, status,
                delay_seconds=min(1800, max(60, 2 ** min(attempt["post_count"] + 6, 10))))
            return {"attempt_id": attempt["id"], "status": status}
        if status not in TERMINAL:
            raise ValueError("unknown_batch_status")
        intent = _read_sealed(Path(attempt["payload_path"]), attempt["payload_sha256"])
        attestation = _terminal_route_attestation(client, intent, remote)
        if attestation is None:
            ledger.defer_batch_poll(attempt["id"], owner, delay_seconds=300,
                                    error_code="terminal_route_identity_unverified")
            return {"attempt_id": attempt["id"], "status": "poll_deferred",
                    "reason": "terminal_route_identity_unverified"}
        attestation_path = (private_root / "source_first" / workflow["semantic_key"] /
                            "route_attestations" / (attempt["id"] + ".json"))
        _json_file_once(attestation_path, {
            "attempt_id": attempt["id"], "remote_batch_id": attempt["remote_id"],
            **attestation})
        parsed = parse_batch_items(remote, _expected_ids(ledger, attempt),
                                   expected_batch_id=attempt["remote_id"])
        reported_counts = remote.get("request_counts")
        corrupt_item_identity = (bool(parsed["extra_ids"])
            or bool(parsed["invalid_result_count"])
            or (isinstance(reported_counts, dict)
                and reported_counts.get("total") != len(_expected_ids(ledger, attempt))))
        base = private_root / "source_first" / workflow["semantic_key"] / "terminal"
        terminal_path = base / (attempt["id"] + ".json")
        terminal_sha = _json_file_once(terminal_path, remote)
        outcomes = {}
        for custom_id, item in parsed["items"].items():
            item_raw = item["raw"]
            raw_path = base / (attempt["id"] + "-" + custom_id + ".json") if item_raw is not None else None
            raw_sha = _json_file_once(raw_path, {"item": item_raw}) if raw_path else None
            usage = item.get("usage") if isinstance(item.get("usage"), dict) else {}
            item_status = "invalid" if corrupt_item_identity else item["status"]
            if item_status == "ok":
                item_status = "completed"
            elif item_status not in {"item_error", "refusal", "length", "invalid",
                                     "missing", "duplicate", "unavailable", "http_error"}:
                item_status = "invalid"
            outcomes[custom_id] = {
                "status": item_status,
                "billed_microusd": _money_micros(usage.get("cost")),
                "prompt_tokens": _token_count(usage.get("prompt_tokens")),
                "completion_tokens": _token_count(usage.get("completion_tokens")),
                "raw_path": raw_path, "raw_sha256": raw_sha,
                "error_code": None if item_status == "completed" else item_status,
            }
        batch_usage = parsed["batch_usage"] or {}
        ledger.record_batch_terminal(attempt["id"], owner, status,
            batch_cost_microusd=_money_micros(batch_usage.get("cost")),
            terminal_path=terminal_path, terminal_sha256=terminal_sha,
            item_outcomes=outcomes)
        return {"attempt_id": attempt["id"], "status": status,
                "item_statuses": {key: value["status"] for key, value in outcomes.items()}}
    except CredentialError:
        ledger.defer_batch_poll(attempt["id"], owner, delay_seconds=600,
                                error_code="credential_required", credential_required=True)
        return {"attempt_id": attempt["id"], "status": "credential_required"}
    except (BatchError, ValueError, OSError) as exc:
        if isinstance(exc, BatchError) and exc.code == 404:
            try:
                return _missing_batch_observation(client=client, attempt=attempt,
                    workflow=workflow, ledger=ledger, owner=owner, private_root=private_root)
            except (BatchError, CredentialError, ValueError, OSError):
                reason = "remote_batch_404_recovery_unconfirmed"
        else:
            reason = ("batch_http_" + str(exc.code) if isinstance(exc, BatchError)
                      and exc.code is not None else "batch_get_or_parse_failed")
        ledger.defer_batch_poll(attempt["id"], owner, delay_seconds=300,
                                error_code=reason)
        return {"attempt_id": attempt["id"], "status": "poll_deferred"}


def _cleanup_terminal_once(*, ledger: Ledger, attempt: dict,
                           workflow: dict, private_root: Path,
                           client_factory) -> dict | None:
    """DELETE the captured terminal gateway record under existing retention policy.

    This does not claim provider files or separate I/O logs were deleted.
    A failed DELETE leaves the sealed raw intact and can be retried next tick.
    """
    if attempt["status"] not in TERMINAL or not attempt["remote_id"]:
        return None
    try:
        _read_sealed(Path(attempt["terminal_path"]), attempt["terminal_sha256"])
    except (OSError, ValueError, TypeError):
        return {"attempt_id": attempt["id"],
                "status": "gateway_cleanup_capture_invalid"}
    receipt = private_root / "source_first" / workflow["semantic_key"] / "cleanup" / (
        attempt["id"] + ".json")
    if receipt.exists():
        return None
    try:
        store = CredentialStore(private_root / "credentials.sqlite3")
        token = _credential_for_read(store, attempt, workflow)
        reply = client_factory(token).delete(attempt["remote_id"])
        deletion = reply.body.get("deletion") if isinstance(reply.body, dict) else None
        if (reply.status_code != 200 or reply.body.get("id") != attempt["remote_id"]
                or not isinstance(deletion, dict)
                or deletion.get("openrouter") != "deleted"):
            raise ValueError("gateway_delete_unconfirmed")
        _json_file_once(receipt, {"batch_id": attempt["remote_id"],
            "http_status": reply.status_code, "gateway_reply": reply.body,
            "provider_files": "per_gateway_deletion_response",
            "io_logs": "unchanged_by_batch_delete"})
        return {"attempt_id": attempt["id"], "status": "gateway_cleanup_recorded"}
    except (CredentialError, BatchError, ValueError, OSError):
        return {"attempt_id": attempt["id"], "status": "gateway_cleanup_pending"}


def _stage_attempt(ledger: Ledger, workflow_id: str, stage: str) -> dict | None:
    attempts = [row for row in ledger.list_batch_attempts(workflow_id)
                if row["stage"] == stage]
    if len(attempts) > 1:
        raise ValueError("multiple_stage_attempts_require_bounded_recovery")
    return attempts[0] if attempts else None


def _verify_reserved_dispatch(*, ledger: Ledger, workflow: dict,
                              manifest: dict, attempt: dict,
                              client: BatchClient) -> None:
    """Re-admit an unposted durable intent after a scheduler restart.

    The old reservation protects the app budget, but it does not prove that
    the key, account policy, endpoint, price or key balance is still valid.
    For wave one, both pending POSTs compete for the same key balance.
    """
    route = verify_source_first_batch_route(client)
    if (attempt["post_count"] != 0 or attempt["status"] != "reserved"
            or attempt["credential_id"] != workflow["credential_id"]
            or attempt["credential_version"] != workflow["credential_version"]
            or attempt["workspace_id"] != workflow["workspace_id"]
            or route.workspace_id != workflow["workspace_id"]
            or route.model != manifest.get("model_submit_slug")
            or route.batch_endpoint_model != manifest.get("resolved_batch_endpoint")
            or route.provider_endpoint_tag != manifest.get("provider_endpoint_tag")):
        raise RouteBlocked("reserved_batch_route_identity_changed")
    if (manifest.get("prompt_hashes") !=
            {name: prompt_sha256_for_stage(name) for name in STAGES}
            or manifest.get("schema_hashes") !=
            {name: digest(schema_for_stage(name)) for name in STAGES}):
        raise RouteBlocked("stage_contract_changed_during_job")
    pending = [attempt]
    if attempt["stage"] in {"writer", "extract"}:
        pending = [row for row in ledger.list_batch_attempts(workflow["id"])
                   if row["stage"] in {"writer", "extract"}
                   and row["status"] == "reserved" and row["post_count"] == 0]
    pricing_route = replace(route, key_limit_remaining_usd=None)
    total_pending = 0
    for row in pending:
        envelope = _read_sealed(Path(row["payload_path"]), row["payload_sha256"])
        saved = {item["custom_id"]: item for item in ledger.batch_items(row["id"])}
        requests = envelope.get("requests")
        if (not isinstance(requests, list) or len(requests) != len(saved)
                or len({entry.get("custom_id") for entry in requests
                        if isinstance(entry, dict)}) != len(saved)):
            raise RouteBlocked("reserved_batch_items_changed")
        for entry in requests:
            if not isinstance(entry, dict) or entry.get("custom_id") not in saved:
                raise RouteBlocked("reserved_batch_items_changed")
            body = entry.get("body")
            item = saved[entry["custom_id"]]
            if not isinstance(body, dict) or digest(body) != item["request_sha256"]:
                raise RouteBlocked("reserved_batch_items_changed")
            current_price = pricing_route.reserve_microusd(
                canonical_bytes(body), max_completion_tokens=item["max_output_tokens"],
                authorized_job_cap_microusd=workflow["planned_reserve_microusd"])
            if current_price > item["reserved_microusd"]:
                raise RouteBlocked("reserved_batch_price_increased")
        total_pending += row["reserved_microusd"]
    if (route.key_limit_remaining_usd is not None
            and Decimal(total_pending) > route.key_limit_remaining_usd * 1_000_000):
        raise RouteBlocked("key_budget_insufficient")


def _stage_native(ledger: Ledger, attempt: dict) -> tuple[dict[str, dict], dict[str, str]]:
    """Decode only uniquely completed custom IDs from an immutable GET capture."""
    if attempt and attempt["status"] == "remote_unavailable":
        return {}, {identity: "remote_batch_not_found" for identity in _expected_ids(ledger, attempt)}
    if not attempt or attempt["status"] not in TERMINAL:
        return {}, {}
    remote = _read_sealed(Path(attempt["terminal_path"]), attempt["terminal_sha256"])
    parsed = parse_batch_items(remote, _expected_ids(ledger, attempt),
                               expected_batch_id=attempt["remote_id"])
    expected = _expected_ids(ledger, attempt)
    counts = remote.get("request_counts")
    if (parsed["extra_ids"] or parsed["invalid_result_count"]
            or (isinstance(counts, dict) and counts.get("total") != len(expected))):
        return {}, {custom_id: "terminal_item_identity_invalid" for custom_id in expected}
    recorded = {row["custom_id"]: row for row in ledger.batch_items(attempt["id"])}
    reports, failures = {}, {}
    for custom_id, outcome in parsed["items"].items():
        if outcome["status"] != "ok" or recorded[custom_id]["status"] != "completed":
            failures[custom_id] = (outcome["status"] if outcome["status"] != "ok"
                                   else recorded[custom_id]["status"])
            continue
        try:
            content = outcome["body"]["choices"][0]["message"]["content"]
            native = json.loads(content)
            if not isinstance(native, dict):
                raise ValueError("report_not_object")
            reports[custom_id] = native
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            failures[custom_id] = "invalid_json"
    return reports, failures


def _packet_custom_id(ledger: Ledger, attempt: dict, packet: dict) -> str:
    requests = _read_sealed(Path(attempt["payload_path"]), attempt["payload_sha256"])["requests"]
    matches = [row["custom_id"] for row in requests
               if row["custom_id"].endswith("-" + packet["packet_id"] + "-a1")]
    if len(matches) != 1:
        raise ValueError("packet_custom_id_missing")
    return matches[0]


def _inventory_or_placeholder(raw: dict | None, packet: dict,
                              snapshot: SourceSnapshot) -> tuple[dict, str | None]:
    if raw is not None:
        try:
            return validate_inventory(raw, packet, snapshot), None
        except (ValueError, KeyError, TypeError):
            try:
                return salvage_inventory(raw, packet, snapshot), "inventory_salvaged_incomplete"
            except (ValueError, KeyError, TypeError):
                reason = "inventory_validation_failed"
    else:
        reason = "inventory_item_unavailable"
    return {
        "packet_id": packet["packet_id"], "complete": False,
        "evidence": {}, "units": {}, "facets": {}, "open_links": [],
        "source_accounting": {u_id: {"u_id": u_id, "disposition": "uncertain",
                                "unit_ids": [], "note": reason}
                              for u_id in packet["core_ids"]},
        "unprocessed_ids": packet["core_ids"], "native_report": raw,
    }, reason


def _validated_wave_context(ledger: Ledger, workflow: dict, snapshot: SourceSnapshot,
                            packets: tuple[dict, ...]) -> dict:
    writer = _stage_attempt(ledger, workflow["id"], "writer")
    extraction = _stage_attempt(ledger, workflow["id"], "extract")
    writer_raw, writer_failure = _stage_native(ledger, writer)
    writer_id = _expected_ids(ledger, writer)[0]
    document = writer_raw.get(writer_id)
    failures = list(writer_failure.values())
    if document is not None:
        try:
            validate_document(document, snapshot.index)
        except (ValueError, KeyError, TypeError):
            document = None
            failures.append("writer_document_invalid")
    if document is None:
        return {"document": None, "failures": failures or ["writer_unavailable"]}
    registry = build_surfaces(document, snapshot.index)
    partitions = partition_surfaces(registry, packets)
    extract_raw, extract_failures = _stage_native(ledger, extraction)
    failures.extend(extract_failures.values())
    inventories = []
    for packet in packets:
        custom_id = _packet_custom_id(ledger, extraction, packet)
        inventory, failure = _inventory_or_placeholder(
            extract_raw.get(custom_id), packet, snapshot)
        inventories.append(inventory)
        if failure:
            failures.append(packet["packet_id"] + ":" + failure)
        elif not inventory["complete"]:
            failures.append(packet["packet_id"] + ":inventory_incomplete")
    combined = normalize_inventories(inventories, packets, snapshot)
    return {"document": document, "registry": registry, "partitions": partitions,
            "inventories": inventories, "combined": combined, "failures": failures}


def _audit_payload(packet: dict, inventory: dict, document: dict,
                   surfaces: list[dict]) -> dict:
    units = []
    for unit_id, unit in inventory["units"].items():
        presented = deepcopy(unit)
        presented["facets"] = [deepcopy(facet) for facet in inventory["facets"].values()
                               if facet["unit_id"] == unit_id]
        units.append(presented)
    return {
        "SOURCE_PACKET": packet["records"],
        "SOURCE_UNITS": units,
        "SOURCE_EVIDENCE": inventory["evidence"],
        "DRAFT": document,
        "DOCUMENT_SURFACES": surfaces,
        "EXPECTED_UNIT_IDS": list(inventory["units"]),
        "EXPECTED_SURFACE_IDS": [row["surface_id"] for row in surfaces],
    }


def _validated_audits(ledger: Ledger, workflow: dict, snapshot: SourceSnapshot,
                      packets: tuple[dict, ...], context: dict) -> tuple[list[dict], list[str]]:
    attempt = _stage_attempt(ledger, workflow["id"], "audit")
    raw, failed = _stage_native(ledger, attempt)
    failures = list(failed.values())
    audits = []
    for number, packet in enumerate(packets):
        report = raw.get(_packet_custom_id(ledger, attempt, packet))
        if report is None:
            failures.append(packet["packet_id"] + ":audit_unavailable")
            continue
        arguments = dict(packet=packet, packet_inventory=context["inventories"][number],
            surfaces=context["partitions"][number], full_registry=context["registry"], snapshot=snapshot)
        try:
            checked = validate_audit(report, **arguments)
        except (ValueError, KeyError, TypeError) as exc:
            try:
                inventory = context["inventories"][number]
                checked = salvage_review(report, stage="audit", snapshot=snapshot,
                    registry=context["registry"], packet=packet,
                    known_evidence=inventory["evidence"], known_units=set(inventory["units"]),
                    known_context_targets=set(inventory["units"]) | set(inventory["facets"]) | set(context["registry"]))
                checked["validation_error"] = str(exc)
            except (ValueError, KeyError, TypeError):
                failures.append(packet["packet_id"] + ":audit_validation_failed")
                continue
        audits.append(checked)
        normalized_path = Path(ledger.root) / "source_first" / workflow["semantic_key"] / "normalized" / (packet["packet_id"] + "-audit.json")
        _json_file_once(normalized_path, checked)
        if not checked["complete"]:
            failures.append(packet["packet_id"] + ":audit_incomplete")
    return audits, failures


def _links_with_ids(links: list[dict]) -> list[dict]:
    return [{"link_id": "link-" + str(number + 1), **link}
            for number, link in enumerate(links)]


def _global_payload(snapshot: SourceSnapshot, context: dict,
                    audits: list[dict]) -> dict:
    requests = [row for audit in audits for row in audit["context_requests"]]
    links = _links_with_ids(context["combined"]["open_links"])
    findings = [row for audit in audits for row in audit["findings"]]
    additional = [unit for audit in audits for unit in audit["additional_units"].values()]
    source_evidence = dict(context["combined"]["evidence"])
    for audit in audits:
        source_evidence.update(audit["evidence"])
    return {
        "FULL_SOURCE": list(snapshot.records),
        "FULL_DRAFT": context["document"],
        "UNITS": list(context["combined"]["units"].values()) + additional,
        "SOURCE_EVIDENCE": source_evidence,
        "ALL_TASKS": context["document"]["tasks"],
        "CONTEXT_REQUESTS": requests,
        "OPEN_LINKS": links,
        "COMPACT_FINDINGS": findings,
        "EXPECTED_REVIEW_IDS": [row["request_id"] for row in requests],
    }


def _validated_global(ledger: Ledger, workflow: dict, snapshot: SourceSnapshot,
                      context: dict, audits: list[dict]) -> tuple[dict | None, list[str]]:
    attempt = _stage_attempt(ledger, workflow["id"], "global")
    raw, failed = _stage_native(ledger, attempt)
    report = next(iter(raw.values()), None)
    if report is None:
        return None, list(failed.values()) or ["global_unavailable"]
    expected_context = {row["request_id"] for audit in audits
                        for row in audit["context_requests"]}
    links = _links_with_ids(context["combined"]["open_links"])
    known_evidence = dict(context["combined"]["evidence"])
    for audit in audits:
        known_evidence.update(audit["evidence"])
    unit_ids = set(context["combined"]["units"]) | {unit_id for audit in audits
                                                   for unit_id in audit["additional_units"]}
    link_ids = {row["link_id"] for row in links}
    try:
        checked = validate_global(report, snapshot=snapshot,
            expected_context_ids=expected_context, full_registry=context["registry"],
            expected_link_ids=link_ids, expected_unit_ids=unit_ids, known_evidence=known_evidence)
    except (ValueError, KeyError, TypeError) as exc:
        try:
            checked = salvage_review(report, stage="global", snapshot=snapshot,
                registry=context["registry"], known_evidence=known_evidence,
                known_units=unit_ids, known_contexts=expected_context, known_links=link_ids)
            checked["validation_error"] = str(exc)
        except (ValueError, KeyError, TypeError):
            return None, ["global_validation_failed"]
    normalized_path = Path(ledger.root) / "source_first" / workflow["semantic_key"] / "normalized" / "global.json"
    _json_file_once(normalized_path, checked)
    return checked, [] if checked["complete"] else ["global_incomplete"]


def _dispatch_wave(*, ledger: Ledger, workflow: dict, manifest: dict,
                   stage: str, payloads: list[tuple[str, dict]],
                   private_root: Path, client_factory) -> dict:
    """One new Batch create POST for one dependency-ready stage."""
    existing = _stage_attempt(ledger, workflow["id"], stage)
    if existing is not None:
        return {"status": existing["status"], "attempt_id": existing["id"]}
    store = CredentialStore(private_root / "credentials.sqlite3")
    try:
        with credential_dispatch_guard(store.path):
            token = store.reveal_for_dispatch(workflow["credential_id"],
                                              workflow["credential_version"])
            client = client_factory(token)
            route = verify_source_first_batch_route(client)
            if (route.workspace_id != workflow["workspace_id"]
                    or route.model != manifest["model_submit_slug"]
                    or route.batch_endpoint_model != manifest["resolved_batch_endpoint"]
                    or route.provider_endpoint_tag != manifest["provider_endpoint_tag"]):
                raise RouteBlocked("batch_route_identity_changed")
            if (manifest["prompt_hashes"] !=
                    {name: prompt_sha256_for_stage(name) for name in STAGES}
                    or manifest["schema_hashes"] !=
                    {name: digest(schema_for_stage(name)) for name in STAGES}):
                raise RouteBlocked("stage_contract_changed_during_job")
            items = _items_for_wave(workflow["id"], stage, payloads)
            items = _capacity_items(ledger=ledger, workflow=workflow, route=route,
                snapshot=_snapshot_from_manifest(manifest), packets=tuple(manifest["packets"]),
                stage=stage, items=items)
            decision, path, payload_sha = _batch_intent(
                ledger=ledger, workflow=workflow, stage=stage, items=items, route=route)
            if decision.kind == "blocked":
                return {"status": "budget_blocked", "stage": stage,
                        "reason": decision.reason}
            if decision.kind == "pending":
                return {"status": "pending", "stage": stage,
                        "attempt_id": decision.attempt_id}
            return _post_reserved(ledger=ledger, attempt_id=decision.attempt_id,
                                  payload_path=path, payload_sha=payload_sha,
                                  client=client)
    except RouteBlocked as exc:
        reason = str(exc)
        return {"status": ("budget_blocked" if reason in
                {"job_budget_exceeded", "context_capacity_unverified"}
                else "dispatch_blocked"), "stage": stage, "reason": reason}
    except (CredentialError, BatchError) as exc:
        return {"status": "dispatch_blocked", "stage": stage,
                "reason": str(exc)}


def _repair_payload(snapshot: SourceSnapshot, context: dict, findings: dict,
                    global_report: dict | None, evidence: dict) -> dict:
    return {
        "DRAFT": context["document"],
        "SOURCE_CONTEXT": list(snapshot.records),
        "FINDINGS": list(findings.values()),
        "SOURCE_EVIDENCE": evidence,
        "RELATION_RESOLUTIONS": list(global_report["resolutions"].values()) if global_report else [],
        "AFFECTED_SURFACES": list(context["registry"].values()),
        "ALLOWED_OPERATIONS": ["replace_field", "update_task", "add_section_item", "create_task"],
        "TASK_SCHEMA_DESCRIPTION": WRITER_SCHEMA["$defs"]["task"],
        "OPERATION_CONTRACT": {
            "replace_field": "target_id = exact surface_id; field_key = its exact field_key; value_json = serialized field value",
            "update_task": "Use one operation per changed business field, addressed by exact surface_id and field_key; retain task identity",
            "unresolved": "Only exact FINDINGS finding_id strings; explanations belong in preservation_notes",
            "shared_targets": "Put all findings changing the same field in one bundle and return one combined value",
        },
    }


def _verification_payload(snapshot: SourceSnapshot, context: dict,
                          plan: dict, candidate: dict, before_hashes: dict) -> dict:
    return {
        "ORIGINAL_DRAFT": context["document"],
        "CANDIDATE_DOCUMENT": candidate,
        "PATCH_BUNDLES": list(plan["bundles"].values()),
        "EXACT_DIFF": before_hashes,
        "SOURCE_CONTEXT": list(snapshot.records),
        "EXPECTED_BUNDLE_IDS": list(plan["bundles"]),
    }


def _validated_patch(ledger: Ledger, workflow: dict, context: dict,
                     findings: dict[str, dict], evidence: dict[str, dict],
                     snapshot: SourceSnapshot) -> tuple[dict | None, dict | None,
                                                        dict | None, list[str]]:
    attempt = _stage_attempt(ledger, workflow["id"], "repair")
    raw, failed = _stage_native(ledger, attempt)
    report = next(iter(raw.values()), None)
    if report is None:
        return None, None, None, list(failed.values()) or ["repair_unavailable"]
    try:
        plan, provenance = normalize_patch_report(report, document=context["document"],
            findings=findings, registry=context["registry"], evidence=evidence,
            source_index=snapshot.index)
        candidate, before = stage_patch_candidate(context["document"], plan,
            context["registry"], snapshot.index, evidence)
        normalized = {"plan": plan, "before": before,
                      "candidate_sha256": digest(candidate), "provenance": provenance}
        normalized_path = Path(ledger.root) / "source_first" / workflow["semantic_key"] / "normalized" / (NORMALIZATION_VERSION + ".json")
        normalized_sha = _json_file_once(normalized_path, normalized)
        if workflow.get("repair_recovery_path"):
            recovery = _read_sealed(Path(workflow["repair_recovery_path"]),
                                    workflow["repair_recovery_sha256"])
            if recovery["normalized_patch_sha256"] != normalized_sha:
                raise ValueError("repair_recovery_candidate_changed")
        return plan, candidate, before, [] if plan["complete"] else ["repair_incomplete"]
    except (ValueError, KeyError, TypeError) as exc:
        if workflow.get("repair_recovery_sha256"):
            # A registered candidate is immutable. Do not downgrade a changed
            # recovery artifact into another publication of D0.
            raise
        diagnostic = Path(ledger.root) / "source_first" / workflow["semantic_key"] / "diagnostics" / (NORMALIZATION_VERSION + "-failure.json")
        _json_file_once(diagnostic, {"error": str(exc), "native_sha256": digest(report)})
        return None, None, None, ["repair_validation_failed"]


def _draft_root(ledger: Ledger, workflow: dict) -> Path:
    root = Path(ledger.root) / "source_first" / workflow["semantic_key"]
    if workflow.get("repair_recovery_sha256"):
        root = root / "recovery" / workflow["repair_recovery_sha256"]
    return root / "drafts"


def resume_saved_repair(*, private_root: Path, workflow_id: str) -> dict:
    """Prepare one continuation from sealed responses; never dispatch here.

    Invoked for a published draft whose repair failed bookkeeping validation.
    Original generations and native responses stay sealed. A fresh candidate
    must still pass the normal paid verification and publication transaction.
    """
    ledger = Ledger(Path(private_root))
    try:
        workflow = ledger.get_batch_workflow(workflow_id)
        if workflow is None:
            raise ValueError("repair_recovery_workflow_unknown")
        if workflow.get("repair_recovery_sha256"):
            _read_sealed(Path(workflow["repair_recovery_path"]), workflow["repair_recovery_sha256"])
            return {"status": workflow["status"], "workflow_id": workflow_id,
                    "recovery_sha256": workflow["repair_recovery_sha256"]}
        accepted = Path(workflow["accepted_document_path"] or "")
        if (workflow["status"] != "accepted" or not accepted.is_file()
                or _sha(accepted.read_bytes()) != workflow["accepted_document_sha256"]):
            raise ValueError("repair_recovery_not_accepted")
        sidecar = json.loads((accepted.parent / "review_sidecar.json").read_text())
        if "repair_validation_failed" not in sidecar.get("technical_failures", []):
            raise ValueError("repair_recovery_no_validation_failure")
        pointer_path = Path(workflow["output_dir"]) / "summary_current.json"
        pointer = json.loads(pointer_path.read_text())
        if pointer["generation_id"] != accepted.parent.name:
            raise ValueError("repair_recovery_newer_generation_selected")
        manifest = _read_sealed(Path(workflow["manifest_path"]), workflow["manifest_sha256"])
        snapshot = _snapshot_from_manifest(manifest)
        if _sha(snapshot.transcript_path.read_bytes()) != snapshot.source_sha256:
            raise ValueError("repair_recovery_source_changed")
        context = _validated_wave_context(ledger, workflow, snapshot, tuple(manifest["packets"]))
        audits, _ = _validated_audits(ledger, workflow, snapshot, tuple(manifest["packets"]), context)
        global_report, _ = _validated_global(ledger, workflow, snapshot, context, audits)
        findings = {row["finding_id"]: row for audit in audits for row in audit["findings"]}
        evidence = dict(context["combined"]["evidence"])
        for report in [*audits, *([global_report] if global_report else [])]:
            findings.update({row["finding_id"]: row for row in report["findings"]})
            evidence.update(report["evidence"])
        plan, candidate, before, failures = _validated_patch(ledger, workflow, context,
                                                            findings, evidence, snapshot)
        if plan is None or not plan["bundles"]:
            raise ValueError("repair_recovery_no_valid_bundles")
        TaskStore(Path(private_root) / "tasks.sqlite3").preview_reconcile(snapshot.source_sha256, candidate["tasks"])
        root = Path(private_root) / "source_first" / workflow["semantic_key"]
        receipt_path = root / "repair_recovery.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
        else:
            receipt = {"workflow_id": workflow_id, "source_sha256": snapshot.source_sha256,
                "accepted_document_path": str(accepted),
                "accepted_document_sha256": workflow["accepted_document_sha256"],
                "expected_pointer": pointer, "normalization_version": NORMALIZATION_VERSION,
                "normalized_patch_sha256": _sha((root / "normalized" / (NORMALIZATION_VERSION + ".json")).read_bytes()),
                "repair_terminal_sha256": _stage_attempt(ledger, workflow_id, "repair")["terminal_sha256"],
                "generation_id": datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:12]}
        receipt_sha = _json_file_once(receipt_path, receipt)
        ledger.resume_saved_repair(workflow_id, receipt_path=receipt_path, receipt_sha256=receipt_sha)
        return {"status": "active", "workflow_id": workflow_id,
            "recovery_sha256": receipt_sha, "bundle_count": len(plan["bundles"]),
            "native_bundle_count": sum(len(v) for v in plan["normalization"]["bundle_lineage"].values()),
            "next_stage": "verify", "new_dispatches": 0}
    finally:
        ledger.close()


def _validated_verify(ledger: Ledger, workflow: dict, plan: dict,
                      snapshot: SourceSnapshot, registry: dict) -> tuple[dict | None, list[str]]:
    attempt = _stage_attempt(ledger, workflow["id"], "verify")
    raw, failed = _stage_native(ledger, attempt)
    report = next(iter(raw.values()), None)
    if report is None:
        return None, list(failed.values()) or ["verification_unavailable"]
    try:
        checked = validate_verification(report, plan=plan, snapshot=snapshot,
                                        registry=registry)
        return checked, [] if checked["complete"] else ["verification_incomplete"]
    except (ValueError, KeyError, TypeError):
        return None, ["verification_validation_failed"]


def _publish(*, ledger: Ledger, workflow: dict, manifest: dict,
             snapshot: SourceSnapshot, context: dict, audits: list[dict],
             global_report: dict | None, findings: dict[str, dict],
             evidence: dict[str, dict], failures: list[str],
             plan: dict | None, verification: dict | None,
             synthetic_marks: list[dict],
             private_root: Path) -> dict:
    document = context["document"]
    accepted = set(verification["accepted"]) if verification else set()
    rejected = set(plan["bundles"]) - accepted if plan else set()
    if plan and verification:
        document = accepted_patch_document(document, plan, verification,
            context["registry"], snapshot.index, evidence)
    unresolved_ids = set(findings)
    if plan:
        for bundle_id in accepted:
            unresolved_ids.difference_update(plan["bundles"][bundle_id]["finding_ids"])
    unresolved = [findings[identity] for identity in sorted(unresolved_ids)]
    relation_marks = []
    if global_report:
        for row in global_report["resolutions"].values():
            if row["status"] == "unresolved":
                relation_marks.append({
                    "finding_id": "unresolved_context:" + row["request_id"],
                    "problem": "Связь между репликами остаётся неустановленной",
                    "affected_surface_ids": row["affected_surface_ids"],
                    "evidence_ids": row["evidence_ids"],
                })
        for row in global_report["link_checks"]:
            if row["relation"] == "unresolved":
                refs = list(row["evidence_ids"])
                for unit_id in row["unit_ids"]:
                    refs.extend(context["combined"]["units"].get(unit_id, {}).get(
                        "evidence_ids", []))
                relation_marks.append({
                    "finding_id": "unresolved_link:" + row["link_id"],
                    "problem": "Отношение между действиями или утверждениями не установлено",
                    "affected_surface_ids": [], "evidence_ids": list(dict.fromkeys(refs)),
                })
    if unresolved or relation_marks or synthetic_marks:
        new_registry = build_surfaces(document, snapshot.index)
        try:
            marks_to_apply = remap_mark_targets(
                unresolved + relation_marks + synthetic_marks,
                plan=plan, accepted=accepted,
                old_registry=context["registry"], new_registry=new_registry)
        except ValueError:
            # Never qualify a different claim after a positional insertion.
            # Fall back to the intact D0 with every finding still unresolved.
            failures.append("mark_target_rebase_failed")
            accepted.clear()
            rejected = set(plan["bundles"]) if plan else set()
            document = context["document"]
            new_registry = context["registry"]
            unresolved = [findings[identity] for identity in sorted(findings)]
            marks_to_apply = unresolved + relation_marks + synthetic_marks
        document, marks = safe_mark_document(document,
            marks_to_apply, new_registry, snapshot, evidence)
    else:
        marks = []
    projection_root = _draft_root(ledger, workflow)
    projection_sha = _json_file_once(projection_root / "publication_projection.json", document)
    global_ambiguities = []
    if global_report:
        global_ambiguities = [row["request_id"] for row in global_report["resolutions"].values()
                              if row["status"] == "unresolved"]
        global_ambiguities.extend(row["link_id"] for row in global_report["link_checks"]
                                  if row["relation"] == "unresolved")
    if failures:
        review_status, quality_status = "review_incomplete", "review_incomplete"
    elif unresolved or global_ambiguities:
        review_status, quality_status = "reviewed_with_uncertainties", "unresolved"
    else:
        review_status, quality_status = "review_completed", "source_first_checked"
    attempts = ledger.list_batch_attempts(workflow["id"])
    held = sum(row["reserved_microusd"] for row in attempts
               if row["billed_microusd"] is None and row["status"] not in
               {"rejected_no_charge", "cancelled_before_submit"})
    billed = sum(row["billed_microusd"] for row in attempts
                 if row["billed_microusd"] is not None)
    unknown_bills = [row["id"] for row in attempts
                     if row["billed_microusd"] is None and row["post_count"]]
    checked = set()
    complete_inventories = {inventory["packet_id"] for inventory in context["inventories"]
                            if inventory["complete"]}
    audited_packets = {audit["packet_id"] for audit in audits
                       if audit["complete"] and audit["packet_id"] in complete_inventories}
    for packet in manifest["packets"]:
        if packet["packet_id"] in audited_packets:
            checked.update(packet["core_ids"])
    sidecar = {
        "schema_version": "luna_review_sidecar_v1", "generation_id": "",
        "source_revision": snapshot.source_sha256, "execution_status": "completed",
        "review_status": review_status, "source_scope_ids": list(snapshot.ids),
        "checked_ids": [u_id for u_id in snapshot.ids if u_id in checked],
        "unreviewed_ids": [u_id for u_id in snapshot.ids if u_id not in checked],
        "accepted_bundle_ids": sorted(accepted), "rejected_bundle_ids": sorted(rejected),
        "unresolved_bundle_ids": sorted(rejected),
        "source_ambiguities": global_ambiguities,
        "model_disagreements": [row["finding_id"] for row in unresolved + synthetic_marks],
        "technical_failures": list(dict.fromkeys(failures)),
        "billed_cost_microusd": billed if not unknown_bills else None,
        "held_cost_microusd": held, "unknown_bill_ids": unknown_bills,
        "artifact_hashes": [], "generation_lineage": [
            workflow["id"], "D0:" + context["d0_sha256"],
            "publication_projection:" + projection_sha,
        ] + (["D1:" + context["d1_sha256"]] if context.get("d1_sha256") else []),
    }
    quality = {"status": quality_status, "unresolved_count": len(unresolved) +
               len(global_ambiguities) + len(synthetic_marks),
               "applied_bundle_count": len(accepted),
               "technical_failures": sidecar["technical_failures"],
               "reviewed_source_ids": sidecar["checked_ids"],
               "local_marks": marks, "policy_version": POLICY_VERSION,
               "source_draft_sha256": context["d0_sha256"],
               "effective_projection_sha256": projection_sha}
    if plan and plan.get("normalization"):
        quality["patch_normalization"] = plan["normalization"]
        quality["applied_native_bundle_ids"] = [native for identity in sorted(accepted)
            for native in plan["normalization"]["bundle_lineage"][identity]]
    task_store = TaskStore(private_root / "tasks.sqlite3")
    task_plan = task_store.preview_reconcile(snapshot.source_sha256, document["tasks"])
    generation_id = datetime.fromtimestamp(workflow["created_at"], timezone.utc).strftime(
        "%Y%m%d-%H%M%S") + "-" + workflow["id"][:12]
    recovery = None
    if workflow.get("repair_recovery_path"):
        recovery = _read_sealed(Path(workflow["repair_recovery_path"]), workflow["repair_recovery_sha256"])
        generation_id = recovery["generation_id"]
        sidecar["generation_lineage"].append("previous_generation:" + recovery["expected_pointer"]["generation_id"])
    def commit_tasks():
        # publish_document holds the meeting's publication lock here. A newer
        # user generation must not be overwritten by this bounded continuation.
        if recovery:
            current = json.loads((Path(workflow["output_dir"]) / "summary_current.json").read_text())
            if current != recovery["expected_pointer"]:
                raise RevisionConflict("repair_recovery_current_changed")
        task_store.commit_reconcile(task_plan)
    writer = _stage_attempt(ledger, workflow["id"], "writer")
    try:
        generation, target = publish_document(
            document=document, source_index=snapshot.index,
            transcript_path=snapshot.transcript_path,
            output_dir=Path(workflow["output_dir"]),
            semantic_key=workflow["semantic_key"], job_id=workflow["id"],
            remote_batch_id=writer["remote_id"],
            credential_id=workflow["credential_id"],
            prompt_sha256=manifest["prompt_hashes"]["writer"],
            schema_sha256=manifest["schema_hashes"]["writer"],
            effective_tasks=task_plan.effective_tasks, generation_id=generation_id,
            quality_review=quality, review_sidecar=sidecar,
            before_pointer=commit_tasks,
        )
    except (RevisionConflict, ReconciliationConflict):
        return {"status": "publication_revision_conflict", "workflow_id": workflow["id"]}
    result_path = target / "model_document.json"
    result_sha = _sha(result_path.read_bytes())
    ledger.finish_batch_workflow(workflow["id"], status="accepted",
        result_path=result_path, result_sha256=result_sha)
    return {"status": "accepted", "workflow_id": workflow["id"],
            "generation_id": generation, "review_status": review_status,
            "result_path": str(target)}


def _ready(attempt: dict | None) -> bool:
    return bool(attempt and attempt["status"] in
                set(TERMINAL) | {"rejected_no_charge", "cancelled_before_submit", "remote_unavailable"})


def _advance_workflow(*, ledger: Ledger, workflow: dict,
                      private_root: Path, client_factory) -> dict:
    manifest = _read_sealed(Path(workflow["manifest_path"]),
                            workflow["manifest_sha256"])
    if (manifest["source_revision"] != workflow["source_sha256"]
            or manifest["workspace_id"] != workflow["workspace_id"]):
        raise ValueError("source_first_manifest_identity_changed")
    snapshot = _snapshot_from_manifest(manifest)
    packets = tuple(manifest["packets"])

    def obsolete_if_source_changed() -> dict | None:
        try:
            current_sha = _sha(snapshot.transcript_path.read_bytes())
        except OSError:
            current_sha = None
        if current_sha == snapshot.source_sha256:
            return None
        # A prepared intent has never left the process and can release its
        # hold. A submitted or ambiguous POST must still be recovered/polled.
        for attempt in ledger.list_batch_attempts(workflow["id"]):
            if attempt["status"] == "reserved" and attempt["post_count"] == 0:
                ledger.cancel_batch_before_submit(attempt["id"],
                                                  "source_revision_changed")
        unsettled = [attempt["id"] for attempt in ledger.list_batch_attempts(workflow["id"])
                     if attempt["status"] not in set(TERMINAL) |
                     {"rejected_no_charge", "cancelled_before_submit", "remote_unavailable"}]
        if unsettled:
            return {"status": "source_revision_obsolete_pending",
                    "workflow_id": workflow["id"], "pending_attempt_ids": unsettled}
        ledger.finish_batch_workflow(workflow["id"], status="failed",
                                     error_code="source_revision_changed")
        return {"status": "source_revision_changed", "workflow_id": workflow["id"]}

    obsolete = obsolete_if_source_changed()
    if obsolete is not None:
        return obsolete
    writer = _stage_attempt(ledger, workflow["id"], "writer")
    extract = _stage_attempt(ledger, workflow["id"], "extract")
    if not writer or not extract:
        try:
            store = CredentialStore(private_root / "credentials.sqlite3")
            with credential_dispatch_guard(store.path):
                token = store.reveal_for_dispatch(workflow["credential_id"],
                                                  workflow["credential_version"])
                client = client_factory(token)
                route = verify_source_first_batch_route(client)
                if (route.workspace_id != workflow["workspace_id"]
                        or route.model != manifest["model_submit_slug"]
                        or route.batch_endpoint_model != manifest["resolved_batch_endpoint"]):
                    raise RouteBlocked("batch_route_identity_changed")
                obsolete = obsolete_if_source_changed()
                if obsolete is not None:
                    return obsolete
                if not workflow["plan_sha256"]:
                    reserved = ledger.reserve_source_first_plan(workflow["id"],
                        plan_sha256=workflow["manifest_sha256"],
                        reserve_microusd=manifest["planned_capacity_microusd"])
                    if reserved is False or getattr(reserved, "kind", None) == "blocked":
                        return {"status": "budget_blocked", "workflow_id": workflow["id"],
                                "reason": getattr(reserved, "reason", None)
                                          or "rolling_week_budget_exceeded"}
                    workflow = ledger.get_batch_workflow(workflow["id"])
                return _reserve_and_post_wave1(ledger=ledger, workflow=workflow,
                    snapshot=snapshot, packets=packets, route=route, client=client)
        except (CredentialError, RouteBlocked, BatchError) as exc:
            return {"status": "wave1_dispatch_blocked", "workflow_id": workflow["id"],
                    "reason": str(exc)}
    if not _ready(writer) or not _ready(extract):
        return {"status": "wave1_pending", "workflow_id": workflow["id"]}
    context = _validated_wave_context(ledger, workflow, snapshot, packets)
    if context["document"] is None:
        ledger.finish_batch_workflow(workflow["id"], status="failed",
                                     error_code="writer_generation_failed")
        return {"status": "generation_failed", "workflow_id": workflow["id"],
                "source_path": str(snapshot.transcript_path)}
    draft_root = _draft_root(ledger, workflow)
    context["d0_sha256"] = _json_file_once(draft_root / "d0.json", context["document"])
    failures = context["failures"]
    audit_attempt = _stage_attempt(ledger, workflow["id"], "audit")
    if audit_attempt is None:
        obsolete = obsolete_if_source_changed()
        if obsolete is not None:
            return obsolete
        dispatched = _dispatch_wave(
            ledger=ledger, workflow=workflow, manifest=manifest, stage="audit",
            payloads=[(packet["packet_id"], _audit_payload(packet,
                context["inventories"][number], context["document"],
                context["partitions"][number]))
                for number, packet in enumerate(packets)],
            private_root=private_root, client_factory=client_factory)
        if dispatched["status"] == "budget_blocked":
            return _publish(ledger=ledger, workflow=workflow, manifest=manifest,
                snapshot=snapshot, context=context, audits=[], global_report=None,
                findings={}, evidence=context["combined"]["evidence"],
                failures=failures + ["audit_budget_blocked"], plan=None,
                verification=None, synthetic_marks=[], private_root=private_root)
        return {"workflow_id": workflow["id"], **dispatched}
    if not _ready(audit_attempt):
        return {"status": "audit_pending", "workflow_id": workflow["id"]}
    audits, audit_failures = _validated_audits(ledger, workflow, snapshot,
                                               packets, context)
    failures.extend(audit_failures)
    global_attempt = _stage_attempt(ledger, workflow["id"], "global")
    if global_attempt is None:
        obsolete = obsolete_if_source_changed()
        if obsolete is not None:
            return obsolete
        dispatched = _dispatch_wave(
            ledger=ledger, workflow=workflow, manifest=manifest, stage="global",
            payloads=[("full", _global_payload(snapshot, context, audits))],
            private_root=private_root, client_factory=client_factory)
        if dispatched["status"] == "budget_blocked":
            local_findings = {row["finding_id"]: row for audit in audits
                              for row in audit["findings"]}
            local_evidence = dict(context["combined"]["evidence"])
            for audit in audits:
                local_evidence.update(audit["evidence"])
            return _publish(ledger=ledger, workflow=workflow, manifest=manifest,
                snapshot=snapshot, context=context, audits=audits, global_report=None,
                findings=local_findings, evidence=local_evidence,
                failures=failures + ["global_budget_blocked"], plan=None,
                verification=None, synthetic_marks=[], private_root=private_root)
        return {"workflow_id": workflow["id"], **dispatched}
    if not _ready(global_attempt):
        return {"status": "global_pending", "workflow_id": workflow["id"]}
    global_report, global_failures = _validated_global(ledger, workflow,
        snapshot, context, audits)
    failures.extend(global_failures)
    findings = {row["finding_id"]: row for audit in audits
                for row in audit["findings"]}
    evidence = {key: value for audit in audits
                for key, value in audit["evidence"].items()}
    evidence.update(context["combined"]["evidence"])
    if global_report:
        findings.update({row["finding_id"]: row for row in global_report["findings"]})
        evidence.update(global_report["evidence"])
    additional_units = {unit_id: unit for audit in audits
                        for unit_id, unit in audit["additional_units"].items()}
    if global_report:
        additional_units.update(global_report["additional_units"])
    referenced_units = {unit_id for finding in findings.values()
                        for unit_id in finding["affected_unit_ids"]}
    synthetic_marks = [{
        "finding_id": "missing_finding_for_additional_unit:" + unit_id,
        "problem": "Обнаруженный при сверке исходный смысл не получил отдельной проверки покрытия",
        "affected_surface_ids": [], "evidence_ids": unit["evidence_ids"],
    } for unit_id, unit in additional_units.items() if unit_id not in referenced_units]
    if synthetic_marks:
        failures.append("additional_units_without_coverage_finding")
    plan, verification = None, None
    if findings:
        repair_attempt = _stage_attempt(ledger, workflow["id"], "repair")
        if repair_attempt is None:
            obsolete = obsolete_if_source_changed()
            if obsolete is not None:
                return obsolete
            dispatched = _dispatch_wave(
                ledger=ledger, workflow=workflow, manifest=manifest, stage="repair",
                payloads=[("findings", _repair_payload(snapshot, context,
                    findings, global_report, evidence))], private_root=private_root,
                client_factory=client_factory)
            if dispatched["status"] == "budget_blocked":
                return _publish(ledger=ledger, workflow=workflow, manifest=manifest,
                    snapshot=snapshot, context=context, audits=audits,
                    global_report=global_report, findings=findings, evidence=evidence,
                    failures=failures + ["repair_budget_blocked"], plan=None,
                    verification=None, synthetic_marks=synthetic_marks,
                    private_root=private_root)
            return {"workflow_id": workflow["id"], **dispatched}
        if not _ready(repair_attempt):
            return {"status": "repair_pending", "workflow_id": workflow["id"]}
        plan, candidate, before, patch_failures = _validated_patch(
            ledger, workflow, context, findings, evidence, snapshot)
        failures.extend(patch_failures)
        if plan and plan["bundles"]:
            context["d1_sha256"] = _json_file_once(draft_root / "d1.json", candidate)
            verify_attempt = _stage_attempt(ledger, workflow["id"], "verify")
            if verify_attempt is None:
                obsolete = obsolete_if_source_changed()
                if obsolete is not None:
                    return obsolete
                dispatched = _dispatch_wave(
                    ledger=ledger, workflow=workflow, manifest=manifest, stage="verify",
                    payloads=[("bundles", _verification_payload(snapshot, context,
                        plan, candidate, before))], private_root=private_root,
                    client_factory=client_factory)
                if dispatched["status"] == "budget_blocked":
                    return _publish(ledger=ledger, workflow=workflow, manifest=manifest,
                        snapshot=snapshot, context=context, audits=audits,
                        global_report=global_report, findings=findings, evidence=evidence,
                        failures=failures + ["verification_budget_blocked"],
                        plan=plan, verification=None,
                        synthetic_marks=synthetic_marks, private_root=private_root)
                return {"workflow_id": workflow["id"], **dispatched}
            if not _ready(verify_attempt):
                return {"status": "verify_pending", "workflow_id": workflow["id"]}
            verification, verify_failures = _validated_verify(
                ledger, workflow, plan, snapshot, context["registry"])
            failures.extend(verify_failures)
            if verification:
                evidence.update(verification["evidence"])
                synthetic_marks.extend(verification["new_findings"])
                if verification["new_findings"]:
                    failures.append("verification_new_findings_unresolved")
    return _publish(ledger=ledger, workflow=workflow, manifest=manifest,
        snapshot=snapshot, context=context, audits=audits,
        global_report=global_report, findings=findings, evidence=evidence,
        failures=failures, plan=plan, verification=verification,
        synthetic_marks=synthetic_marks,
        private_root=private_root)


def poll_source_first_once(*, private_root: Path,
                           client_factory=BatchClient) -> list[dict]:
    """One scheduler tick: recover/resume/GET due attempts, then advance jobs.

    This is deliberately nonblocking and never waits for a remote Batch.
    Leases and next_poll_at make a process restart use the existing remote ID.
    """
    private_root = Path(private_root)
    ledger = Ledger(private_root)
    outcomes = []
    try:
        for attempt in ledger.list_pending_batch_attempts():
            workflow = ledger.get_batch_workflow(attempt["workflow_id"])
            if workflow is None or workflow["status"] != "active":
                continue
            if attempt["status"] == "reserved":
                # A crash may have left a later-wave intent reserved. Check
                # the frozen source again before the scheduler resumes its POST.
                try:
                    manifest = _read_sealed(Path(workflow["manifest_path"]),
                                            workflow["manifest_sha256"])
                    source_path = Path(manifest["snapshot"]["transcript_path"])
                    source_current = (manifest["source_revision"] == workflow["source_sha256"]
                                      and _sha(source_path.read_bytes()) == workflow["source_sha256"])
                except (OSError, ValueError, KeyError, TypeError):
                    source_current = False
                if not source_current:
                    ledger.cancel_batch_before_submit(attempt["id"],
                                                      "source_revision_changed")
                    outcomes.append({"status": "source_revision_changed_before_submit",
                                     "attempt_id": attempt["id"]})
                    continue
                if attempt["stage"] == "extract":
                    writer = _stage_attempt(ledger, workflow["id"], "writer")
                    if writer and writer["status"] in {"rejected_no_charge",
                                                       "cancelled_before_submit",
                                                       "failed", "expired", "cancelled"}:
                        ledger.cancel_batch_before_submit(attempt["id"],
                                                          "writer_peer_not_submitted")
                        outcomes.append({"status": "cancelled_before_submit",
                                         "attempt_id": attempt["id"]})
                        continue
                    if (writer is None
                            or writer["status"] not in {"submitted", "polling",
                                                        "credential_required", "completed"}
                            or not writer["remote_id"]):
                        outcomes.append({"status": "wave1_peer_pending",
                                         "attempt_id": attempt["id"]})
                        continue
                if attempt["stage"] == "writer":
                    extract = _stage_attempt(ledger, workflow["id"], "extract")
                    if extract and extract["status"] == "cancelled_before_submit":
                        ledger.cancel_batch_before_submit(attempt["id"],
                                                          "extract_peer_not_submitted")
                        outcomes.append({"status": "cancelled_before_submit",
                                         "attempt_id": attempt["id"]})
                        continue
                if (attempt["stage"] in {"writer", "extract"}
                        and (_stage_attempt(ledger, workflow["id"], "writer") is None
                             or _stage_attempt(ledger, workflow["id"], "extract") is None)):
                    # Resume preflight/reservation of the complete first wave
                    # before either physical POST.
                    continue
                try:
                    store = CredentialStore(private_root / "credentials.sqlite3")
                    with credential_dispatch_guard(store.path):
                        token = store.reveal_for_dispatch(attempt["credential_id"],
                                                          attempt["credential_version"])
                        client = client_factory(token)
                        _verify_reserved_dispatch(ledger=ledger, workflow=workflow,
                            manifest=manifest, attempt=attempt, client=client)
                        try:
                            source_still_current = (
                                _sha(source_path.read_bytes()) == workflow["source_sha256"])
                        except OSError:
                            source_still_current = False
                        if not source_still_current:
                            ledger.cancel_batch_before_submit(attempt["id"],
                                                              "source_revision_changed")
                            outcomes.append({"status": "source_revision_changed_before_submit",
                                             "attempt_id": attempt["id"]})
                            continue
                        outcomes.append(_post_reserved(ledger=ledger,
                            attempt_id=attempt["id"],
                            payload_path=Path(attempt["payload_path"]),
                            payload_sha=attempt["payload_sha256"],
                            client=client))
                except CredentialError:
                    outcomes.append({"status": "credential_required",
                                     "attempt_id": attempt["id"]})
                except (RouteBlocked, BatchError, ValueError, OSError) as exc:
                    outcomes.append({"status": "dispatch_blocked",
                                     "attempt_id": attempt["id"],
                                     "reason": (str(exc) if isinstance(exc, RouteBlocked)
                                                else "reserved_batch_preflight_failed")})
            elif attempt["status"] == "submitting" and attempt["remote_id"] is None:
                ledger.defer_batch_unknown(attempt["id"], delay_seconds=60,
                                           error_code="post_outcome_unknown_after_restart")
                outcomes.append({"status": "submission_unknown",
                                 "attempt_id": attempt["id"]})
            elif attempt["status"] == "submission_unknown":
                try:
                    store = CredentialStore(private_root / "credentials.sqlite3")
                    token = _credential_for_read(store, attempt, workflow)
                    outcomes.append(_recover_unknown(ledger=ledger,
                        attempt=attempt, client=client_factory(token),
                        private_root=private_root))
                except CredentialError:
                    ledger.defer_batch_unknown(attempt["id"], delay_seconds=600,
                                               error_code="credential_required_for_recovery")
                    outcomes.append({"status": "credential_required",
                                     "attempt_id": attempt["id"]})
            elif attempt["remote_id"] is not None:
                outcomes.append(_poll_attempt(ledger=ledger, attempt=attempt,
                    workflow=workflow, private_root=private_root,
                    client_factory=client_factory))
        for workflow in ledger.list_source_first_workflows(states=("active",)):
            try:
                outcomes.append(_advance_workflow(ledger=ledger,
                    workflow=workflow, private_root=private_root,
                    client_factory=client_factory))
            except (RevisionConflict, ReconciliationConflict) as exc:
                outcomes.append({"status": "publication_reconciliation_conflict",
                                 "workflow_id": workflow["id"],
                                 "reason": type(exc).__name__})
            except ValueError as exc:
                if str(exc) in {"source_changed_before_publication",
                                "source_changed_during_publication"}:
                    ledger.finish_batch_workflow(workflow["id"], status="failed",
                                                 error_code="source_revision_changed")
                    outcomes.append({"status": "source_revision_changed",
                                     "workflow_id": workflow["id"]})
                else:
                    outcomes.append({"status": "workflow_integrity_error",
                                     "workflow_id": workflow["id"],
                                     "reason": str(exc)[:120]})
        for workflow in ledger.list_source_first_workflows(
                states=("accepted", "failed")):
            for attempt in ledger.list_batch_attempts(workflow["id"]):
                cleanup = _cleanup_terminal_once(ledger=ledger, attempt=attempt,
                    workflow=workflow, private_root=private_root,
                    client_factory=client_factory)
                if cleanup is not None:
                    outcomes.append(cleanup)
        return outcomes
    finally:
        ledger.close()
