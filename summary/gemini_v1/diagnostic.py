"""Two fixed Gemini Batch probes against one frozen source/draft excerpt.

This is an isolated, nonpublishing diagnostic. It uses the application's
credential store and spending ledger, while the ordinary scheduler collects
its two Batch results. Neither response can change a summary generation.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from pathlib import Path

from scripts.summary_credentials import CredentialError, CredentialStore, credential_dispatch_guard
from summary.luna_v1.ledger import (DIAGNOSTIC_GROUP_CAP_MICROUSD,
                                    DIAGNOSTIC_KINDS, Ledger,
                                    write_private_json)

from .batch import (BATCH_MODEL_IDS, MODEL, PROVIDER, BatchClient, BatchError,
                    batch_id_from_submit, canonical_request, extract_one_completed)
from .route import RouteBlocked, verify_batch_route
from .reconcile_contract import _unit_evidence_texts, draft_units


VERSION = "gemini_two_prompt_diagnostic_v1"
PRIVACY_MODE = "batch_gateway_retention_up_to_30d_provider_zdr_off_user_authorized"
SLOTS = ("baseline", "compact")
_SHA = re.compile(r"[a-f0-9]{64}\Z")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{7,127}\Z")


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _pin_json(path: Path, value: object) -> None:
    """Create one immutable artifact, or verify an earlier identical write."""
    if path.is_symlink():
        raise ValueError("diagnostic artifact symlink")
    if path.exists():
        if _json_bytes(json.loads(path.read_text(encoding="utf-8"))) != _json_bytes(value):
            raise ValueError("diagnostic artifact identity changed")
        return
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(_json_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if path.is_symlink() or _json_bytes(json.loads(path.read_text(encoding="utf-8"))) != _json_bytes(value):
                raise ValueError("diagnostic artifact identity changed")
    finally:
        temporary.unlink(missing_ok=True)


def _load_plan(path: Path) -> tuple[dict, dict[str, dict]]:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(plan, dict) or set(plan) != {
            "run_id", "source_sha256", "draft_sha256", "cases", "requests"}
            or not isinstance(plan["run_id"], str)
            or _RUN_ID.fullmatch(plan["run_id"]) is None
            or any(not isinstance(plan[key], str) or _SHA.fullmatch(plan[key]) is None
                   for key in ("source_sha256", "draft_sha256"))
            or not isinstance(plan["cases"], list) or not plan["cases"]
            or any(not isinstance(x, str) or not x or len(x) > 80 for x in plan["cases"])
            or len(set(plan["cases"])) != len(plan["cases"])
            or not isinstance(plan["requests"], dict)
            or set(plan["requests"]) != set(SLOTS)):
        raise ValueError("invalid diagnostic plan")
    requests = {}
    for slot in SLOTS:
        name = plan["requests"][slot]
        if not isinstance(name, str) or not name:
            raise ValueError("invalid diagnostic request path")
        request_path = (Path(path).parent / name).resolve()
        if request_path.stat().st_size >= 20_000_000:
            raise ValueError("diagnostic request too large")
        requests[slot] = canonical_request(json.loads(request_path.read_text(encoding="utf-8")))
    first, second = (requests[slot] for slot in SLOTS)
    if (first["messages"][1] != second["messages"][1]
            or first["max_completion_tokens"] != second["max_completion_tokens"]
            or first.get("reasoning") != second.get("reasoning")
            or first.get("tool_choice", "none") != second.get("tool_choice", "none")
            or first.get("plugins", []) != second.get("plugins", [])
            or first.get("modalities", ["text"]) != second.get("modalities", ["text"])):
        raise ValueError("diagnostic requests use different source or decode settings")
    if first["max_completion_tokens"] > 5000:
        raise ValueError("diagnostic output cap exceeds frozen trial")
    return plan, requests


def submit_pair(*, plan_path: Path, private_root: Path, output_dir: Path,
                client_factory=BatchClient) -> dict:
    """Preflight and reserve both comparisons before either paid dispatch."""
    private_root = Path(private_root)
    if not (private_root / "luna.sqlite3").is_file() or not (private_root / "credentials.sqlite3").is_file():
        raise ValueError("existing application ledger and credential store required")
    plan, requests = _load_plan(plan_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(output_dir, 0o700)
    run_dir = output_dir / plan["run_id"]
    run_dir.mkdir(mode=0o700, exist_ok=True)
    os.chmod(run_dir, 0o700)

    def blocked(reason: str, reserves: dict | None = None) -> dict:
        outcome = {"status": "blocked", "run_id": plan["run_id"], "reason": reason}
        if reserves is not None:
            outcome["reserve_microusd"] = reserves
        write_private_json(run_dir / "preflight_status.json", outcome)
        return outcome

    ledger = Ledger(private_root)
    try:
        store = CredentialStore(private_root / "credentials.sqlite3")
        with credential_dispatch_guard(store.path):
            candidates = store.dispatch_candidates(role="judge")
            if not candidates:
                return blocked("judge_credential_unavailable")
            selected = candidates[0]
            token = store.reveal_for_dispatch(selected["id"], selected["version"], role="judge")
            routes = {slot: verify_batch_route(
                client_factory(token), requests[slot],
                max_output_tokens=requests[slot]["max_completion_tokens"])
                for slot in SLOTS}
            if any(route.workspace_id != selected.get("workspace_id") for route in routes.values()):
                return blocked("judge_workspace_changed")
            reserves = {slot: routes[slot].reserve_microusd() for slot in SLOTS}
            if sum(reserves.values()) > DIAGNOSTIC_GROUP_CAP_MICROUSD:
                return blocked("diagnostic_group_budget_exceeded", reserves)
            remaining = [route.key_limit_remaining_usd for route in routes.values()
                         if route.key_limit_remaining_usd is not None]
            if remaining and sum(reserves.values()) > int(min(remaining) * 1_000_000):
                return blocked("judge_key_budget_insufficient", reserves)
            input_sha = _sha(requests[SLOTS[0]]["messages"][1]["content"].encode("utf-8"))
            identities = {slot: _sha(_json_bytes({
                "version": VERSION, "run_id": plan["run_id"], "slot": slot,
                "source_sha256": plan["source_sha256"],
                "draft_sha256": plan["draft_sha256"],
                "input_sha256": input_sha, "request": requests[slot],
                "workspace_id": routes[slot].workspace_id,
            })) for slot in SLOTS}
            run_manifest = {
                "version": VERSION, "run_id": plan["run_id"],
                "source_sha256": plan["source_sha256"],
                "draft_sha256": plan["draft_sha256"],
                "input_sha256": input_sha, "cases": plan["cases"],
                "model": MODEL, "provider_only": [PROVIDER],
                "privacy_mode": PRIVACY_MODE, "publication": "forbidden",
                "credential_id": selected["id"], "credential_version": selected["version"],
                "workspace_id": routes[SLOTS[0]].workspace_id,
                "slots": {slot: {"semantic_key": identities[slot],
                                  "request_sha256": _sha(_json_bytes(requests[slot])),
                                  "prompt_sha256": _sha(requests[slot]["messages"][0]["content"].encode("utf-8")),
                                  "schema_sha256": _sha(_json_bytes(requests[slot]["response_format"]["json_schema"]["schema"])),
                                  "reserve_microusd": reserves[slot]}
                          for slot in SLOTS},
            }
            run_manifest_sha = _sha(_json_bytes(run_manifest))
            _pin_json(run_dir / "run_manifest.json", run_manifest)
            stages = {}
            for slot in SLOTS:
                kind = "diagnostic_" + slot
                decision = ledger.reserve_diagnostic(
                    run_id=plan["run_id"], kind=kind, semantic_key=identities[slot],
                    source_sha256=plan["source_sha256"], output_dir=run_dir,
                    credential_id=selected["id"], credential_version=selected["version"],
                    workspace_id=routes[slot].workspace_id,
                    max_cost_microusd=reserves[slot])
                if decision.kind == "blocked":
                    for prior in stages.values():
                        if prior["status"] == "reserved":
                            ledger.cancel_before_submit(prior["id"], "diagnostic_pair_preflight_blocked")
                    return blocked(decision.reason, reserves)
                stages[slot] = ledger.get(decision.job_id)
                job = stages[slot]
                manifest = {
                    "job_id": job["id"], "kind": kind, "diagnostic_run_id": plan["run_id"],
                    "run_manifest_sha256": run_manifest_sha,
                    "semantic_key": identities[slot], "source_sha256": plan["source_sha256"],
                    "draft_sha256": plan["draft_sha256"], "input_sha256": input_sha,
                    "request_sha256": _sha(_json_bytes(requests[slot])),
                    "prompt_sha256": _sha(requests[slot]["messages"][0]["content"].encode("utf-8")),
                    "schema_sha256": _sha(_json_bytes(requests[slot]["response_format"]["json_schema"]["schema"])),
                    "runner_sha256": _sha(Path(__file__).read_bytes()),
                    "custom_id": job["custom_id"], "inline_request_count": 1,
                    "credential_id": job["credential_id"],
                    "credential_version": job["credential_version"],
                    "workspace_id": job["workspace_id"], "model": MODEL,
                    "provider": "openrouter_gemini", "provider_only": [PROVIDER],
                    "privacy_mode": PRIVACY_MODE,
                    "reserve_microusd": job["reserved_microusd"],
                    "max_output_tokens": requests[slot]["max_completion_tokens"],
                    "publication": "forbidden",
                }
                _pin_json(Path(job["artifact_dir"]) / "manifest.json", manifest)
                _pin_json(Path(job["artifact_dir"]) / "request.json", requests[slot])
            results = []
            for slot in SLOTS:
                job = stages[slot]
                if job["status"] != "reserved":
                    results.append({"slot": slot, "status": job["status"], "job_id": job["id"]})
                    continue
                try:
                    token = store.reveal_for_dispatch(job["credential_id"],
                                                      job["credential_version"], role="judge")
                except CredentialError:
                    ledger.cancel_before_submit(job["id"], "judge_credential_changed_before_post")
                    results.append({"slot": slot, "status": "cancelled_before_submit",
                                    "job_id": job["id"]})
                    continue
                if not ledger.mark_submitting(job["id"]):
                    results.append({"slot": slot, "status": "submitting", "job_id": job["id"]})
                    continue
                try:
                    reply = client_factory(token).submit(job["custom_id"], requests[slot])
                    write_private_json(Path(job["artifact_dir"]) / "submit_response.json", reply.body)
                    try:
                        remote_id = batch_id_from_submit(reply.body)
                    except ValueError:
                        remote_id = None
                    if reply.status_code != 202 or remote_id is None:
                        ledger.submission_result(job["id"], remote_id=None,
                                                 error_code="unexpected_submit_response")
                        results.append({"slot": slot, "status": "submission_unknown",
                                        "job_id": job["id"]})
                    else:
                        ledger.submission_result(job["id"], remote_id=remote_id)
                        results.append({"slot": slot, "status": "submitted",
                                        "job_id": job["id"], "remote_id": remote_id})
                except BatchError as exc:
                    definite = exc.code in {400, 401, 402, 403, 404, 422, 429}
                    ledger.submission_result(job["id"], remote_id=None,
                                             error_code=exc.reason,
                                             definite_rejection=definite)
                    write_private_json(Path(job["artifact_dir"]) / "submit_error.json", {
                        "code": exc.code, "reason": exc.reason,
                        "retry_after": exc.retry_after, "definite_rejection": definite,
                    })
                    results.append({"slot": slot,
                                    "status": "rejected_before_submit" if definite else "submission_unknown",
                                    "job_id": job["id"], "reason": exc.reason})
            outcome = {"status": "registered", "run_id": plan["run_id"], "slots": results,
                       "reserve_microusd": reserves}
            write_private_json(run_dir / "submit_status.json", outcome)
            return outcome
    except (CredentialError, RouteBlocked, BatchError) as exc:
        return blocked(str(exc)[:120])
    finally:
        ledger.close()


def _validate_schema(value: object, schema: dict, path: str = "response") -> None:
    """Check the bounded JSON Schema subset used by the two frozen contracts."""
    kinds = schema.get("type")
    kinds = [kinds] if isinstance(kinds, str) else kinds
    if not isinstance(kinds, list) or not kinds:
        raise ValueError("diagnostic response schema unsupported")
    matches = {
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: type(v) is int,
        "number": lambda v: isinstance(v, (int, float)) and type(v) is not bool,
        "boolean": lambda v: type(v) is bool,
        "null": lambda v: v is None,
    }
    if any(kind not in matches for kind in kinds) or not any(matches[kind](value) for kind in kinds):
        raise ValueError(f"{path}: diagnostic response type mismatch")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: diagnostic response enum mismatch")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if (not isinstance(properties, dict) or not isinstance(required, list)
                or any(key not in value for key in required)
                or (schema.get("additionalProperties") is False
                    and any(key not in properties for key in value))):
            raise ValueError(f"{path}: diagnostic response object fields mismatch")
        for key, child in value.items():
            if key in properties:
                _validate_schema(child, properties[key], f"{path}.{key}")
    if isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, child in enumerate(value):
                _validate_schema(child, item_schema, f"{path}[{index}]")


def _source_references(value: object, visible: set[str], path: str = "response") -> None:
    """Reject any source coordinate invented outside the frozen excerpts."""
    if isinstance(value, dict):
        for key, child in value.items():
            where = f"{path}.{key}"
            if key == "source_ids":
                if (not isinstance(child, list)
                        or any(not isinstance(source_id, str) or source_id not in visible
                               for source_id in child)):
                    raise ValueError(f"{where}: source outside diagnostic excerpts")
            elif key in {"start_id", "end_id"}:
                if child is not None and (not isinstance(child, str) or child not in visible):
                    raise ValueError(f"{where}: source outside diagnostic excerpts")
            elif key == "field_sources":
                if not isinstance(child, dict):
                    raise ValueError(f"{where}: invalid field sources")
                for field, ids in child.items():
                    if (not isinstance(ids, list)
                            or any(not isinstance(source_id, str) or source_id not in visible
                                   for source_id in ids)):
                        raise ValueError(f"{where}.{field}: source outside diagnostic excerpts")
            else:
                _source_references(child, visible, where)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _source_references(child, visible, f"{path}[{index}]")


def _validate_diagnostic_report(report: dict, request: dict, run_manifest: dict,
                                kind: str) -> None:
    payload = json.loads(request["messages"][1]["content"])
    if not isinstance(payload, dict):
        raise ValueError("diagnostic input payload invalid")
    contract = request["response_format"]["json_schema"]
    expected_version = ("gemini_inventory_reconcile_v2" if kind == "diagnostic_baseline"
                        else "gemini_diagnostic_compact_v1")
    if (contract["name"] != expected_version
            or report.get("schema_version") != expected_version):
        raise ValueError("diagnostic response version mismatch")
    _validate_schema(report, contract["schema"])
    inventory = payload.get("INDEPENDENT_SOURCE_INVENTORY")
    source = payload.get("SOURCE_EXCERPTS")
    draft = payload.get("DRAFT_DOCUMENT")
    request_units = payload.get("DRAFT_UNITS")
    windows = payload.get("SOURCE_WINDOWS_TO_RECHECK")
    if (not isinstance(inventory, dict) or not isinstance(inventory.get("items"), list)
            or not isinstance(source, list) or not isinstance(draft, dict)
            or not isinstance(request_units, list) or not isinstance(windows, list)):
        raise ValueError("diagnostic input scope invalid")
    expected_items = run_manifest.get("cases")
    input_items = [item.get("item_id") if isinstance(item, dict) else None
                   for item in inventory["items"]]
    if (not isinstance(expected_items, list) or not expected_items
            or input_items != expected_items or len(set(input_items)) != len(input_items)):
        raise ValueError("diagnostic frozen case identity mismatch")
    visible_ids = [row.get("id") if isinstance(row, dict) else None for row in source]
    if (not visible_ids or any(not isinstance(source_id, str) for source_id in visible_ids)
            or len(set(visible_ids)) != len(visible_ids)):
        raise ValueError("diagnostic source excerpt identity invalid")
    visible = set(visible_ids)
    for item in inventory["items"]:
        cited = item.get("source_ids") if isinstance(item, dict) else None
        if (not isinstance(cited, list) or not cited
                or any(not isinstance(source_id, str) or source_id not in visible
                       for source_id in cited)):
            raise ValueError("diagnostic inventory cites outside frozen excerpts")
    expected_units = [row.get("unit_id") if isinstance(row, dict) else None
                      for row in request_units]
    document_units = {row["unit_id"]: row for row in draft_units(draft)}
    if (any(unit_id not in document_units for unit_id in expected_units)
            or any(row != document_units.get(row.get("unit_id"))
                   for row in request_units if isinstance(row, dict))
            or len(set(expected_units)) != len(expected_units)):
        raise ValueError("diagnostic draft unit identity invalid")
    reference_ids = payload.get("DRAFT_REFERENCE_UNIT_IDS", expected_units)
    if (not isinstance(reference_ids, list)
            or any(unit_id not in document_units for unit_id in reference_ids)):
        raise ValueError("diagnostic draft reference identity invalid")
    allowed_units = set(expected_units) | set(reference_ids)
    if kind == "diagnostic_baseline":
        received_items = [row["item_id"] for row in report["inventory_assessments"]]
        received_units = [row["unit_id"] for row in report["draft_assessments"]]
        expected_windows = [row.get("window_id") if isinstance(row, dict) else None
                            for row in windows]
        received_windows = [row["window_id"] for row in report["source_window_assessments"]]
        if (received_items != expected_items or received_units != expected_units
                or received_windows != expected_windows):
            raise ValueError("diagnostic baseline assessment coverage mismatch")
        assessment_rows = report["inventory_assessments"]
        for patch in report["patches"]:
            if patch["item_json"] is not None:
                _source_references(json.loads(patch["item_json"]), visible, "response.patch")
    else:
        assessment_rows = report["item_assessments"]
        if [row["item_id"] for row in assessment_rows] != expected_items:
            raise ValueError("diagnostic compact assessment coverage mismatch")
        for row in assessment_rows:
            if not row["source_ids"]:
                raise ValueError("diagnostic compact item has no source citation")
    _source_references(report, visible)
    for row in assessment_rows:
        evidence = row["draft_evidence"]
        status = row["status"] if kind == "diagnostic_baseline" else row["draft_status"]
        if status == "represented" and not evidence:
            raise ValueError("diagnostic represented item lacks draft evidence")
        for entry in evidence:
            unit = document_units.get(entry["unit_id"])
            quote = entry["quote"]
            if (unit is None or entry["unit_id"] not in allowed_units
                    or not quote or not any(quote in text for text in _unit_evidence_texts(draft, unit))):
                raise ValueError("diagnostic draft evidence is not an exact unit quote")


def finish_raw(ledger: Ledger, job: dict) -> dict:
    """Parse a completed diagnostic Batch without applying or publishing it."""
    if job["kind"] not in DIAGNOSTIC_KINDS:
        raise ValueError("not a diagnostic job")
    artifacts = Path(job["artifact_dir"])
    manifest = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
    request = json.loads((artifacts / "request.json").read_text(encoding="utf-8"))
    run_manifest_path = Path(job["output_dir"]) / "run_manifest.json"
    if run_manifest_path.is_symlink():
        raise ValueError("diagnostic run manifest symlink")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    submission = json.loads((artifacts / "submit_response.json").read_text(encoding="utf-8"))
    batch = json.loads((artifacts / "batch_terminal.json").read_text(encoding="utf-8"))
    if (manifest.get("kind") != job["kind"]
            or manifest.get("publication") != "forbidden"
            or run_manifest.get("publication") != "forbidden"
            or run_manifest.get("run_id") != manifest.get("diagnostic_run_id")
            or manifest.get("run_manifest_sha256") != _sha(_json_bytes(run_manifest))
            or run_manifest.get("source_sha256") != job["source_sha256"]
            or manifest.get("draft_sha256") != run_manifest.get("draft_sha256")
            or manifest.get("input_sha256") != _sha(request["messages"][1]["content"].encode("utf-8"))
            or manifest.get("semantic_key") != job["semantic_key"]
            or manifest.get("source_sha256") != job["source_sha256"]
            or manifest.get("prompt_sha256") != _sha(request["messages"][0]["content"].encode("utf-8"))
            or manifest.get("schema_sha256") != _sha(_json_bytes(request["response_format"]["json_schema"]["schema"]))
            or manifest.get("credential_id") != job["credential_id"]
            or manifest.get("credential_version") != job["credential_version"]
            or manifest.get("workspace_id") != job["workspace_id"]
            or manifest.get("reserve_microusd") != job["reserved_microusd"]
            or manifest.get("model") != MODEL
            or manifest.get("provider_only") != [PROVIDER]
            or manifest.get("privacy_mode") != PRIVACY_MODE
            or submission.get("id") != job["remote_id"]
            or batch.get("id") != job["remote_id"]
            or submission.get("model") not in BATCH_MODEL_IDS
            or batch.get("model") != submission.get("model")):
        raise ValueError("diagnostic batch identity mismatch")
    body, usage = extract_one_completed(
        batch, job["custom_id"], expected_batch_id=job["remote_id"],
        manifest=manifest, saved_request=request)
    if body.get("model") not in BATCH_MODEL_IDS | {MODEL.removesuffix(":batch")}:
        raise ValueError("diagnostic response model mismatch")
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("diagnostic response choices invalid")
    choice = choices[0]
    message = choice.get("message") or {}
    raw_text = message.get("content")
    if not isinstance(raw_text, str):
        raise ValueError("diagnostic response content invalid")
    write_private_json(artifacts / "native_response.json", {
        "text": raw_text, "usage": usage, "finish_reason": choice.get("finish_reason"),
        "refusal": bool(message.get("refusal")),
    })
    if choice.get("finish_reason") != "stop" or message.get("refusal"):
        raise ValueError("diagnostic response incomplete or refused")
    report = json.loads(raw_text)
    if not isinstance(report, dict):
        raise ValueError("diagnostic response not object")
    _validate_diagnostic_report(report, request, run_manifest, job["kind"])
    report_path = artifacts / "diagnostic_report.json"
    _pin_json(report_path, report)
    ledger.diagnostic_completed(job["id"], report_path)
    return {"status": "diagnostic_complete", "kind": job["kind"],
            "job_id": job["id"], "report_path": str(report_path)}


def inspect_run(*, private_root: Path, output_dir: Path, run_id: str) -> dict:
    """Read saved state without GET, POST, or private response content."""
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise ValueError("invalid diagnostic run id")
    run_dir = Path(output_dir) / run_id
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("diagnostic run manifest unavailable")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("run_id") != run_id or manifest.get("publication") != "forbidden":
        raise ValueError("diagnostic run identity mismatch")
    ledger = Ledger(private_root)
    try:
        rows = ledger.db.execute("SELECT * FROM jobs WHERE billing_group_id=? ORDER BY kind",
                                 ("diagnostic-" + run_id,)).fetchall()
        stages = []
        for row in rows:
            if row["kind"] not in DIAGNOSTIC_KINDS or row["output_dir"] != str(run_dir):
                raise ValueError("diagnostic ledger identity mismatch")
            stages.append({"kind": row["kind"], "job_id": row["id"],
                           "status": row["status"], "remote_id": row["remote_id"],
                           "dispatches": row["dispatches"],
                           "reserved_microusd": row["reserved_microusd"],
                           "billed_microusd": row["billed_microusd"],
                           "error_code": row["error_code"],
                           "artifact_dir": row["artifact_dir"]})
        return {"run_id": run_id, "version": manifest["version"],
                "source_sha256": manifest["source_sha256"],
                "draft_sha256": manifest["draft_sha256"],
                "publication": "forbidden", "stages": stages}
    finally:
        ledger.close()
