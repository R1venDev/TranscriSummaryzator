"""One-item Claude Opus 5.5 judge through OpenRouter's Chat Batch API.

Caller records intent and reserves budget before POST. An unknown POST is not
retried here. OpenRouter retains Batch inputs and results until deletion or its
retention limit. No direct Google API key is used.

Contract: https://openrouter.ai/docs/batch-quickstart
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

API_BASE = "https://openrouter.ai/api/v1"
MODEL = "anthropic/claude-opus-5.5:batch"
# OpenRouter may report the alias, its ordinary Chat slug, or the dated
# revision of the same Opus 5.5 release in terminal metadata.
RESOLVED_MODEL = "anthropic/claude-opus-5.5-20260921"
BATCH_MODEL_IDS = frozenset({MODEL, MODEL.removesuffix(":batch"),
                             RESOLVED_MODEL, RESOLVED_MODEL + ":batch"})
PROVIDER = "anthropic"
# Workspace input/output logs are separate from the Batch artifact lifetime.
WORKSPACE_IO_LOGGING_ENABLED = True
PRIVACY_MODE = (
    "openrouter_batch_30d_anthropic_"
    "workspace_input_output_logging_on_min_3mo_or_longer_user_authorized"
)
# Versioned with opus_summary_audit_v1. The cap includes adaptive thinking
# plus the JSON report; medium effort can allocate about half to thinking.
OUTPUT_CAP_AUDIT = 10_000
OUTPUT_CAP_VERIFY = 6_000
REASONING_EFFORT = "medium"
MAX_REQUEST_BYTES = 20_000_000
MAX_RESPONSE_BYTES = 32_000_000
_BATCH_ID = re.compile(r"batch[-_][A-Za-z0-9_-]{3,128}\Z")
_CUSTOM_ID = re.compile(r"[A-Za-z0-9_.-]{1,100}\Z")
_STATES = frozenset({"validating", "in_progress", "finalizing", "cancelling",
                     "completed", "failed", "expired", "cancelled"})


def valid_batch_id(value: object) -> bool:
    return isinstance(value, str) and _BATCH_ID.fullmatch(value) is not None


class BatchError(RuntimeError):
    """Redacted transport/provider error without source or token."""

    def __init__(self, code: int | None, reason: str, retry_after: str | None = None):
        super().__init__(f"OpenRouter Batch HTTP {code}: {reason}")
        self.code, self.reason, self.retry_after = code, reason, retry_after


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        raise BatchError(code, "unexpected_redirect")


@dataclass(frozen=True)
class Reply:
    status_code: int
    body: dict[str, Any]


def canonical_request(request_body: dict[str, Any]) -> dict[str, Any]:
    """Accept only a text-only Opus Chat judge with bounded JSON output."""
    if not isinstance(request_body, dict):
        raise ValueError("invalid_chat_request")
    allowed = {"messages", "response_format", "max_completion_tokens",
               "reasoning", "plugins", "modalities"}
    if set(request_body) - allowed:
        raise ValueError("unapproved_chat_parameter")
    messages = request_body.get("messages")
    if (not isinstance(messages, list) or len(messages) != 2
            or [m.get("role") if isinstance(m, dict) else None for m in messages]
            != ["system", "user"]):
        raise ValueError("invalid_messages")
    for message in messages:
        if (set(message) != {"role", "content"}
                or not isinstance(message["content"], str)
                or not message["content"].strip()):
            raise ValueError("text_only_required")
    cap = request_body.get("max_completion_tokens")
    if type(cap) is not int or not 1 <= cap <= 128_000:
        raise ValueError("invalid_output_cap")
    fmt = request_body.get("response_format")
    if (not isinstance(fmt, dict) or set(fmt) != {"type", "json_schema"}
            or fmt["type"] != "json_schema"):
        raise ValueError("structured_output_required")
    schema = fmt["json_schema"]
    if (not isinstance(schema, dict)
            or set(schema) != {"name", "strict", "schema"}
            or not isinstance(schema["name"], str) or not schema["name"]
            or schema["strict"] is not True
            or not isinstance(schema["schema"], dict)):
        raise ValueError("invalid_output_schema")
    if (not isinstance(request_body.get("reasoning"), dict)
            or set(request_body["reasoning"]) != {"effort"}
            or request_body["reasoning"]["effort"] not in {"low", "medium", "high", "xhigh", "max"}):
        raise ValueError("invalid_reasoning_profile")
    if request_body.get("plugins", []) != []:
        raise ValueError("plugins_not_allowed")
    if request_body.get("modalities", ["text"]) != ["text"]:
        raise ValueError("text_only_required")
    return dict(request_body)


class BatchClient:
    """Selected OpenRouter inference key; never expose it in a URL or log."""

    def __init__(self, token: str, *, opener=None):
        if (not isinstance(token, str) or not token or not token.isascii()
                or not token.isprintable() or any(ch.isspace() for ch in token)):
            raise ValueError("invalid credential")
        self._token = token
        self._opener = opener or urllib.request.build_opener(_NoRedirect())

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Reply:
        if not path.startswith("/") or ".." in path or "?" in path or "#" in path:
            raise ValueError("invalid Batch API path")
        try:
            data = (None if payload is None else
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        except (TypeError, ValueError):
            raise ValueError("invalid_request_body") from None
        if data is not None and len(data) >= MAX_REQUEST_BYTES:
            raise ValueError("batch_payload_too_large")
        request = urllib.request.Request(
            API_BASE + path, data=data,
            headers={"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"},
            method=method,
        )
        try:
            with self._opener.open(request, timeout=45) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise BatchError(response.status, "response_too_large")
                if not raw and response.status == 204:
                    return Reply(response.status, {})
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise BatchError(response.status, "response_not_object")
                return Reply(response.status, value)
        except urllib.error.HTTPError as exc:
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            exc.close()
            if retry_after is not None and not re.fullmatch(r"[0-9]{1,8}", retry_after):
                retry_after = None
            raise BatchError(exc.code, "request_rejected", retry_after) from None
        except (urllib.error.URLError, OSError):
            raise BatchError(None, "transport_unknown") from None
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise BatchError(None, "malformed_response") from None

    def current_key(self) -> Reply:
        return self._request("GET", "/key")

    def models_for_key(self) -> Reply:
        return self._request("GET", "/models/user")

    def model_details(self) -> Reply:
        return self._request("GET", "/model/anthropic/claude-opus-5.5:batch")

    def model_endpoints(self) -> Reply:
        # The endpoints API requires the variant colon percent-encoded.
        return self._request("GET", "/models/anthropic/claude-opus-5.5%3Abatch/endpoints")

    def submit(self, custom_id: str, request_body: dict[str, Any]) -> Reply:
        if not isinstance(custom_id, str) or _CUSTOM_ID.fullmatch(custom_id) is None:
            raise ValueError("invalid custom_id")
        request = canonical_request(request_body)
        # OpenRouter stream-parses the object: headers must precede requests.
        payload = {
            "endpoint": "/v1/chat/completions", "model": MODEL,
            "provider": {"only": [PROVIDER]}, "completion_window": "24h",
            "requests": [{"custom_id": custom_id, "body": request}],
        }
        return self._request("POST", "/batches", payload)

    def get(self, batch_id: str) -> Reply:
        if not valid_batch_id(batch_id):
            raise ValueError("invalid batch id")
        reply = self._request("GET", f"/batches/{batch_id}")
        if reply.body.get("id") != batch_id:
            raise BatchError(reply.status_code, "batch_identity_mismatch")
        return reply

    def delete(self, batch_id: str) -> Reply:
        if not valid_batch_id(batch_id):
            raise ValueError("invalid batch id")
        return self._request("DELETE", f"/batches/{batch_id}")


def batch_id_from_submit(body: dict[str, Any]) -> str:
    batch_id = body.get("id") if isinstance(body, dict) else None
    if not valid_batch_id(batch_id):
        raise ValueError("batch_id_missing")
    return batch_id


def normalize_batch_status(body: dict[str, Any]) -> str:
    if not isinstance(body, dict):
        return "unknown"
    status = body.get("status")
    return status if status in _STATES else "unknown"


def extract_one_completed(
    batch: dict[str, Any], custom_id: str, *,
    expected_batch_id: str | None = None,
    manifest: dict[str, Any] | None = None,
    saved_request: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the exact chat response and actual Batch usage for one request."""
    if normalize_batch_status(batch) != "completed":
        raise ValueError("batch_not_completed")
    if expected_batch_id is not None and batch.get("id") != expected_batch_id:
        raise ValueError("batch_identity_mismatch")
    if batch.get("model") not in BATCH_MODEL_IDS or batch.get("endpoint") != "/v1/chat/completions":
        raise ValueError("batch_route_mismatch")
    if manifest is not None or saved_request is not None:
        if not isinstance(manifest, dict) or not isinstance(saved_request, dict):
            raise ValueError("saved_request_mismatch")
        digest = hashlib.sha256(json.dumps(
            saved_request, ensure_ascii=False, sort_keys=True,
            separators=(",", ":")).encode("utf-8")).hexdigest()
        if (manifest.get("inline_request_count") != 1
                or manifest.get("custom_id") != custom_id
                or manifest.get("request_sha256") != digest):
            raise ValueError("saved_request_mismatch")
    results = batch.get("results")
    if not isinstance(results, list):
        raise ValueError("batch_results_missing")
    matches = [item for item in results if isinstance(item, dict)
               and item.get("custom_id") == custom_id]
    if len(matches) != 1 or len(results) != 1:
        raise ValueError("batch_custom_id_mismatch")
    item = matches[0]
    if item.get("error") is not None:
        raise ValueError("batch_item_failed")
    wrapped = item.get("response")
    if (not isinstance(wrapped, dict) or wrapped.get("status_code") != 200
            or not isinstance(wrapped.get("body"), dict)):
        raise ValueError("batch_item_invalid")
    usage = batch.get("usage")
    return wrapped["body"], usage if isinstance(usage, dict) else {}
