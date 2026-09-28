"""Small OpenRouter Batch client for the summary-only route.

The selected route is explicitly provider-ZDR-off and stores Batch input and
results at OpenRouter for up to 30 days. No transcript is sent by this module
until a caller has reserved budget and selected a verified credential.
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Sequence


API_BASE = "https://openrouter.ai/api/v1"
MODEL = "openai/gpt-6-luna:batch"
# The public Batch API selects the eligible :batch endpoint for this base
# slug. Keep MODEL for the pinned one-item policy that already uses it.
SUBMIT_MODEL = "openai/gpt-6-luna"
PROVIDER = "openai"
TERMINAL = frozenset({"completed", "failed", "expired", "cancelled"})
_BATCH_ID = re.compile(r"batch[-_][A-Za-z0-9_-]{3,128}\Z")
_CUSTOM_ID = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
_LIST_STATUSES = frozenset({"validating", "in_progress", "completed", "failed", "expired", "cancelled"})
_REASONING_EFFORTS = frozenset({"medium", "high"})


def valid_batch_id(value: object) -> bool:
    # Current OpenRouter responses use batch-..., while examples in the
    # official quickstart use batch_.... Both are opaque remote identifiers.
    return isinstance(value, str) and _BATCH_ID.fullmatch(value) is not None


class BatchError(RuntimeError):
    def __init__(self, code: int | None, reason: str, retry_after: str | None = None):
        super().__init__(f"OpenRouter Batch HTTP {code}: {reason}")
        self.code = code
        self.reason = reason
        self.retry_after = retry_after


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        raise BatchError(code, "unexpected_redirect")


@dataclass(frozen=True)
class Reply:
    status_code: int
    body: dict[str, Any]


class BatchClient:
    """Use one selected backend key; callers save bodies in a private ledger."""

    def __init__(self, token: str, *, opener=None):
        if not token or "\n" in token or "\r" in token:
            raise ValueError("invalid credential")
        self._token = token
        self._opener = opener or urllib.request.build_opener(_NoRedirect())

    def _request(self, method: str, path: str, payload: dict | None = None, *,
                 query: Sequence[tuple[str, str]] = (),
                 prepared_body: bytes | None = None) -> Reply:
        if not path.startswith("/") or ".." in path or "?" in path or "#" in path:
            raise ValueError("invalid Batch API path")
        if payload is not None and prepared_body is not None:
            raise ValueError("ambiguous_request_body")
        body = (prepared_body if prepared_body is not None else None if payload is None else
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        url = API_BASE + path
        if query:
            url += "?" + urllib.parse.urlencode(query)
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"},
            method=method,
        )
        try:
            with self._opener.open(request, timeout=45) as response:
                raw = response.read(16 * 1024 * 1024 + 1)
                if len(raw) > 16 * 1024 * 1024:
                    raise BatchError(response.status, "response_too_large")
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise BatchError(response.status, "response_not_object")
                return Reply(response.status, value)
        except urllib.error.HTTPError as exc:
            # Provider errors may echo input or headers. Keep only a fixed code
            # and Retry-After; private raw is never printed to logs.
            raise BatchError(exc.code, "request_rejected", exc.headers.get("Retry-After")) from None
        except (urllib.error.URLError, OSError):
            raise BatchError(None, "transport_unknown") from None
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise BatchError(None, "malformed_response") from None

    def submit(self, custom_id: str, request_body: dict[str, Any]) -> Reply:
        if not custom_id or len(custom_id) > 128:
            raise ValueError("invalid custom_id")
        # Insertion order is required by OpenRouter's streaming Batch parser.
        payload = {
            "endpoint": "/v1/chat/completions",
            "model": MODEL,
            "provider": {"only": [PROVIDER]},
            "completion_window": "24h",
            "requests": [{"custom_id": custom_id, "body": request_body}],
        }
        return self._request("POST", "/batches", payload)

    def submit_many(self, items: list[tuple[str, dict[str, Any]]]) -> Reply:
        """Submit independent Chat items in one Batch, never through sync Chat.

        This method deliberately accepts no provider/model override. The
        account and endpoint policy must be verified by the caller before
        sending private inputs. No hidden retry is performed after POST.
        """
        payload = build_batch_payload(items)
        return self._request("POST", "/batches", payload)

    def submit_prepared(self, payload_bytes: bytes, *, expected_sha256: str) -> Reply:
        """POST the exact, previously reserved envelope bytes once.

        Canonical parsing alone is insufficient: OpenRouter's Batch parser
        requires requests after endpoint/model/provider/window.  The caller
        seals this byte string and records its digest before dispatch.
        """
        if (not isinstance(payload_bytes, bytes) or len(payload_bytes) > 16 * 1024 * 1024
                or hashlib.sha256(payload_bytes).hexdigest() != expected_sha256):
            raise ValueError("prepared_batch_payload_changed")
        try:
            value = json.loads(payload_bytes)
            requests = value["requests"]
            items = [(item["custom_id"], item["body"]) for item in requests]
        except (TypeError, KeyError, ValueError, UnicodeDecodeError):
            raise ValueError("prepared_batch_payload_invalid") from None
        expected = build_batch_payload(items)
        encoded = json.dumps(expected, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if value != expected or payload_bytes != encoded:
            raise ValueError("prepared_batch_order_or_shape_invalid")
        return self._request("POST", "/batches", prepared_body=payload_bytes)

    def list_batches(self, *, limit: int = 100, after: str | None = None,
                     created_after: int | str | None = None,
                     created_before: int | str | None = None,
                     statuses: Sequence[str] = ()) -> Reply:
        """List workspace metadata to investigate an unknown create outcome.

        List responses omit item results/custom_ids. A timestamp/model match
        alone must never authorize a replacement POST or claim identity.
        """
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("invalid_batch_list_limit")
        query: list[tuple[str, str]] = [("limit", str(limit))]
        if after is not None:
            if not valid_batch_id(after):
                raise ValueError("invalid_batch_list_cursor")
            query.append(("after", after))
        for name, value in (("created_after", created_after),
                            ("created_before", created_before)):
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, (int, str)):
                    raise ValueError("invalid_batch_list_time")
                rendered = str(value)
                if not rendered or len(rendered) > 64:
                    raise ValueError("invalid_batch_list_time")
                query.append((name, rendered))
        for status in statuses:
            if status not in _LIST_STATUSES:
                raise ValueError("invalid_batch_list_status")
            query.append(("status", status))
        return self._request("GET", "/batches", query=query)

    def current_key(self) -> Reply:
        return self._request("GET", "/key")

    def models_for_key(self) -> Reply:
        return self._request("GET", "/models/user")

    def model_details(self) -> Reply:
        return self._request("GET", "/model/openai/gpt-6-luna:batch")

    def model_endpoints(self) -> Reply:
        # The model's variant colon is percent-encoded in the documented
        # /models/{author}/{slug}/endpoints route.
        return self._request("GET", "/models/openai/gpt-6-luna%3Abatch/endpoints")

    def get(self, batch_id: str) -> Reply:
        if not valid_batch_id(batch_id):
            raise ValueError("invalid batch id")
        return self._request("GET", f"/batches/{batch_id}")

    def delete(self, batch_id: str) -> Reply:
        if not valid_batch_id(batch_id):
            raise ValueError("invalid batch id")
        return self._request("DELETE", f"/batches/{batch_id}")


def _canonical_chat_item(body: dict[str, Any]) -> dict[str, Any]:
    """Reject unsupported controls before the gateway can silently drop them."""
    if not isinstance(body, dict):
        raise ValueError("invalid_chat_body")
    allowed = {"messages", "response_format", "max_completion_tokens", "max_tokens",
               "reasoning", "reasoning_effort", "prompt_cache_options"}
    if set(body) - allowed:
        raise ValueError("unapproved_chat_parameter")
    messages = body.get("messages")
    if (not isinstance(messages, list) or len(messages) != 2
            or [m.get("role") if isinstance(m, dict) else None for m in messages]
            != ["developer", "user"]):
        raise ValueError("invalid_messages")
    for message in messages:
        if (set(message) != {"role", "content"}
                or not isinstance(message["content"], str)
                or not message["content"].strip()):
            raise ValueError("text_only_required")
    if ("max_completion_tokens" in body) == ("max_tokens" in body):
        raise ValueError("ambiguous_output_cap")
    cap = body.get("max_completion_tokens", body.get("max_tokens"))
    if type(cap) is not int or not 1 <= cap <= 128_000:
        raise ValueError("invalid_output_cap")
    fmt = body.get("response_format")
    if (not isinstance(fmt, dict) or set(fmt) != {"type", "json_schema"}
            or fmt.get("type") != "json_schema"):
        raise ValueError("structured_output_required")
    schema = fmt["json_schema"]
    if (not isinstance(schema, dict)
            or set(schema) != {"name", "strict", "schema"}
            or not isinstance(schema.get("name"), str) or not schema["name"]
            or schema.get("strict") is not True
            or not isinstance(schema.get("schema"), dict)):
        raise ValueError("invalid_output_schema")
    has_reasoning = "reasoning" in body
    has_shortcut = "reasoning_effort" in body
    if has_reasoning == has_shortcut:
        raise ValueError("invalid_reasoning_profile")
    if has_reasoning:
        reasoning = body["reasoning"]
        if (not isinstance(reasoning, dict) or set(reasoning) != {"effort"}
                or reasoning["effort"] not in _REASONING_EFFORTS):
            raise ValueError("invalid_reasoning_profile")
    elif body["reasoning_effort"] not in _REASONING_EFFORTS:
        raise ValueError("invalid_reasoning_profile")
    if "prompt_cache_options" in body:
        options = body["prompt_cache_options"]
        if not isinstance(options, dict) or options != {"mode": "explicit"}:
            raise ValueError("unapproved_cache_control")
    return dict(body)


def build_batch_payload(items: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """Create the ordered, single-provider Batch wire envelope."""
    if not isinstance(items, list) or not items:
        raise ValueError("batch_items_required")
    requests: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError("invalid_batch_item")
        custom_id, body = item
        if not isinstance(custom_id, str) or _CUSTOM_ID.fullmatch(custom_id) is None:
            raise ValueError("invalid_custom_id")
        if custom_id in seen:
            raise ValueError("duplicate_custom_id")
        seen.add(custom_id)
        requests.append({"custom_id": custom_id, "body": _canonical_chat_item(body)})
    # OpenRouter stream-parses this object; requests must come last.
    return {"endpoint": "/v1/chat/completions", "model": SUBMIT_MODEL,
            "provider": {"only": [PROVIDER]}, "completion_window": "24h",
            "requests": requests}


def _item_outcome(custom_id: str, raw: Any, *, status: str) -> dict[str, Any]:
    return {"custom_id": custom_id, "status": status, "raw": raw,
            "body": None, "usage": None, "request_id": None,
            "generation_id": None, "finish_reason": None, "error": None}


def _parse_item(custom_id: str, raw: dict[str, Any]) -> dict[str, Any]:
    outcome = _item_outcome(custom_id, raw, status="invalid")
    error = raw.get("error")
    if error is not None:
        outcome["status"] = "item_error"
        outcome["error"] = error
        return outcome
    response = raw.get("response")
    if not isinstance(response, dict):
        return outcome
    request_id = response.get("request_id")
    if isinstance(request_id, str):
        outcome["request_id"] = request_id
    if response.get("status_code") != 200:
        outcome["status"] = "http_error"
        outcome["error"] = response.get("body")
        return outcome
    body = response.get("body")
    if not isinstance(body, dict):
        return outcome
    outcome["body"] = body
    generation_id = body.get("id")
    if isinstance(generation_id, str):
        outcome["generation_id"] = generation_id
    usage = body.get("usage")
    if isinstance(usage, dict):
        outcome["usage"] = usage
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return outcome
    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    if isinstance(finish_reason, str):
        outcome["finish_reason"] = finish_reason
    message = choice.get("message")
    if not isinstance(message, dict):
        return outcome
    if message.get("refusal"):
        outcome["status"] = "refusal"
    elif finish_reason == "length":
        outcome["status"] = "length"
    elif finish_reason == "content_filter":
        outcome["status"] = "content_filter"
    elif finish_reason != "stop":
        outcome["status"] = "incomplete"
    elif (message.get("role") == "assistant"
          and isinstance(message.get("content"), str)
          and message["content"].strip()):
        outcome["status"] = "ok"
    return outcome


def parse_batch_items(batch: dict[str, Any], expected_ids: Sequence[str], *,
                      expected_batch_id: str | None = None) -> dict[str, Any]:
    """Preserve independent item outcomes; never infer success from envelope.

    `ok` means only that Chat transport returned one non-refused `stop`
    message. The caller still checks JSON Schema, source IDs and evidence.
    Batch usage is separate from item usage and must not be added to it.
    """
    if not isinstance(batch, dict):
        raise ValueError("invalid_batch_object")
    if expected_batch_id is not None and batch.get("id") != expected_batch_id:
        raise ValueError("batch_identity_mismatch")
    if isinstance(expected_ids, str):
        raise ValueError("invalid_expected_custom_ids")
    expected = tuple(expected_ids)
    if (not expected or any(not isinstance(value, str)
                            or _CUSTOM_ID.fullmatch(value) is None for value in expected)
            or len(set(expected)) != len(expected)):
        raise ValueError("invalid_expected_custom_ids")
    batch_status = batch.get("status")
    if batch_status not in {"validating", "in_progress", "finalizing", "cancelling", *TERMINAL}:
        batch_status = "unknown"
    usage = batch.get("usage")
    batch_usage = usage if isinstance(usage, dict) else None
    results = batch.get("results")
    if not isinstance(results, list):
        return {"batch_status": batch_status, "batch_usage": batch_usage,
                "batch_model": batch.get("model"), "batch_endpoint": batch.get("endpoint"),
                "results_available": False,
                "items": {custom_id: _item_outcome(custom_id, None, status="unavailable")
                          for custom_id in expected},
                "missing_ids": (), "unavailable_ids": expected,
                "extra_ids": (), "duplicate_ids": (), "invalid_result_count": 0}
    grouped: dict[str, list[dict[str, Any]]] = {}
    invalid_result_count = 0
    for raw in results:
        if (not isinstance(raw, dict) or not isinstance(raw.get("custom_id"), str)
                or _CUSTOM_ID.fullmatch(raw["custom_id"]) is None):
            invalid_result_count += 1
            continue
        grouped.setdefault(raw["custom_id"], []).append(raw)
    expected_set = set(expected)
    duplicates = tuple(sorted(custom_id for custom_id, matches in grouped.items()
                              if len(matches) > 1))
    extras = tuple(sorted(set(grouped) - expected_set))
    missing = tuple(custom_id for custom_id in expected if custom_id not in grouped)
    items: dict[str, dict[str, Any]] = {}
    for custom_id in expected:
        matches = grouped.get(custom_id, [])
        if not matches:
            items[custom_id] = _item_outcome(custom_id, None, status="missing")
        elif len(matches) > 1:
            items[custom_id] = _item_outcome(custom_id, matches, status="duplicate")
        else:
            items[custom_id] = _parse_item(custom_id, matches[0])
    return {"batch_status": batch_status, "batch_usage": batch_usage,
            "batch_model": batch.get("model"), "batch_endpoint": batch.get("endpoint"),
            "results_available": True, "items": items,
            "missing_ids": missing, "unavailable_ids": (),
            "extra_ids": extras, "duplicate_ids": duplicates,
            "invalid_result_count": invalid_result_count}


def extract_one_completed(batch: dict[str, Any], custom_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the exact response body and usage for our one logical request."""
    if batch.get("status") != "completed":
        raise ValueError("batch_not_completed")
    results = batch.get("results")
    if not isinstance(results, list):
        raise ValueError("batch_results_missing")
    matches = [item for item in results if isinstance(item, dict) and item.get("custom_id") == custom_id]
    if len(matches) != 1:
        raise ValueError("batch_custom_id_mismatch")
    item = matches[0]
    if item.get("error") is not None:
        raise ValueError("batch_item_failed")
    response = item.get("response")
    if not isinstance(response, dict) or response.get("status_code") != 200 or not isinstance(response.get("body"), dict):
        raise ValueError("batch_item_invalid")
    return response["body"], batch.get("usage") or {}
