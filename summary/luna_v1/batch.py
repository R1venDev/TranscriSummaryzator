"""Small OpenRouter Batch client for the summary-only route.

The selected route is explicitly provider-ZDR-off and stores Batch input and
results at OpenRouter for up to 30 days. No transcript is sent by this module
until a caller has reserved budget and selected a verified credential.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


API_BASE = "https://openrouter.ai/api/v1"
MODEL = "openai/gpt-6-luna:batch"
PROVIDER = "openai"
TERMINAL = frozenset({"completed", "failed", "expired", "cancelled"})
_BATCH_ID = re.compile(r"batch[-_][A-Za-z0-9_-]{3,128}\Z")


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

    def _request(self, method: str, path: str, payload: dict | None = None) -> Reply:
        if not path.startswith("/") or ".." in path or "?" in path or "#" in path:
            raise ValueError("invalid Batch API path")
        body = None if payload is None else json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            API_BASE + path,
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
        except urllib.error.URLError as exc:
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

    def current_key(self) -> Reply:
        return self._request("GET", "/key")

    def models_for_key(self) -> Reply:
        return self._request("GET", "/models/user")

    def model_details(self) -> Reply:
        return self._request("GET", "/model/openai/gpt-6-luna:batch")

    def get(self, batch_id: str) -> Reply:
        if not valid_batch_id(batch_id):
            raise ValueError("invalid batch id")
        return self._request("GET", f"/batches/{batch_id}")

    def delete(self, batch_id: str) -> Reply:
        if not valid_batch_id(batch_id):
            raise ValueError("invalid batch id")
        return self._request("DELETE", f"/batches/{batch_id}")


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
