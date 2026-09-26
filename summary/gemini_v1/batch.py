"""One-item Google Gemini GenerateContent Batch transport.

The caller owns durable intent/budget records.  In particular, a failed or
unknown POST is never retried here: batch creation is not idempotent.

Contract: https://ai.google.dev/api/batch-api and
https://ai.google.dev/gemini-api/docs/batch-api (v1beta, September 2026).
"""
from __future__ import annotations

import json
import hashlib
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


API_BASE = "https://generativelanguage.googleapis.com/v1beta"
MODEL = "gemini-3.7-flash"
MODEL_RESOURCE = f"models/{MODEL}"
MAX_INLINE_BYTES = 20_000_000  # Google's inline limit is under 20 MB.
MAX_RESPONSE_BYTES = 32_000_000
_BATCH_NAME = re.compile(r"batches/[A-Za-z0-9_-]{1,128}\Z")
_CUSTOM_ID = re.compile(r"[A-Za-z0-9_.-]{1,100}\Z")


def valid_batch_id(value: object) -> bool:
    return isinstance(value, str) and _BATCH_NAME.fullmatch(value) is not None


class BatchError(RuntimeError):
    """Redacted transport/provider error; never contains a request or API key."""

    def __init__(self, code: int | None, reason: str, retry_after: str | None = None):
        super().__init__(f"Gemini Batch HTTP {code}: {reason}")
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


def canonical_request(request_body: dict[str, Any]) -> dict[str, Any]:
    """Validate one text-only GenerateContentRequest and pin safe defaults.

    The same result is used in countTokens and Batch submission, so a caller
    cannot count one prompt and send a materially different one accidentally.
    """
    if not isinstance(request_body, dict) or not isinstance(request_body.get("contents"), list) or not request_body["contents"]:
        raise ValueError("invalid_generate_request")
    if request_body.get("model", MODEL_RESOURCE) != MODEL_RESOURCE:
        raise ValueError("wrong_gemini_model")
    if request_body.get("store", False) is not False:
        raise ValueError("request_storage_not_disabled")
    if "cachedContent" in request_body or "cached_content" in request_body:
        raise ValueError("explicit_cache_not_enabled")
    if "tools" in request_body or "toolConfig" in request_body:
        raise ValueError("tools_not_allowed")
    generation = request_body.get("generationConfig")
    if not isinstance(generation, dict):
        raise ValueError("generation_config_missing")
    output_cap = generation.get("maxOutputTokens")
    if type(output_cap) is not int or not 1 <= output_cap <= 65_536:
        raise ValueError("invalid_output_cap")
    # This route processes a compact text transcript only.  No file/media URI
    # and no tool can cause a second, unbudgeted external data transfer.
    for content in request_body["contents"]:
        if not isinstance(content, dict) or not isinstance(content.get("parts"), list) or not content["parts"]:
            raise ValueError("invalid_content")
        for part in content["parts"]:
            if not isinstance(part, dict) or set(part) != {"text"} or not isinstance(part["text"], str):
                raise ValueError("text_only_required")
    instruction = request_body.get("systemInstruction")
    if instruction is not None:
        if not isinstance(instruction, dict) or not isinstance(instruction.get("parts"), list) or not instruction["parts"]:
            raise ValueError("invalid_system_instruction")
        if any(not isinstance(part, dict) or set(part) != {"text"} or not isinstance(part["text"], str)
               for part in instruction["parts"]):
            raise ValueError("text_only_required")
    result = dict(request_body)
    result["model"] = MODEL_RESOURCE
    result["store"] = False  # Request-level override of project logging.
    return result


class BatchClient:
    """HTTP client using a Gemini API key supplied by the secret store."""

    def __init__(self, token: str, *, opener=None):
        if (not isinstance(token, str) or not token or not token.isascii()
                or not token.isprintable() or any(ch.isspace() for ch in token)):
            raise ValueError("invalid credential")
        self._token = token
        self._opener = opener or urllib.request.build_opener(_NoRedirect())

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Reply:
        if not path.startswith("/") or ".." in path or "?" in path or "#" in path:
            raise ValueError("invalid Gemini API path")
        try:
            data = None if payload is None else json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError):
            raise ValueError("invalid_request_body") from None
        if data is not None and len(data) >= MAX_INLINE_BYTES:
            raise ValueError("batch_inline_payload_too_large")
        request = urllib.request.Request(
            API_BASE + path,
            data=data,
            headers={"x-goog-api-key": self._token, "Content-Type": "application/json"},
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
        except urllib.error.URLError:
            raise BatchError(None, "transport_unknown") from None
        except OSError:
            raise BatchError(None, "transport_unknown") from None
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise BatchError(None, "malformed_response") from None

    def model_details(self) -> Reply:
        return self._request("GET", f"/{MODEL_RESOURCE}")

    def count_tokens(self, request_body: dict[str, Any]) -> Reply:
        request = canonical_request(request_body)
        # Count the same systemInstruction, contents and generationConfig that
        # Batch will receive, not just the user-text subset.
        return self._request(
            "POST", f"/{MODEL_RESOURCE}:countTokens", {"generateContentRequest": request}
        )

    def submit(self, custom_id: str, request_body: dict[str, Any]) -> Reply:
        if not isinstance(custom_id, str) or _CUSTOM_ID.fullmatch(custom_id) is None:
            raise ValueError("invalid custom_id")
        request = canonical_request(request_body)
        payload = {
            "batch": {
                "display_name": f"transcri-{custom_id}",
                "input_config": {
                    "requests": {
                        "requests": [{"request": request, "metadata": {"key": custom_id}}]
                    }
                },
            }
        }
        try:
            encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError):
            raise ValueError("invalid_request_body") from None
        if len(encoded) >= MAX_INLINE_BYTES:
            raise ValueError("batch_inline_payload_too_large")
        return self._request("POST", f"/{MODEL_RESOURCE}:batchGenerateContent", payload)

    def get(self, batch_id: str) -> Reply:
        if not valid_batch_id(batch_id):
            raise ValueError("invalid batch id")
        reply = self._request("GET", f"/{batch_id}")
        if reply.body.get("name") != batch_id:
            raise BatchError(reply.status_code, "batch_identity_mismatch")
        return reply

    def cancel(self, batch_id: str) -> Reply:
        if not valid_batch_id(batch_id):
            raise ValueError("invalid batch id")
        return self._request("POST", f"/{batch_id}:cancel")

    def delete(self, batch_id: str) -> Reply:
        """Forget job resource after terminal; API does not promise data erasure."""
        if not valid_batch_id(batch_id):
            raise ValueError("invalid batch id")
        return self._request("DELETE", f"/{batch_id}")


def batch_id_from_submit(body: dict[str, Any]) -> str:
    name = body.get("name") if isinstance(body, dict) else None
    if not valid_batch_id(name):
        raise ValueError("batch_name_missing")
    return name


_STATES = {
    "JOB_STATE_PENDING": "pending", "BATCH_STATE_PENDING": "pending",
    "JOB_STATE_RUNNING": "running", "BATCH_STATE_RUNNING": "running",
    "JOB_STATE_SUCCEEDED": "completed", "BATCH_STATE_SUCCEEDED": "completed",
    "JOB_STATE_FAILED": "failed", "BATCH_STATE_FAILED": "failed",
    "JOB_STATE_CANCELLED": "cancelled", "BATCH_STATE_CANCELLED": "cancelled",
    "JOB_STATE_EXPIRED": "expired", "BATCH_STATE_EXPIRED": "expired",
}


def normalize_batch_status(body: dict[str, Any]) -> str:
    """Read both documented REST Operation and BatchJob view state shapes."""
    if not isinstance(body, dict):
        return "unknown"
    if body.get("done") is False:
        # An Operation cannot contain its response until done=true. Its
        # metadata may already report a terminal batch state while the final
        # operation result is still being assembled; keep polling for it.
        return "running"
    if body.get("done") is True and isinstance(body.get("error"), dict):
        return "failed"
    if body.get("done") is True and _inlined_items(body) is not None:
        return "completed"
    for container in (body, body.get("metadata"), body.get("response")):
        if isinstance(container, dict) and container.get("state") in _STATES:
            return _STATES[container["state"]]
    return "unknown"


def _inlined_items(body: dict[str, Any]) -> list[Any] | None:
    # REST reference: Operation.response.output.inlinedResponses.inlinedResponses.
    response = body.get("response")
    if isinstance(response, dict):
        output = response.get("output")
        if isinstance(output, dict):
            inline = output.get("inlinedResponses")
            if isinstance(inline, list):
                return inline
            if isinstance(inline, dict) and isinstance(inline.get("inlinedResponses"), list):
                return inline["inlinedResponses"]
        inline = response.get("inlinedResponses")
        if isinstance(inline, list):
            return inline
        if isinstance(inline, dict) and isinstance(inline.get("inlinedResponses"), list):
            return inline["inlinedResponses"]
    # BatchJob view in the guide: dest.inlinedResponses is the list.
    dest = body.get("dest")
    if isinstance(dest, dict) and isinstance(dest.get("inlinedResponses"), list):
        return dest["inlinedResponses"]
    return None


def extract_one_completed(
    batch: dict[str, Any], custom_id: str, *,
    expected_batch_id: str | None = None,
    manifest: dict[str, Any] | None = None,
    saved_request: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return matched raw GenerateContentResponse and its usageMetadata."""
    if normalize_batch_status(batch) != "completed":
        raise ValueError("batch_not_completed")
    if isinstance(batch.get("error"), dict):
        raise ValueError("batch_failed")
    items = _inlined_items(batch)
    if items is None:
        raise ValueError("batch_inline_results_missing")
    matches = [item for item in items if isinstance(item, dict)
               and isinstance(item.get("metadata"), dict)
               and item["metadata"].get("key") == custom_id]
    if len(matches) == 1:
        item = matches[0]
    elif (len(matches) == 0 and len(items) == 1 and isinstance(items[0], dict)
          and items[0].get("metadata") is None
          and valid_batch_id(expected_batch_id)
          and batch.get("name") == expected_batch_id
          and isinstance(manifest, dict) and isinstance(saved_request, dict)
          and manifest.get("inline_request_count") == 1
          and manifest.get("custom_id") == custom_id
          and manifest.get("request_sha256") == hashlib.sha256(
              json.dumps(saved_request, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
          ).hexdigest()):
        # Google has returned an inline response without the request metadata.
        # One persisted request, one result and the exact remote identity make
        # positional recovery unambiguous; never use this for a multi-item job.
        item = items[0]
    else:
        raise ValueError("batch_custom_id_mismatch")
    if item.get("error") is not None:
        raise ValueError("batch_item_failed")
    response = item.get("response")
    if not isinstance(response, dict):
        raise ValueError("batch_item_invalid")
    usage = response.get("usageMetadata")
    return response, usage if isinstance(usage, dict) else {}
