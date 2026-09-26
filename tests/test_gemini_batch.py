"""Offline OpenRouter Gemini Batch transport and route contract tests."""
from __future__ import annotations

import hashlib
import io
import json
import unittest
import urllib.error

from summary.gemini_v1.batch import (
    BatchClient, BatchError, MODEL, PROVIDER, batch_id_from_submit,
    canonical_request, extract_one_completed, normalize_batch_status,
    valid_batch_id,
)
from summary.gemini_v1.route import (
    RouteBlocked, estimate_usage_cost_microusd, verify_batch_route,
)


def _request():
    return {
        "messages": [
            {"role": "system", "content": "Return grounded JSON."},
            {"role": "user", "content": "Тестовая фраза."},
        ],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "audit", "strict": True, "schema": {"type": "object"},
        }},
        "max_completion_tokens": 8000,
        "reasoning": {"effort": "medium"},
        "tool_choice": "none", "plugins": [], "modalities": ["text"],
    }


def _route_replies(*, workspace="ws-test", context=1_048_576,
                   endpoint_tag="google-vertex/global", allowed=True, endpoint_price="0.000000375"):
    key = {"data": {"workspace_id": workspace, "limit_remaining": "1"}}
    models = {"data": [{"id": MODEL}] if allowed else []}
    model = {"data": {
        "id": MODEL, "context_length": context,
        "top_provider": {"max_completion_tokens": 65_536},
        "supported_parameters": ["response_format", "reasoning", "max_tokens"],
        "pricing": {"prompt": "0.000000375", "completion": "0.000001875",
                    "input_cache_write": "0.00000004167", "request": "0"},
    }}
    endpoints = {"data": {"id": MODEL, "endpoints": [{
        "tag": endpoint_tag, "provider_name": "Google",
        "context_length": context, "max_completion_tokens": 65_536,
        "supported_parameters": ["response_format", "reasoning"],
        "pricing": {"prompt": endpoint_price, "completion": "0.000001875",
                    "input_cache_write": "0.00000004167", "request": "0"},
    }]}}
    return [(200, x) for x in (key, models, model, endpoints)]


class _Response:
    def __init__(self, status, body):
        self.status = status
        self.body = b"" if body is None else json.dumps(body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, limit):
        return self.body[:limit]


class _Opener:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def open(self, request, timeout):
        self.calls.append((request, timeout))
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return _Response(*item)


class GeminiBatchTests(unittest.TestCase):
    def test_submit_pins_model_provider_and_one_chat_item(self):
        opener = _Opener((202, {"id": "batch_opaque_123", "status": "validating"}))
        client = BatchClient("fake-secret-test-key", opener=opener)
        original = _request()
        reply = client.submit("audit.001", original)
        self.assertEqual(batch_id_from_submit(reply.body), "batch_opaque_123")
        self.assertTrue(valid_batch_id("batch-opaque_123"))
        self.assertFalse(valid_batch_id("batch-../../bad"))
        self.assertEqual(len(opener.calls), 1)
        sent, timeout = opener.calls[0]
        self.assertEqual((timeout, sent.get_method(), sent.full_url),
                         (45, "POST", "https://openrouter.ai/api/v1/batches"))
        self.assertEqual(sent.get_header("Authorization"), "Bearer fake-secret-test-key")
        body = json.loads(sent.data)
        self.assertEqual(list(body), ["endpoint", "model", "provider",
                                      "completion_window", "requests"])
        self.assertEqual(body["endpoint"], "/v1/chat/completions")
        self.assertEqual(body["model"], MODEL)
        self.assertEqual(body["provider"], {"only": [PROVIDER]})
        self.assertEqual(body["completion_window"], "24h")
        self.assertEqual(body["requests"], [{"custom_id": "audit.001", "body": original}])
        self.assertNotIn("store", original)
        self.assertNotIn("cache_control", original)

    def test_read_only_preflight_key_catalog_endpoint_and_price(self):
        opener = _Opener(*_route_replies())
        route = verify_batch_route(BatchClient("fake", opener=opener), _request(),
                                   max_output_tokens=8000)
        self.assertEqual([call[0].get_method() for call in opener.calls], ["GET"] * 4)
        self.assertEqual([call[0].full_url for call in opener.calls], [
            "https://openrouter.ai/api/v1/key",
            "https://openrouter.ai/api/v1/models/user",
            "https://openrouter.ai/api/v1/model/google/gemini-3.7-flash:batch",
            "https://openrouter.ai/api/v1/models/google/gemini-3.7-flash%3Abatch/endpoints",
        ])
        self.assertEqual((route.model, route.provider, route.workspace_id),
                         (MODEL, PROVIDER, "ws-test"))
        self.assertGreaterEqual(route.input_tokens, len(json.dumps(_request(), ensure_ascii=False)))
        self.assertGreater(route.reserve_microusd(), 18_000)
        self.assertLess(route.reserve_microusd(), 25_000)

    def test_wrong_workspace_model_endpoint_capacity_or_price_fails_closed(self):
        request = _request()
        cases = []
        wrong_key = _route_replies()
        wrong_key[0][1]["data"]["workspace_id"] = None
        cases.append((wrong_key, "workspace_scope_unverified"))
        cases.append((_route_replies(allowed=False), "model_not_allowed_for_key"))
        no_vertex = _route_replies()
        no_vertex[3][1]["data"]["endpoints"][0].update(
            tag="other", provider_name="Other")
        cases.append((no_vertex, "google_vertex_endpoint_unverified"))
        cases.append((_route_replies(context=100), "context_capacity_exceeded"))
        no_price = _route_replies()
        no_price[3][1]["data"]["endpoints"][0]["pricing"] = None
        cases.append((no_price, "pricing_unavailable"))
        for replies, reason in cases:
            with self.subTest(reason=reason):
                with self.assertRaisesRegex(RouteBlocked, reason):
                    verify_batch_route(BatchClient("fake", opener=_Opener(*replies)),
                                       request, max_output_tokens=8000)
        with self.assertRaisesRegex(RouteBlocked, "output_cap_mismatch"):
            verify_batch_route(BatchClient("fake", opener=_Opener()), request,
                               max_output_tokens=5000)

    def test_request_rejects_media_tools_plugins_cache_and_unsupported_fields(self):
        for field, value in (
            ("tools", []), ("store", False), ("cache_control", {"type": "ephemeral"}),
            ("plugins", [{"id": "web"}]), ("tool_choice", "auto"),
            ("modalities", ["audio"]),
        ):
            bad = _request()
            bad[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                canonical_request(bad)
        bad = _request()
        bad["messages"][1]["content"] = [{"type": "image_url", "image_url": "http://example"}]
        with self.assertRaisesRegex(ValueError, "text_only"):
            canonical_request(bad)
        bad = _request()
        bad["response_format"]["json_schema"]["strict"] = False
        with self.assertRaisesRegex(ValueError, "invalid_output_schema"):
            canonical_request(bad)

    def test_transport_error_redacted_and_post_not_retried(self):
        secret, source = "fake-secret-do-not-log", "PRIVATE CONTENT DO NOT LOG"
        error = urllib.error.HTTPError(
            "https://openrouter.ai/api/v1/batches", 401, secret + source,
            {"Retry-After": "12", "X-Echo": secret}, io.BytesIO(source.encode()),
        )
        opener = _Opener(error)
        request = _request()
        request["messages"][1]["content"] = source
        with self.assertRaises(BatchError) as caught:
            BatchClient(secret, opener=opener).submit("audit-001", request)
        self.assertEqual((caught.exception.code, caught.exception.retry_after), (401, "12"))
        self.assertNotIn(secret, str(caught.exception))
        self.assertNotIn(source, str(caught.exception))
        self.assertEqual(len(opener.calls), 1)

    def test_completion_matches_exact_custom_id_and_billed_cost(self):
        response = {"object": "chat.completion", "model": MODEL,
                    "choices": [{"message": {"role": "assistant", "content": '{"findings":[]}'},
                                 "finish_reason": "stop"}]}
        request = _request()
        manifest = {"inline_request_count": 1, "custom_id": "audit-001",
                    "request_sha256": hashlib.sha256(json.dumps(
                        request, ensure_ascii=False, sort_keys=True,
                        separators=(",", ":")).encode("utf-8")).hexdigest()}
        batch = {"id": "batch_abc123", "status": "completed", "model": MODEL,
                 "endpoint": "/v1/chat/completions",
                 "usage": {"prompt_tokens": 35522, "completion_tokens": 5964,
                           "total_tokens": 41486, "cost": 0.024504},
                 "results": [{"custom_id": "audit-001", "error": None,
                              "response": {"status_code": 200, "body": response}}]}
        self.assertEqual(normalize_batch_status(batch), "completed")
        body, usage = extract_one_completed(batch, "audit-001", expected_batch_id="batch_abc123",
                                            manifest=manifest, saved_request=request)
        self.assertEqual(body, response)
        self.assertEqual(estimate_usage_cost_microusd(usage), 24504)
        self.assertIsNone(estimate_usage_cost_microusd({"prompt_tokens": 5}))
        self.assertEqual(normalize_batch_status({"status": "in_progress"}), "in_progress")
        with self.assertRaisesRegex(ValueError, "batch_custom_id_mismatch"):
            extract_one_completed(batch, "wrong")
        with self.assertRaisesRegex(ValueError, "saved_request_mismatch"):
            extract_one_completed(batch, "audit-001", manifest=manifest,
                                  saved_request=dict(request, extra=True))
        with self.assertRaisesRegex(ValueError, "batch_identity_mismatch"):
            extract_one_completed(batch, "audit-001", expected_batch_id="batch_other")
        batch["results"].append(batch["results"][0])
        with self.assertRaisesRegex(ValueError, "batch_custom_id_mismatch"):
            extract_one_completed(batch, "audit-001")

    def test_get_remote_identity_delete_204_and_no_path_injection(self):
        opener = _Opener((200, {"id": "batch_other", "status": "in_progress"}),
                         (204, None))
        client = BatchClient("fake", opener=opener)
        with self.assertRaisesRegex(BatchError, "batch_identity_mismatch"):
            client.get("batch_abc123")
        self.assertEqual(client.delete("batch_abc123").body, {})
        self.assertEqual(opener.calls[1][0].get_method(), "DELETE")
        with self.assertRaisesRegex(ValueError, "invalid batch id"):
            client.get("../other")


if __name__ == "__main__":
    unittest.main()
