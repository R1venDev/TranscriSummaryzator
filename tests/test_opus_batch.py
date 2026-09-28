"""Offline OpenRouter Opus Batch route, body, and transport tests."""

from __future__ import annotations

import hashlib
import io
import json
import unittest
import urllib.error

from summary.opus_v1 import (
    BATCH_MODEL_IDS, BatchClient, BatchError, MODEL, PROVIDER,
    RouteBlocked, batch_id_from_submit, canonical_request,
    extract_one_completed, verify_batch_route,
)
from summary.luna_v1.ledger import OPUS_STAGE_CAP_MICROUSD


def _request(cap=1000):
    return {
        "messages": [{"role": "system", "content": "Return grounded JSON."},
                     {"role": "user", "content": "Короткая учебная стенограмма и черновик."}],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "opus_summary_audit_v1", "strict": True,
            "schema": {"type": "object", "additionalProperties": False},
        }},
        "max_completion_tokens": cap,
        "reasoning": {"effort": "medium"},
    }


class _Response:
    def __init__(self, status, body):
        self.status = status
        self.body = json.dumps(body).encode("utf-8")

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
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return _Response(*reply)


def _route_replies(*, allowed=True, provider="anthropic/us", context=1_000_000):
    price = {"prompt": "0.000002", "completion": "0.00001", "request": "0"}
    return [(200, body) for body in (
        {"data": {"workspace_id": "ws-test", "limit_remaining": "1"}},
        {"data": [{"id": MODEL}] if allowed else []},
        {"data": {"id": MODEL, "context_length": context,
                  "top_provider": {"max_completion_tokens": 128_000},
                  "supported_parameters": ["response_format", "reasoning", "max_tokens"],
                  "pricing": price}},
        {"data": {"id": MODEL, "endpoints": [{
            "tag": provider, "provider_name": "Anthropic" if provider.startswith("anthropic") else "Other",
            "context_length": context, "max_completion_tokens": 128_000,
            "supported_parameters": ["response_format", "reasoning"],
            "pricing": price,
        }]}}
    )]


class OpusBatchTests(unittest.TestCase):
    def test_submit_pins_batch_model_anthropic_and_chat_contract(self):
        opener = _Opener((202, {"id": "batch_opaque_123", "status": "validating"}))
        request = _request()
        reply = BatchClient("fake-secret", opener=opener).submit("audit.001", request)
        self.assertEqual(batch_id_from_submit(reply.body), "batch_opaque_123")
        sent, timeout = opener.calls[0]
        self.assertEqual((sent.get_method(), sent.full_url, timeout),
                         ("POST", "https://openrouter.ai/api/v1/batches", 45))
        self.assertEqual(sent.get_header("Authorization"), "Bearer fake-secret")
        payload = json.loads(sent.data)
        self.assertEqual(list(payload), ["endpoint", "model", "provider", "completion_window", "requests"])
        self.assertEqual(payload["endpoint"], "/v1/chat/completions")
        self.assertEqual(payload["model"], MODEL)
        self.assertEqual(payload["provider"], {"only": [PROVIDER]})
        self.assertEqual(payload["completion_window"], "24h")
        self.assertEqual(payload["requests"], [{"custom_id": "audit.001", "body": request}])

    def test_read_only_route_verifies_key_model_endpoint_and_estimate(self):
        opener = _Opener(*_route_replies())
        route = verify_batch_route(BatchClient("fake", opener=opener), _request(),
                                   max_output_tokens=1000)
        self.assertEqual([call[0].get_method() for call in opener.calls], ["GET"] * 4)
        self.assertEqual([call[0].full_url for call in opener.calls], [
            "https://openrouter.ai/api/v1/key",
            "https://openrouter.ai/api/v1/models/user",
            "https://openrouter.ai/api/v1/model/anthropic/claude-opus-5.5:batch",
            "https://openrouter.ai/api/v1/models/anthropic/claude-opus-5.5%3Abatch/endpoints",
        ])
        self.assertEqual((route.model, route.provider, route.workspace_id), (MODEL, PROVIDER, "ws-test"))
        self.assertGreater(route.reserve_microusd(), 0)
        self.assertLessEqual(route.reserve_microusd(), OPUS_STAGE_CAP_MICROUSD)
        self.assertGreater(route.context_bound_tokens, route.input_tokens)

    def test_unsupported_route_and_over_budget_fail_before_post(self):
        for replies, reason in (
            (_route_replies(allowed=False), "model_not_allowed_for_key"),
            (_route_replies(provider="other"), "anthropic_endpoint_unverified"),
            (_route_replies(context=100), "context_capacity_exceeded"),
        ):
            with self.subTest(reason=reason), self.assertRaisesRegex(RouteBlocked, reason):
                verify_batch_route(BatchClient("fake", opener=_Opener(*replies)),
                                   _request(), max_output_tokens=1000)
        audit = verify_batch_route(BatchClient("fake", opener=_Opener(*_route_replies())),
                                   _request(cap=10_000), max_output_tokens=10_000)
        self.assertGreater(audit.estimated_reserve_microusd(), 100_000)
        self.assertLessEqual(audit.reserve_microusd(), OPUS_STAGE_CAP_MICROUSD)
        oversized = verify_batch_route(BatchClient("fake", opener=_Opener(*_route_replies())),
                                       _request(cap=50_000), max_output_tokens=50_000)
        self.assertGreater(oversized.estimated_reserve_microusd(), OPUS_STAGE_CAP_MICROUSD)
        with self.assertRaisesRegex(RouteBlocked, "job_budget_exceeded"):
            oversized.reserve_microusd()

    def test_rejects_temperature_tools_cache_media_and_missing_effort(self):
        for field, value in (
            ("temperature", 0.0), ("tools", []), ("tool_choice", "none"),
            ("cache_control", {"type": "ephemeral"}), ("plugins", [{"id": "web"}]),
            ("modalities", ["audio"]),
        ):
            bad = _request()
            bad[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                canonical_request(bad)
        bad = _request()
        del bad["reasoning"]
        with self.assertRaisesRegex(ValueError, "invalid_reasoning_profile"):
            canonical_request(bad)

    def test_prompt_only_json_is_explicit_direct_writer_exception(self):
        request = _request()
        del request["response_format"]
        with self.assertRaisesRegex(ValueError, "structured_output_required"):
            canonical_request(request)
        self.assertEqual(canonical_request(request, allow_prompt_json=True), request)
        opener = _Opener((202, {"id": "batch_direct_001", "status": "validating"}))
        client = BatchClient("fake-secret", opener=opener)
        with self.assertRaisesRegex(ValueError, "structured_output_required"):
            client.submit("direct.001", request)
        self.assertEqual(opener.calls, [])
        client.submit("direct.001", request, allow_prompt_json=True)
        self.assertEqual(json.loads(opener.calls[0][0].data)["requests"][0]["body"], request)
        bad = dict(request, tools=[])
        with self.assertRaisesRegex(ValueError, "unapproved_chat_parameter"):
            canonical_request(bad, allow_prompt_json=True)
        route = verify_batch_route(
            BatchClient("fake-secret", opener=_Opener(*_route_replies())),
            request, max_output_tokens=1000, allow_prompt_json=True)
        self.assertEqual(route.provider, PROVIDER)
        with self.assertRaisesRegex(ValueError, "structured_output_required"):
            verify_batch_route(
                BatchClient("fake-secret", opener=_Opener(*_route_replies())),
                request, max_output_tokens=1000)

    def test_redacted_transport_and_exact_terminal_item(self):
        secret, source = "fake-secret-do-not-log", "PRIVATE CONTENT DO NOT LOG"
        error = urllib.error.HTTPError(
            "https://openrouter.ai/api/v1/batches", 401, secret + source,
            {"Retry-After": "12"}, io.BytesIO(source.encode()),
        )
        opener = _Opener(error)
        request = _request()
        request["messages"][1]["content"] = source
        with self.assertRaises(BatchError) as caught:
            BatchClient(secret, opener=opener).submit("audit-001", request)
        self.assertEqual(len(opener.calls), 1)
        self.assertNotIn(secret, str(caught.exception))
        self.assertNotIn(source, str(caught.exception))

        digest = hashlib.sha256(json.dumps(request, ensure_ascii=False, sort_keys=True,
                                           separators=(",", ":")).encode("utf-8")).hexdigest()
        chat = {"object": "chat.completion", "model": MODEL,
                "choices": [{"message": {"role": "assistant", "content": '{"findings":[]}'},
                             "finish_reason": "stop"}]}
        batch = {"id": "batch_abc123", "status": "completed", "model": MODEL,
                 "endpoint": "/v1/chat/completions", "usage": {"cost": 0.01},
                 "results": [{"custom_id": "audit-001", "error": None,
                              "response": {"status_code": 200, "body": chat}}]}
        response, usage = extract_one_completed(
            batch, "audit-001", expected_batch_id="batch_abc123",
            manifest={"inline_request_count": 1, "custom_id": "audit-001",
                      "request_sha256": digest}, saved_request=request,
        )
        self.assertEqual(response, chat)
        self.assertEqual(usage["cost"], 0.01)
        self.assertIn(MODEL, BATCH_MODEL_IDS)


if __name__ == "__main__":
    unittest.main()
