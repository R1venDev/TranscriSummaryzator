"""Offline OpenRouter Batch transport tests with invented text only."""
import json
import hashlib
import unittest
import urllib.error
import urllib.parse

from summary.luna_v1.batch import (
    BatchClient, BatchError, MODEL, SUBMIT_MODEL, build_batch_payload,
    parse_batch_items,
)


def _body(effort="medium"):
    return {
        "messages": [
            {"role": "developer", "content": "Return only JSON."},
            {"role": "user", "content": '{"source":"Synthetic example."}'},
        ],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "synthetic", "strict": True,
            "schema": {"type": "object", "properties": {"answer": {"type": "string"}},
                       "required": ["answer"], "additionalProperties": False},
        }},
        "max_completion_tokens": 25_000,
        "reasoning": {"effort": effort},
    }


class _Response:
    status = 202

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self, _cap):
        return b'{"id":"batch_synthetic123","status":"validating"}'


class _Opener:
    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def open(self, request, timeout):
        self.calls.append((request, timeout))
        if self.error:
            raise self.error
        return _Response()


def _result(custom_id, *, content='{"answer":"yes"}', finish="stop",
            refusal=None, status_code=200, error=None):
    response = {"status_code": status_code, "request_id": f"request_{custom_id}",
                "body": {"id": f"gen_{custom_id}",
                         "usage": {"prompt_tokens": 4, "completion_tokens": 5},
                         "choices": [{"finish_reason": finish,
                                      "message": {"role": "assistant", "content": content,
                                                  "refusal": refusal}}]}}
    return {"custom_id": custom_id, "response": None if error else response,
            "error": error}


class MultiBatchTransportTests(unittest.TestCase):
    def test_prepared_batch_posts_exact_reserved_bytes_and_rejects_reordered_keys(self):
        opener = _Opener()
        client = BatchClient("synthetic-token", opener=opener)
        payload = build_batch_payload([("writer", _body())])
        sealed = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        client.submit_prepared(sealed, expected_sha256=hashlib.sha256(sealed).hexdigest())
        self.assertEqual(opener.calls[0][0].data, sealed)
        reordered = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":")).encode()
        with self.assertRaisesRegex(ValueError, "prepared_batch_order"):
            client.submit_prepared(reordered, expected_sha256=hashlib.sha256(reordered).hexdigest())
        self.assertEqual(len(opener.calls), 1)

    def test_only_batch_post_with_ordered_headers_and_independent_items(self):
        opener = _Opener()
        client = BatchClient("synthetic-token", opener=opener)
        reply = client.submit_many([("stage-writer", _body()),
                                    ("stage-extract", _body("high"))])
        self.assertEqual(reply.status_code, 202)
        self.assertEqual(len(opener.calls), 1)
        request = opener.calls[0][0]
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.full_url, "https://openrouter.ai/api/v1/batches")
        payload = json.loads(request.data)
        self.assertEqual(list(payload), ["endpoint", "model", "provider",
                                         "completion_window", "requests"])
        self.assertEqual(payload["endpoint"], "/v1/chat/completions")
        self.assertEqual(payload["model"], SUBMIT_MODEL)
        self.assertEqual(payload["provider"], {"only": ["openai"]})
        self.assertEqual(payload["completion_window"], "24h")
        self.assertEqual([x["custom_id"] for x in payload["requests"]],
                         ["stage-writer", "stage-extract"])
        self.assertNotIn("model", payload["requests"][0]["body"])
        self.assertEqual(MODEL, "openai/gpt-6-luna:batch")  # legacy pinned route

    def test_unsupported_controls_and_duplicate_id_fail_before_post(self):
        opener = _Opener()
        client = BatchClient("synthetic-token", opener=opener)
        for unsupported in ({"temperature": 0}, {"tools": []}, {"stream": False},
                            {"provider": {"only": ["openai"]}},
                            {"model": SUBMIT_MODEL}, {"store": False},
                            {"plugins": []}):
            with self.subTest(unsupported=unsupported):
                with self.assertRaises(ValueError):
                    client.submit_many([("a", {**_body(), **unsupported})])
        with self.assertRaisesRegex(ValueError, "duplicate_custom_id"):
            client.submit_many([("a", _body()), ("a", _body())])
        self.assertEqual(opener.calls, [])

    def test_optional_explicit_cache_control_and_chat_reasoning_shortcut(self):
        body = _body()
        body.pop("reasoning")
        body["reasoning_effort"] = "high"
        body["prompt_cache_options"] = {"mode": "explicit"}
        payload = build_batch_payload([("a", body)])
        self.assertEqual(payload["requests"][0]["body"], body)
        body["prompt_cache_options"] = {"mode": "implicit"}
        with self.assertRaisesRegex(ValueError, "unapproved_cache_control"):
            build_batch_payload([("a", body)])

    def test_reordered_item_results_are_not_mapped_by_array_position(self):
        batch = {"id": "batch_synthetic123", "status": "completed",
                 "usage": {"cost": 0.012},
                 "results": [_result("extract"), _result("writer")]}
        parsed = parse_batch_items(batch, ["writer", "extract"],
                                   expected_batch_id="batch_synthetic123")
        self.assertEqual(list(parsed["items"]), ["writer", "extract"])
        self.assertEqual(parsed["items"]["writer"]["generation_id"], "gen_writer")
        self.assertEqual(parsed["items"]["extract"]["request_id"], "request_extract")
        self.assertEqual(parsed["items"]["writer"]["status"], "ok")
        self.assertEqual(parsed["batch_usage"], {"cost": 0.012})
        self.assertEqual(parsed["items"]["writer"]["usage"],
                         {"prompt_tokens": 4, "completion_tokens": 5})
        self.assertIs(parsed["items"]["writer"]["raw"], batch["results"][1])

    def test_duplicate_missing_extra_and_error_keep_good_items(self):
        batch = {"status": "completed", "results": [
            _result("good"), _result("dup"), _result("foreign"),
            _result("dup"), _result("failed", error={"code": "rate_limit"}),
        ]}
        parsed = parse_batch_items(batch, ["good", "dup", "failed", "missing"])
        self.assertEqual(parsed["items"]["good"]["status"], "ok")
        self.assertEqual(parsed["items"]["dup"]["status"], "duplicate")
        self.assertEqual(len(parsed["items"]["dup"]["raw"]), 2)
        self.assertEqual(parsed["items"]["failed"]["status"], "item_error")
        self.assertEqual(parsed["items"]["missing"]["status"], "missing")
        self.assertEqual(parsed["missing_ids"], ("missing",))
        self.assertEqual(parsed["extra_ids"], ("foreign",))
        self.assertEqual(parsed["duplicate_ids"], ("dup",))

    def test_refusal_length_bad_chat_and_http_error_are_distinct(self):
        batch = {"status": "completed", "results": [
            _result("length", finish="length"),
            _result("refused", refusal="Cannot comply"),
            _result("http", status_code=429),
            {"custom_id": "invalid", "response": {"status_code": 200, "body": {}},
             "error": None},
        ]}
        parsed = parse_batch_items(batch, ["length", "refused", "http", "invalid"])
        self.assertEqual([parsed["items"][key]["status"] for key in
                          ("length", "refused", "http", "invalid")],
                         ["length", "refusal", "http_error", "invalid"])

    def test_results_null_means_unavailable_not_zero_or_missing(self):
        for status in ("failed", "expired", "cancelled", "completed"):
            with self.subTest(status=status):
                parsed = parse_batch_items({"status": status, "results": None,
                                            "usage": None}, ["a", "b"])
                self.assertFalse(parsed["results_available"])
                self.assertEqual(parsed["unavailable_ids"], ("a", "b"))
                self.assertEqual(parsed["missing_ids"], ())
                self.assertIsNone(parsed["batch_usage"])
                self.assertEqual(parsed["items"]["a"]["status"], "unavailable")

    def test_workspace_list_for_unknown_post_uses_only_metadata_get(self):
        opener = _Opener()
        client = BatchClient("synthetic-token", opener=opener)
        client.list_batches(limit=50, after="batch_synthetic123", created_after=123,
                            created_before="2026-09-28T12:00:00Z",
                            statuses=("completed", "failed"))
        request = opener.calls[0][0]
        parsed = urllib.parse.urlparse(request.full_url)
        query = urllib.parse.parse_qs(parsed.query)
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(parsed.path, "/api/v1/batches")
        self.assertEqual(query["status"], ["completed", "failed"])
        self.assertEqual(query["after"], ["batch_synthetic123"])
        self.assertEqual(query["limit"], ["50"])

    def test_unknown_create_transport_is_not_automatically_retried(self):
        opener = _Opener(urllib.error.URLError("synthetic disconnect"))
        with self.assertRaises(BatchError) as caught:
            BatchClient("synthetic-token", opener=opener).submit_many([("a", _body())])
        self.assertEqual(caught.exception.reason, "transport_unknown")
        self.assertEqual(len(opener.calls), 1)


if __name__ == "__main__":
    unittest.main()
