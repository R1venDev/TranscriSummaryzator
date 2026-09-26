import io
import hashlib
import json
import unittest
import urllib.error
from datetime import date
from unittest.mock import patch

from summary.gemini_v1.batch import (
    BatchClient, BatchError, batch_id_from_submit, canonical_request,
    extract_one_completed, normalize_batch_status, valid_batch_id,
)
from summary.gemini_v1.route import (
    RouteBlocked, estimate_usage_cost_microusd, verify_batch_route,
)


def _request():
    return {
        "systemInstruction": {"parts": [{"text": "Return grounded JSON."}]},
        "contents": [{"role": "user", "parts": [{"text": "Тестовая фраза."}]}],
        "generationConfig": {
            "maxOutputTokens": 8000,
            "thinkingConfig": {"thinkingLevel": "medium"},
            "responseFormat": {"text": {"mimeType": "application/json", "schema": {"type": "object"}}},
        },
    }


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
    def test_submit_is_one_inline_rest_request_with_key_and_no_logging(self):
        opener = _Opener((200, {"name": "batches/opaque_123", "done": False}))
        client = BatchClient("fake-secret-test-key", opener=opener)
        original = _request()
        reply = client.submit("audit.001", original)
        self.assertEqual(batch_id_from_submit(reply.body), "batches/opaque_123")
        self.assertTrue(valid_batch_id("batches/opaque_123"))
        self.assertFalse(valid_batch_id("batches/../../bad"))
        self.assertEqual(len(opener.calls), 1)
        sent, timeout = opener.calls[0]
        self.assertEqual(timeout, 45)
        self.assertEqual(sent.get_method(), "POST")
        self.assertEqual(sent.full_url, "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.7-flash:batchGenerateContent")
        self.assertEqual(sent.get_header("X-goog-api-key"), "fake-secret-test-key")
        body = json.loads(sent.data)
        self.assertEqual(body["batch"]["display_name"], "transcri-audit.001")
        self.assertNotIn("file_name", body["batch"]["input_config"])
        item = body["batch"]["input_config"]["requests"]["requests"][0]
        self.assertEqual(item["metadata"], {"key": "audit.001"})
        self.assertEqual(item["request"]["model"], "models/gemini-3.7-flash")
        self.assertIs(item["request"]["store"], False)
        self.assertEqual(item["request"]["contents"], original["contents"])
        self.assertNotIn("store", original)

    def test_get_model_and_count_tokens_exact_generate_request(self):
        opener = _Opener(
            (200, {"name": "models/gemini-3.7-flash", "inputTokenLimit": 1048576,
                   "outputTokenLimit": 65536, "supportedGenerationMethods": ["generateContent", "countTokens"]}),
            (200, {"totalTokens": 35522}),
        )
        client = BatchClient("fake", opener=opener)
        route = verify_batch_route(client, _request(), max_output_tokens=8000,
                                   as_of=date(2026, 9, 26))
        self.assertEqual(route.model, "gemini-3.7-flash")
        self.assertEqual(route.input_tokens, 35522)
        self.assertEqual(route.provider, "google")
        self.assertEqual(route.reserve_microusd(), 33985)
        self.assertEqual(opener.calls[0][0].get_method(), "GET")
        counted = opener.calls[1][0]
        self.assertEqual(counted.full_url, "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.7-flash:countTokens")
        self.assertEqual(json.loads(counted.data), {"generateContentRequest": canonical_request(_request())})

    def test_price_cutoff_and_capacity_block_before_dispatch(self):
        client = BatchClient("fake", opener=_Opener())
        with self.assertRaisesRegex(RouteBlocked, "price_review_required"):
            verify_batch_route(client, _request(), max_output_tokens=8000,
                               as_of=date(2027, 1, 1))
        opener = _Opener(
            (200, {"name": "models/gemini-3.7-flash", "inputTokenLimit": 20,
                   "outputTokenLimit": 65536}),
            (200, {"totalTokens": 100}),
        )
        with self.assertRaisesRegex(RouteBlocked, "input_capacity_exceeded"):
            verify_batch_route(BatchClient("fake", opener=opener), _request(),
                               max_output_tokens=8000, as_of=date(2026, 9, 26))

    def test_transport_error_is_redacted_and_post_is_not_retried(self):
        secret = "fake-secret-do-not-log"
        source = "PRIVATE CONTENT DO NOT LOG"
        error = urllib.error.HTTPError(
            "https://generativelanguage.googleapis.com/v1beta/x", 401,
            secret + source, {"Retry-After": "12", "X-Echo": secret}, io.BytesIO(source.encode()),
        )
        opener = _Opener(error)
        request = _request()
        request["contents"][0]["parts"][0]["text"] = source
        with self.assertRaises(BatchError) as caught:
            BatchClient(secret, opener=opener).submit("audit-001", request)
        self.assertEqual(caught.exception.code, 401)
        self.assertEqual(caught.exception.retry_after, "12")
        self.assertNotIn(secret, str(caught.exception))
        self.assertNotIn(source, str(caught.exception))
        self.assertEqual(len(opener.calls), 1)

    def test_reject_large_payload_file_refs_tools_and_storage(self):
        client = BatchClient("fake", opener=_Opener())
        bad = _request()
        bad["store"] = True
        with self.assertRaisesRegex(ValueError, "storage_not_disabled"):
            client.submit("x", bad)
        bad = _request()
        bad["contents"][0]["parts"][0]["fileData"] = {"fileUri": "files/secret"}
        with self.assertRaisesRegex(ValueError, "text_only_required"):
            client.submit("x", bad)
        bad = _request()
        bad["tools"] = [{"googleSearch": {}}]
        with self.assertRaisesRegex(ValueError, "tools_not_allowed"):
            client.submit("x", bad)
        with patch("summary.gemini_v1.batch.MAX_INLINE_BYTES", 150):
            with self.assertRaisesRegex(ValueError, "payload_too_large"):
                client.submit("x", _request())

    def test_lro_and_batchjob_status_and_identity_preserving_extract(self):
        generated = {"candidates": [{"content": {"parts": [{"text": '{"findings":[]}' }],
                                                      "role": "model"}, "finishReason": "STOP"}],
                     "usageMetadata": {"promptTokenCount": 35522, "candidatesTokenCount": 4000,
                                       "thoughtsTokenCount": 1964, "totalTokenCount": 41486}}
        lro = {
            "name": "batches/abc123", "done": True,
            "metadata": {"state": "BATCH_STATE_SUCCEEDED"},
            "response": {"output": {"inlinedResponses": {"inlinedResponses": [
                {"metadata": {"key": "other"}, "response": {"candidates": []}},
                {"metadata": {"key": "audit-001"}, "response": generated},
            ]}}},
        }
        self.assertEqual(normalize_batch_status(lro), "completed")
        response, usage = extract_one_completed(lro, "audit-001")
        self.assertEqual(response, generated)
        self.assertEqual(usage["totalTokenCount"], 41486)
        self.assertEqual(estimate_usage_cost_microusd(usage), 24504)
        job_view = {"name": "batches/abc123", "state": "JOB_STATE_SUCCEEDED",
                    "dest": {"inlinedResponses": [{"metadata": {"key": "audit-001"},
                                                   "response": generated}]}}
        self.assertEqual(normalize_batch_status(job_view), "completed")
        self.assertEqual(extract_one_completed(job_view, "audit-001")[0], generated)
        direct_lro = {"name": "batches/abc123", "done": True,
                      "metadata": {"state": "JOB_STATE_SUCCEEDED"},
                      "response": {"inlinedResponses": [{"metadata": {"key": "audit-001"},
                                                        "response": generated}]}}
        self.assertEqual(extract_one_completed(direct_lro, "audit-001")[0], generated)
        self.assertEqual(normalize_batch_status({"done": False, "metadata": {"state": "BATCH_STATE_RUNNING"}}), "running")
        self.assertEqual(normalize_batch_status({"done": False, "metadata": {"state": "BATCH_STATE_SUCCEEDED"}}), "running")
        self.assertEqual(normalize_batch_status({"done": True, "metadata": {"state": "BATCH_STATE_RUNNING"},
                                                 "response": direct_lro["response"]}), "completed")
        self.assertEqual(normalize_batch_status({"done": True, "error": {"code": 13}}), "failed")
        with self.assertRaisesRegex(ValueError, "custom_id_mismatch"):
            extract_one_completed(lro, "wrong")
        lro["response"]["output"]["inlinedResponses"]["inlinedResponses"].append(
            {"metadata": {"key": "audit-001"}, "response": generated})
        with self.assertRaisesRegex(ValueError, "custom_id_mismatch"):
            extract_one_completed(lro, "audit-001")

    def test_single_inline_result_without_metadata_requires_saved_request_proof(self):
        request = canonical_request(_request())
        custom_id = "audit-001"
        batch_id = "batches/abc123"
        manifest = {
            "inline_request_count": 1,
            "custom_id": custom_id,
            "request_sha256": hashlib.sha256(json.dumps(
                request, ensure_ascii=False, sort_keys=True,
                separators=(",", ":")).encode("utf-8")).hexdigest(),
        }
        result = {"candidates": [], "usageMetadata": {"promptTokenCount": 100}}
        batch = {"name": batch_id, "done": True,
                 "response": {"output": {"inlinedResponses": {"inlinedResponses": [
                     {"response": result}]}}}}
        with self.assertRaisesRegex(ValueError, "custom_id_mismatch"):
            extract_one_completed(batch, custom_id)
        self.assertIs(extract_one_completed(
            batch, custom_id, expected_batch_id=batch_id,
            manifest=manifest, saved_request=request)[0], result)
        for changed in (
            {"expected_batch_id": "batches/other", "manifest": manifest,
             "saved_request": request},
            {"expected_batch_id": batch_id, "manifest": dict(manifest, inline_request_count=2),
             "saved_request": request},
            {"expected_batch_id": batch_id, "manifest": manifest,
             "saved_request": dict(request, store=True)},
        ):
            with self.assertRaisesRegex(ValueError, "custom_id_mismatch"):
                extract_one_completed(batch, custom_id, **changed)
        batch["response"]["output"]["inlinedResponses"]["inlinedResponses"].append({"response": result})
        with self.assertRaisesRegex(ValueError, "custom_id_mismatch"):
            extract_one_completed(batch, custom_id, expected_batch_id=batch_id,
                                  manifest=manifest, saved_request=request)
        batch["response"]["output"]["inlinedResponses"]["inlinedResponses"] = [
            {"metadata": {"key": "other"}, "response": result}]
        with self.assertRaisesRegex(ValueError, "custom_id_mismatch"):
            extract_one_completed(batch, custom_id, expected_batch_id=batch_id,
                                  manifest=manifest, saved_request=request)

    def test_unknown_usage_is_not_zero_and_cache_discount_not_assumed(self):
        self.assertIsNone(estimate_usage_cost_microusd(None))
        self.assertIsNone(estimate_usage_cost_microusd({}))
        self.assertIsNone(estimate_usage_cost_microusd({"promptTokenCount": 100}))
        base = {"promptTokenCount": 35522, "candidatesTokenCount": 4000,
                "thoughtsTokenCount": 1964, "totalTokenCount": 41486}
        with_cache = dict(base, cachedContentTokenCount=30000)
        self.assertEqual(estimate_usage_cost_microusd(base),
                         estimate_usage_cost_microusd(with_cache))
        self.assertIsNone(estimate_usage_cost_microusd(dict(base, cachedContentTokenCount=99999)))

    def test_delete_accepts_empty_204_and_cancel_has_empty_body(self):
        opener = _Opener((200, {}), (204, None))
        client = BatchClient("fake", opener=opener)
        self.assertEqual(client.cancel("batches/abc123").body, {})
        self.assertIsNone(opener.calls[0][0].data)
        self.assertEqual(client.delete("batches/abc123").body, {})
        self.assertEqual(opener.calls[1][0].get_method(), "DELETE")

    def test_get_checks_remote_identity_and_os_errors_are_redacted(self):
        opener = _Opener((200, {"name": "batches/other", "done": False}), OSError("private path"))
        client = BatchClient("fake", opener=opener)
        with self.assertRaisesRegex(BatchError, "batch_identity_mismatch"):
            client.get("batches/abc123")
        with self.assertRaises(BatchError) as caught:
            client.get("batches/abc123")
        self.assertEqual(caught.exception.reason, "transport_unknown")
        self.assertNotIn("private path", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
