"""Additive source-first endpoint checks; all metadata is synthetic."""
import copy
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from summary.luna_v1.batch import MODEL, SUBMIT_MODEL
from summary.luna_v1.route import RouteBlocked, verify_source_first_batch_route


class _MetadataClient:
    def __init__(self):
        self.key = {"data": {"workspace_id": "workspace-synthetic",
                             "limit_remaining": 1}}
        self.models = {"data": [{"id": MODEL}]}
        self.details = {"data": {
            "id": MODEL, "supported_parameters": ["response_format"],
            "context_length": 1_050_000,
            "top_provider": {"max_completion_tokens": 128_000},
            "pricing": {"prompt": "0.00000005", "completion": "0.00000025",
                        "input_cache_write": "0.0000000625", "request": "0",
                        "overrides": [{"min_prompt_tokens": 272000,
                                       "prompt": "0.0000001",
                                       "completion": "0.000000375",
                                       "input_cache_write": "0.000000125"}]},
        }}
        self.endpoints = {"data": {
            "id": MODEL, "endpoints": [{
                "model_id": MODEL, "tag": "openai", "provider_name": "OpenAI",
                "context_length": 1_050_000,
                "max_completion_tokens": 128_000,
                "supported_parameters": ["response_format", "reasoning",
                                         "max_completion_tokens", "prompt_cache_options"],
                "pricing": {"prompt": "0.00000005",
                            "completion": "0.00000025",
                            "input_cache_write": "0.0000000625", "request": "0"},
            }]}}

    def current_key(self):
        return SimpleNamespace(body=copy.deepcopy(self.key))

    def models_for_key(self):
        return SimpleNamespace(body=copy.deepcopy(self.models))

    def model_details(self):
        return SimpleNamespace(body=copy.deepcopy(self.details))

    def model_endpoints(self):
        return SimpleNamespace(body=copy.deepcopy(self.endpoints))


class SourceFirstRouteTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {
            "TRANSCRI_SUMMARY_VERIFIED_WORKSPACE_ID": "workspace-synthetic"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_base_submit_slug_selected_only_after_exact_openai_batch_metadata(self):
        route = verify_source_first_batch_route(_MetadataClient(),
                                                require_cache_disabled=True)
        self.assertEqual(route.model, SUBMIT_MODEL)
        self.assertEqual(route.batch_endpoint_model, MODEL)
        self.assertEqual(route.provider_endpoint_tag, "openai")
        self.assertTrue(route.supports_explicit_cache)
        self.assertEqual(str(route.prompt_usd_per_token), "1E-7")
        self.assertEqual(str(route.cache_write_usd_per_token), "1.25E-7")
        self.assertGreater(route.reserve_microusd(b"synthetic request",
                                                   max_completion_tokens=25_000), 0)

    def test_wrong_or_multiple_openai_endpoint_blocks(self):
        client = _MetadataClient()
        client.endpoints["data"]["endpoints"][0]["tag"] = "amazon-bedrock"
        with self.assertRaisesRegex(RouteBlocked, "openai_batch_endpoint_not_unique"):
            verify_source_first_batch_route(client)
        client = _MetadataClient()
        client.endpoints["data"]["endpoints"].append(
            copy.deepcopy(client.endpoints["data"]["endpoints"][0]))
        with self.assertRaisesRegex(RouteBlocked, "openai_batch_endpoint_not_unique"):
            verify_source_first_batch_route(client)

    def test_missing_required_chat_capability_or_cache_control_blocks(self):
        client = _MetadataClient()
        client.endpoints["data"]["endpoints"][0]["supported_parameters"].remove("reasoning")
        with self.assertRaisesRegex(RouteBlocked, "endpoint_chat_parameters_unavailable"):
            verify_source_first_batch_route(client)
        client = _MetadataClient()
        client.endpoints["data"]["endpoints"][0]["supported_parameters"].remove(
            "prompt_cache_options")
        with self.assertRaisesRegex(RouteBlocked, "explicit_cache_control_unavailable"):
            verify_source_first_batch_route(client, require_cache_disabled=True)

    def test_missing_endpoint_tariff_does_not_reuse_model_tariff(self):
        client = _MetadataClient()
        del client.endpoints["data"]["endpoints"][0]["pricing"]["input_cache_write"]
        with self.assertRaisesRegex(RouteBlocked, "input_cache_write_price"):
            verify_source_first_batch_route(client)

    def test_model_base_tariff_remains_in_reserve_when_endpoint_quotes_less(self):
        client = _MetadataClient()
        client.details["data"]["pricing"]["prompt"] = "0.0000002"
        client.details["data"]["pricing"]["completion"] = "0.0000004"
        client.details["data"]["pricing"]["input_cache_write"] = "0.0000003"
        client.details["data"]["pricing"]["request"] = "0.01"
        route = verify_source_first_batch_route(client)
        self.assertEqual(str(route.prompt_usd_per_token), "2E-7")
        self.assertEqual(str(route.completion_usd_per_token), "4E-7")
        self.assertEqual(str(route.cache_write_usd_per_token), "3E-7")
        self.assertEqual(str(route.request_usd), "0.01")
        with self.assertRaisesRegex(RouteBlocked, "invalid_output_cap"):
            route.reserve_microusd(b"synthetic request", max_completion_tokens=True)

    def test_missing_endpoint_response_blocks_without_inference(self):
        client = _MetadataClient()
        client.endpoints = {"data": {"id": MODEL, "endpoints": None}}
        with self.assertRaisesRegex(RouteBlocked, "batch_endpoints_unavailable"):
            verify_source_first_batch_route(client)


if __name__ == "__main__":
    unittest.main()
