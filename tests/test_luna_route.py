"""Free route checks; no live key, private source, or paid request."""

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from summary.luna_v1.route import RouteBlocked, verify_batch_route


class _MetadataClient:
    def current_key(self):
        return SimpleNamespace(body={"data": {"workspace_id": "workspace-example", "limit_remaining": 1}})

    def models_for_key(self):
        return SimpleNamespace(body={"data": [{"id": "openai/gpt-6-luna:batch"}]})

    def model_details(self):
        return SimpleNamespace(body={"data": {
            "id": "openai/gpt-6-luna:batch", "supported_parameters": ["response_format"],
            "context_length": 1_050_000,
            "top_provider": {"max_completion_tokens": 128_000},
            "pricing": {"prompt": "0.00000005", "completion": "0.00000025",
                        "input_cache_write": "0.0000000625", "request": "0",
                        "overrides": [{"min_prompt_tokens": 272000, "prompt": "0.0000001",
                                       "completion": "0.000000375", "input_cache_write": "0.000000125"}]},
        }})


class RouteTests(unittest.TestCase):
    def test_private_dispatch_requires_exact_verified_workspace(self):
        client = _MetadataClient()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RouteBlocked, "account_policy_unverified"):
                verify_batch_route(client)
        with patch.dict(os.environ, {"TRANSCRI_SUMMARY_VERIFIED_WORKSPACE_ID": "other-workspace"}):
            with self.assertRaisesRegex(RouteBlocked, "account_policy_unverified"):
                verify_batch_route(client)
        with patch.dict(os.environ, {"TRANSCRI_SUMMARY_VERIFIED_WORKSPACE_ID": "workspace-example"}):
            route = verify_batch_route(client)
        self.assertGreater(route.cache_write_usd_per_token, route.prompt_usd_per_token)
        self.assertEqual(str(route.cache_write_usd_per_token), "1.25E-7")
        self.assertGreater(route.reserve_microusd(b"tiny"), 0)

    def test_missing_cache_write_tariff_blocks_instead_of_under_reserving(self):
        class MissingTariff(_MetadataClient):
            def model_details(self):
                result = super().model_details()
                del result.body["data"]["pricing"]["input_cache_write"]
                return result

        with patch.dict(os.environ, {"TRANSCRI_SUMMARY_VERIFIED_WORKSPACE_ID": "workspace-example"}):
            with self.assertRaisesRegex(RouteBlocked, "cache_write_price"):
                verify_batch_route(MissingTariff())


if __name__ == "__main__":
    unittest.main()
