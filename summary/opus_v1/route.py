"""Read-only Opus Batch route and conservative full-price preflight.

The selected OpenRouter key, batch variant, Anthropic endpoint, and pricing
are checked before private text is submitted. No cache benefit is assumed.
Terminal usage.cost is authoritative when it includes the provider charge.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_UP
from typing import Any

from .batch import MODEL, PROVIDER, BatchClient, canonical_request
from ..luna_v1.ledger import OPUS_STAGE_CAP_MICROUSD

RESERVE_SAFETY = Decimal("1.2")
_CHAT_WRAPPER_MARGIN = 2048


class RouteBlocked(RuntimeError):
    """Stable safe reason code; no request text in the error."""


def _decimal(value: object, field: str) -> Decimal:
    try:
        number = Decimal(str(value))
    except (TypeError, InvalidOperation):
        raise RouteBlocked(f"missing_or_invalid_{field}") from None
    if not number.is_finite() or number < 0:
        raise RouteBlocked(f"missing_or_invalid_{field}")
    return number


def _positive_int(value: object) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 and type(value) is not bool else None


def _microusd(cost_usd: Decimal) -> int:
    return int((cost_usd * 1_000_000).to_integral_value(rounding=ROUND_UP))


def estimate_usage_cost_microusd(usage: dict[str, Any] | None) -> int | None:
    """Use OpenRouter's billed Batch cost; unknown charge is never zeroed."""
    if not isinstance(usage, dict) or "cost" not in usage:
        return None
    try:
        return _microusd(_decimal(usage["cost"], "batch_cost"))
    except RouteBlocked:
        return None


def _tiers(pricing: dict[str, Any]) -> list[dict[str, Any]]:
    overrides = pricing.get("overrides", [])
    if not isinstance(overrides, list) or any(not isinstance(t, dict) for t in overrides):
        raise RouteBlocked("pricing_overrides_unavailable")
    return [pricing, *overrides]


def _highest(tiers: list[dict[str, Any]], field: str, *aliases: str,
             optional: bool = False) -> Decimal:
    values = []
    for tier in tiers:
        present = next((name for name in (field, *aliases) if name in tier), None)
        if present is None:
            if optional:
                values.append(Decimal(0))
                continue
            raise RouteBlocked(f"missing_or_invalid_{field}_price")
        values.append(_decimal(tier[present], f"{field}_price"))
    return max(values)


@dataclass(frozen=True)
class Route:
    model: str
    provider: str
    workspace_id: str
    max_input_tokens: int
    max_output_tokens: int
    counted_input_tokens: int  # Cost estimate, not tokenizer measurement.
    context_bound_tokens: int  # Separate conservative capacity bound.
    requested_max_output_tokens: int
    prompt_usd_per_token: Decimal
    completion_usd_per_token: Decimal
    request_usd: Decimal
    key_limit_remaining_usd: Decimal | None

    @property
    def input_tokens(self) -> int:
        return self.counted_input_tokens

    def estimated_reserve_microusd(self) -> int:
        """Conservative estimate, including all reasoning tokens in output."""
        cost = (Decimal(self.counted_input_tokens) *
                self.prompt_usd_per_token
                + Decimal(self.requested_max_output_tokens) * self.completion_usd_per_token
                + self.request_usd)
        return _microusd(cost * RESERVE_SAFETY)

    def reserve_microusd(self, *, limit_microusd: int = OPUS_STAGE_CAP_MICROUSD) -> int:
        reserve = self.estimated_reserve_microusd()
        if reserve < 1 or reserve > limit_microusd:
            raise RouteBlocked("job_budget_exceeded")
        if (self.key_limit_remaining_usd is not None
                and Decimal(reserve) / 1_000_000 > self.key_limit_remaining_usd):
            raise RouteBlocked("key_budget_insufficient")
        return reserve

    def actual_cost_microusd(self, usage: dict[str, Any] | None) -> int | None:
        return estimate_usage_cost_microusd(usage)


def verify_batch_route(
    client: BatchClient, request_body: dict[str, Any], *,
    max_output_tokens: int, as_of=None, allow_prompt_json: bool = False,
) -> Route:
    """Verify this key/model/endpoint and reserve a full-miss price bound.

    `as_of` is retained for test/caller compatibility; live prices come from
    current route metadata, not a calendar-dated tariff constant.
    """
    request = canonical_request(request_body, allow_prompt_json=allow_prompt_json)
    if (type(max_output_tokens) is not int
            or request["max_completion_tokens"] != max_output_tokens):
        raise RouteBlocked("output_cap_mismatch")
    key = client.current_key().body.get("data")
    allowed = client.models_for_key().body.get("data")
    model = client.model_details().body.get("data")
    endpoint_data = client.model_endpoints().body.get("data")
    if (not isinstance(key, dict) or not isinstance(allowed, list)
            or not isinstance(model, dict) or not isinstance(endpoint_data, dict)):
        raise RouteBlocked("route_metadata_unavailable")
    if key.get("is_management_key") or key.get("is_provisioning_key"):
        raise RouteBlocked("not_inference_key")
    workspace_id = key.get("workspace_id")
    if not isinstance(workspace_id, str) or not workspace_id:
        raise RouteBlocked("workspace_scope_unverified")
    if not any(isinstance(item, dict) and item.get("id") == MODEL for item in allowed):
        raise RouteBlocked("model_not_allowed_for_key")
    if model.get("id") != MODEL or endpoint_data.get("id") != MODEL:
        raise RouteBlocked("batch_model_identity_changed")
    required_params = ({"reasoning"} if allow_prompt_json
                       else {"response_format", "reasoning"})
    params = model.get("supported_parameters")
    if not isinstance(params, list) or not required_params.issubset(set(params)):
        raise RouteBlocked("required_parameters_unavailable")
    endpoints = endpoint_data.get("endpoints")
    if not isinstance(endpoints, list):
        raise RouteBlocked("anthropic_endpoint_unverified")
    anthropic = [item for item in endpoints if isinstance(item, dict) and (
        item.get("tag") == PROVIDER or
        (isinstance(item.get("tag"), str) and item["tag"].startswith(PROVIDER + "/")) or
        item.get("provider_slug") == PROVIDER or
        item.get("provider_name") == "Anthropic")]
    if len(anthropic) != 1:
        raise RouteBlocked("anthropic_endpoint_unverified")
    endpoint = anthropic[0]
    ep_params = endpoint.get("supported_parameters")
    if isinstance(ep_params, list) and not required_params.issubset(set(ep_params)):
        raise RouteBlocked("endpoint_parameters_unavailable")
    context = _positive_int(model.get("context_length"))
    top = model.get("top_provider")
    output = _positive_int(top.get("max_completion_tokens")) if isinstance(top, dict) else None
    endpoint_context = _positive_int(endpoint.get("context_length"))
    endpoint_output = _positive_int(endpoint.get("max_completion_tokens"))
    if context is None or output is None:
        raise RouteBlocked("model_limits_unverified")
    context = min(context, endpoint_context) if endpoint_context is not None else context
    output = min(output, endpoint_output) if endpoint_output is not None else output
    if max_output_tokens > output:
        raise RouteBlocked("output_capacity_exceeded")
    serialized_text = json.dumps(request, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":"))
    byte_count = len(serialized_text.encode("utf-8"))
    # Capacity and cost are separate conservative estimates, not tokenizer
    # measurements. The output cap includes Opus's adaptive thinking tokens.
    context_upper = (byte_count * 6 + 4) // 5 + _CHAT_WRAPPER_MARGIN
    estimated_input = max(len(serialized_text), (byte_count * 3 + 3) // 4) + _CHAT_WRAPPER_MARGIN
    if context_upper + max_output_tokens > context:
        raise RouteBlocked("context_capacity_exceeded")
    model_pricing = model.get("pricing")
    endpoint_pricing = endpoint.get("pricing")
    if not isinstance(model_pricing, dict) or not isinstance(endpoint_pricing, dict):
        raise RouteBlocked("pricing_unavailable")
    tiers = [*_tiers(model_pricing), *_tiers(endpoint_pricing)]
    prompt = _highest(tiers, "prompt")
    completion = _highest(tiers, "completion")
    # Anthropic needs explicit cache_control for prompt-cache writes; the
    # canonical request forbids it, so charge ordinary prompt price once.
    request_usd = _highest(tiers, "request", optional=True)
    remaining = key.get("limit_remaining")
    return Route(
        model=MODEL, provider=PROVIDER, workspace_id=workspace_id,
        max_input_tokens=context - max_output_tokens,
        max_output_tokens=output, counted_input_tokens=estimated_input,
        context_bound_tokens=context_upper,
        requested_max_output_tokens=max_output_tokens,
        prompt_usd_per_token=prompt, completion_usd_per_token=completion,
        request_usd=request_usd,
        key_limit_remaining_usd=(_decimal(remaining, "key_remaining")
                                 if remaining is not None else None),
    )
