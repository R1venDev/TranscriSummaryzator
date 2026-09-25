"""Free route and price preflight before a private paid Batch submission."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_UP
import os

from .batch import MODEL, BatchClient
from .ledger import JOB_CAP_MICROUSD


MAX_COMPLETION_TOKENS = 32_000


class RouteBlocked(RuntimeError):
    pass


@dataclass(frozen=True)
class Route:
    model: str
    provider: str
    workspace_id: str
    max_context_tokens: int
    max_completion_tokens: int
    prompt_usd_per_token: Decimal
    completion_usd_per_token: Decimal
    cache_write_usd_per_token: Decimal
    request_usd: Decimal
    key_limit_remaining_usd: Decimal | None

    def reserve_microusd(self, serialized_request: bytes) -> int:
        # A deliberately loose source upper bound: UTF-8 bytes plus 20% for
        # transport/tokenizer overhead. No promised cache hit is subtracted.
        input_upper = (len(serialized_request) * 6 + 4) // 5
        if input_upper + self.max_completion_tokens > self.max_context_tokens:
            raise RouteBlocked("context_capacity_unverified")
        cost = (Decimal(input_upper) * max(self.prompt_usd_per_token,
                                           self.cache_write_usd_per_token)
                + Decimal(self.max_completion_tokens) * self.completion_usd_per_token
                + self.request_usd)
        reserve = int((cost * 1_000_000).to_integral_value(rounding=ROUND_UP))
        if reserve < 1 or reserve > JOB_CAP_MICROUSD:
            raise RouteBlocked("job_budget_exceeded")
        if self.key_limit_remaining_usd is not None and cost > self.key_limit_remaining_usd:
            raise RouteBlocked("key_budget_insufficient")
        return reserve


def _decimal(value, field: str) -> Decimal:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError):
        raise RouteBlocked(f"missing_or_invalid_{field}") from None
    if not number.is_finite() or number < 0:
        raise RouteBlocked(f"missing_or_invalid_{field}")
    return number


def verify_batch_route(client: BatchClient) -> Route:
    """Require the exact Batch variant in this key's filtered catalog.

    GET /key alone establishes that a key exists, not that its route and
    structured-output parameters are eligible. POST remains authoritative;
    its failures still produce durable attempt evidence.
    """
    key = client.current_key().body.get("data")
    allowed = client.models_for_key().body.get("data")
    details = client.model_details().body.get("data")
    if not isinstance(key, dict) or not isinstance(allowed, list) or not isinstance(details, dict):
        raise RouteBlocked("metadata_unavailable")
    if key.get("is_management_key") or key.get("is_provisioning_key"):
        raise RouteBlocked("not_inference_key")
    workspace_id = key.get("workspace_id")
    if not isinstance(workspace_id, str) or not workspace_id:
        raise RouteBlocked("workspace_scope_unverified")
    # An inference key cannot enumerate workspace BYOK or I/O logging policy.
    # The operator must verify those account settings for this exact workspace
    # before private Batch dispatch; a key from another workspace is blocked.
    if os.environ.get("TRANSCRI_SUMMARY_VERIFIED_WORKSPACE_ID") != workspace_id:
        raise RouteBlocked("account_policy_unverified_for_workspace")
    if not any(isinstance(item, dict) and item.get("id") == MODEL for item in allowed):
        raise RouteBlocked("model_not_allowed_for_key")
    if details.get("id") != MODEL:
        raise RouteBlocked("batch_model_identity_changed")
    parameters = set(details.get("supported_parameters") or [])
    if "response_format" not in parameters:
        raise RouteBlocked("structured_output_unavailable")
    pricing = details.get("pricing")
    if not isinstance(pricing, dict):
        raise RouteBlocked("pricing_unavailable")
    overrides = pricing.get("overrides", [])
    if not isinstance(overrides, list) or any(not isinstance(tier, dict) for tier in overrides):
        raise RouteBlocked("pricing_overrides_unavailable")
    # Reserve against the highest published tier even when the current input
    # is below its threshold. That keeps the cap conservative if token counts
    # or the catalog threshold interpretation differ from the byte bound.
    tiers = [pricing, *overrides]

    def highest_price(field: str, *aliases: str) -> Decimal:
        names = (field, *aliases)
        values = []
        for tier in tiers:
            value = next((tier[name] for name in names if name in tier), None)
            values.append(_decimal(value, field + "_price"))
        return max(values)
    context = details.get("context_length")
    provider = details.get("top_provider") or {}
    completion_cap = provider.get("max_completion_tokens") or 0
    try:
        context = int(context)
        completion_cap = int(completion_cap)
    except (TypeError, ValueError):
        raise RouteBlocked("context_capacity_unverified") from None
    if context < 1 or completion_cap < MAX_COMPLETION_TOKENS:
        raise RouteBlocked("context_capacity_unverified")
    remaining = key.get("limit_remaining")
    return Route(
        model=MODEL, provider="openai", workspace_id=workspace_id,
        max_context_tokens=context,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        prompt_usd_per_token=highest_price("prompt"),
        completion_usd_per_token=highest_price("completion"),
        # Luna can bill automatic cache writes above the plain prompt rate.
        # A missing cache-write tariff cannot yield a valid upper reserve.
        cache_write_usd_per_token=highest_price("input_cache_write", "cache_write"),
        request_usd=max(_decimal(tier.get("request", "0"), "request_price") for tier in tiers),
        key_limit_remaining_usd=_decimal(remaining, "key_remaining") if remaining is not None else None,
    )
