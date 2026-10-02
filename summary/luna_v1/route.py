"""Free route and price preflight before a private paid Batch submission."""
from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation, ROUND_UP
import os

from .batch import MODEL, PROVIDER, SUBMIT_MODEL, BatchClient
from .ledger import JOB_CAP_MICROUSD, WEEK_CAP_MICROUSD


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
    # Additive metadata for the source-first policy. Legacy Route instances
    # retain their existing MODEL while new jobs record both submit and
    # resolved Batch identities.
    batch_endpoint_model: str = MODEL
    provider_endpoint_tag: str = PROVIDER
    supports_explicit_cache: bool = False

    def reserve_microusd(self, serialized_request: bytes, *,
                         max_completion_tokens: int | None = None,
                         authorized_job_cap_microusd: int = JOB_CAP_MICROUSD) -> int:
        # A deliberately loose source upper bound: UTF-8 bytes plus 20% for
        # transport/tokenizer overhead. No promised cache hit is subtracted.
        output_cap = self.max_completion_tokens if max_completion_tokens is None else max_completion_tokens
        if type(output_cap) is not int or not 1 <= output_cap <= self.max_completion_tokens:
            raise RouteBlocked("invalid_output_cap")
        input_upper = (len(serialized_request) * 6 + 4) // 5
        if input_upper + output_cap > self.max_context_tokens:
            raise RouteBlocked("context_capacity_unverified")
        cost = (Decimal(input_upper) * max(self.prompt_usd_per_token,
                                           self.cache_write_usd_per_token)
                + Decimal(output_cap) * self.completion_usd_per_token
                + self.request_usd)
        reserve = int((cost * 1_000_000).to_integral_value(rounding=ROUND_UP))
        if (type(authorized_job_cap_microusd) is not int
                or not 0 < authorized_job_cap_microusd <= WEEK_CAP_MICROUSD
                or reserve < 1 or reserve > authorized_job_cap_microusd):
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
    if context < 1 or completion_cap < 1:
        raise RouteBlocked("context_capacity_unverified")
    remaining = key.get("limit_remaining")
    return Route(
        model=MODEL, provider="openai", workspace_id=workspace_id,
        max_context_tokens=context,
        max_completion_tokens=completion_cap,
        prompt_usd_per_token=highest_price("prompt"),
        completion_usd_per_token=highest_price("completion"),
        # Luna can bill automatic cache writes above the plain prompt rate.
        # A missing cache-write tariff cannot yield a valid upper reserve.
        cache_write_usd_per_token=highest_price("input_cache_write", "cache_write"),
        request_usd=max(_decimal(tier.get("request", "0"), "request_price") for tier in tiers),
        key_limit_remaining_usd=_decimal(remaining, "key_remaining") if remaining is not None else None,
    )


def verify_source_first_batch_route(client: BatchClient, *,
                                    require_cache_disabled: bool = False) -> Route:
    """Verify the single OpenAI :batch endpoint used by the new policy.

    /models/user establishes account eligibility, while the variant's
    /endpoints response establishes provider identity, parameters and prices.
    The documented endpoint response is data={id,endpoints:[...]}. If any
    required fact is unavailable, private dispatch remains blocked.
    """
    legacy = verify_batch_route(client)
    data = client.model_endpoints().body.get("data")
    if (not isinstance(data, dict) or data.get("id") not in {MODEL, SUBMIT_MODEL}
            or not isinstance(data.get("endpoints"), list)):
        raise RouteBlocked("batch_endpoints_unavailable")
    endpoints = data["endpoints"]
    eligible = [entry for entry in endpoints if isinstance(entry, dict)
                and entry.get("tag") == PROVIDER
                and entry.get("provider_name") == "OpenAI"
                and entry.get("model_id") in {MODEL, SUBMIT_MODEL}]
    if len(eligible) != 1:
        raise RouteBlocked("openai_batch_endpoint_not_unique")
    endpoint = eligible[0]
    parameters = endpoint.get("supported_parameters")
    if not isinstance(parameters, list) or not all(isinstance(p, str) for p in parameters):
        raise RouteBlocked("endpoint_capabilities_unavailable")
    # This provider's advertised Chat skin currently lists max_tokens,
    # despite the gateway's newer generic max_completion_tokens alias.
    # Send only the cap that this specific Batch endpoint declares.
    required = {"response_format", "reasoning", "max_tokens"}
    if not required.issubset(parameters):
        raise RouteBlocked("endpoint_chat_parameters_unavailable")
    supports_cache = "prompt_cache_options" in parameters
    if require_cache_disabled and not supports_cache:
        raise RouteBlocked("explicit_cache_control_unavailable")
    context = endpoint.get("context_length")
    completion_cap = endpoint.get("max_completion_tokens")
    try:
        context = int(context)
        completion_cap = int(completion_cap)
    except (TypeError, ValueError):
        raise RouteBlocked("endpoint_capacity_unavailable") from None
    if context < 1 or completion_cap < 1:
        raise RouteBlocked("endpoint_capacity_unavailable")
    endpoint_pricing = endpoint.get("pricing")
    if not isinstance(endpoint_pricing, dict):
        raise RouteBlocked("endpoint_pricing_unavailable")
    endpoint_overrides = endpoint_pricing.get("overrides", [])
    if (not isinstance(endpoint_overrides, list)
            or any(not isinstance(tier, dict) for tier in endpoint_overrides)):
        raise RouteBlocked("endpoint_pricing_overrides_unavailable")
    # The legacy route already holds the highest model-level base and tier
    # prices. Keep those bounds if endpoint metadata quotes a lower tariff.
    tiers = [endpoint_pricing, *endpoint_overrides]

    def highest_price(field: str, *aliases: str) -> Decimal:
        values = []
        for tier in tiers:
            value = next((tier[name] for name in (field, *aliases) if name in tier), None)
            values.append(_decimal(value, field + "_price"))
        return max(values)

    request_usd = max(legacy.request_usd, *(
        _decimal(tier.get("request", "0"), "request_price") for tier in tiers))
    return replace(
        legacy, model=SUBMIT_MODEL, max_context_tokens=min(legacy.max_context_tokens, context),
        max_completion_tokens=min(legacy.max_completion_tokens, completion_cap),
        prompt_usd_per_token=max(legacy.prompt_usd_per_token, highest_price("prompt")),
        completion_usd_per_token=max(legacy.completion_usd_per_token,
                                     highest_price("completion")),
        cache_write_usd_per_token=max(legacy.cache_write_usd_per_token,
                                      highest_price("input_cache_write", "cache_write")),
        request_usd=request_usd, batch_endpoint_model=MODEL,
        provider_endpoint_tag=PROVIDER, supports_explicit_cache=supports_cache,
    )
