"""Read-only model/token preflight and conservative Gemini Batch accounting.

Prices are Google's published paid Batch tariff for gemini-3.7-flash through
2026-12-31: https://ai.google.dev/gemini-api/docs/pricing .  The reservation
does not assume a cache hit.  Explicit cache is disabled in batch.py because
its token-hour storage charge may exceed savings for dependent Batch calls.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_UP
from typing import Any

from .batch import MODEL, MODEL_RESOURCE, BatchClient, canonical_request


INPUT_USD_PER_MILLION = Decimal("0.375")
OUTPUT_USD_PER_MILLION = Decimal("1.875")  # Includes thinking tokens.
PRICE_VALID_THROUGH = date(2026, 12, 31)
RESERVE_SAFETY = Decimal("1.2")


class RouteBlocked(RuntimeError):
    """Safe reason code for a preflight failure, without source text."""


def _nonnegative_int(value: object) -> int | None:
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        return int(value)
    return None


def _microusd(cost: Decimal) -> int:
    return int((cost * Decimal(1_000_000)).to_integral_value(rounding=ROUND_UP))


def estimate_usage_cost_microusd(usage: dict[str, Any] | None) -> int | None:
    """Conservative token-priced charge estimate, or unknown.

    Google returns no invoice amount in GenerateContentResponse.  We count
    *all* prompt tokens at ordinary Batch input price, even if a cache hit is
    reported.  This deliberately overstates rather than silently deducting an
    unverified cache discount.  Unknown usage remains an unknown charge.
    """
    if not isinstance(usage, dict):
        return None
    prompt = _nonnegative_int(usage.get("promptTokenCount"))
    total = _nonnegative_int(usage.get("totalTokenCount"))
    candidates = _nonnegative_int(usage.get("candidatesTokenCount", 0))
    thoughts = _nonnegative_int(usage.get("thoughtsTokenCount", 0))
    cached = _nonnegative_int(usage.get("cachedContentTokenCount", 0))
    if prompt is None or prompt < 1 or candidates is None or thoughts is None or cached is None or cached > prompt:
        return None
    if total is None and "candidatesTokenCount" not in usage:
        return None
    if total is not None and total < prompt:
        return None
    # totalTokenCount can include otherwise unnamed billable output tokens.
    output = max(candidates + thoughts, (total - prompt) if total is not None else 0)
    cost = (Decimal(prompt) * INPUT_USD_PER_MILLION
            + Decimal(output) * OUTPUT_USD_PER_MILLION) / Decimal(1_000_000)
    return _microusd(cost)


@dataclass(frozen=True)
class Route:
    model: str
    model_version: str | None
    provider: str
    max_input_tokens: int
    max_output_tokens: int
    counted_input_tokens: int
    requested_max_output_tokens: int

    @property
    def input_tokens(self) -> int:
        return self.counted_input_tokens

    def reserve_microusd(self) -> int:
        """Hold max-output cost plus 20% of the whole estimate before POST."""
        cost = (Decimal(self.counted_input_tokens) * INPUT_USD_PER_MILLION
                + Decimal(self.requested_max_output_tokens) * OUTPUT_USD_PER_MILLION)
        return _microusd(cost * RESERVE_SAFETY / Decimal(1_000_000))

    def actual_cost_microusd(self, usage: dict[str, Any] | None) -> int | None:
        return estimate_usage_cost_microusd(usage)


def verify_batch_route(
    client: BatchClient,
    request_body: dict[str, Any],
    *,
    max_output_tokens: int,
    as_of: date | None = None,
) -> Route:
    """Free metadata/countTokens preflight for the exact Batch item body.

    These calls do not generate text.  countTokens still transfers the prompt
    to Google, so the caller must apply its credential/privacy policy first.
    """
    if (as_of or date.today()) > PRICE_VALID_THROUGH:
        raise RouteBlocked("gemini_batch_price_review_required")
    request = canonical_request(request_body)
    if type(max_output_tokens) is not int or request["generationConfig"]["maxOutputTokens"] != max_output_tokens:
        raise RouteBlocked("output_cap_mismatch")

    model = client.model_details().body
    if not isinstance(model, dict) or model.get("name") != MODEL_RESOURCE:
        raise RouteBlocked("model_identity_unverified")
    input_limit = _nonnegative_int(model.get("inputTokenLimit"))
    output_limit = _nonnegative_int(model.get("outputTokenLimit"))
    if input_limit is None or output_limit is None or input_limit < 1 or output_limit < 1:
        raise RouteBlocked("model_limits_unverified")
    methods = model.get("supportedGenerationMethods")
    if isinstance(methods, list) and "generateContent" not in methods:
        raise RouteBlocked("generate_content_unavailable")
    if max_output_tokens < 1 or max_output_tokens > output_limit:
        raise RouteBlocked("output_capacity_exceeded")

    counted = client.count_tokens(request).body
    input_tokens = _nonnegative_int(counted.get("totalTokens") if isinstance(counted, dict) else None)
    if input_tokens is None or input_tokens < 1:
        raise RouteBlocked("token_count_unavailable")
    # 20% headroom covers a small accounting mismatch without changing the
    # actual prompt.  Gemini's input and output limits are separate.
    if Decimal(input_tokens) * RESERVE_SAFETY > input_limit:
        raise RouteBlocked("input_capacity_exceeded")
    version = model.get("version")
    return Route(
        model=MODEL,
        model_version=version if isinstance(version, str) else None,
        provider="google",
        max_input_tokens=input_limit,
        max_output_tokens=output_limit,
        counted_input_tokens=input_tokens,
        requested_max_output_tokens=max_output_tokens,
    )
