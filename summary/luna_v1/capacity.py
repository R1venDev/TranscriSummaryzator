"""Output capacity from verified endpoint, real input and authorized dollars.

There is no stage token constant. Reasoning and visible output share this
capacity. Future calls retain a share of the money; unused shares are reclaimed
after their actual bill arrives. This is an upper reservation, not a target.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_UP
import json

from .route import Route, RouteBlocked


CAPACITY_POLICY = "endpoint_budget_capacity_v1"


def input_upper(body: dict) -> int:
    raw = json.dumps(body, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return (len(raw) * 6 + 4) // 5


def input_cost(route: Route, tokens: int) -> int:
    charge = Decimal(tokens) * max(route.prompt_usd_per_token,
                                   route.cache_write_usd_per_token)
    return int((charge * 1_000_000).to_integral_value(rounding=ROUND_UP))


def allocate_outputs(route: Route, bodies: list[dict], *, available_microusd: int,
                     future_items: int, future_input_tokens: int) -> tuple[list[dict], dict]:
    """Price actual bodies and divide remaining output money across open items.

    Future input is a source-based planning floor, not a promise about an
    unknown report. Every later wave prices its real payload again. The ledger
    remains the atomic money gate. Zero pricing still respects context/model.
    """
    if not bodies or future_items < 0 or future_input_tokens < 0:
        raise RouteBlocked("invalid_capacity_plan")
    # Include the eventual parameter with the endpoint's largest digit width.
    priced = [{**body, "max_tokens": route.max_completion_tokens} for body in bodies]
    inputs = [input_upper(body) for body in priced]
    request_cost = int((route.request_usd * 1_000_000).to_integral_value(rounding=ROUND_UP))
    input_holds = [input_cost(route, count) + request_cost for count in inputs]
    future_hold = input_cost(route, future_input_tokens) + future_items * request_cost
    output_money = available_microusd - sum(input_holds) - future_hold
    if output_money <= 0:
        raise RouteBlocked("job_budget_exceeded")
    output_share = output_money // (len(bodies) + future_items)
    rate = route.completion_usd_per_token * 1_000_000
    money_cap = int(Decimal(output_share) / rate) if rate > 0 else route.max_completion_tokens
    caps = [min(route.max_completion_tokens, route.max_context_tokens - count, money_cap)
            for count in inputs]
    if any(cap < 1 for cap in caps):
        raise RouteBlocked("context_capacity_unverified" if any(
            route.max_context_tokens <= count for count in inputs) else "job_budget_exceeded")
    result = [{**body, "max_tokens": cap} for body, cap in zip(bodies, caps, strict=True)]
    reserves = [route.reserve_microusd(json.dumps(body, ensure_ascii=False,
        sort_keys=True, separators=(",", ":")).encode("utf-8"),
        max_completion_tokens=body["max_tokens"],
        authorized_job_cap_microusd=available_microusd) for body in result]
    if sum(reserves) > available_microusd:
        raise RouteBlocked("job_budget_exceeded")
    return result, {"policy": CAPACITY_POLICY, "available_microusd": available_microusd,
        "input_upper_tokens": inputs, "future_input_hold_microusd": future_hold,
        "future_items": future_items, "endpoint_output_limit": route.max_completion_tokens,
        "output_caps": caps, "reserved_microusd": reserves}
