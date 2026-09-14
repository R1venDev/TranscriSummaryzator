"""One source of truth for planner-to-publication routing."""

TECHNICAL_KINDS = frozenset({
    "trading_rule", "system_rule", "definition", "experimental_result",
    "metric", "design_choice", "constraint", "dependency",
})

RULE_KINDS = frozenset({"trading_rule", "system_rule"})

