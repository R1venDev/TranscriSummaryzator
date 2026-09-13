"""Versioned artifact manifests and deterministic compatibility gates."""
from __future__ import annotations
from datetime import datetime, timezone
from contracts import SCHEMA_VERSIONS


def manifest(artifact, schema, producer_stage, producer_version, input_hashes=None, model_versions=None):
    if schema not in SCHEMA_VERSIONS:
        raise ValueError(f"unknown schema: {schema}")
    return {"artifact": artifact, "schema": schema, "schema_version": SCHEMA_VERSIONS[schema], "producer_stage": producer_stage, "producer_version": producer_version, "input_hashes": input_hashes or {}, "model_versions": model_versions or {}, "created_at": datetime.now(timezone.utc).isoformat()}


def require_compatible(value, schema):
    expected = SCHEMA_VERSIONS[schema]
    if value.get("schema") != schema or value.get("schema_version") != expected:
        raise ValueError(f"incompatible {schema}: expected v{expected}; explicit migration or cache invalidation required")
    return value
