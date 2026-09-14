"""Immutable model identities used by cache keys and run manifests."""
from __future__ import annotations
import hashlib, json

def model_identity(repository, revision, digest, runtime, quantization=None, config=None):
    if not revision or not digest: raise ValueError("model revision and digest must be pinned")
    value = {"repository": repository, "revision": revision, "digest": digest, "runtime": runtime, "quantization": quantization, "config_hash": hashlib.sha256(json.dumps(config or {}, sort_keys=True).encode()).hexdigest()}
    value["identity_digest"] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    return value

def cache_identity(stage_version, inputs, models):
    payload = {"stage_version": stage_version, "inputs": inputs, "models": [x["identity_digest"] for x in models]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
