"""Local HTTP liveness probe; never calls a paid API or inspects secrets."""
import json
import urllib.request

with urllib.request.urlopen("http://127.0.0.1:8765/api/status", timeout=3) as response:
    payload = json.load(response)
if not isinstance(payload, dict) or not isinstance(payload.get("jobs"), list):
    raise SystemExit(1)
