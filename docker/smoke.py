#!/usr/bin/env python3
"""Offline Docker smoke with synthetic input and disposable, unique volumes."""
import argparse
import base64
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Already built summary image")
    parser.add_argument("--port", type=int, default=18767)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    origin = f"http://127.0.0.1:{args.port}"
    project = "transcri-smoke-" + uuid.uuid4().hex[:12]
    environment = dict(os.environ, TRANSCRI_IMAGE=args.image, TRANSCRI_PORT=str(args.port),
                       TRANSCRI_PUBLIC_ORIGIN=origin, TRANSCRI_SUMMARY_VERIFIED_WORKSPACE_ID="")
    compose = ["docker", "compose", "-p", project, "-f", str(root / "compose.yaml")]

    def run(arguments, *, input=None):
        result = subprocess.run(compose + arguments, env=environment, cwd=root,
                                input=input, text=True, capture_output=True)
        if result.returncode:
            # Bootstrap output can contain a password; never echo command output.
            raise RuntimeError("Compose operation failed: " + " ".join(arguments[:3]))
        return result.stdout

    def request(path, headers=None):
        try:
            with urllib.request.urlopen(urllib.request.Request(origin + path, headers=headers or {}), timeout=5) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()

    try:
        run(["config", "--quiet"])
        first = run(["run", "--rm", "-T", "--no-deps", "bootstrap"])
        prefix = "Password (shown once): "
        password = next(line[len(prefix):] for line in first.splitlines() if line.startswith(prefix))
        repeated = run(["run", "--rm", "-T", "--no-deps", "bootstrap"])
        assert prefix not in repeated
        run(["up", "-d", "--no-build", "--wait", "--wait-timeout", "50", "app", "scheduler"])
        assert request("/api/status")[0] == 200
        assert request("/api/summary/credentials")[0] == 401
        authorization = "Basic " + base64.b64encode(("admin:" + password).encode()).decode()
        assert request("/api/summary/credentials", {"Authorization": authorization})[0] == 200
        assert request("/api/summary/credentials", {"Authorization": authorization, "Host": "wrong.invalid"})[0] == 403
        synthetic = json.dumps({"source": "synthetic.wav", "duration_seconds": 2,
            "speakers": {"S1": "Test"}, "utterances": [
                {"speaker": "S1", "start": 0, "end": 2, "text": "Synthetic container test."}]})
        imported = json.loads(run(["exec", "-T", "app", "python", "/app/docker/import-transcript.py", "--name", "Container smoke"], input=synthetic))
        assert imported["summary_status"] == "not_started"
        run(["exec", "-T", "app", "python", "-m", "unittest", "discover", "-s", "docker", "-p", "test_*.py"])
        run(["restart", "app", "scheduler"])
        for attempt in range(20):
            try:
                status, raw = request("/api/status")
                if status == 200:
                    break
            except OSError:
                pass
            time.sleep(0.5)
        else:
            raise RuntimeError("App did not recover after restart")
        jobs = json.loads(raw)["jobs"]
        assert len(jobs) == 1 and jobs[0]["id"] == imported["job_id"]
        assert jobs[0]["summary_status"] == "not_started"
        assert request("/api/summary/credentials", {"Authorization": authorization})[0] == 200
        services = {row["Service"]: row for row in
                    (json.loads(line) for line in run(["ps", "--format", "json"]).splitlines())}
        assert set(services) == {"app", "scheduler"}
        assert all(row["State"] == "running" for row in services.values())
        print(json.dumps({"result": "passed", "image": args.image, "checks": [
            "bootstrap-idempotence", "admin-auth", "host-rejection", "offline-tests",
            "transcript-import", "restart-persistence", "no-summary-dispatch"]}))
    finally:
        run(["down", "-v", "--remove-orphans"])


if __name__ == "__main__":
    main()
