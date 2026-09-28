"""The release switch gates new private Batch submissions at the real CLI."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class WorkerPolicyTests(unittest.TestCase):
    def test_new_policy_flag_off_blocks_before_source_or_network(self):
        with tempfile.TemporaryDirectory() as name:
            base = Path(name)
            output = base / "output"
            output.mkdir()
            env = dict(os.environ)
            env.pop("TRANSCRI_LUNA_SOURCE_FIRST_ENABLED", None)
            env["PYTHONPATH"] = str(ROOT)
            completed = subprocess.run([
                sys.executable, str(ROOT / "scripts" / "luna_summary_worker.py"),
                "submit", "--transcript", str(base / "missing-transcript.json"),
                "--output", str(output), "--private-root", str(base / "private"),
            ], env=env, capture_output=True, text=True, timeout=20, check=False)
            self.assertEqual(completed.returncode, 2, completed.stderr)
            status = json.loads((output / "summary_luna_attempt.json").read_text())
            self.assertEqual(status["status"], "policy_disabled")
            self.assertFalse((base / "private").exists())


if __name__ == "__main__":
    unittest.main()
