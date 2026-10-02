"""Offline checks for registry publication identity guards."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location("publish", Path(__file__).with_name("publish.py"))
publish = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publish)


class PublishIdentityTests(unittest.TestCase):
    def environment(self, **extra):
        return mock.patch.dict(os.environ, {
            "GITHUB_REPOSITORY": "R1venDev/TranscriSummaryzator",
            "GITHUB_SHA": "a" * 40, "GITHUB_REF": "refs/heads/main",
            "IMAGE_VARIANT": "speech", "RELEASE_VERSION": "2026.10.02", **extra})

    def test_existing_commit_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            with self.environment(GITHUB_OUTPUT=str(output)), \
                 mock.patch.object(publish.sys, "argv", ["publish.py", "prepare"]), \
                 mock.patch.object(publish, "manifest", return_value="sha256:" + "1" * 64), \
                 mock.patch.object(publish.subprocess, "run") as command:
                publish.main()
            self.assertIn("exists=true", output.read_text())
            command.assert_not_called()

    def test_release_conflict_blocks_every_alias_write(self):
        with self.environment(), \
             mock.patch.object(publish.sys, "argv", ["publish.py", "validate"]), \
             mock.patch.object(publish, "manifest", side_effect=["sha256:" + "1" * 64, "sha256:" + "2" * 64]), \
             mock.patch.object(publish.subprocess, "run") as command:
            with self.assertRaisesRegex(RuntimeError, "already names another image"):
                publish.main()
            command.assert_not_called()


if __name__ == "__main__":
    unittest.main()
