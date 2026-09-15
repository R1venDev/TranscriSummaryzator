"""Release checks for public generations, links and stuck stage processes."""
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

from pipeline import current_summary_output, run_command
from scripts.summary_worker import build_public_document, render_public_document, time_link
from summary.verifier import verify_public_document


class GenerationTests(unittest.TestCase):
    def test_only_committed_generation_is_public(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            generation = "20260915-120000-abcdef123456"
            pending = base / "summary_generations" / (generation + ".pending")
            pending.mkdir(parents=True)
            (pending / "summary.md").write_text("partial", encoding="utf-8")
            self.assertIsNone(current_summary_output(base))
            final = pending.with_name(generation)
            pending.rename(final)
            (final / "generation_manifest.json").write_text(json.dumps({
                "generation_id": generation, "artifact_sha256": {"summary.md": "hash"}}), encoding="utf-8")
            (base / "summary_current.json").write_text(json.dumps({"generation_id": generation}), encoding="utf-8")
            self.assertEqual(current_summary_output(base), final)
            (base / "summary_current.json").write_text(json.dumps({"generation_id": "../unsafe"}), encoding="utf-8")
            self.assertIsNone(current_summary_output(base))

    def test_subprocess_partial_line_cannot_stall_watchdog(self):
        with tempfile.TemporaryDirectory() as directory:
            start = time.monotonic()
            command = [sys.executable, "-c", "import sys,time;sys.stdout.write('partial');sys.stdout.flush();time.sleep(8)"]
            with self.assertRaises(TimeoutError):
                run_command(command, Path(directory) / "stage.log", deadline_seconds=2, idle_seconds=.5)
            self.assertLess(time.monotonic() - start, 4)

    def test_portable_transcript_link_without_private_server_address(self):
        self.assertEqual(time_link(10.125, job_id=9), "[00:00:10](transcript.html#t-10125)")

    def test_final_document_mutations_are_rejected(self):
        item = {"public_id": "PI00001", "section": "overview", "text": "Нужно проверить качество сигналов.",
                "claim_ids": ["C1"], "evidence_ids": ["U1"], "start": 10.125}
        document = build_public_document([item], {"job_id": 9, "project": "Проект"})
        artifact = render_public_document(document)
        self.assertTrue(verify_public_document(document, artifact, [item])["passed"])
        invented = json.loads(json.dumps(document))
        invented["title"]["text"] = "Подписана сделка без согласования"
        self.assertFalse(verify_public_document(invented, artifact, [item])["passed"])
        invented = json.loads(json.dumps(document))
        invented["overview"][0]["text"] += " Сервер уничтожен."
        self.assertFalse(verify_public_document(invented, artifact, [item])["passed"])


if __name__ == "__main__":
    unittest.main()
