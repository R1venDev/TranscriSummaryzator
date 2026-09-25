"""Crash recovery of a sealed Luna generation without another model call."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from summary.luna_v1 import load_source
from summary.luna_v1.publication import publish_document
from summary.luna_v1.tasks import TaskStore
from tests.test_luna_task_api import fake_document


class PublicationRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.output = self.root / "output"
        self.output.mkdir()
        self.transcript = self.output / "transcript.json"
        self.transcript.write_text(json.dumps({
            "source": "01.09.2026 — Test.mkv", "duration_seconds": 8,
            "speakers": {"p1": "А"},
            "utterances": [{"start": 0.4, "end": 7.5, "speaker": "p1",
                            "text": "Предлагаю проверить X или Y, не оба."}],
        }, ensure_ascii=False), encoding="utf-8")
        self.document = fake_document()
        _, self.index, self.source_sha = load_source(self.transcript)
        self.store = TaskStore(self.root / "private" / "tasks.sqlite3")
        self.generation_id = "20260925-010203-" + "a" * 12

    def _publish(self, plan, callback):
        return publish_document(
            document=self.document, source_index=self.index,
            transcript_path=self.transcript, output_dir=self.output,
            semantic_key="synthetic-key", job_id="synthetic-job",
            remote_batch_id="batch_synthetic", credential_id="synthetic-credential",
            prompt_sha256="a" * 64, schema_sha256="b" * 64,
            effective_tasks=plan.effective_tasks, generation_id=self.generation_id,
            before_pointer=callback,
        )

    def test_target_sealed_before_task_commit_is_recovered_idempotently(self):
        plan = self.store.preview_reconcile(self.source_sha, self.document["tasks"])
        with self.assertRaisesRegex(RuntimeError, "crash before task commit"):
            self._publish(plan, lambda: (_ for _ in ()).throw(RuntimeError("crash before task commit")))
        target = self.output / "summary_generations" / self.generation_id
        self.assertTrue((target / "generation_manifest.json").is_file())
        self.assertFalse((self.output / "summary_current.json").exists())
        recovered = self.store.preview_reconcile(self.source_sha, self.document["tasks"])
        generation_id, selected = self._publish(recovered, lambda: self.store.commit_reconcile(recovered))
        self.assertEqual(generation_id, self.generation_id)
        self.assertEqual(selected, target)
        self.assertEqual(json.loads((self.output / "summary_current.json").read_text())["generation_id"], generation_id)
        self.assertEqual(len(self.store.effective_for_sealed(self.source_sha, self.document["tasks"],
            [item["action_id"] for item in json.loads((target / "tasks.json").read_text())])), 1)

    def test_task_commit_before_pointer_swap_recovers_without_duplicate_action(self):
        plan = self.store.preview_reconcile(self.source_sha, self.document["tasks"])
        from summary.luna_v1 import publication
        real_replace = publication.os.replace

        def interrupt_pointer(source, destination):
            if Path(destination) == self.output / "summary_current.json":
                raise OSError("simulated pointer interruption")
            return real_replace(source, destination)

        with patch.object(publication.os, "replace", side_effect=interrupt_pointer):
            with self.assertRaisesRegex(OSError, "simulated pointer interruption"):
                self._publish(plan, lambda: self.store.commit_reconcile(plan))
        self.assertFalse((self.output / "summary_current.json").exists())
        before = json.loads((self.output / "summary_generations" / self.generation_id / "tasks.json").read_text())
        recovered = self.store.preview_reconcile(self.source_sha, self.document["tasks"])
        self._publish(recovered, lambda: self.store.commit_reconcile(recovered))
        after = json.loads((self.output / "summary_generations" / self.generation_id / "tasks.json").read_text())
        self.assertEqual(before, after)
        self.assertEqual(json.loads((self.output / "summary_current.json").read_text())["generation_id"], self.generation_id)
        self.assertEqual(len(self.store.history(before[0]["action_id"])), 0)

    def test_tampered_staged_target_cannot_be_repointed(self):
        plan = self.store.preview_reconcile(self.source_sha, self.document["tasks"])
        with self.assertRaises(RuntimeError):
            self._publish(plan, lambda: (_ for _ in ()).throw(RuntimeError("crash")))
        target = self.output / "summary_generations" / self.generation_id
        (target / "summary.md").write_text("modified", encoding="utf-8")
        recovered = self.store.preview_reconcile(self.source_sha, self.document["tasks"])
        with self.assertRaisesRegex(ValueError, "staged_generation_artifact_changed"):
            self._publish(recovered, lambda: self.store.commit_reconcile(recovered))
        self.assertFalse((self.output / "summary_current.json").exists())

    def test_older_published_generation_survives_crash_until_recovery(self):
        self.generation_id = "20260925-000000-" + "b" * 12
        old_plan = self.store.preview_reconcile(self.source_sha, self.document["tasks"])
        self._publish(old_plan, lambda: self.store.commit_reconcile(old_plan))
        old_id = self.generation_id
        self.generation_id = "20260925-010203-" + "a" * 12
        new_plan = self.store.preview_reconcile(self.source_sha, self.document["tasks"])
        with self.assertRaisesRegex(RuntimeError, "simulated commit interruption"):
            self._publish(new_plan, lambda: (_ for _ in ()).throw(RuntimeError("simulated commit interruption")))
        self.assertEqual(json.loads((self.output / "summary_current.json").read_text())["generation_id"], old_id)
        recovered = self.store.preview_reconcile(self.source_sha, self.document["tasks"])
        self._publish(recovered, lambda: self.store.commit_reconcile(recovered))
        self.assertEqual(json.loads((self.output / "summary_current.json").read_text())["generation_id"], self.generation_id)
        self.assertTrue((self.output / "summary_generations" / old_id / "summary.md").is_file())


if __name__ == "__main__":
    unittest.main()
