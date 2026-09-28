"""Publication and identity boundaries for the source-first Batch path."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from summary.luna_v1 import load_source
from summary.luna_v1.publication import publish_document
from summary.luna_v1.render import render_document
from summary.luna_v1.tasks import ReconciliationConflict, TaskStore
from tests.test_luna_task_api import fake_document
from tests.test_luna_tasks import task


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sidecar() -> dict:
    return {
        "schema_version": "luna_review_sidecar_v1",
        "execution_status": "terminal",
        "review_status": "review_completed",
        "source_scope_ids": ["U00001"],
        "checked_ids": ["U00001"],
        "unreviewed_ids": [],
        "accepted_bundle_ids": [],
        "rejected_bundle_ids": [],
        "unresolved_bundle_ids": [],
        "source_ambiguities": [],
        "model_disagreements": [],
        "technical_failures": [],
        "billed_cost_microusd": 1000,
        "held_cost_microusd": 0,
        "unknown_bill_ids": [],
        "generation_lineage": ["synthetic-writer"],
    }


class BatchPublicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.output = self.root / "output"
        self.output.mkdir()
        self.source = self.output / "transcript.json"
        self.source.write_text(json.dumps({
            "source": "01.09.2026 — Test.mkv", "duration_seconds": 8,
            "speakers": {"p1": "А"},
            "utterances": [
                {"start": 0.4, "end": 3.0, "speaker": "p1",
                 "text": "Предлагаю проверить X или Y, не оба."},
                {"start": 0.4, "end": 7.5, "speaker": "p1",
                 "text": "Поздняя реплика с тем же началом."},
            ],
        }, ensure_ascii=False), encoding="utf-8")
        _, self.index, self.source_sha = load_source(self.source)
        self.document = fake_document()
        self.generation_id = "20260928-010203-" + "a" * 12

    def _publish(self, *, sidecar=None, status=None, before_pointer=None):
        return publish_document(
            document=self.document, source_index=self.index,
            transcript_path=self.source, output_dir=self.output,
            semantic_key="synthetic-key", job_id="synthetic-job",
            remote_batch_id="batch-synthetic", credential_id="synthetic-key",
            prompt_sha256="a" * 64, schema_sha256="b" * 64,
            generation_id=self.generation_id,
            quality_review={"status": status, "unresolved_count": 0,
                            "coverage_warning_count": 0} if status else None,
            review_sidecar=sidecar, before_pointer=before_pointer,
        )

    def test_u_anchors_are_unique_when_two_utterances_share_a_time(self):
        document = deepcopy(self.document)
        document["main"][0]["source_ids"].append("U00002")
        rendered = render_document(document, self.index)
        for name in ("transcript.html", "summary.html"):
            text = rendered[name]
            self.assertEqual(text.count('id="u-U00001"'), 1)
            self.assertEqual(text.count('id="u-U00002"'), 1)
            self.assertEqual(text.count('id="t-400"'), 1)
        self.assertIn('href="transcript.html#u-U00001"', rendered["summary.fragment.html"])
        self.assertIn('href="#u-U00001"', rendered["summary.html"])

    def test_source_first_notices_state_limits_honestly(self):
        expected = {
            "source_first_checked": "не гарантирует полноту и точность",
            "review_incomplete": "Часть пунктов или реплик осталась без сверки",
            "unresolved": "Полнота и точность не подтверждены",
        }
        for status, phrase in expected.items():
            with self.subTest(status=status):
                rendered = render_document(self.document, self.index,
                    quality_review={"status": status, "unresolved_count": 1,
                                    "coverage_warning_count": 0})
                self.assertIn(phrase, rendered["summary.fragment.html"])

    def test_review_sidecar_is_sealed_before_pointer_and_recovers(self):
        report = _sidecar()
        def interrupt():
            generation = self.output / "summary_generations" / self.generation_id
            self.assertTrue((generation / "review_sidecar.json").is_file())
            self.assertFalse((self.output / "summary_current.json").exists())
            raise RuntimeError("before pointer")

        with self.assertRaisesRegex(RuntimeError, "before pointer"):
            self._publish(sidecar=report, status="source_first_checked",
                          before_pointer=interrupt)
        generation = self.output / "summary_generations" / self.generation_id
        manifest = json.loads((generation / "generation_manifest.json").read_text())
        sidecar = json.loads((generation / "review_sidecar.json").read_text())
        self.assertEqual(sidecar["generation_id"], self.generation_id)
        self.assertEqual(sidecar["source_revision"], self.source_sha)
        self.assertEqual(set(x["name"] for x in sidecar["artifact_hashes"]),
                         set(manifest["artifact_sha256"]) - {"review_sidecar.json"})
        self.assertEqual(manifest["artifact_sha256"]["review_sidecar.json"],
                         _sha(generation / "review_sidecar.json"))
        for item in sidecar["artifact_hashes"]:
            self.assertEqual(item["sha256"], _sha(generation / item["name"]))
        self._publish(sidecar=report, status="source_first_checked")
        pointer = json.loads((self.output / "summary_current.json").read_text())
        self.assertEqual(pointer["generation_id"], self.generation_id)
        from pipeline import current_summary_output
        self.assertEqual(current_summary_output(self.output), generation)

    def test_inconsistent_or_pending_review_cannot_publish(self):
        for review_status, quality_status in (
            ("review_completed", "review_incomplete"),
            ("review_pending", "source_first_checked"),
        ):
            with self.subTest(review_status=review_status):
                sidecar = _sidecar()
                sidecar["review_status"] = review_status
                with self.assertRaisesRegex(ValueError, "review_sidecar_quality_status_mismatch"):
                    self._publish(sidecar=sidecar, status=quality_status)
                self.assertFalse((self.output / "summary_current.json").exists())


class SourceActionIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = TaskStore(Path(self.tmp.name) / "private" / "tasks.sqlite3")
        self.source_sha = hashlib.sha256(b"synthetic source revision").hexdigest()
        self.first = task("Проверить X или Y", "Проверить один из вариантов после выбора.", "U00001")

    def test_unedited_anchor_keeps_id_across_rewording(self):
        first_id = self.store.reconcile(self.source_sha, [self.first])[0]["action_id"]
        renamed = deepcopy(self.first)
        renamed["title"] = "Исследовать один вариант"
        renamed["description"] = "Исследовать выбранный вариант в следующем цикле."
        self.assertEqual(self.store.reconcile(self.source_sha, [renamed])[0]["action_id"], first_id)

    def test_renderer_preview_id_agrees_with_persistent_source_anchor(self):
        index = {
            "source_sha256": self.source_sha, "meeting_date": None,
            "participants": ["А"], "unattributed_speech": False,
            "by_id": {"U00001": {"start_ms": 0, "end_ms": 1000,
                                 "speaker": "А", "text": "Проверить X или Y."}},
        }
        document = fake_document()
        self.assertEqual(
            render_document(document, index)["tasks.json"][0]["action_id"],
            self.store.reconcile(self.source_sha, document["tasks"])[0]["action_id"],
        )

    def test_title_alone_cannot_move_a_manual_override_to_new_source_action(self):
        initial = self.store.reconcile(self.source_sha, [self.first])[0]
        self.store.update(initial["action_id"], 0, {"assignee": "А"}, "admin")
        other = deepcopy(self.first)
        other["source_ids"] = ["U00002"]
        other["field_sources"]["action"] = ["U00002"]
        other["field_sources"]["discussion_status"] = ["U00002"]
        with self.assertRaises(ReconciliationConflict):
            self.store.reconcile(self.source_sha, [other])
        self.assertEqual(self.store.get(initial["action_id"])["assignee"], "А")

    def test_edited_action_rewording_at_same_anchor_is_an_explicit_conflict(self):
        initial = self.store.reconcile(self.source_sha, [self.first])[0]
        self.store.update(initial["action_id"], 0, {"assignee": "А"}, "admin")
        changed = deepcopy(self.first)
        changed["description"] = "Проверить оба варианта."
        with self.assertRaisesRegex(ReconciliationConflict, "edited action content changed"):
            self.store.reconcile(self.source_sha, [changed])
        self.assertEqual(self.store.get(initial["action_id"])["assignee"], "А")

    def test_two_actions_on_one_utterance_keep_distinct_edits_when_reordered(self):
        second = task("Подготовить Y", "Подготовить только вариант Y.", "U00001")
        first, other = self.store.reconcile(self.source_sha, [self.first, second])
        self.store.update(first["action_id"], 0, {"assignee": "А"}, "admin")
        reordered = self.store.reconcile(self.source_sha, [second, self.first])
        self.assertEqual(reordered[0]["action_id"], other["action_id"])
        self.assertEqual(reordered[1]["action_id"], first["action_id"])
        self.assertEqual(reordered[1]["assignee"], "А")
        self.assertIsNone(reordered[0]["assignee"])


if __name__ == "__main__":
    unittest.main()
