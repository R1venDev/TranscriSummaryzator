"""Plane integration tests: synthetic secrets, no external writes."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import uuid

from cryptography.fernet import Fernet
from summary.plane import PlaneClient, PlaneError, PlaneHTTPError, PlanePreflightError, PlaneStore, safe_html

PROJECT = "b99da5a6-d63f-4375-bccb-034213f7db57"
COLLECTION = "b69dc630-39d1-4cfc-9e63-d2786c71c32c"
REMOTE = "920ccfda-64b4-452a-ab6b-9eeaa85081e4"
KEY = "plane-fake-secret-not-real-111111"


class FakeClient:
    def __init__(self):
        self.created = []
        self.recovered = []
        self.effect = None
        self.recover_response = None
        self.keys = []
    def factory(self, settings, key):
        self.keys.append(key)
        return self
    def check(self):
        return {"projects": [{"id": PROJECT, "name": "Test"}], "collections": []}
    def create(self, row, settings):
        self.created.append(dict(row))
        if self.effect:
            raise self.effect
        return {"id": REMOTE}
    def recover(self, row):
        self.recovered.append(dict(row))
        return self.recover_response


class PlaneStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.fake = FakeClient()
        self.key = Fernet.generate_key()
        self.store = PlaneStore(Path(self.temp.name) / "private" / "plane.sqlite3", self.key, self.fake.factory)
        self.configure()
        self.cards = [dict(kind="task", item_id="task-1", title="Проверить формат", description="Самостоятельная работа",
                           description_html="<p>Проверить формат без изменения источника.</p>"),
                      dict(kind="hypothesis", item_id="idea-1", title="Другой формат", description_html="<p>Предложено сравнить другой формат.</p>")]
        self.page = dict(title="Техническая встреча", description_html="<h2>Главное</h2><p>Описание.</p>")
    def tearDown(self):
        self.temp.cleanup()
    def configure(self, **fields):
        previous = self.store.public_settings()["settings"]
        data = dict(expected_revision=previous["revision"], base_url="https://plane.example.test", workspace_slug="test", project_id=PROJECT)
        if not previous["configured"]:
            data["api_key"] = KEY
        data.update(fields)
        return self.store.save_settings(data)
    def observe(self, generation="g1", auto=False, source="source-1", cards=None):
        return self.store.observe_generation(1, generation, self.cards if cards is None else cards, self.page, source, allow_auto=auto)
    def force_due(self):
        with self.store._db() as db:
            db.execute("UPDATE deliveries SET next_attempt=0")
    def states(self):
        with self.store._db() as db:
            return [r[0] for r in db.execute("SELECT state FROM deliveries ORDER BY created")]

    def test_settings_encrypted_masked_cas_no_redirected_key(self):
        self.assertNotIn(KEY.encode(), self.store.path.read_bytes())
        self.assertNotIn(KEY, json.dumps(self.store.public_settings()))
        with self.assertRaises(PlaneError):
            self.store.save_settings({"expected_revision": 0, "auto_tasks": True})
        with self.assertRaises(PlaneError):
            self.configure(base_url="https://other.example.test")
        with self.assertRaises(PlaneError):
            self.configure(base_url="https://user:pass@plane.example.test")
        with self.assertRaises(PlaneError):
            self.configure(collection_id=COLLECTION, parent_page_id=REMOTE)

    def test_check_is_read_only_and_incomplete_config_allowed(self):
        self.configure(project_id="")
        checked = self.store.check()
        self.assertEqual(checked["status"]["state"], "ready")
        self.assertEqual(len(checked["options"]["projects"]), 1)
        self.assertEqual(self.fake.created, [])

    def test_baseline_and_get_do_not_consume_new_auto_generation(self):
        self.configure(auto_tasks=True, auto_hypotheses=True)
        self.observe(auto=False)
        self.assertIsNone(self.store.known_generation(1))
        self.store.mark_baseline_complete()
        self.assertTrue(self.store.baseline_complete())
        self.assertEqual(self.store.known_generation(1), "g1")
        self.observe(auto=True)
        self.assertEqual(self.states(), [])
        self.observe("g2", auto=False)
        self.assertEqual(self.store.known_generation(1), "g1")
        self.observe("g2", auto=True)
        self.assertEqual(self.states(), ["queued", "queued", "queued"])

    def test_independent_toggles_manual_idempotency_and_proposal(self):
        self.configure(auto_tasks=True, auto_hypotheses=False, auto_meeting_page=False)
        self.observe(auto=True)
        self.assertEqual(self.states(), ["queued"])
        self.store.enqueue(1, "hypothesis", "idea-1", "g1")
        self.store.enqueue(1, "hypothesis", "idea-1", "g1")
        self.assertEqual(len(self.states()), 2)
        self.store.drain_one()
        self.store.drain_one()
        payload = json.loads(self.fake.created[1]["payload"])
        self.assertTrue(payload["name"].startswith("Гипотеза: "))
        self.assertIn("Предложено", payload["description_html"])
        self.assertEqual(payload["assignees"], [])
        self.observe("g2", auto=True)
        self.store.enqueue(1, "hypothesis", "idea-1", "g2")
        self.store.drain_one()
        self.assertEqual(len(self.fake.created), 2)

    def test_disabled_auto_pending_can_still_manual_create(self):
        self.configure(auto_tasks=True, auto_meeting_page=False)
        self.observe(auto=True)
        self.configure(auto_tasks=False)
        self.assertIsNone(self.store.drain_one())
        self.assertEqual(self.states(), ["auto_disabled"])
        self.store.enqueue(1, "task", "task-1", "g1")
        self.assertEqual(self.store.drain_one()["state"], "created")

    def test_revived_auto_task_still_respects_disabled_toggle(self):
        self.configure(auto_tasks=True, auto_meeting_page=False)
        self.observe("g1", auto=True)
        self.observe("g2", auto=True, cards=[])
        self.assertEqual(self.states(), ["cancelled_obsolete"])
        self.observe("g3", auto=True)
        with self.store._db() as db:
            row = db.execute("SELECT state,automatic FROM deliveries").fetchone()
        self.assertEqual(tuple(row), ("queued", 1))
        self.configure(auto_tasks=False)
        self.assertEqual(self.states(), ["auto_disabled"])
        self.assertIsNone(self.store.drain_one())
        self.assertEqual(self.fake.created, [])

    def test_page_link_uses_installed_wiki_route(self):
        self.observe()
        self.store.enqueue(1, "page", "meeting", "g1")
        result = self.store.drain_one()
        self.assertEqual(result["remote_url"], "https://plane.example.test/test/wiki/" + REMOTE)

    def test_pending_refresh_removed_cancelled_and_unknown_immutable(self):
        self.observe()
        self.store.enqueue(1, "task", "task-1", "g1")
        self.cards[0]["description_html"] = "<p>Правка пользователя</p>"
        self.observe()  # same generation, effective manual revision changed
        with self.store._db() as db:
            payload = db.execute("SELECT payload FROM deliveries").fetchone()[0]
        self.assertIn("Правка пользователя", payload)
        self.observe("g2", cards=[])
        self.assertEqual(self.states(), ["cancelled_obsolete"])
        self.observe("g3")
        self.store.enqueue(1, "hypothesis", "idea-1", "g3")
        self.fake.effect = PlaneError("timeout")
        self.store.drain_one()
        before = self.fake.created[0]["payload"]
        self.observe("g4", cards=[])
        with self.store._db() as db:
            payload = db.execute("SELECT payload FROM deliveries WHERE state='submission_unknown'").fetchone()[0]
        self.assertEqual(before, payload)

    def test_unconfigured_status(self):
        self.configure(project_id="")
        items = self.observe()
        self.assertEqual(items["items"][0]["state"], "unconfigured")
        self.assertEqual(items["meeting_page"]["state"], "not_created")

    def test_stale_generation_blocked(self):
        self.observe()
        with self.assertRaises(PlaneError):
            self.store.enqueue(1, "task", "task-1", "stale")
        self.assertEqual(self.states(), [])

    def test_unknown_post_recovers_without_second_post(self):
        self.observe()
        self.store.enqueue(1, "task", "task-1", "g1")
        self.fake.effect = PlaneError("Транспорт не ответил")
        self.assertEqual(self.store.drain_one()["state"], "submission_unknown")
        self.force_due()
        self.store.enqueue(1, "task", "task-1", "g1")
        self.store.drain_one()  # not found is not permission to POST
        self.assertEqual(len(self.fake.created), 1)
        self.assertEqual(self.states(), ["submission_unknown"])
        self.fake.recover_response = {"id": REMOTE}
        self.force_due()
        reopened = PlaneStore(self.store.path, self.key, self.fake.factory)
        self.assertEqual(reopened.drain_one()["state"], "created")
        self.assertEqual(len(self.fake.created), 1)

    def test_definite_failure_manual_retry_and_unknown_rotation(self):
        self.observe()
        self.store.enqueue(1, "task", "task-1", "g1")
        self.fake.effect = PlaneHTTPError(401)
        self.assertEqual(self.store.drain_one()["state"], "failed")
        self.configure(api_key=KEY + "rotated")
        self.fake.effect = None
        self.store.enqueue(1, "task", "task-1", "g1")
        self.assertEqual(self.store.drain_one()["state"], "created")
        self.assertEqual(self.fake.keys[-1], KEY + "rotated")

    def test_preflight_get_timeout_is_known_unsent_and_retryable(self):
        self.observe()
        self.store.enqueue(1, "task", "task-1", "g1")
        self.store.client_factory = PlaneClient
        with patch.object(PlaneClient, "request", side_effect=PlaneError("read timeout")) as request:
            self.assertEqual(self.store.drain_one()["state"], "failed")
            self.assertEqual([c.args[0] for c in request.call_args_list], ["GET"])
        self.store.enqueue(1, "task", "task-1", "g1")
        with patch.object(PlaneClient, "request", side_effect=[{"default_assignee": None}, {"id": REMOTE}]) as request:
            self.assertEqual(self.store.drain_one()["state"], "created")
            self.assertEqual([c.args[0] for c in request.call_args_list], ["GET", "POST"])

    def test_destination_freeze_and_collection_does_not_duplicate_tasks(self):
        self.observe()
        self.store.enqueue(1, "task", "task-1", "g1")
        self.configure(collection_id=COLLECTION)
        self.assertEqual(self.store.drain_one()["state"], "created")
        self.store.enqueue(1, "task", "task-1", "g1")
        self.assertEqual(len(self.states()), 1)
        self.store.enqueue(1, "page", "meeting", "g1")
        self.configure(workspace_slug="another")
        self.assertEqual(self.store.drain_one()["state"], "destination_changed")
        self.assertEqual(len(self.fake.created), 1)

    def test_remote_body_preserved_and_meeting_identity_survives_source_revision(self):
        self.observe()
        self.store.enqueue(1, "page", "meeting", "g1")
        self.store.drain_one()
        self.page["description_html"] = "<p>Исправленное описание.</p>"
        self.observe("g2", source="corrected-source")
        self.store.enqueue(1, "page", "meeting", "g2")
        self.assertEqual(self.states(), ["update_available"])
        self.assertEqual(len(self.fake.created), 1)

    def test_manual_exports_warn_on_local_edits_without_automation(self):
        self.configure(auto_tasks=False, auto_hypotheses=False, auto_meeting_page=False)
        self.observe()
        for kind, item in [("task", "task-1"), ("hypothesis", "idea-1"), ("page", "meeting")]:
            self.store.enqueue(1, kind, item, "g1")
            self.store.drain_one()
        old_task = self.cards[0]["description_html"]
        self.cards[0]["description_html"] = "<p>Ручная правка задачи.</p>"
        self.cards[1]["description_html"] = "<p>Уточнение гипотезы.</p>"
        self.page["description_html"] = "<p>Новая версия встречи.</p>"
        # Also works without a usable key: no HTTP is required for comparison.
        self.configure(remove_key=True)
        self.observe(auto=False)
        self.assertEqual(self.states(), ["update_available"] * 3)
        self.assertEqual(len(self.fake.created), 3)
        self.cards[0]["description_html"] = old_task
        self.observe(auto=False)
        self.assertEqual(self.states(), ["created", "update_available", "update_available"])
        self.assertEqual(len(self.fake.created), 3)

    def test_hypothesis_identity_multiple_predicates_and_ambiguous_change(self):
        ideas = [{"text": "Проверить A", "source_ids": ["U1"]}, {"text": "Проверить B", "source_ids": ["U1"]}]
        ids = self.store.hypothesis_ids("s", ideas)
        self.assertEqual(len(set(ids)), 2)
        self.assertEqual(self.store.hypothesis_ids("s", list(reversed(ideas))), list(reversed(ids)))
        changed = [{"text": "Проверить C", "source_ids": ["U1"]}]
        ambiguous = self.store.hypothesis_ids("s", changed)[0]
        cards = [{"kind": "hypothesis", "item_id": ambiguous, "title": "C", "description_html": "<p>C</p>"}]
        self.configure(auto_hypotheses=True, auto_meeting_page=False)
        result = self.observe(auto=True, source="s", cards=cards)
        self.assertEqual(result["items"][0]["state"], "identity_ambiguous")
        self.assertEqual(self.states(), [])
        with self.assertRaises(PlaneError):
            self.store.enqueue(1, "hypothesis", ambiguous, "g1")

    def test_no_raw_error_or_key_from_arbitrary_exception(self):
        self.observe()
        self.store.enqueue(1, "task", "task-1", "g1")
        self.fake.effect = RuntimeError(KEY)
        result = self.store.drain_one()
        self.assertNotIn(KEY, json.dumps(result))
        self.assertNotIn(KEY, json.dumps(self.store.items(1)))


class PlaneClientTests(unittest.TestCase):
    def client(self):
        return PlaneClient({"base_url": "https://plane.example.test", "workspace_slug": "test", "project_id": PROJECT}, KEY)
    def row(self, kind="task"):
        return {"kind": kind, "external_id": "ts-123", "title": "Meeting", "payload": json.dumps({"name": "Name", "assignees": []})}
    def test_project_default_never_accidentally_assigns(self):
        client = self.client()
        with patch.object(client, "request", return_value={"default_assignee": REMOTE}) as request:
            with self.assertRaises(PlanePreflightError):
                client.create(self.row(), {})
            self.assertEqual([c.args[0] for c in request.call_args_list], ["GET"])
    def test_work_item_recovery_supported_v1_dict_and_404(self):
        client = self.client()
        with patch.object(client, "request", return_value={"id": REMOTE}) as request:
            self.assertEqual(client.recover(self.row())["id"], REMOTE)
            self.assertIn("external_source=transcrisummaryzator", request.call_args.args[1])
        with patch.object(client, "request", side_effect=PlaneHTTPError(404)):
            self.assertIsNone(client.recover(self.row()))
    def test_wiki_recovery_uses_exact_marker_not_title(self):
        client = self.client()
        with patch.object(client, "list_all", return_value=[{"id": REMOTE}]), patch.object(client, "request", return_value={"id": REMOTE, "description_html": "<p>Other</p>"}):
            self.assertIsNone(client.recover(self.row("page")))
        with patch.object(client, "list_all", return_value=[{"id": REMOTE}]), patch.object(client, "request", return_value={"id": REMOTE, "description_html": "<p>transcrisummaryzator-id:ts-123</p>"}):
            self.assertEqual(client.recover(self.row("page"))["id"], REMOTE)
    def test_html_sanitizes_script_handlers_and_unsafe_links(self):
        html = safe_html('<h2 onclick="hack()">Задачи</h2><script>secret()</script><a href="javascript:run()">x</a><a href="https://safe.test/?a=1&b=2">Источник</a>')
        self.assertNotIn("secret", html)
        self.assertNotIn("onclick", html)
        self.assertNotIn("javascript", html)
        self.assertIn("https://safe.test/", html)


if __name__ == "__main__":
    unittest.main()
