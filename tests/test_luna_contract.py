"""No HTTP, model, audio, Plane or private transcript is needed for this contract."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from summary.luna_v1 import SCHEMA, SCHEMA_ID, load_source, render_document, validate_document


def _task(title, description, sources):
    return {
        "title": title,
        "description": description,
        "discussion_status": "proposed",
        "assignee": None,
        "due": None,
        "priority": None,
        "recipient": None,
        "source_ids": sources,
        "field_sources": {
            "action": sources, "assignee": [], "due": [], "priority": [],
            "recipient": [], "discussion_status": sources,
        },
    }


def _document():
    return {
        "schema_version": SCHEMA_ID,
        "meeting": {"topic": "Проверка <script>данных", "project": None},
        "main": [{"text": "Команда обсудила вход & альтернативы.", "source_ids": ["U00001", "U00002"]}],
        "timecodes": [{"topic": "Проверка данных", "start_id": "U00001", "end_id": "U00002"}],
        "tasks": [
            _task("Проверить X или Y", "Предложено проверить X или Y; выбрать одну альтернативу после проверки.", ["U00001"]),
            _task("Подготовить данные", "Подготовить данные встречи для следующего обсуждения.", ["U00002"]),
        ],
        "questions": [],
        "technical": [{"text": "Лимит равен 30 минутам.", "source_ids": ["U00001"]}],
        "ideas": [],
        "verification": [],
        "chapters": [{
            "topic": "Обсуждение проверки", "start_id": "U00001", "end_id": "U00002",
            "summary": "Участники обсудили варианты и оставили выбор открытым.",
            "source_ids": ["U00001", "U00002"],
            "details": [{"text": "Срок пока не указан.", "source_ids": ["U00002"]}],
        }],
    }


class LunaContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "transcript.json"
        source = {
            "source": "02.01.2030 — Учебная встреча.mkv",
            "duration_seconds": 21,
            "speakers": {"p1": "@Алекс"},
            "utterances": [
                {"start": 0.125, "end": 9.5, "speaker": "p1",
                 "text": "Можно проверить X или Y, не оба. <script>alert(1)</script>",
                 "uncertainty": {"needs_review": False}},
                {"start": 12.0, "end": 20.2, "speaker": None,
                 "text": "Данные подготовим позже.", "uncertainty": {"needs_review": True}},
            ],
            "words": [{"text": "PRIVATE_WORD_METADATA"}],
            "audio_path": "/private/audio.wav",
        }
        self.path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
        self.payload, self.index, self.sha = load_source(self.path)

    def test_canonical_source_is_complete_and_minimal(self):
        again, second_index, digest = load_source(self.path)
        self.assertEqual((self.payload, self.index, self.sha), (again, second_index, digest))
        self.assertEqual(self.sha, hashlib.sha256(self.path.read_bytes()).hexdigest())
        data = json.loads(self.payload)
        self.assertEqual([u["id"] for u in data["utterances"]], ["U00001", "U00002"])
        self.assertEqual(data["utterances"][0]["start_ms"], 125)
        self.assertEqual(data["meeting_date"], "2030-01-02")
        self.assertEqual(data["participants_by_transcript"], ["@Алекс"])
        self.assertTrue(data["unattributed_speech"])
        self.assertNotIn("PRIVATE_WORD_METADATA", self.payload)
        self.assertNotIn("/private/audio.wav", self.payload)

    def test_schema_uses_required_nullable_business_fields(self):
        task = SCHEMA["$defs"]["task"]
        self.assertEqual(set(task["properties"]), set(task["required"]))
        self.assertEqual(task["properties"]["assignee"]["type"], ["string", "null"])
        self.assertFalse(task["additionalProperties"])

    def test_accepted_unassigned_task_and_all_eight_sections(self):
        doc = _document()
        self.assertIs(validate_document(doc, self.index), doc)
        rendered = render_document(doc, self.index)
        self.assertEqual(set(rendered), {"summary.md", "summary.html", "summary.fragment.html", "summary.json", "tasks.json", "transcript.html"})
        md = rendered["summary.md"]
        self.assertEqual(md.count("\n## "), 8)
        self.assertIn("**Исполнитель:** Не назначен", md)
        self.assertIn("Предложено / нужно распределить", md)
        self.assertIn("[U00001 00:00:00](transcript.html#t-125)", md)
        self.assertIn("<details>", md)
        self.assertEqual(md.count("</details>"), 1)
        self.assertEqual(len(rendered["tasks.json"]), 2)
        self.assertNotEqual(rendered["tasks.json"][0]["action_id"], rendered["tasks.json"][1]["action_id"])
        self.assertIsNone(rendered["tasks.json"][0]["assignee"])
        self.assertEqual(rendered["summary.json"]["source_sha256"], self.sha)

    def test_html_escapes_model_and_source_and_links_actual_time(self):
        rendered = render_document(_document(), self.index)
        for name in ("summary.html", "summary.fragment.html"):
            self.assertNotIn("<script>", rendered[name])
            self.assertIn("&lt;script&gt;", rendered[name])
        self.assertIn('href="transcript.html#t-125"', rendered["summary.fragment.html"])
        self.assertIn('href="#t-125"', rendered["summary.html"])
        self.assertIn('id="t-125"', rendered["summary.html"])
        self.assertIn('id="t-125"', rendered["transcript.html"])
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", rendered["transcript.html"])

    def test_rejects_unknown_source_and_unsupported_assignee(self):
        doc = _document()
        doc["technical"][0]["source_ids"] = ["U99999"]
        with self.assertRaisesRegex(ValueError, "unknown source ID"):
            validate_document(doc, self.index)
        doc = _document()
        doc["tasks"][0]["assignee"] = "@Алекс"
        with self.assertRaisesRegex(ValueError, "assignee and its source references disagree"):
            validate_document(doc, self.index)

    def test_effective_manual_edit_keeps_action_id_and_sealed_evidence(self):
        doc = _document()
        generated = render_document(doc, self.index)["tasks.json"]
        manual = json.loads(json.dumps(generated))
        manual[0]["title"] = "Проверить выбранный вариант"
        manual[0]["description"] = "Человек уточнил описание после чтения записи."
        manual[0]["assignee"] = "@Алекс"
        manual[0]["revision"] = 1
        rendered = render_document(doc, self.index, manual)
        self.assertEqual(rendered["tasks.json"][0]["action_id"], generated[0]["action_id"])
        self.assertIn("@Алекс", rendered["summary.md"])
        self.assertIsNone(doc["tasks"][0]["assignee"])
        manual[0]["source_ids"] = ["U00002"]
        with self.assertRaisesRegex(ValueError, "sealed source coordinates"):
            render_document(doc, self.index, manual)

    def test_source_rejects_bad_time_and_does_not_modify_file(self):
        original = self.path.read_bytes()
        source = json.loads(original)
        source["utterances"][1]["start"] = 0.01
        self.path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
        invalid = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "chronological"):
            load_source(self.path)
        self.assertEqual(self.path.read_bytes(), invalid)


if __name__ == "__main__":
    unittest.main()
