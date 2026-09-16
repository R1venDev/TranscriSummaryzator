import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from contracts.meeting import PublicItemContract
from pipeline import REQUIRED_GENERATION_FILES, current_summary_output
from scripts.speech_acts import primary_speech_act
from scripts.summary_worker import build_public_document, deterministic_fact_check, render_public_document
from semantics.entities import EntityRegistry
from semantics.meeting_graph import build_meeting_graph
from semantics.propositions import proposition_from_record
from summary.planner import plan
from summary.verifier import build_public_items, verify_public_document


def rec(number, kind, statement, act="assert", **extra):
    return {
        "record_id": f"F{number}", "kind": kind, "statement": statement,
        "speech_act": act, "modality": extra.pop("modality", "asserted"),
        "attributed_speakers": extra.pop("attributed_speakers", ["@A"]),
        "evidence_ids": [f"U{number}"], "source_word_ids": [f"W{number}"],
        "start": float(number), "end": float(number) + 1, **extra,
    }


class Run13SemanticRegressions(unittest.TestCase):
    def test_contextual_commitment_and_acceptance_are_detected(self):
        self.assertEqual(primary_speech_act("Потом встрою это в методичку"), "commit")
        self.assertEqual(primary_speech_act("А, месяца. Ну ладно"), "accept")
        self.assertEqual(primary_speech_act("Да-да-да. Дальше этап апробации"), "accept")

    def test_ambiguous_alias_does_not_resolve_last_write_wins(self):
        registry = EntityRegistry()
        registry.register("Максим", "person", entity_id="E1")
        registry.register("Максим", "person", entity_id="E2")
        self.assertIsNone(registry.resolve("Максим", "person"))
        self.assertEqual({x["entity_id"] for x in registry.resolve_candidates("Максим")}, {"E1", "E2"})

    def test_direction_is_part_of_proposition_identity(self):
        base = {"kind": "metric", "statement": "Конверсия изменилась на 10%", "evidence_ids": ["U1"], "quantities": [{"value": 10, "unit": "%"}]}
        up = proposition_from_record({**base, "quantities": [{"value": 10, "unit": "%", "direction": "increase"}]})
        down = proposition_from_record({**base, "quantities": [{"value": 10, "unit": "%", "direction": "decrease"}]})
        self.assertNotEqual(up["proposition_id"], down["proposition_id"])

    def test_question_event_survives_shared_proposition(self):
        graph = build_meeting_graph([
            rec(1, "current_state", "Сервис готов."),
            rec(2, "question", "Сервис готов?", "ask", requested_slots=["yes_no"]),
        ])
        self.assertEqual(len(graph["propositions"]), 1)
        self.assertEqual(len(graph["question_states"]), 1)
        self.assertEqual(graph["question_states"][0]["status"], "unanswered")

    def test_semantic_revision_changes_generation_id(self):
        first = build_meeting_graph([rec(1, "observation", "Сервис работает", verification_status="supported")], provenance={"recording_id": "R"})
        second = build_meeting_graph([rec(1, "observation", "Сервис работает", modality="tentative", verification_status="insufficient_evidence")], provenance={"recording_id": "R"})
        self.assertNotEqual(first["generation_id"], second["generation_id"])

    def test_argument_negation_is_not_predicate_contradiction(self):
        graph = build_meeting_graph([
            rec(1, "current_state", "Сервис работает"),
            rec(2, "current_state", "Сервис работает без ошибок"),
        ])
        self.assertFalse(any(x["type"] == "contradicts" for x in graph["relations"]))

    def test_number_time_and_profile_normalization(self):
        fact = {"statement": "@Misha. проверит четыре сигнала после 00:00", "evidence": [{"speaker": "@Misha", "text": "Проверю 4 сигнала после 00"}]}
        self.assertEqual(deterministic_fact_check(fact), (True, None))


class Run13PublicationRegressions(unittest.TestCase):
    def test_public_contract_matches_runtime_fields(self):
        graph = build_meeting_graph([rec(1, "action", "Я подготовлю отчёт", "commit", assignees=["@A"])])
        result = plan(graph["claims"], graph["episodes"], graph["relations"], lambda _item: 1)
        item = next(x for x in build_public_items(graph, result) if x["section"] == "tasks")
        PublicItemContract.model_validate(item)

    def test_unknown_evidence_and_unrendered_card_are_rejected(self):
        item = {"public_id": "PI1", "section": "overview", "text": "Сервис работает", "claim_ids": ["C1"], "evidence_ids": ["U1"], "source_word_ids": ["W1"], "content_kind": "current_state", "social_state": "candidate", "start": 1, "end": 2}
        document = build_public_document([item])
        artifact = render_public_document(document)
        document["title"]["evidence_ids"] = ["U999"]
        self.assertFalse(verify_public_document(document, artifact, [item])["passed"])
        document = build_public_document([item])
        document["outcome_cards"][0]["fields"]["current_state"]["value"] = "Выдуманное состояние"
        self.assertFalse(verify_public_document(document, artifact, [item])["passed"])

    def test_chapter_uses_episode_end(self):
        item = {"public_id": "PI1", "section": "minutes", "text": "Сервис работает", "claim_ids": ["C1"], "evidence_ids": ["U1"], "source_word_ids": ["W1"], "content_kind": "current_state", "social_state": "candidate", "start": 10, "end": 18, "episode_id": "E1"}
        chapter = build_public_document([item])["chronology"][0]
        self.assertEqual(chapter["end"], 18)
        self.assertGreater(chapter["end"], chapter["start"])

    def test_incomplete_generation_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); gid = "20260915-120000-abcdef123456"
            target = base / "summary_generations" / gid; target.mkdir(parents=True)
            (target / "summary.md").write_text("ok")
            digest = hashlib.sha256(b"ok").hexdigest()
            (target / "generation_manifest.json").write_text(json.dumps({"generation_id": gid, "artifact_sha256": {"summary.md": digest}}))
            (base / "summary_current.json").write_text(json.dumps({"generation_id": gid}))
            self.assertTrue(REQUIRED_GENERATION_FILES - {"summary.md"})
            self.assertIsNone(current_summary_output(base))


if __name__ == "__main__":
    unittest.main()
