"""Versioned structured output schema and deterministic local checks.

These checks establish JSON shape and source-coordinate integrity only. They
do not assert that a cited utterance actually entails the generated claim.
"""

from __future__ import annotations

import json
from pathlib import Path


SCHEMA_ID = "luna_summary_v1"
PROMPT_PATH = Path(__file__).with_name("prompt_v1.md")
SCHEMA_PATH = Path(__file__).with_name("output_schema_v1.json")
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _check_schema(value: object, schema: dict, location: str) -> None:
    reference = schema.get("$ref")
    if reference:
        if not isinstance(reference, str) or not reference.startswith("#/$defs/"):
            raise ValueError(f"{location}: unsupported schema reference")
        _check_schema(value, SCHEMA["$defs"][reference.split("/")[-1]], location)
        return
    types = schema.get("type")
    if isinstance(types, str):
        types = [types]
    matched = any(
        (kind == "object" and isinstance(value, dict))
        or (kind == "array" and isinstance(value, list))
        or (kind == "string" and isinstance(value, str))
        or (kind == "null" and value is None)
        for kind in types or []
    )
    if not matched:
        raise ValueError(f"{location}: wrong JSON type")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{location}: value is outside enum")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        missing = set(schema.get("required", [])) - set(value)
        extra = set(value) - set(properties)
        if missing or (schema.get("additionalProperties") is False and extra):
            raise ValueError(f"{location}: missing {sorted(missing)} or extra {sorted(extra)}")
        for key, item in value.items():
            if key in properties:
                _check_schema(item, properties[key], f"{location}.{key}")
    elif isinstance(value, list):
        item_schema = schema.get("items", {})
        for number, item in enumerate(value):
            _check_schema(item, item_schema, f"{location}[{number}]")


def _text(value: str, location: str) -> None:
    if not value.strip():
        raise ValueError(f"{location}: empty text")


def _source_ids(ids: list[str], index: dict, location: str, *, required: bool = False) -> None:
    if required and not ids:
        raise ValueError(f"{location}: missing source IDs")
    known = index.get("by_id", {})
    for source_id in ids:
        if source_id not in known:
            raise ValueError(f"{location}: unknown source ID {source_id!r}")


def _navigation(item: dict, index: dict, location: str) -> int:
    _text(item["topic"], f"{location}.topic")
    _source_ids([item["start_id"]], index, f"{location}.start_id", required=True)
    start = index["by_id"][item["start_id"]]["start_ms"]
    if item["end_id"] is not None:
        _source_ids([item["end_id"]], index, f"{location}.end_id", required=True)
        end = index["by_id"][item["end_id"]]["end_ms"]
        if end < start:
            raise ValueError(f"{location}: end precedes start")
    return start


def validate_document(document: dict, source_index: dict) -> dict:
    """Raise ValueError on malformed output or fabricated coordinates.

    Returns the same object on success; no meaning is repaired, filled in, or
    re-generated. A separate source review is required before publication.
    """
    _check_schema(document, SCHEMA, "document")
    _text(document["meeting"]["topic"], "meeting.topic")
    project = document["meeting"]["project"]
    if project is not None:
        _text(project, "meeting.project")
    if not document["main"]:
        raise ValueError("main: no meeting overview")

    for key in ("main", "questions", "technical", "ideas"):
        for number, item in enumerate(document[key]):
            where = f"{key}[{number}]"
            _text(item["text"], f"{where}.text")
            _source_ids(item["source_ids"], source_index, where, required=True)

    previous = -1
    for number, item in enumerate(document["timecodes"]):
        start = _navigation(item, source_index, f"timecodes[{number}]")
        if start < previous:
            raise ValueError("timecodes are not chronological")
        previous = start

    for number, item in enumerate(document["tasks"]):
        where = f"tasks[{number}]"
        _text(item["title"], f"{where}.title")
        _text(item["description"], f"{where}.description")
        if item["title"].strip() == item["description"].strip():
            raise ValueError(f"{where}: description repeats title")
        _source_ids(item["source_ids"], source_index, where, required=True)
        sources = item["field_sources"]
        if not sources["action"]:
            raise ValueError(f"{where}: action has no source")
        for field, ids in sources.items():
            _source_ids(ids, source_index, f"{where}.field_sources.{field}")
            if not set(ids).issubset(item["source_ids"]):
                raise ValueError(f"{where}: field evidence is absent from task sources")
            if field in ("assignee", "due", "priority", "recipient"):
                # Evidence may explain why an optional value remains unknown.
                # A populated value, however, must always cite its source.
                if item[field] is not None and not ids:
                    raise ValueError(f"{where}: {field} and its source references disagree")

    for number, item in enumerate(document["verification"]):
        where = f"verification[{number}]"
        _text(item["text"], f"{where}.text")
        _text(item["why_unresolved"], f"{where}.why_unresolved")
        _source_ids(item["source_ids"], source_index, where, required=True)

    previous = -1
    for number, item in enumerate(document["chapters"]):
        where = f"chapters[{number}]"
        start = _navigation(item, source_index, where)
        if start < previous:
            raise ValueError("chapters are not chronological")
        previous = start
        _text(item["summary"], f"{where}.summary")
        _source_ids(item["source_ids"], source_index, where, required=True)
        for detail_number, detail in enumerate(item["details"]):
            _text(detail["text"], f"{where}.details[{detail_number}].text")
            _source_ids(detail["source_ids"], source_index, f"{where}.details[{detail_number}]", required=True)
    return document
