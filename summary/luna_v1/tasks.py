"""Private, local task identities and versioned human edits.

The model document is never changed here. Reconciliation records a copy of
each newly generated task; user changes live in a separate append-only edit
history and an effective-state projection. Ambiguous identities stop the new
projection instead of silently moving a person's edit to another action.
"""

from __future__ import annotations

from contextlib import contextmanager, closing
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Iterator


_EDITABLE = frozenset({
    "title", "description", "discussion_status", "assignee", "due",
    "priority", "recipient",
})
_OPTIONAL = frozenset({"assignee", "due", "priority", "recipient"})
_STATUSES = frozenset({"proposed", "committed", "in_progress", "unknown"})
_TASK_FIELDS = _EDITABLE | {"source_ids", "field_sources"}
_SOURCE_SHA = re.compile(r"[0-9a-f]{64}\Z")
_WORDS = re.compile(r"\w+", re.UNICODE)


class RevisionConflict(ValueError):
    """The submitted revision is stale; the caller must reload the card."""


class ReconciliationConflict(ValueError):
    """A regenerated task cannot safely inherit or discard a human edit."""

    def __init__(self, reason: str, action_ids: list[str]):
        self.reason = reason
        self.action_ids = tuple(action_ids)
        super().__init__(f"{reason}: {', '.join(action_ids)}")


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _normal(value: str) -> str:
    return " ".join(_WORDS.findall(value.casefold()))


def _action_sources(task: dict) -> tuple[str, ...]:
    return tuple(sorted(set(task["field_sources"]["action"])))


def _fingerprint(task: dict) -> str:
    return _digest({
        "action_sources": _action_sources(task),
        "title": _normal(task["title"]),
        "description": _normal(task["description"]),
    })


def _validate_generated(task: object) -> dict:
    if not isinstance(task, dict) or set(task) != _TASK_FIELDS:
        raise ValueError("generated task must have the exact Luna v1 task fields")
    for field in ("title", "description"):
        if not isinstance(task[field], str) or not task[field].strip():
            raise ValueError(f"generated task has empty {field}")
    if task["discussion_status"] not in _STATUSES:
        raise ValueError("generated task has invalid discussion status")
    for field in _OPTIONAL:
        if task[field] is not None and (not isinstance(task[field], str) or not task[field].strip()):
            raise ValueError(f"generated task has invalid {field}")
    sources = task["source_ids"]
    evidence = task["field_sources"]
    if not isinstance(sources, list) or not sources or not all(isinstance(x, str) and x for x in sources):
        raise ValueError("generated task has no source IDs")
    if len(sources) != len(set(sources)):
        raise ValueError("generated task repeats a source ID")
    if not isinstance(evidence, dict) or set(evidence) != {"action", "assignee", "due", "priority", "recipient", "discussion_status"}:
        raise ValueError("generated task has invalid field evidence")
    for field, ids in evidence.items():
        if not isinstance(ids, list) or not all(isinstance(x, str) and x in sources for x in ids):
            raise ValueError(f"generated task has invalid {field} evidence")
    if not evidence["action"]:
        raise ValueError("generated task has no action evidence")
    return deepcopy(task)


def _valid_edit(changes: object) -> dict:
    if not isinstance(changes, dict) or not changes or set(changes) - _EDITABLE:
        raise ValueError("edit must change only task content or optional business fields")
    for field, value in changes.items():
        if field in ("title", "description"):
            if not isinstance(value, str) or not value.strip() or len(value) > (300 if field == "title" else 20000):
                raise ValueError(f"invalid {field}")
        elif field == "discussion_status":
            if value not in _STATUSES:
                raise ValueError("invalid discussion_status")
        elif value is not None and (not isinstance(value, str) or not value.strip() or len(value) > 1000):
            raise ValueError(f"invalid {field}; use null to clear it")
    return deepcopy(changes)


def _may_match(old: dict, fresh: dict) -> bool:
    old_sources, new_sources = set(_action_sources(old)), set(_action_sources(fresh))
    if not old_sources & new_sources:
        return False
    old_title, new_title = _normal(old["title"]), _normal(fresh["title"])
    old_description, new_description = _normal(old["description"]), _normal(fresh["description"])
    # Lexical similarity is unsafe here: "X или Y" and "X и Y" differ by
    # one word but mean different actions. Preserve identity only while one
    # complete human-readable action field remains stable. If both change,
    # an edited action becomes an explicit reconciliation conflict.
    return old_title == new_title or old_description == new_description


def _effective(generated: dict, action_id: str, revision: int, patch: dict) -> dict:
    item = deepcopy(generated)
    item.update(deepcopy(patch))
    item["action_id"] = action_id
    item["revision"] = revision
    return item


@dataclass(frozen=True)
class ReconciliationPlan:
    """Read-only candidate projection; commit before switching the pointer."""

    source_sha: str
    generated_json: str
    store_snapshot: str
    effective_json: str

    @property
    def effective_tasks(self) -> list[dict]:
        return json.loads(self.effective_json)


class TaskStore:
    """SQLite task store for one private application instance.

    The containing directory must be private. All writes use SQLite's
    immediate transaction so concurrent workers share identity and revision
    decisions. The returned effective tasks can be passed directly to
    ``render_document``; source coordinates always come from the generation.
    """

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.path.parent.stat().st_mode & 0o077:
            raise ValueError("task store directory must be private (0700)")
        existed = self.path.exists()
        with self._connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS task_identity (
                    action_id TEXT PRIMARY KEY,
                    source_sha TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    generated_json TEXT NOT NULL,
                    generated_sha TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS task_identity_source ON task_identity(source_sha);
                CREATE TABLE IF NOT EXISTS task_override (
                    action_id TEXT PRIMARY KEY REFERENCES task_identity(action_id),
                    revision INTEGER NOT NULL DEFAULT 0,
                    patch_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS task_edit_history (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action_id TEXT NOT NULL REFERENCES task_identity(action_id),
                    revision INTEGER NOT NULL,
                    actor TEXT NOT NULL,
                    edited_at TEXT NOT NULL,
                    changes_json TEXT NOT NULL,
                    generated_sha TEXT NOT NULL,
                    source_sha TEXT NOT NULL,
                    UNIQUE(action_id, revision)
                );
            """)
        if not existed:
            self.path.chmod(0o600)
        if self.path.stat().st_mode & 0o077:
            raise ValueError("task store file must be private (0600)")

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self.path, timeout=15, isolation_level=None)) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA busy_timeout=15000")
            try:
                yield db
            finally:
                pass

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
            except BaseException:
                db.rollback()
                raise
            else:
                db.commit()

    @staticmethod
    def _rows(db: sqlite3.Connection, source_sha: str) -> list[sqlite3.Row]:
        return db.execute("""
            SELECT i.*, o.revision, o.patch_json FROM task_identity AS i
            JOIN task_override AS o USING(action_id)
            WHERE i.source_sha=? ORDER BY i.created_at, i.action_id
        """, (source_sha,)).fetchall()

    @staticmethod
    def _snapshot(rows: list[sqlite3.Row]) -> str:
        return _digest([
            [row[key] for key in (
                "action_id", "fingerprint", "generated_sha", "updated_at",
                "revision", "patch_json",
            )] for row in rows
        ])

    @staticmethod
    def _validate_input(source_sha: str, generated_tasks: list[dict]) -> list[dict]:
        if not isinstance(source_sha, str) or not _SOURCE_SHA.fullmatch(source_sha):
            raise ValueError("invalid source SHA-256")
        if not isinstance(generated_tasks, list):
            raise ValueError("generated tasks must be a list")
        fresh = [_validate_generated(item) for item in generated_tasks]
        fingerprints = [_fingerprint(item) for item in fresh]
        if len(fingerprints) != len(set(fingerprints)):
            raise ReconciliationConflict("duplicate generated action content", [])
        return fresh

    @staticmethod
    def _plan(source_sha: str, fresh: list[dict], rows: list[sqlite3.Row]) -> tuple[dict[int, str], list[dict]]:
        fingerprints = [_fingerprint(item) for item in fresh]
        old = {row["action_id"]: row for row in rows}
        assignments: dict[int, str] = {}
        used: set[str] = set()

        for position, fingerprint in enumerate(fingerprints):
            candidates = [row["action_id"] for row in rows if row["fingerprint"] == fingerprint and row["action_id"] not in used]
            if len(candidates) > 1:
                raise ReconciliationConflict("ambiguous exact identity", candidates)
            if candidates:
                assignments[position] = candidates[0]
                used.add(candidates[0])

        possibilities: dict[int, list[str]] = {}
        for position, task in enumerate(fresh):
            if position in assignments:
                continue
            possibilities[position] = [
                row["action_id"] for row in rows if row["action_id"] not in used
                and _may_match(json.loads(row["generated_json"]), task)
            ]
        for position, candidates in possibilities.items():
            if len(candidates) > 1:
                raise ReconciliationConflict("ambiguous source/action match", candidates)
            if candidates and sum(candidates[0] in other for other in possibilities.values()) > 1:
                raise ReconciliationConflict("multiple generated tasks match one action", candidates)
            if candidates:
                assignments[position] = candidates[0]
                used.add(candidates[0])

        missing_edits = [row["action_id"] for row in rows if row["action_id"] not in used and row["revision"] > 0]
        if missing_edits:
            raise ReconciliationConflict("manually edited action missing from regeneration", missing_edits)

        result: list[dict] = []
        for position, task in enumerate(fresh):
            action_id = assignments.get(position)
            if action_id is None:
                action_id = "A-" + hashlib.sha256(f"{source_sha}:{fingerprints[position]}".encode()).hexdigest()[:24]
                if action_id in old:
                    raise ReconciliationConflict("action ID collision", [action_id])
                assignments[position] = action_id
                revision, patch = 0, {}
            else:
                row = old[action_id]
                revision, patch = row["revision"], json.loads(row["patch_json"])
            result.append(_effective(task, action_id, revision, patch))
        return assignments, result

    def preview_reconcile(self, source_sha: str, generated_tasks: list[dict]) -> ReconciliationPlan:
        """Project a candidate without changing persisted identities or edits."""
        fresh = self._validate_input(source_sha, generated_tasks)
        with self._connection() as db:
            rows = self._rows(db, source_sha)
        _, effective = self._plan(source_sha, fresh, rows)
        return ReconciliationPlan(source_sha, _json(fresh), self._snapshot(rows), _json(effective))

    def commit_reconcile(self, plan: ReconciliationPlan) -> list[dict]:
        """Persist a sealed plan if no task/edit changed since preview.

        The caller must seal the candidate, hold its publication/write lock,
        call this method, then switch the current pointer last. Existing
        readers use ``effective_for_sealed`` to keep the old selected model
        fields until that pointer changes.
        """
        if not isinstance(plan, ReconciliationPlan):
            raise TypeError("expected a ReconciliationPlan")
        fresh = self._validate_input(plan.source_sha, json.loads(plan.generated_json))
        now = datetime.now(timezone.utc).isoformat()
        with self._write() as db:
            rows = self._rows(db, plan.source_sha)
            if self._snapshot(rows) != plan.store_snapshot:
                raise RevisionConflict("task store changed since candidate preview")
            assignments, effective = self._plan(plan.source_sha, fresh, rows)
            if _json(effective) != plan.effective_json:
                raise RevisionConflict("candidate projection changed since preview")
            old = {row["action_id"]: row for row in rows}
            for position, task in enumerate(fresh):
                action_id = assignments[position]
                if action_id not in old:
                    db.execute("""INSERT INTO task_identity
                        (action_id,source_sha,fingerprint,generated_json,generated_sha,created_at,updated_at)
                        VALUES (?,?,?,?,?,?,?)""",
                        (action_id, plan.source_sha, _fingerprint(task), _json(task), _digest(task), now, now))
                    db.execute("INSERT INTO task_override (action_id,revision,patch_json,updated_at) VALUES (?,0,'{}',?)", (action_id, now))
                else:
                    db.execute("""UPDATE task_identity SET fingerprint=?,generated_json=?,generated_sha=?,updated_at=?
                        WHERE action_id=?""", (_fingerprint(task), _json(task), _digest(task), now, action_id))
            return effective

    def reconcile(self, source_sha: str, generated_tasks: list[dict]) -> list[dict]:
        """Convenience for an already accepted generation, never a candidate."""
        return self.commit_reconcile(self.preview_reconcile(source_sha, generated_tasks))

    def effective_for_sealed(self, source_sha: str, generated_tasks: list[dict], action_ids: list[str]) -> list[dict]:
        """Overlay human edits on the *selected sealed generation* only.

        Generated content and evidence come from ``generated_tasks`` rather
        than the latest identity row, so even a failed pointer switch cannot
        mix candidate model fields into the previously selected generation.
        """
        fresh = self._validate_input(source_sha, generated_tasks)
        if not isinstance(action_ids, list) or len(action_ids) != len(fresh):
            raise ValueError("sealed action IDs and tasks differ in count")
        if any(not isinstance(action_id, str) or not action_id for action_id in action_ids):
            raise ValueError("sealed action ID is invalid")
        if len(set(action_ids)) != len(action_ids):
            raise ValueError("sealed action IDs repeat")
        if not action_ids:
            return []
        placeholders = ",".join("?" for _ in action_ids)
        with self._connection() as db:
            rows = db.execute(f"""SELECT i.action_id,i.source_sha,o.revision,o.patch_json
                FROM task_identity AS i JOIN task_override AS o USING(action_id)
                WHERE i.action_id IN ({placeholders})""", action_ids).fetchall()
        by_id = {row["action_id"]: row for row in rows}
        if set(by_id) != set(action_ids) or any(row["source_sha"] != source_sha for row in rows):
            raise ValueError("sealed action ID is absent from this source revision")
        return [
            _effective(task, action_id, by_id[action_id]["revision"], json.loads(by_id[action_id]["patch_json"]))
            for task, action_id in zip(fresh, action_ids)
        ]

    def get(self, action_id: str) -> dict:
        """Get latest matching material for editor diagnostics, not a sealed view."""
        with self._connection() as db:
            row = db.execute("""SELECT i.generated_json,o.revision,o.patch_json FROM task_identity AS i
                JOIN task_override AS o USING(action_id) WHERE action_id=?""", (action_id,)).fetchone()
        if row is None:
            raise KeyError(action_id)
        return _effective(json.loads(row["generated_json"]), action_id, row["revision"], json.loads(row["patch_json"]))

    def update(self, action_id: str, expected_revision: int, changes: dict, actor: str) -> dict:
        """Apply a user edit with compare-and-swap revision and provenance."""
        if not isinstance(action_id, str) or not action_id:
            raise ValueError("invalid action ID")
        if not isinstance(expected_revision, int) or isinstance(expected_revision, bool) or expected_revision < 0:
            raise ValueError("invalid expected revision")
        if not isinstance(actor, str) or not actor.strip() or len(actor) > 200:
            raise ValueError("invalid actor")
        updates = _valid_edit(changes)
        now = datetime.now(timezone.utc).isoformat()
        with self._write() as db:
            row = db.execute("""SELECT i.source_sha,i.generated_json,i.generated_sha,o.revision,o.patch_json
                FROM task_identity AS i JOIN task_override AS o USING(action_id)
                WHERE action_id=?""", (action_id,)).fetchone()
            if row is None:
                raise KeyError(action_id)
            if row["revision"] != expected_revision:
                raise RevisionConflict(f"stale task revision for {action_id}: current {row['revision']}")
            patch = json.loads(row["patch_json"])
            if all(field in patch and patch[field] == value for field, value in updates.items()):
                return _effective(json.loads(row["generated_json"]), action_id, row["revision"], patch)
            patch.update(updates)
            revision = row["revision"] + 1
            db.execute("UPDATE task_override SET revision=?,patch_json=?,updated_at=? WHERE action_id=?",
                       (revision, _json(patch), now, action_id))
            db.execute("""INSERT INTO task_edit_history
                (action_id,revision,actor,edited_at,changes_json,generated_sha,source_sha)
                VALUES (?,?,?,?,?,?,?)""",
                (action_id, revision, actor.strip(), now, _json(updates), row["generated_sha"], row["source_sha"]))
            return _effective(json.loads(row["generated_json"]), action_id, revision, patch)

    def history(self, action_id: str) -> list[dict]:
        """Return local provenance; do not include this in public exports."""
        with self._connection() as db:
            rows = db.execute("""SELECT revision,actor,edited_at,changes_json,generated_sha,source_sha
                FROM task_edit_history WHERE action_id=? ORDER BY revision""", (action_id,)).fetchall()
        return [{**{key: row[key] for key in ("revision", "actor", "edited_at", "generated_sha", "source_sha")},
                 "changes": json.loads(row["changes_json"])} for row in rows]
