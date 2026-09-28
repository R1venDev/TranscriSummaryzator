"""Durable single-flight and shared USD budget for Luna summary requests."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


WEEK_SECONDS = 7 * 24 * 60 * 60
WEEK_CAP_MICROUSD = 1_000_000
JOB_CAP_MICROUSD = 100_000
MAX_DISPATCHES_PER_JOB = 6
MAX_BATCH_ITEMS_PER_WORKFLOW = 12
MAX_BATCH_POSTS_PER_WORKFLOW = 6


def usd_micros(amount: float) -> int:
    if amount < 0 or amount != amount:
        raise ValueError("invalid USD amount")
    return round(amount * 1_000_000)


def _secure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)


def write_private_json(path: Path, value) -> None:
    _secure_dir(path.parent)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sealed_sha256(path: Path) -> str:
    """Hash the exact saved bytes; a path or abbreviated digest is not proof."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("sealed artifact is unavailable")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _full_hash(value: str) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _canonical_hash(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class StartDecision:
    kind: str  # new, pending, accepted, blocked
    job_id: str | None
    reason: str | None = None


@dataclass(frozen=True)
class BatchItemIntent:
    """One separately billable generation inside a remote Batch envelope."""

    custom_id: str
    request_sha256: str
    input_sha256: str
    prompt_sha256: str
    schema_sha256: str
    max_output_tokens: int
    reserve_microusd: int


@dataclass(frozen=True)
class BatchIntentDecision:
    kind: str  # new, pending, blocked
    attempt_id: str | None
    reason: str | None = None


def batch_item_intent_for_body(custom_id: str, body: dict, *,
                               reserve_microusd: int) -> BatchItemIntent:
    """Bind an item's hold to its *actual* text, prompt, schema and cap.

    The pricing/route preflight computes ``reserve_microusd`` from this same
    body. This helper never estimates token counts or assumes a cache hit.
    """
    if not isinstance(body, dict):
        raise ValueError("invalid batch body")
    messages = body.get("messages")
    if (not isinstance(messages, list) or len(messages) != 2
            or not all(isinstance(m, dict) for m in messages)
            or [m.get("role") for m in messages] != ["developer", "user"]
            or not all(isinstance(m.get("content"), str) for m in messages)):
        raise ValueError("invalid batch messages")
    fmt = body.get("response_format")
    if not isinstance(fmt, dict) or not isinstance(fmt.get("json_schema"), dict):
        raise ValueError("invalid batch output schema")
    if ("max_completion_tokens" in body) == ("max_tokens" in body):
        raise ValueError("ambiguous batch output cap")
    cap = body.get("max_completion_tokens", body.get("max_tokens"))
    if type(cap) is not int or not 1 <= cap <= 128_000:
        raise ValueError("invalid batch output cap")
    return BatchItemIntent(
        custom_id=custom_id,
        request_sha256=_canonical_hash(body),
        input_sha256=_canonical_hash(messages[1]["content"]),
        prompt_sha256=_canonical_hash(messages[0]["content"]),
        schema_sha256=_canonical_hash(fmt["json_schema"]),
        max_output_tokens=cap,
        reserve_microusd=reserve_microusd,
    )


class Ledger:
    """One app-wide ledger, independent of API-key count and worker count.

    All mutating decisions run inside SQLite BEGIN IMMEDIATE. Pending/unknown
    charges continue to reserve their upper bound through restart. Only a
    confirmed terminal cost can replace a reservation.
    """

    def __init__(self, private_root: Path):
        self.root = Path(private_root)
        _secure_dir(self.root)
        self.db_path = self.root / "luna.sqlite3"
        self.db = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=30000")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            semantic_key TEXT NOT NULL UNIQUE,
            source_sha256 TEXT NOT NULL,
            output_dir TEXT NOT NULL,
            artifact_dir TEXT NOT NULL,
            status TEXT NOT NULL,
            credential_id TEXT NOT NULL,
            credential_version INTEGER NOT NULL,
            workspace_id TEXT,
            custom_id TEXT NOT NULL UNIQUE,
            remote_id TEXT UNIQUE,
            reserved_microusd INTEGER NOT NULL,
            billed_microusd INTEGER,
            dispatches INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            next_poll_at REAL,
            error_code TEXT,
            accepted_document_path TEXT,
            generation_id TEXT
          );
          CREATE TABLE IF NOT EXISTS consumers (
            semantic_key TEXT NOT NULL,
            output_dir TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            generation_id TEXT,
            error_code TEXT,
            updated_at REAL,
            PRIMARY KEY(semantic_key,output_dir)
          );
          CREATE TABLE IF NOT EXISTS source_first_workflows (
            id TEXT PRIMARY KEY,
            semantic_key TEXT NOT NULL UNIQUE,
            source_sha256 TEXT NOT NULL,
            output_dir TEXT NOT NULL,
            manifest_path TEXT NOT NULL,
            manifest_sha256 TEXT NOT NULL,
            plan_sha256 TEXT,
            planned_reserve_microusd INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL,
            credential_id TEXT NOT NULL,
            credential_version INTEGER NOT NULL,
            workspace_id TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            error_code TEXT,
            accepted_document_path TEXT,
            accepted_document_sha256 TEXT
          );
          CREATE TABLE IF NOT EXISTS batch_attempts (
            id TEXT PRIMARY KEY,
            workflow_id TEXT NOT NULL REFERENCES source_first_workflows(id),
            intent_key TEXT NOT NULL UNIQUE,
            stage TEXT NOT NULL,
            status TEXT NOT NULL,
            credential_id TEXT NOT NULL,
            credential_version INTEGER NOT NULL,
            workspace_id TEXT,
            payload_path TEXT NOT NULL,
            payload_sha256 TEXT NOT NULL,
            remote_id TEXT UNIQUE,
            remote_status TEXT,
            reserved_microusd INTEGER NOT NULL,
            billed_microusd INTEGER,
            post_count INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            next_poll_at REAL,
            error_code TEXT,
            lease_owner TEXT,
            lease_until REAL,
            terminal_path TEXT,
            terminal_sha256 TEXT,
            recovery_evidence_path TEXT,
            recovery_evidence_sha256 TEXT
          );
          CREATE TABLE IF NOT EXISTS batch_items (
            attempt_id TEXT NOT NULL REFERENCES batch_attempts(id),
            custom_id TEXT NOT NULL,
            request_sha256 TEXT NOT NULL,
            input_sha256 TEXT NOT NULL,
            prompt_sha256 TEXT NOT NULL,
            schema_sha256 TEXT NOT NULL,
            max_output_tokens INTEGER NOT NULL,
            reserved_microusd INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            billed_microusd INTEGER,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            raw_path TEXT,
            raw_sha256 TEXT,
            error_code TEXT,
            updated_at REAL NOT NULL,
            PRIMARY KEY(attempt_id,custom_id)
          );
          CREATE INDEX IF NOT EXISTS jobs_by_created ON jobs(created_at);
          CREATE INDEX IF NOT EXISTS jobs_by_credential ON jobs(credential_id,status);
          CREATE INDEX IF NOT EXISTS batch_attempts_by_workflow ON batch_attempts(workflow_id,created_at);
          CREATE INDEX IF NOT EXISTS batch_attempts_due ON batch_attempts(status,next_poll_at,lease_until);
        """)
        # Existing isolated ledgers had only the semantic/output pair. Keep
        # those rows pending until their selected generations are verified.
        self.db.execute("BEGIN IMMEDIATE")
        try:
            columns = {row[1] for row in self.db.execute("PRAGMA table_info(consumers)")}
            for name, definition in (
                ("status", "TEXT NOT NULL DEFAULT 'pending'"),
                ("generation_id", "TEXT"),
                ("error_code", "TEXT"),
                ("updated_at", "REAL"),
            ):
                if name not in columns:
                    self.db.execute(f"ALTER TABLE consumers ADD COLUMN {name} {definition}")
            workflow_columns = {row[1] for row in self.db.execute(
                "PRAGMA table_info(source_first_workflows)")}
            if "plan_sha256" not in workflow_columns:
                self.db.execute("ALTER TABLE source_first_workflows ADD COLUMN plan_sha256 TEXT")
            if "planned_reserve_microusd" not in workflow_columns:
                self.db.execute("ALTER TABLE source_first_workflows ADD COLUMN "
                                "planned_reserve_microusd INTEGER NOT NULL DEFAULT 0")
            self.db.execute("COMMIT")
        except Exception:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise
        os.chmod(self.db_path, 0o600)

    def close(self) -> None:
        self.db.close()

    def _rolling_spent_microusd(self, now: float) -> int:
        """One account-wide window for legacy jobs and source-first Batch items.

        Call only inside a BEGIN IMMEDIATE transaction before a reservation.
        A missing terminal cost keeps the full pre-dispatch hold, including a
        submission whose remote identity could not be recovered.
        """
        since = now - WEEK_SECONDS
        old = self.db.execute("""SELECT COALESCE(SUM(CASE WHEN billed_microusd IS NULL
            THEN reserved_microusd ELSE billed_microusd END),0) FROM jobs
            WHERE (created_at>=? OR (billed_microusd IS NULL AND dispatches>0))
            AND status NOT IN
            ('rejected_before_submit','cancelled_before_submit')""",
            (since,)).fetchone()[0]
        new = 0
        for workflow in self.db.execute("""SELECT id,status,planned_reserve_microusd
            FROM source_first_workflows"""):
            if workflow["status"] == "active":
                # The full-chain hold remains in the rolling budget while
                # later stages are still pending. Stage reserves draw from
                # this pool and must never be added to it a second time.
                attempts = self.db.execute("""SELECT COALESCE(SUM(CASE WHEN
                    billed_microusd IS NULL THEN reserved_microusd ELSE
                    billed_microusd END),0) FROM batch_attempts WHERE workflow_id=?
                    AND status NOT IN ('rejected_no_charge','cancelled_before_submit')""",
                    (workflow["id"],)).fetchone()[0]
                new += max(workflow["planned_reserve_microusd"], attempts)
            else:
                # A terminal workflow releases unused capacity. A POST with
                # unknown cost keeps its original hold even past seven days;
                # time alone is not evidence of a zero charge.
                new += self.db.execute("""SELECT COALESCE(SUM(CASE WHEN
                    billed_microusd IS NULL THEN reserved_microusd ELSE
                    billed_microusd END),0) FROM batch_attempts WHERE workflow_id=?
                    AND status NOT IN ('rejected_no_charge','cancelled_before_submit')
                    AND (created_at>=? OR (post_count=1 AND billed_microusd IS NULL))""",
                    (workflow["id"], since)).fetchone()[0]
        return old + new

    def create_source_first_job(self, *, semantic_key: str, source_sha256: str,
                                output_dir: Path, manifest_path: Path,
                                manifest_sha256: str, credential_id: str,
                                credential_version: int,
                                workspace_id: str | None) -> StartDecision:
        """Register a source-first logical job without a billable dispatch.

        It has its own table, so legacy single-item scheduler queries cannot
        mistake the zero-cost root for a pending remote Batch. Its saved
        manifest must exist before the row is exposed to the scheduler.
        """
        if (not _full_hash(semantic_key) or not _full_hash(source_sha256)
                or not _full_hash(manifest_sha256)
                or not isinstance(credential_id, str) or not credential_id
                or type(credential_version) is not int or credential_version < 1
                or workspace_id is not None and (not isinstance(workspace_id, str)
                                                  or not workspace_id)):
            raise ValueError("invalid source-first identity")
        output_dir = Path(output_dir)
        manifest_path = Path(manifest_path)
        if _sealed_sha256(manifest_path) != manifest_sha256:
            raise ValueError("source-first manifest hash changed")
        now = time.time()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            prior = self.db.execute("SELECT * FROM source_first_workflows WHERE semantic_key=?",
                                    (semantic_key,)).fetchone()
            if prior is not None:
                if (prior["source_sha256"] != source_sha256
                        or prior["output_dir"] != str(output_dir)
                        or prior["manifest_path"] != str(manifest_path)
                        or prior["manifest_sha256"] != manifest_sha256
                        or prior["credential_id"] != credential_id
                        or prior["credential_version"] != credential_version
                        or prior["workspace_id"] != workspace_id):
                    raise ValueError("source-first semantic identity changed")
                self.db.execute("COMMIT")
                return StartDecision("accepted" if prior["status"] == "accepted"
                                     else "pending", prior["id"], prior["status"])
            if self.db.execute("SELECT 1 FROM jobs WHERE semantic_key=?",
                               (semantic_key,)).fetchone() is not None:
                raise ValueError("semantic identity belongs to a legacy job")
            workflow_id = uuid.uuid4().hex
            self.db.execute("""INSERT INTO source_first_workflows
                (id,semantic_key,source_sha256,output_dir,manifest_path,manifest_sha256,
                 status,credential_id,credential_version,workspace_id,created_at,updated_at)
                VALUES (?,?,?,?,?,?,'active',?,?,?,?,?)""",
                (workflow_id, semantic_key, source_sha256, str(output_dir),
                 str(manifest_path), manifest_sha256, credential_id,
                 credential_version, workspace_id, now, now))
            self.db.execute("COMMIT")
            return StartDecision("new", workflow_id)
        except Exception:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def get_batch_workflow(self, workflow_id: str) -> dict | None:
        row = self.db.execute("SELECT * FROM source_first_workflows WHERE id=?",
                              (workflow_id,)).fetchone()
        return dict(row) if row else None

    def reactivate_source_first_budget_refusal(self, workflow_id: str, *,
                                               semantic_key: str,
                                               source_sha256: str,
                                               output_dir: Path,
                                               manifest_sha256: str) -> bool:
        """Retry an old weekly refusal only when no Batch intent ever existed.

        Earlier workers finalized an unreserved logical job on a temporary
        rolling-week refusal, pinning its unique semantic key forever. The
        caller checks the sealed manifest's current policy before this narrow
        state transition; the ledger checks its durable identity and lack of
        billable or prepared attempts atomically.
        """
        if (not _full_hash(semantic_key) or not _full_hash(source_sha256)
                or not _full_hash(manifest_sha256)):
            raise ValueError("invalid source-first retry identity")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT * FROM source_first_workflows WHERE id=?",
                                  (workflow_id,)).fetchone()
            eligible = (row is not None
                and row["status"] == "failed"
                and row["error_code"] == "rolling_week_budget_exceeded"
                and row["semantic_key"] == semantic_key
                and row["source_sha256"] == source_sha256
                and row["output_dir"] == str(Path(output_dir))
                and row["manifest_sha256"] == manifest_sha256
                and row["plan_sha256"] is None
                and row["planned_reserve_microusd"] == 0
                and row["accepted_document_path"] is None
                and row["accepted_document_sha256"] is None
                and self.db.execute("SELECT 1 FROM batch_attempts WHERE workflow_id=? LIMIT 1",
                                    (workflow_id,)).fetchone() is None)
            if not eligible:
                self.db.execute("ROLLBACK")
                return False
            if _sealed_sha256(Path(row["manifest_path"])) != manifest_sha256:
                raise ValueError("source-first retry manifest changed")
            changed = self.db.execute("""UPDATE source_first_workflows
                SET status='active',error_code=NULL,updated_at=?
                WHERE id=? AND status='failed'""", (time.time(), workflow_id))
            if changed.rowcount != 1:
                raise ValueError("source-first budget retry transition lost")
            self.db.execute("COMMIT")
            return True
        except Exception:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def reserve_source_first_plan(self, workflow_id: str, *,
                                  plan_sha256: str,
                                  reserve_microusd: int) -> StartDecision:
        """Hold the entire quoted workflow before its first physical POST.

        The plan identity is the saved root manifest's exact hash. Every later
        Batch-item reservation spends from this one held pool; neither stages
        nor API keys create another weekly allowance. Repeating the same plan
        is an exact no-op, including after a process restart.
        """
        if (not _full_hash(plan_sha256) or type(reserve_microusd) is not int
                or not 0 < reserve_microusd <= JOB_CAP_MICROUSD):
            raise ValueError("invalid source-first plan reserve")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT * FROM source_first_workflows WHERE id=?",
                                  (workflow_id,)).fetchone()
            if row is None or row["status"] != "active":
                raise ValueError("source-first workflow is not active")
            if (row["manifest_sha256"] != plan_sha256
                    or _sealed_sha256(Path(row["manifest_path"])) != plan_sha256):
                raise ValueError("source-first plan manifest changed")
            try:
                manifest = json.loads(Path(row["manifest_path"]).read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ValueError("source-first plan manifest is not JSON") from exc
            if (not isinstance(manifest, dict)
                    or manifest.get("planned_capacity_microusd") != reserve_microusd
                    or not isinstance(manifest.get("capacity_basis"), dict)
                    or not manifest["capacity_basis"]):
                raise ValueError("source-first plan quote differs from sealed manifest")
            if row["plan_sha256"] is not None:
                if (row["plan_sha256"] != plan_sha256
                        or row["planned_reserve_microusd"] != reserve_microusd):
                    raise ValueError("different source-first plan already held")
                self.db.execute("COMMIT")
                return StartDecision("pending", workflow_id, "plan_reserved")
            if self.db.execute("SELECT 1 FROM batch_attempts WHERE workflow_id=? LIMIT 1",
                               (workflow_id,)).fetchone() is not None:
                raise ValueError("full plan must precede Batch item reservations")
            if self._rolling_spent_microusd(time.time()) + reserve_microusd > WEEK_CAP_MICROUSD:
                self.db.execute("ROLLBACK")
                return StartDecision("blocked", None, "weekly_budget_exceeded")
            self.db.execute("""UPDATE source_first_workflows SET plan_sha256=?,
                planned_reserve_microusd=?,updated_at=? WHERE id=?""",
                (plan_sha256, reserve_microusd, time.time(), workflow_id))
            self.db.execute("COMMIT")
            return StartDecision("new", workflow_id)
        except Exception:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def list_source_first_workflows(self, *, states: Sequence[str] = ("active",)) -> list[dict]:
        allowed = {"active", "accepted", "failed", "cancelled"}
        if not states or any(state not in allowed for state in states):
            raise ValueError("invalid workflow states")
        placeholders = ",".join("?" for _ in states)
        return [dict(row) for row in self.db.execute(
            f"SELECT * FROM source_first_workflows WHERE status IN ({placeholders}) ORDER BY created_at",
            tuple(states))]

    @staticmethod
    def _validate_batch_payload(path: Path, expected_sha256: str,
                                items: Sequence[BatchItemIntent]) -> None:
        if _sealed_sha256(path) != expected_sha256:
            raise ValueError("Batch payload hash changed")
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError("Batch payload is not JSON") from exc
        requests = envelope.get("requests") if isinstance(envelope, dict) else None
        if (not isinstance(requests, list) or len(requests) != len(items)
                or envelope.get("endpoint") != "/v1/chat/completions"
                or not isinstance(envelope.get("model"), str)
                or not isinstance(envelope.get("provider"), dict)
                or not isinstance(envelope["provider"].get("only"), list)
                or not envelope["provider"]["only"]):
            raise ValueError("Batch payload envelope changed")
        expected = {item.custom_id: item for item in items}
        if len(expected) != len(items):
            raise ValueError("duplicate Batch custom ID")
        seen: set[str] = set()
        for request in requests:
            if not isinstance(request, dict) or set(request) != {"custom_id", "body"}:
                raise ValueError("Batch request shape changed")
            custom_id = request["custom_id"]
            if custom_id not in expected or custom_id in seen:
                raise ValueError("Batch custom IDs changed")
            seen.add(custom_id)
            actual = batch_item_intent_for_body(custom_id, request["body"],
                reserve_microusd=expected[custom_id].reserve_microusd)
            if actual != expected[custom_id]:
                raise ValueError("Batch item request or output cap changed")

    def reserve_batch_intent(self, *, workflow_id: str, intent_key: str,
                             stage: str, credential_id: str,
                             credential_version: int, workspace_id: str | None,
                             payload_path: Path, payload_sha256: str,
                             items: list[BatchItemIntent]) -> BatchIntentDecision:
        """Hold worst-case cost of every generation before an HTTP POST.

        One attempt is one physical Batch POST; items are its separately
        counted generations. No discount from a hoped-for prompt-cache hit is
        applied here. Repeating an identical intent returns its existing row.
        """
        if (not isinstance(workflow_id, str) or not workflow_id
                or not _full_hash(intent_key) or not _full_hash(payload_sha256)
                or stage not in {"writer", "extract", "audit", "global", "repair", "verify"}
                or not isinstance(credential_id, str) or not credential_id
                or type(credential_version) is not int or credential_version < 1
                or workspace_id is not None and (not isinstance(workspace_id, str)
                                                  or not workspace_id)
                or not isinstance(items, list) or not 1 <= len(items) <= MAX_BATCH_ITEMS_PER_WORKFLOW):
            raise ValueError("invalid Batch intent")
        for item in items:
            if (not isinstance(item, BatchItemIntent)
                    or not isinstance(item.custom_id, str)
                    or re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}",
                                    item.custom_id) is None
                    or any(not _full_hash(value) for value in
                           (item.request_sha256, item.input_sha256,
                            item.prompt_sha256, item.schema_sha256))
                    or type(item.max_output_tokens) is not int
                    or not 1 <= item.max_output_tokens <= 128_000
                    or type(item.reserve_microusd) is not int
                    or not 0 < item.reserve_microusd <= JOB_CAP_MICROUSD):
                raise ValueError("invalid Batch item identity or reserve")
        total = sum(item.reserve_microusd for item in items)
        payload_path = Path(payload_path)
        self._validate_batch_payload(payload_path, payload_sha256, items)
        now = time.time()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            workflow = self.db.execute("SELECT * FROM source_first_workflows WHERE id=?",
                                       (workflow_id,)).fetchone()
            if workflow is None:
                raise ValueError("unknown source-first workflow")
            prior = self.db.execute("SELECT * FROM batch_attempts WHERE intent_key=?",
                                    (intent_key,)).fetchone()
            if prior is not None:
                prior_items = [dict(row) for row in self.db.execute(
                    "SELECT * FROM batch_items WHERE attempt_id=? ORDER BY custom_id",
                    (prior["id"],))]
                if (prior["workflow_id"] != workflow_id or prior["stage"] != stage
                        or prior["credential_id"] != credential_id
                        or prior["credential_version"] != credential_version
                        or prior["workspace_id"] != workspace_id
                        or prior["payload_path"] != str(payload_path)
                        or prior["payload_sha256"] != payload_sha256
                        or prior["reserved_microusd"] != total
                        or [(row["custom_id"], row["request_sha256"], row["input_sha256"],
                             row["prompt_sha256"], row["schema_sha256"],
                             row["max_output_tokens"], row["reserved_microusd"])
                            for row in prior_items] != sorted(
                            (item.custom_id, item.request_sha256, item.input_sha256,
                             item.prompt_sha256, item.schema_sha256,
                             item.max_output_tokens, item.reserve_microusd)
                            for item in items)):
                    raise ValueError("Batch intent identity changed")
                self.db.execute("COMMIT")
                return BatchIntentDecision("pending", prior["id"], prior["status"])
            if workflow["status"] != "active":
                self.db.execute("ROLLBACK")
                return BatchIntentDecision("blocked", None, "workflow_not_active")
            if (workflow["plan_sha256"] != workflow["manifest_sha256"]
                    or not 0 < workflow["planned_reserve_microusd"] <= JOB_CAP_MICROUSD):
                self.db.execute("ROLLBACK")
                return BatchIntentDecision("blocked", None, "workflow_plan_not_reserved")
            if stage == "writer" and (
                    credential_id != workflow["credential_id"]
                    or credential_version != workflow["credential_version"]
                    or workspace_id != workflow["workspace_id"]):
                raise ValueError("writer credential identity changed")
            if workspace_id != workflow["workspace_id"]:
                manifest = json.loads(Path(workflow["manifest_path"]).read_text(encoding="utf-8"))
                if (_sealed_sha256(Path(workflow["manifest_path"]))
                        != workflow["manifest_sha256"]
                        or manifest.get("judge_workspace_id") != workspace_id
                        or not isinstance(workspace_id, str) or not workspace_id):
                    raise ValueError("judge workspace was not pinned")
            rows = self.db.execute("SELECT status,post_count,reserved_microusd,billed_microusd "
                                   "FROM batch_attempts WHERE workflow_id=?",
                                   (workflow_id,)).fetchall()
            count = self.db.execute("SELECT COUNT(*) FROM batch_items WHERE attempt_id IN "
                                    "(SELECT id FROM batch_attempts WHERE workflow_id=?)",
                                    (workflow_id,)).fetchone()[0]
            if count + len(items) > MAX_BATCH_ITEMS_PER_WORKFLOW:
                self.db.execute("ROLLBACK")
                return BatchIntentDecision("blocked", None, "logical_job_item_limit")
            if len(rows) >= MAX_BATCH_POSTS_PER_WORKFLOW or sum(
                    row["post_count"] for row in rows) >= MAX_BATCH_POSTS_PER_WORKFLOW:
                self.db.execute("ROLLBACK")
                return BatchIntentDecision("blocked", None, "logical_job_dispatch_limit")
            group_cost = sum(row["billed_microusd"] if row["billed_microusd"] is not None
                             else row["reserved_microusd"] for row in rows
                             if row["status"] not in {"rejected_no_charge",
                                                     "cancelled_before_submit"})
            if (total > JOB_CAP_MICROUSD
                    or group_cost + total > workflow["planned_reserve_microusd"]):
                self.db.execute("ROLLBACK")
                return BatchIntentDecision("blocked", None, "logical_job_budget_exceeded")
            # The full workflow already owns this weekly allowance. Adding
            # item holds a second time would double-count the same money.
            if self._rolling_spent_microusd(now) > WEEK_CAP_MICROUSD:
                self.db.execute("ROLLBACK")
                return BatchIntentDecision("blocked", None, "weekly_budget_exceeded")
            attempt_id = uuid.uuid4().hex
            self.db.execute("""INSERT INTO batch_attempts
                (id,workflow_id,intent_key,stage,status,credential_id,credential_version,
                 workspace_id,payload_path,payload_sha256,reserved_microusd,
                 post_count,created_at,updated_at)
                VALUES (?,?,?,?,'reserved',?,?,?,?,?,?,0,?,?)""",
                (attempt_id, workflow_id, intent_key, stage, credential_id,
                 credential_version, workspace_id, str(payload_path),
                 payload_sha256, total, now, now))
            self.db.executemany("""INSERT INTO batch_items
                (attempt_id,custom_id,request_sha256,input_sha256,prompt_sha256,
                 schema_sha256,max_output_tokens,reserved_microusd,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                [(attempt_id, item.custom_id, item.request_sha256,
                  item.input_sha256, item.prompt_sha256, item.schema_sha256,
                  item.max_output_tokens, item.reserve_microusd, now)
                 for item in items])
            self.db.execute("UPDATE source_first_workflows SET updated_at=? WHERE id=?",
                            (now, workflow_id))
            self.db.execute("COMMIT")
            return BatchIntentDecision("new", attempt_id)
        except Exception:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def get_batch_attempt(self, attempt_id: str) -> dict | None:
        row = self.db.execute("SELECT * FROM batch_attempts WHERE id=?",
                              (attempt_id,)).fetchone()
        return dict(row) if row else None

    def list_batch_attempts(self, workflow_id: str) -> list[dict]:
        return [dict(row) for row in self.db.execute(
            "SELECT * FROM batch_attempts WHERE workflow_id=? ORDER BY created_at,id",
            (workflow_id,))]

    def batch_items(self, attempt_id: str) -> list[dict]:
        return [dict(row) for row in self.db.execute(
            "SELECT * FROM batch_items WHERE attempt_id=? ORDER BY custom_id",
            (attempt_id,))]

    def list_pending_batch_attempts(self, *, now: float | None = None) -> list[dict]:
        """Include ambiguous submissions for recovery, never for blind POST."""
        now = time.time() if now is None else now
        return [dict(row) for row in self.db.execute("""SELECT * FROM batch_attempts
            WHERE status IN ('reserved','submitting','submission_unknown','submitted',
                             'polling','credential_required')
              AND (next_poll_at IS NULL OR next_poll_at<=?)
              AND (lease_until IS NULL OR lease_until<=?)
            ORDER BY created_at""", (now, now))]

    def cancel_batch_before_submit(self, attempt_id: str, reason: str) -> bool:
        """Release a prepared hold only while no physical POST was attempted."""
        if not isinstance(reason, str) or not reason:
            raise ValueError("invalid cancellation reason")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT status,post_count FROM batch_attempts WHERE id=?",
                                  (attempt_id,)).fetchone()
            if row is None:
                raise ValueError("unknown Batch attempt")
            if row["status"] == "cancelled_before_submit":
                self.db.execute("COMMIT")
                return False
            if row["status"] != "reserved" or row["post_count"] != 0:
                raise ValueError("cannot release an attempted Batch hold")
            self.db.execute("UPDATE batch_attempts SET status='cancelled_before_submit',"
                            "error_code=?,updated_at=? WHERE id=?",
                            (reason[:120], time.time(), attempt_id))
            self.db.execute("COMMIT")
            return True
        except Exception:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def mark_batch_submitting(self, attempt_id: str,
                              expected_payload_sha256: str) -> bool:
        """Persist one physical POST intent before touching the network.

        A process that crashes between this commit and the HTTP response must
        treat the submission as unknown. No automatic second POST is allowed.
        """
        if not _full_hash(expected_payload_sha256):
            raise ValueError("invalid payload hash")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT * FROM batch_attempts WHERE id=?",
                                  (attempt_id,)).fetchone()
            if (row is None or row["status"] != "reserved" or row["post_count"] != 0
                    or row["payload_sha256"] != expected_payload_sha256):
                self.db.execute("ROLLBACK")
                return False
            workflow = self.db.execute("SELECT * FROM source_first_workflows WHERE id=?",
                                       (row["workflow_id"],)).fetchone()
            if workflow is None or workflow["status"] != "active":
                self.db.execute("ROLLBACK")
                return False
            if (workflow["plan_sha256"] != workflow["manifest_sha256"]
                    or workflow["planned_reserve_microusd"] <= 0):
                self.db.execute("ROLLBACK")
                return False
            items = [BatchItemIntent(item["custom_id"], item["request_sha256"],
                item["input_sha256"], item["prompt_sha256"], item["schema_sha256"],
                item["max_output_tokens"], item["reserved_microusd"])
                for item in self.db.execute("SELECT * FROM batch_items WHERE attempt_id=?",
                                            (attempt_id,))]
            self._validate_batch_payload(Path(row["payload_path"]),
                                         expected_payload_sha256, items)
            counts = self.db.execute("SELECT COALESCE(SUM(post_count),0) FROM batch_attempts "
                                     "WHERE workflow_id=?", (row["workflow_id"],)).fetchone()[0]
            if counts >= MAX_BATCH_POSTS_PER_WORKFLOW:
                self.db.execute("ROLLBACK")
                return False
            group_cost = self.db.execute("""SELECT COALESCE(SUM(CASE WHEN billed_microusd
                IS NULL THEN reserved_microusd ELSE billed_microusd END),0)
                FROM batch_attempts WHERE workflow_id=? AND status NOT IN
                ('rejected_no_charge','cancelled_before_submit')""",
                (row["workflow_id"],)).fetchone()[0]
            if (group_cost > workflow["planned_reserve_microusd"]
                    or self._rolling_spent_microusd(time.time()) > WEEK_CAP_MICROUSD):
                self.db.execute("ROLLBACK")
                return False
            changed = self.db.execute("UPDATE batch_attempts SET status='submitting',"
                "post_count=1,updated_at=? WHERE id=? AND status='reserved' AND post_count=0",
                (time.time(), attempt_id))
            if changed.rowcount != 1:
                raise ValueError("Batch submission transition lost")
            self.db.execute("COMMIT")
            return True
        except Exception:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def record_batch_submission(self, attempt_id: str, *,
                                remote_id: str | None = None,
                                error_code: str | None = None,
                                definite_rejection: bool = False) -> None:
        """Map a POST to one remote ID or keep its potential charge held."""
        if remote_id is not None and re.fullmatch(
                r"batch[-_][A-Za-z0-9_-]{3,128}", remote_id) is None:
            raise ValueError("invalid remote Batch ID")
        if remote_id is not None and definite_rejection:
            raise ValueError("remote identity cannot be a definite rejection")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT status FROM batch_attempts WHERE id=?",
                                  (attempt_id,)).fetchone()
            if row is None or row["status"] != "submitting":
                raise ValueError("invalid Batch submission transition")
            status = ("submitted" if remote_id else
                      "rejected_no_charge" if definite_rejection else "submission_unknown")
            now = time.time()
            self.db.execute("""UPDATE batch_attempts SET status=?,remote_id=?,error_code=?,
                next_poll_at=?,updated_at=? WHERE id=?""",
                (status, remote_id, error_code[:120] if error_code else None,
                 now + 120 if status != "rejected_no_charge" else None,
                 now, attempt_id))
            self.db.execute("COMMIT")
        except Exception:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def defer_batch_unknown(self, attempt_id: str, *, delay_seconds: int,
                            error_code: str) -> None:
        """Mark an ambiguous POST for human/reconciliation recovery, not retry."""
        if not 30 <= delay_seconds <= 3600 or not error_code:
            raise ValueError("invalid unknown-submission deferral")
        changed = self.db.execute("""UPDATE batch_attempts
            SET status='submission_unknown',next_poll_at=?,error_code=?,updated_at=?
            WHERE id=? AND remote_id IS NULL AND post_count=1
              AND status IN ('submitting','submission_unknown')""",
            (time.time() + delay_seconds, error_code[:120], time.time(), attempt_id))
        if changed.rowcount != 1:
            raise ValueError("invalid unknown-submission state")

    def attach_recovered_batch_remote(self, attempt_id: str, *, remote_id: str,
                                      evidence_path: Path,
                                      evidence_sha256: str) -> bool:
        """Recover a known remote ID from sealed evidence; never resend POST."""
        if (re.fullmatch(r"batch[-_][A-Za-z0-9_-]{3,128}", remote_id or "") is None
                or not _full_hash(evidence_sha256)
                or _sealed_sha256(Path(evidence_path)) != evidence_sha256):
            raise ValueError("invalid remote recovery evidence")
        try:
            evidence = json.loads(Path(evidence_path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError("remote recovery evidence is not JSON") from exc
        if not isinstance(evidence, dict) or evidence.get("id") != remote_id:
            raise ValueError("remote recovery ID differs from evidence")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT * FROM batch_attempts WHERE id=?",
                                  (attempt_id,)).fetchone()
            if row is None:
                raise ValueError("unknown Batch attempt")
            if row["remote_id"] == remote_id:
                if (row["recovery_evidence_path"] != str(evidence_path)
                        or row["recovery_evidence_sha256"] != evidence_sha256):
                    raise ValueError("different remote recovery evidence")
                self.db.execute("COMMIT")
                return False
            if (row["status"] != "submission_unknown" or row["remote_id"] is not None
                    or row["post_count"] != 1):
                raise ValueError("Batch submission was not ambiguous")
            now = time.time()
            self.db.execute("""UPDATE batch_attempts SET status='submitted',remote_id=?,
                recovery_evidence_path=?,recovery_evidence_sha256=?,next_poll_at=?,
                error_code=NULL,updated_at=? WHERE id=?""",
                (remote_id, str(evidence_path), evidence_sha256, now, now, attempt_id))
            self.db.execute("COMMIT")
            return True
        except Exception:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def claim_batch_poll(self, attempt_id: str, owner_id: str, *,
                         lease_seconds: int = 180,
                         now: float | None = None) -> dict | None:
        """Grant at most one worker a bounded GET/terminal-write lease."""
        if (not isinstance(owner_id, str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{7,127}", owner_id) is None
                or type(lease_seconds) is not int
                or not 30 <= lease_seconds <= 1800):
            raise ValueError("invalid Batch poll lease")
        now = time.time() if now is None else now
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT * FROM batch_attempts WHERE id=?",
                                  (attempt_id,)).fetchone()
            if (row is None or row["remote_id"] is None
                    or row["status"] not in {"submitted", "polling", "credential_required"}
                    or row["next_poll_at"] is not None and row["next_poll_at"] > now
                    or row["lease_until"] is not None and row["lease_until"] > now):
                self.db.execute("COMMIT")
                return None
            until = now + lease_seconds
            self.db.execute("""UPDATE batch_attempts
                SET status='polling',lease_owner=?,lease_until=?,updated_at=?
                WHERE id=?""", (owner_id, until, now, attempt_id))
            self.db.execute("COMMIT")
            result = dict(row)
            result.update(status="polling", lease_owner=owner_id, lease_until=until)
            return result
        except Exception:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def record_batch_poll(self, attempt_id: str, owner_id: str,
                          remote_status: str, *, delay_seconds: int = 180,
                          error_code: str | None = None) -> None:
        """Defer another GET; terminal responses use record_batch_terminal."""
        if remote_status not in {"validating", "in_progress", "finalizing", "cancelling"}:
            raise ValueError("terminal Batch response requires saved raw")
        if type(delay_seconds) is not int or not 30 <= delay_seconds <= 3600:
            raise ValueError("invalid Batch poll delay")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT status,lease_owner,lease_until FROM batch_attempts WHERE id=?",
                                  (attempt_id,)).fetchone()
            if (row is None or row["status"] != "polling"
                    or row["lease_owner"] != owner_id
                    or row["lease_until"] is None or row["lease_until"] < time.time()):
                raise ValueError("Batch poll lease expired or changed")
            now = time.time()
            self.db.execute("""UPDATE batch_attempts SET remote_status=?,
                next_poll_at=?,error_code=?,lease_owner=NULL,lease_until=NULL,
                updated_at=? WHERE id=?""",
                (remote_status, now + delay_seconds,
                 error_code[:120] if error_code else None, now, attempt_id))
            self.db.execute("COMMIT")
        except Exception:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def defer_batch_poll(self, attempt_id: str, owner_id: str, *,
                         delay_seconds: int, error_code: str,
                         credential_required: bool = False) -> None:
        """Release a GET lease after transport/auth failure without new POST."""
        if type(delay_seconds) is not int or not 30 <= delay_seconds <= 3600 or not error_code:
            raise ValueError("invalid Batch poll deferral")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT status,lease_owner,lease_until FROM batch_attempts WHERE id=?",
                                  (attempt_id,)).fetchone()
            if (row is None or row["status"] != "polling"
                    or row["lease_owner"] != owner_id
                    or row["lease_until"] is None or row["lease_until"] < time.time()):
                raise ValueError("Batch poll lease expired or changed")
            now = time.time()
            self.db.execute("""UPDATE batch_attempts SET status=?,next_poll_at=?,
                error_code=?,lease_owner=NULL,lease_until=NULL,updated_at=? WHERE id=?""",
                ("credential_required" if credential_required else "polling",
                 now + delay_seconds, error_code[:120], now, attempt_id))
            self.db.execute("COMMIT")
        except Exception:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def record_batch_terminal(self, attempt_id: str, owner_id: str,
                              remote_status: str, *,
                              batch_cost_microusd: int | None,
                              terminal_path: Path, terminal_sha256: str,
                              item_outcomes: dict[str, dict]) -> None:
        """Settle an envelope once, retaining every item's separate outcome.

        Individual item costs are diagnostic only; the envelope's cost is the
        single billed amount. If it is absent, the full reserve stays held.
        The exact raw terminal response must be sealed before this transition.
        """
        if remote_status not in {"completed", "failed", "expired", "cancelled"}:
            raise ValueError("Batch is not terminal")
        if (batch_cost_microusd is not None and
                (type(batch_cost_microusd) is not int or batch_cost_microusd < 0)):
            raise ValueError("invalid Batch cost")
        if not _full_hash(terminal_sha256) or _sealed_sha256(Path(terminal_path)) != terminal_sha256:
            raise ValueError("terminal raw response hash changed")
        try:
            terminal_envelope = json.loads(Path(terminal_path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError("terminal Batch response is not JSON") from exc
        if not isinstance(terminal_envelope, dict) or terminal_envelope.get("status") != remote_status:
            raise ValueError("terminal Batch status mismatch")
        if not isinstance(item_outcomes, dict):
            raise ValueError("missing Batch item outcomes")
        allowed_statuses = {"completed", "item_error", "refusal", "refused", "length",
                            "invalid", "missing", "duplicate", "unavailable", "http_error"}
        cleaned: dict[str, tuple] = {}
        for custom_id, outcome in item_outcomes.items():
            if not isinstance(outcome, dict) or outcome.get("status") not in allowed_statuses:
                raise ValueError("invalid Batch item outcome")
            billed = outcome.get("billed_microusd")
            prompt_tokens = outcome.get("prompt_tokens")
            completion_tokens = outcome.get("completion_tokens")
            for amount in (billed, prompt_tokens, completion_tokens):
                if amount is not None and (type(amount) is not int or amount < 0):
                    raise ValueError("invalid Batch item usage")
            raw_path = outcome.get("raw_path")
            raw_sha256 = outcome.get("raw_sha256")
            if (raw_path is None) != (raw_sha256 is None):
                raise ValueError("incomplete Batch item raw identity")
            if raw_path is not None and (
                    not _full_hash(raw_sha256)
                    or _sealed_sha256(Path(raw_path)) != raw_sha256):
                raise ValueError("Batch item raw hash changed")
            error_code = outcome.get("error_code")
            if error_code is not None and not isinstance(error_code, str):
                raise ValueError("invalid Batch item error code")
            cleaned[custom_id] = (outcome["status"], billed, prompt_tokens,
                completion_tokens, str(raw_path) if raw_path is not None else None,
                raw_sha256, error_code[:120] if error_code else None)
        known_item_cost = sum(values[1] for values in cleaned.values()
                              if values[1] is not None)
        if (batch_cost_microusd is not None
                and known_item_cost > batch_cost_microusd + len(cleaned)):
            raise ValueError("Batch envelope cost is below known item charges")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT * FROM batch_attempts WHERE id=?",
                                  (attempt_id,)).fetchone()
            expected = {item["custom_id"] for item in self.db.execute(
                "SELECT custom_id FROM batch_items WHERE attempt_id=?", (attempt_id,))}
            if (row is None or row["status"] != "polling"
                    or row["lease_owner"] != owner_id
                    or row["lease_until"] is None or row["lease_until"] < time.time()
                    or terminal_envelope.get("id") != row["remote_id"]
                    or set(cleaned) != expected):
                raise ValueError("Batch terminal lease or items changed")
            now = time.time()
            for custom_id, values in cleaned.items():
                self.db.execute("""UPDATE batch_items SET status=?,billed_microusd=?,
                    prompt_tokens=?,completion_tokens=?,raw_path=?,raw_sha256=?,
                    error_code=?,updated_at=? WHERE attempt_id=? AND custom_id=?""",
                    (*values, now, attempt_id, custom_id))
            self.db.execute("""UPDATE batch_attempts SET status=?,remote_status=?,
                billed_microusd=?,terminal_path=?,terminal_sha256=?,next_poll_at=NULL,
                error_code=NULL,lease_owner=NULL,lease_until=NULL,updated_at=?
                WHERE id=?""",
                (remote_status, remote_status, batch_cost_microusd,
                 str(terminal_path), terminal_sha256, now, attempt_id))
            self.db.execute("UPDATE source_first_workflows SET updated_at=? WHERE id=?",
                            (now, row["workflow_id"]))
            self.db.execute("COMMIT")
        except Exception:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def finish_batch_workflow(self, workflow_id: str, *, status: str,
                              error_code: str | None = None,
                              result_path: Path | None = None,
                              result_sha256: str | None = None) -> bool:
        """Finish only after every physical attempt is terminal or cancelled.

        ``accepted`` binds a saved local result; this row does not itself
        publish or alter any legacy current pointer.
        """
        if status not in {"accepted", "failed", "cancelled"}:
            raise ValueError("invalid workflow terminal status")
        if error_code is not None and not isinstance(error_code, str):
            raise ValueError("invalid workflow error code")
        error_code = error_code[:120] if error_code else None
        if status == "accepted":
            if (result_path is None or not _full_hash(result_sha256)
                    or _sealed_sha256(Path(result_path)) != result_sha256):
                raise ValueError("accepted result is not sealed")
        elif result_path is not None or result_sha256 is not None:
            raise ValueError("failed workflow cannot bind an accepted result")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT * FROM source_first_workflows WHERE id=?",
                                  (workflow_id,)).fetchone()
            if row is None:
                raise ValueError("unknown source-first workflow")
            if row["status"] != "active":
                if (row["status"] != status or row["error_code"] != error_code
                        or row["accepted_document_path"] !=
                        (str(result_path) if result_path is not None else None)
                        or row["accepted_document_sha256"] != result_sha256):
                    raise ValueError("different workflow terminal state")
                self.db.execute("COMMIT")
                return False
            open_attempt = self.db.execute("""SELECT id FROM batch_attempts
                WHERE workflow_id=? AND status NOT IN
                ('completed','failed','expired','cancelled','rejected_no_charge',
                 'cancelled_before_submit') LIMIT 1""",
                (workflow_id,)).fetchone()
            if open_attempt is not None:
                raise ValueError("workflow still has an active Batch attempt")
            now = time.time()
            self.db.execute("""UPDATE source_first_workflows SET status=?,error_code=?,
                accepted_document_path=?,accepted_document_sha256=?,updated_at=?
                WHERE id=?""",
                (status, error_code,
                 str(result_path) if result_path is not None else None,
                 result_sha256, now, workflow_id))
            self.db.execute("COMMIT")
            return True
        except Exception:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def reserve(self, *, semantic_key: str, source_sha256: str, output_dir: Path,
                credential_id: str, credential_version: int, workspace_id: str | None,
                max_cost_microusd: int) -> StartDecision:
        if not (0 < max_cost_microusd <= JOB_CAP_MICROUSD):
            return StartDecision("blocked", None, "job_budget_exceeded")
        if len(semantic_key) != 64 or len(source_sha256) != 64:
            raise ValueError("invalid semantic identity")
        now = time.time()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            prior = self.db.execute("SELECT * FROM jobs WHERE semantic_key=?", (semantic_key,)).fetchone()
            if prior:
                self.db.execute("""INSERT INTO consumers (semantic_key,output_dir,updated_at) VALUES (?,?,?)
                    ON CONFLICT(semantic_key,output_dir) DO UPDATE SET updated_at=excluded.updated_at""",
                    (semantic_key, str(output_dir), now))
                self.db.execute("COMMIT")
                if prior["status"] == "accepted":
                    return StartDecision("accepted", prior["id"])
                return StartDecision("pending", prior["id"], prior["status"])
            if self.db.execute("SELECT 1 FROM source_first_workflows WHERE semantic_key=?",
                               (semantic_key,)).fetchone() is not None:
                raise ValueError("semantic identity belongs to a source-first workflow")
            if self._rolling_spent_microusd(now) + max_cost_microusd > WEEK_CAP_MICROUSD:
                self.db.execute("ROLLBACK")
                return StartDecision("blocked", None, "weekly_budget_exceeded")
            job_id = uuid.uuid4().hex
            artifact_dir = self.root / "jobs" / job_id
            custom_id = "summary-" + job_id
            self.db.execute("""INSERT INTO jobs
              (id,semantic_key,source_sha256,output_dir,artifact_dir,status,
               credential_id,credential_version,workspace_id,custom_id,remote_id,
               reserved_microusd,billed_microusd,dispatches,created_at,updated_at)
              VALUES (?,?,?,?,?,'reserved',?,?,?,?,NULL,?,NULL,0,?,?)""",
              (job_id, semantic_key, source_sha256, str(output_dir), str(artifact_dir),
               credential_id, credential_version, workspace_id, custom_id,
               max_cost_microusd, now, now))
            self.db.execute("INSERT INTO consumers (semantic_key,output_dir,updated_at) VALUES (?,?,?)",
                            (semantic_key, str(output_dir), now))
            self.db.execute("COMMIT")
            _secure_dir(artifact_dir)
            return StartDecision("new", job_id)
        except Exception:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def get(self, job_id: str) -> dict | None:
        row = self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def by_semantic_key(self, semantic_key: str) -> dict | None:
        row = self.db.execute("SELECT * FROM jobs WHERE semantic_key=?", (semantic_key,)).fetchone()
        return dict(row) if row else None

    def attach_consumer(self, semantic_key: str, source_sha256: str, output_dir: Path) -> dict | None:
        """Persist a same-input consumer before returning an in-flight job."""
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT * FROM jobs WHERE semantic_key=?", (semantic_key,)).fetchone()
            if row is None:
                self.db.execute("COMMIT")
                return None
            if row["source_sha256"] != source_sha256:
                raise ValueError("semantic_source_identity_mismatch")
            self.db.execute("""INSERT INTO consumers (semantic_key,output_dir,updated_at) VALUES (?,?,?)
                ON CONFLICT(semantic_key,output_dir) DO UPDATE SET updated_at=excluded.updated_at""",
                (semantic_key, str(output_dir), time.time()))
            self.db.execute("COMMIT")
            return dict(row)
        except Exception:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def consumers(self, semantic_key: str) -> list[Path]:
        return [Path(row[0]) for row in self.db.execute("SELECT output_dir FROM consumers WHERE semantic_key=?", (semantic_key,))]

    def consumer_rows(self, semantic_key: str, *, status: str | None = None) -> list[dict]:
        query = "SELECT * FROM consumers WHERE semantic_key=?"
        arguments = [semantic_key]
        if status is not None:
            query += " AND status=?"
            arguments.append(status)
        query += " ORDER BY output_dir"
        return [dict(row) for row in self.db.execute(query, arguments)]

    def mark_consumer(self, semantic_key: str, output_dir: Path, *, generation_id: str | None = None,
                      error_code: str | None = None) -> None:
        if (generation_id is None) == (error_code is None):
            raise ValueError("consumer outcome must have generation or error")
        status = "published" if generation_id is not None else "failed"
        changed = self.db.execute("""UPDATE consumers SET status=?,generation_id=?,error_code=?,updated_at=?
            WHERE semantic_key=? AND output_dir=?""",
            (status, generation_id, error_code[:120] if error_code else None, time.time(),
             semantic_key, str(output_dir)))
        if changed.rowcount != 1:
            raise ValueError("consumer registration missing")

    def accepted_with_pending_consumers(self) -> list[dict]:
        return [dict(row) for row in self.db.execute("""SELECT DISTINCT j.* FROM jobs AS j
            JOIN consumers AS c ON c.semantic_key=j.semantic_key
            WHERE j.status='accepted' AND c.status='pending' ORDER BY j.created_at""")]

    def active_summary_jobs_for_credential(self, credential_id: str) -> int:
        return self.db.execute("""SELECT COUNT(*) FROM jobs WHERE credential_id=?
          AND status IN ('reserved','submitting','submission_unknown','submitted','polling','credential_required')""",
          (credential_id,)).fetchone()[0]

    def mark_submitting(self, job_id: str) -> bool:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT status,dispatches FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row or row["status"] != "reserved" or row["dispatches"] >= MAX_DISPATCHES_PER_JOB:
                self.db.execute("ROLLBACK")
                return False
            self.db.execute("UPDATE jobs SET status='submitting',dispatches=dispatches+1,updated_at=? WHERE id=?", (time.time(), job_id))
            self.db.execute("COMMIT")
            return True
        except Exception:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def cancel_before_submit(self, job_id: str, code: str) -> None:
        self.db.execute("UPDATE jobs SET status='cancelled_before_submit',error_code=?,updated_at=? WHERE id=? AND status='reserved'",
                        (code, time.time(), job_id))

    def submission_result(self, job_id: str, *, remote_id: str | None, error_code: str | None = None,
                          definite_rejection: bool = False) -> None:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row or row["status"] != "submitting":
                raise ValueError("invalid submission transition")
            status = "submitted" if remote_id else ("rejected_before_submit" if definite_rejection else "submission_unknown")
            self.db.execute("UPDATE jobs SET status=?,remote_id=?,error_code=?,updated_at=?,next_poll_at=? WHERE id=?",
                            (status, remote_id, error_code, time.time(), time.time()+120 if remote_id else None, job_id))
            self.db.execute("COMMIT")
        except Exception:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def pending_remote(self, *, now: float | None = None) -> list[dict]:
        now = time.time() if now is None else now
        return [dict(row) for row in self.db.execute("""SELECT * FROM jobs
          WHERE remote_id IS NOT NULL AND status IN ('submitted','polling','credential_required')
          AND (next_poll_at IS NULL OR next_poll_at<=?) ORDER BY created_at""", (now,))]

    def raw_ready(self, *, now: float | None = None) -> list[dict]:
        now = time.time() if now is None else now
        return [dict(row) for row in self.db.execute(
            "SELECT * FROM jobs WHERE status='completed_raw' AND (next_poll_at IS NULL OR next_poll_at<=?) ORDER BY created_at",
            (now,))]

    def defer_raw(self, job_id: str, *, delay_seconds: int, error_code: str) -> None:
        """A sealed but unpublished result stays recoverable without another POST."""
        if not 30 <= delay_seconds <= 3600:
            raise ValueError("invalid raw retry delay")
        self.db.execute("UPDATE jobs SET next_poll_at=?,error_code=?,updated_at=? WHERE id=? AND status='completed_raw'",
            (time.time() + delay_seconds, error_code[:120], time.time(), job_id))

    def poll_result(self, job_id: str, remote_status: str, *, usage_cost_microusd: int | None = None,
                    error_code: str | None = None) -> None:
        if remote_status not in {'validating','in_progress','finalizing','cancelling','completed','failed','expired','cancelled'}:
            raise ValueError("unknown remote status")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row or row["status"] not in {"submitted", "polling", "credential_required"}:
                raise ValueError("invalid poll transition")
            status = "completed_raw" if remote_status == "completed" else ("remote_" + remote_status if remote_status in {"failed","expired","cancelled"} else "polling")
            next_poll = None if status != "polling" else time.time() + 180
            self.db.execute("UPDATE jobs SET status=?,billed_microusd=COALESCE(?,billed_microusd),error_code=?,updated_at=?,next_poll_at=? WHERE id=?",
                (status, usage_cost_microusd, error_code, time.time(), next_poll, job_id))
            self.db.execute("COMMIT")
        except Exception:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def accepted(self, job_id: str, document_path: Path, generation_id: str) -> None:
        changed = self.db.execute("UPDATE jobs SET status='accepted',accepted_document_path=?,generation_id=?,updated_at=? WHERE id=? AND status='completed_raw'",
            (str(document_path), generation_id, time.time(), job_id))
        if changed.rowcount != 1:
            raise ValueError("invalid accepted transition")

    def failed_validation(self, job_id: str, code: str) -> None:
        self.db.execute("UPDATE jobs SET status='failed_validation',error_code=?,updated_at=? WHERE id=? AND status='completed_raw'",
            (code, time.time(), job_id))

    def credential_required(self, job_id: str) -> None:
        self.db.execute("UPDATE jobs SET status='credential_required',updated_at=?,next_poll_at=? WHERE id=? AND status IN ('submitted','polling','credential_required')",
            (time.time(), time.time() + 600, job_id))

    def defer_poll(self, job_id: str, *, delay_seconds: int, error_code: str) -> None:
        """Keep one remote identity while a GET or credential check is unavailable."""
        if not 30 <= delay_seconds <= 3600 or not error_code:
            raise ValueError("invalid poll deferral")
        self.db.execute("UPDATE jobs SET status='polling',next_poll_at=?,error_code=?,updated_at=? "
                        "WHERE id=? AND status IN ('submitted','polling','credential_required')",
                        (time.time() + delay_seconds, error_code[:120], time.time(), job_id))
