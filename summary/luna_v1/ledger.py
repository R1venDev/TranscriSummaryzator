"""Durable single-flight and shared USD budget for Luna summary requests."""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


WEEK_SECONDS = 7 * 24 * 60 * 60
WEEK_CAP_MICROUSD = 1_000_000
JOB_CAP_MICROUSD = 100_000
MAX_DISPATCHES_PER_JOB = 6


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


@dataclass(frozen=True)
class StartDecision:
    kind: str  # new, pending, accepted, blocked
    job_id: str | None
    reason: str | None = None


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
          CREATE INDEX IF NOT EXISTS jobs_by_created ON jobs(created_at);
          CREATE INDEX IF NOT EXISTS jobs_by_credential ON jobs(credential_id,status);
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
            self.db.execute("COMMIT")
        except Exception:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise
        os.chmod(self.db_path, 0o600)

    def close(self) -> None:
        self.db.close()

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
            spent = self.db.execute(
                "SELECT COALESCE(SUM(CASE WHEN billed_microusd IS NULL THEN reserved_microusd ELSE billed_microusd END),0) "
                "FROM jobs WHERE created_at>=? AND status NOT IN ('rejected_before_submit', 'cancelled_before_submit')",
                (now - WEEK_SECONDS,),
            ).fetchone()[0]
            if spent + max_cost_microusd > WEEK_CAP_MICROUSD:
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
