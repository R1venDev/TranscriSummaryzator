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


WEEK_SECONDS = 7 * 24 * 60 * 60
WEEK_CAP_MICROUSD = 1_000_000
# A single explicitly registered Opus v3 run may use the user's $2 rolling
# weekly ceiling. All jobs, including earlier ones, still contribute to the
# same rolling sum; unrelated runs keep WEEK_CAP_MICROUSD.
OPUS_V3_TRIAL_WEEK_CAP_MICROUSD = 2_000_000
OPUS_V3_TRIAL_POLICY = "claude_opus_5_5_partitioned_audit_v3"
OPUS_DIRECT_POLICY = "claude_opus_5_5_direct_writer_v1"
OPUS_DIRECT_KIND = "opus_direct_writer"
OPUS_DIRECT_WEEK_CAP_MICROUSD = 1_600_000
OPUS_DIRECT_CALL_CAP_MICROUSD = 550_000
JOB_CAP_MICROUSD = 100_000
# Opus Batch must reserve the entire full-source input and hidden thinking
# allowance. The user's dynamic-budget decision applies only to this new
# quality policy; the shared rolling $1 cap remains unchanged.
OPUS_STAGE_CAP_MICROUSD = 450_000
OPUS_GROUP_CAP_MICROUSD = 600_000
MAX_DISPATCHES_PER_JOB = 6
DIAGNOSTIC_KINDS = frozenset({"diagnostic_baseline", "diagnostic_compact"})
DIAGNOSTIC_MAX_DISPATCHES = 2
DIAGNOSTIC_GROUP_CAP_MICROUSD = JOB_CAP_MICROUSD
QUALITY_STAGE_KINDS = frozenset({"audit", "verify", "inventory_1", "inventory_2",
                                 "inventory_3", "reconcile", "reconcile_1", "reconcile_2",
                                 "segment_1", "segment_2",
                                 "segment_3"})


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
            generation_id TEXT,
            kind TEXT NOT NULL DEFAULT 'summary',
            root_job_id TEXT,
            billing_group_id TEXT,
            writer_parent_id TEXT
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
          CREATE TABLE IF NOT EXISTS group_spending_authorizations (
            billing_group_id TEXT PRIMARY KEY,
            original_root_id TEXT NOT NULL,
            policy_version TEXT NOT NULL,
            authorization_ref TEXT NOT NULL,
            authorized_at REAL NOT NULL
          );
          CREATE TABLE IF NOT EXISTS weekly_spending_authorizations (
            semantic_key TEXT PRIMARY KEY,
            source_sha256 TEXT NOT NULL,
            output_dir TEXT NOT NULL,
            quality_policy_version TEXT NOT NULL,
            authorization_ref TEXT NOT NULL,
            cap_microusd INTEGER NOT NULL,
            authorized_at REAL NOT NULL,
            root_job_id TEXT UNIQUE
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
            job_columns = {row[1] for row in self.db.execute("PRAGMA table_info(jobs)")}
            if "kind" not in job_columns:
                self.db.execute("ALTER TABLE jobs ADD COLUMN kind TEXT NOT NULL DEFAULT 'summary'")
            if "root_job_id" not in job_columns:
                self.db.execute("ALTER TABLE jobs ADD COLUMN root_job_id TEXT")
            self.db.execute("UPDATE jobs SET root_job_id=id WHERE root_job_id IS NULL")
            if "billing_group_id" not in job_columns:
                self.db.execute("ALTER TABLE jobs ADD COLUMN billing_group_id TEXT")
            if "writer_parent_id" not in job_columns:
                self.db.execute("ALTER TABLE jobs ADD COLUMN writer_parent_id TEXT")
            # Legacy roots and their quality stages already form one logical
            # budget group. A continuation gets a fresh stage namespace while
            # retaining this immutable billing/dispatch history.
            self.db.execute("UPDATE jobs SET billing_group_id=root_job_id WHERE billing_group_id IS NULL")
            self.db.execute("CREATE INDEX IF NOT EXISTS jobs_by_billing_group ON jobs(billing_group_id)")
            self.db.execute("COMMIT")
        except Exception:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise
        os.chmod(self.db_path, 0o600)

    def close(self) -> None:
        self.db.close()

    def authorize_dynamic_group(self, prior_failed_root_id: str,
                                authorization_ref: str) -> bool:
        """Allow tariff-sized future stages for exactly one saved writer group.

        The caller must provide a reference to an explicit user decision. This
        removes only the cumulative $0.10 logical-job limit for the named
        terminal writer lineage. The per-dispatch $0.10, rolling $1/week and
        six-dispatch guards stay in ``reserve`` / ``mark_submitting``.
        """
        if (not isinstance(authorization_ref, str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{7,159}", authorization_ref) is None):
            raise ValueError("invalid dynamic spending authorization reference")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            root = self.db.execute("SELECT * FROM jobs WHERE id=?",
                                   (prior_failed_root_id,)).fetchone()
            if (root is None or root["kind"] != "summary"
                    or root["status"] != "failed_validation"
                    or root["billing_group_id"] != root["id"]
                    or root["writer_parent_id"] is not None
                    or root["remote_id"] is None or root["billed_microusd"] is None
                    or root["dispatches"] != 1
                    or root["accepted_document_path"] is not None
                    or root["generation_id"] is not None):
                raise ValueError("dynamic spending requires a terminal failed original writer")
            existing = self.db.execute("SELECT * FROM group_spending_authorizations WHERE billing_group_id=?",
                                       (root["billing_group_id"],)).fetchone()
            if existing is not None:
                if (existing["original_root_id"] != prior_failed_root_id
                        or existing["policy_version"] != "rolling_week_dynamic_v1"
                        or existing["authorization_ref"] != authorization_ref):
                    raise ValueError("different dynamic spending authorization already recorded")
                self.db.execute("COMMIT")
                return False
            active = self.db.execute("""SELECT id FROM jobs WHERE billing_group_id=? AND status IN
                ('reserved','preparing','quality_pending','submitting','submission_unknown','submitted',
                 'polling','credential_required','completed_raw') LIMIT 1""",
                (root["billing_group_id"],)).fetchone()
            if active is not None:
                raise ValueError("dynamic spending group has an in-flight job")
            self.db.execute("""INSERT INTO group_spending_authorizations
                (billing_group_id,original_root_id,policy_version,authorization_ref,authorized_at)
                VALUES (?,?,?,?,?)""",
                (root["billing_group_id"], root["id"], "rolling_week_dynamic_v1",
                 authorization_ref, time.time()))
            self.db.execute("COMMIT")
            return True
        except Exception:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def logical_group_cap_microusd(self, billing_group_id: str) -> int | None:
        """Return the default cap, or None for an explicitly authorized group.

        None does not mean unmetered: each physical reservation is still
        checked against the rolling weekly cap inside a write transaction.
        """
        if not isinstance(billing_group_id, str) or not billing_group_id:
            raise ValueError("invalid billing group")
        row = self.db.execute("""SELECT policy_version FROM group_spending_authorizations
            WHERE billing_group_id=?""", (billing_group_id,)).fetchone()
        return None if row is not None and row["policy_version"] == "rolling_week_dynamic_v1" else JOB_CAP_MICROUSD

    def authorize_weekly_cap_for_run(self, *, semantic_key: str,
                                     source_sha256: str, output_dir: Path,
                                     quality_policy_version: str,
                                     authorization_ref: str) -> bool:
        """Pin the user's $2 decision to one *future* isolated Opus v3 run.

        This is an administrator-side operation on the private ledger, never a
        request or UI field. It does not alter the shared weekly sum, other
        runs' $1 ceiling, stage limits, or the six-dispatch limit. The exact
        semantic/source/output/policy identity is checked again atomically
        when the writer reserves its first physical request.
        """
        if (not isinstance(semantic_key, str)
                or re.fullmatch(r"[a-f0-9]{64}", semantic_key) is None
                or not isinstance(source_sha256, str)
                or re.fullmatch(r"[a-f0-9]{64}", source_sha256) is None
                or quality_policy_version != OPUS_V3_TRIAL_POLICY
                or not isinstance(authorization_ref, str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{7,159}",
                                authorization_ref) is None):
            raise ValueError("invalid isolated weekly authorization")
        output_dir = str(Path(output_dir))
        self.db.execute("BEGIN IMMEDIATE")
        try:
            existing = self.db.execute(
                "SELECT * FROM weekly_spending_authorizations WHERE semantic_key=?",
                (semantic_key,)).fetchone()
            if existing is not None:
                if (existing["source_sha256"] != source_sha256
                        or existing["output_dir"] != output_dir
                        or existing["quality_policy_version"] != quality_policy_version
                        or existing["authorization_ref"] != authorization_ref
                        or existing["cap_microusd"] != OPUS_V3_TRIAL_WEEK_CAP_MICROUSD):
                    raise ValueError("different isolated weekly authorization already recorded")
                self.db.execute("COMMIT")
                return False
            if self.db.execute("SELECT 1 FROM weekly_spending_authorizations LIMIT 1").fetchone():
                raise ValueError("isolated weekly authorization already assigned to another run")
            if self.db.execute("SELECT 1 FROM jobs WHERE semantic_key=?",
                               (semantic_key,)).fetchone() is not None:
                raise ValueError("isolated weekly authorization requires a future run")
            self.db.execute("""INSERT INTO weekly_spending_authorizations
                (semantic_key,source_sha256,output_dir,quality_policy_version,
                 authorization_ref,cap_microusd,authorized_at,root_job_id)
                VALUES (?,?,?,?,?,?,?,NULL)""",
                (semantic_key, source_sha256, output_dir, quality_policy_version,
                 authorization_ref, OPUS_V3_TRIAL_WEEK_CAP_MICROUSD, time.time()))
            self.db.execute("COMMIT")
            return True
        except Exception:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def weekly_cap_microusd(self, billing_group_id: str) -> int:
        """Return the ceiling allowed for this group, never its own balance."""
        row = self.db.execute("""SELECT cap_microusd,quality_policy_version
            FROM weekly_spending_authorizations WHERE root_job_id=?""",
            (billing_group_id,)).fetchone()
        if row is not None:
            if (row["quality_policy_version"] == OPUS_V3_TRIAL_POLICY
                    and row["cap_microusd"] == OPUS_V3_TRIAL_WEEK_CAP_MICROUSD):
                return OPUS_V3_TRIAL_WEEK_CAP_MICROUSD
            if (row["quality_policy_version"] == OPUS_DIRECT_POLICY
                    and row["cap_microusd"] == OPUS_DIRECT_WEEK_CAP_MICROUSD):
                return OPUS_DIRECT_WEEK_CAP_MICROUSD
        return WEEK_CAP_MICROUSD

    def weekly_authorization_for_root(self, root_job_id: str) -> dict | None:
        row = self.db.execute("""SELECT authorization_ref,cap_microusd,
            quality_policy_version FROM weekly_spending_authorizations
            WHERE root_job_id=?""", (root_job_id,)).fetchone()
        return dict(row) if row is not None else None

    def reserve(self, *, semantic_key: str, source_sha256: str, output_dir: Path,
                credential_id: str, credential_version: int, workspace_id: str | None,
                max_cost_microusd: int, kind: str = "summary",
                root_job_id: str | None = None,
                continuation_of_job_id: str | None = None,
                continuation_mode: str = "failed_writer",
                dynamic_group_authorization_ref: str | None = None,
                quality_policy_version: str | None = None) -> StartDecision:
        is_continuation = continuation_of_job_id is not None
        if (continuation_mode not in {"failed_writer", "accepted_inventory_unavailable"}
                or (continuation_mode != "failed_writer" and not is_continuation)):
            raise ValueError("invalid continuation mode")
        stage_cap = (OPUS_STAGE_CAP_MICROUSD if kind in {"audit", "verify",
                                                       "segment_1", "segment_2", "segment_3"}
                     else JOB_CAP_MICROUSD)
        if not (max_cost_microusd == 0 if is_continuation
                else 0 < max_cost_microusd <= stage_cap):
            return StartDecision("blocked", None, "job_budget_exceeded")
        if kind not in {"summary", *QUALITY_STAGE_KINDS} or (kind == "summary") != (root_job_id is None):
            raise ValueError("invalid summary stage identity")
        if is_continuation and kind != "summary":
            raise ValueError("only a summary root can continue a saved writer")
        if dynamic_group_authorization_ref is not None:
            if (kind != "summary" or root_job_id is not None or is_continuation
                    or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{7,159}",
                                    dynamic_group_authorization_ref) is None):
                raise ValueError("dynamic group authorization requires a new summary root")
        if quality_policy_version is not None and (kind != "summary" or is_continuation):
            raise ValueError("quality policy identity belongs to a new summary root")
        if len(semantic_key) != 64 or len(source_sha256) != 64:
            raise ValueError("invalid semantic identity")
        now = time.time()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            writer_parent_id = None
            if is_continuation:
                parent = self.db.execute("SELECT * FROM jobs WHERE id=?", (continuation_of_job_id,)).fetchone()
                if (parent is None or parent["kind"] != "summary"
                        or parent["source_sha256"] != source_sha256
                        or parent["output_dir"] != str(output_dir)):
                    raise ValueError("saved writer is not a matching summary")
                if continuation_mode == "failed_writer":
                    if (parent["status"] != "failed_validation"
                            or parent["accepted_document_path"] is not None
                            or parent["generation_id"] is not None):
                        raise ValueError("saved writer is not a matching terminal failed summary")
                elif (parent["status"] != "accepted"
                      or parent["credential_id"] != credential_id
                      or parent["credential_version"] != credential_version
                      or parent["workspace_id"] != workspace_id):
                    raise ValueError("accepted inventory continuation identity mismatch")
                else:
                    self._check_accepted_inventory_unavailable(parent)
                writer_parent_id = parent["writer_parent_id"] or parent["id"]
                writer = self.db.execute("SELECT * FROM jobs WHERE id=?", (writer_parent_id,)).fetchone()
                draft_path = (Path(writer["artifact_dir"]) / "draft_document.json") if writer else None
                if (writer is None or writer["kind"] != "summary"
                        or writer["source_sha256"] != source_sha256
                        or writer["output_dir"] != str(output_dir)):
                    raise ValueError("saved writer identity mismatch")
                if (writer["credential_id"] != credential_id
                        or writer["credential_version"] != credential_version
                        or writer["workspace_id"] != workspace_id):
                    raise ValueError("saved writer credential scope mismatch")
                if draft_path.is_symlink() or not draft_path.is_file():
                    raise ValueError("sealed writer draft is unavailable")
            prior = self.db.execute("SELECT * FROM jobs WHERE semantic_key=?", (semantic_key,)).fetchone()
            if prior:
                if (prior["kind"] != kind or prior["source_sha256"] != source_sha256
                        or prior["writer_parent_id"] != writer_parent_id
                        or (is_continuation and prior["output_dir"] != str(output_dir))
                        or (root_job_id is not None and prior["root_job_id"] != root_job_id)):
                    raise ValueError("semantic identity belongs to a different job")
                if kind == "summary":
                    self.db.execute("""INSERT INTO consumers (semantic_key,output_dir,updated_at) VALUES (?,?,?)
                        ON CONFLICT(semantic_key,output_dir) DO UPDATE SET updated_at=excluded.updated_at""",
                        (semantic_key, str(output_dir), now))
                self.db.execute("COMMIT")
                if prior["status"] == "accepted":
                    return StartDecision("accepted", prior["id"])
                return StartDecision("pending", prior["id"], prior["status"])
            if is_continuation:
                billing_group_id = parent["billing_group_id"]
                active = self.db.execute("""SELECT id FROM jobs WHERE billing_group_id=? AND status IN
                    ('reserved','preparing','quality_pending','submitting','submission_unknown','submitted',
                     'polling','credential_required','completed_raw') LIMIT 1""",
                    (billing_group_id,)).fetchone()
                if active is not None:
                    self.db.execute("ROLLBACK")
                    return StartDecision("blocked", None, "continuation_group_inflight")
            elif root_job_id is not None:
                root = self.db.execute("SELECT * FROM jobs WHERE id=?", (root_job_id,)).fetchone()
                if (root is None or root["kind"] != "summary"
                        or root["source_sha256"] != source_sha256):
                    raise ValueError("invalid quality root or workspace")
                if (self.weekly_cap_microusd(root["billing_group_id"])
                        == OPUS_V3_TRIAL_WEEK_CAP_MICROUSD
                        and root["output_dir"] != str(output_dir)):
                    raise ValueError("isolated weekly authorization output changed")
                if root["writer_parent_id"] is not None and root["status"] != "quality_pending":
                    self.db.execute("ROLLBACK")
                    return StartDecision("blocked", None, "continuation_not_ready")
                root_manifest_path = Path(root["artifact_dir"]) / "manifest.json"
                root_manifest = {}
                if root_manifest_path.is_file() and not root_manifest_path.is_symlink():
                    try:
                        root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
                    except (OSError, ValueError) as exc:
                        raise ValueError("quality root manifest unavailable") from exc
                opus_version = root_manifest.get("quality_policy_version")
                opus_policy = (
                    root_manifest.get("quality_provider") == "openrouter_claude_opus"
                    and ((opus_version == "claude_opus_5_5_full_source_audit_v1"
                          and kind in {"audit", "verify"})
                         or (opus_version in {"claude_opus_5_5_partitioned_audit_v2",
                                               "claude_opus_5_5_partitioned_audit_v3"}
                             and kind in {"segment_1", "segment_2", "segment_3"}))
                )
                if max_cost_microusd > JOB_CAP_MICROUSD and not opus_policy:
                    self.db.execute("ROLLBACK")
                    return StartDecision("blocked", None, "job_budget_exceeded")
                if root["workspace_id"] != workspace_id:
                    # The separately selected Gemini judge can use another
                    # OpenRouter workspace only when the writer pinned that
                    # exact scope before its own POST. Legacy Luna audit stays
                    # in the writer workspace.
                    gemini_policy = (root_manifest.get("quality_provider") == "openrouter_gemini"
                                     and root_manifest.get("quality_policy_version") in {
                                         "gemini_openrouter_judge_repair_v2",
                                         "gemini_openrouter_judge_repair_v3_source_inventory",
                                         "gemini_openrouter_source_inventory_reconcile_v1",
                                         "gemini_openrouter_source_inventory_reconcile_v2",
                                         "gemini_openrouter_segment_review_v1",
                                         "gemini_openrouter_segment_review_v2_compact",
                                         "gemini_openrouter_segment_review_v3_packed",
                                     })
                    if (not (gemini_policy or opus_policy)
                            or root_manifest.get("judge_workspace_id") != workspace_id
                            or not isinstance(workspace_id, str) or not workspace_id
                            or credential_id == root["credential_id"]):
                        raise ValueError("invalid quality root or workspace")
                billing_group_id = root["billing_group_id"]
            isolated_weekly = None
            if not is_continuation and root_job_id is None:
                isolated_weekly = self.db.execute("""SELECT * FROM weekly_spending_authorizations
                    WHERE semantic_key=?""", (semantic_key,)).fetchone()
                if isolated_weekly is not None and (
                        isolated_weekly["source_sha256"] != source_sha256
                        or isolated_weekly["output_dir"] != str(output_dir)
                        or isolated_weekly["quality_policy_version"] != quality_policy_version
                        or isolated_weekly["root_job_id"] is not None):
                    raise ValueError("isolated weekly authorization identity changed")
            if is_continuation or root_job_id is not None:
                group_cost = self.db.execute("""SELECT COALESCE(SUM(
                    CASE WHEN billed_microusd IS NULL THEN reserved_microusd ELSE billed_microusd END),0)
                    FROM jobs WHERE billing_group_id=? AND status NOT IN
                    ('rejected_before_submit','cancelled_before_submit')""", (billing_group_id,)).fetchone()[0]
                group_cap = (OPUS_GROUP_CAP_MICROUSD
                             if root_job_id is not None and opus_policy
                             else self.logical_group_cap_microusd(billing_group_id))
                if group_cap is not None and group_cost + max_cost_microusd > group_cap:
                    self.db.execute("ROLLBACK")
                    return StartDecision("blocked", None, "logical_job_budget_exceeded")
                group_dispatches = self.db.execute(
                    "SELECT COALESCE(SUM(dispatches),0) FROM jobs WHERE billing_group_id=?",
                    (billing_group_id,),
                ).fetchone()[0]
                if group_dispatches >= MAX_DISPATCHES_PER_JOB:
                    self.db.execute("ROLLBACK")
                    return StartDecision("blocked", None, "logical_job_dispatch_limit")
            spent = self.db.execute(
                "SELECT COALESCE(SUM(CASE WHEN billed_microusd IS NULL THEN reserved_microusd ELSE billed_microusd END),0) "
                "FROM jobs WHERE created_at>=? AND status NOT IN ('rejected_before_submit', 'cancelled_before_submit')",
                (now - WEEK_SECONDS,),
            ).fetchone()[0]
            weekly_cap = (OPUS_V3_TRIAL_WEEK_CAP_MICROUSD if isolated_weekly is not None
                          else self.weekly_cap_microusd(billing_group_id)
                          if (root_job_id is not None and opus_policy
                              and opus_version == OPUS_V3_TRIAL_POLICY)
                          else WEEK_CAP_MICROUSD)
            if spent + max_cost_microusd > weekly_cap:
                self.db.execute("ROLLBACK")
                return StartDecision("blocked", None, "weekly_budget_exceeded")
            job_id = uuid.uuid4().hex
            group_id = root_job_id or job_id
            billing_group_id = billing_group_id if (is_continuation or root_job_id is not None) else job_id
            artifact_dir = self.root / "jobs" / job_id
            custom_id = kind + "-" + job_id
            self.db.execute("""INSERT INTO jobs
              (id,semantic_key,source_sha256,output_dir,artifact_dir,status,
               credential_id,credential_version,workspace_id,custom_id,remote_id,
               reserved_microusd,billed_microusd,dispatches,created_at,updated_at,kind,root_job_id,
               billing_group_id,writer_parent_id)
              VALUES (?,?,?,?,?,?,?,?,?,?,NULL,?,NULL,0,?,?,?,?,?,?)""",
              (job_id, semantic_key, source_sha256, str(output_dir), str(artifact_dir),
               "preparing" if is_continuation else "reserved",
               credential_id, credential_version, workspace_id, custom_id,
               max_cost_microusd, now, now, kind, group_id, billing_group_id,
               writer_parent_id))
            if dynamic_group_authorization_ref is not None:
                # The user's decision is pinned to this newly created logical
                # run. The weekly, per-dispatch and six-call guards still apply.
                self.db.execute("""INSERT INTO group_spending_authorizations
                    (billing_group_id,original_root_id,policy_version,authorization_ref,authorized_at)
                    VALUES (?,?,?,?,?)""",
                    (job_id, job_id, "rolling_week_dynamic_v1",
                     dynamic_group_authorization_ref, now))
            if isolated_weekly is not None:
                changed = self.db.execute("""UPDATE weekly_spending_authorizations
                    SET root_job_id=? WHERE semantic_key=? AND root_job_id IS NULL""",
                    (job_id, semantic_key))
                if changed.rowcount != 1:
                    raise ValueError("isolated weekly authorization already bound")
            if kind == "summary":
                self.db.execute("INSERT INTO consumers (semantic_key,output_dir,updated_at) VALUES (?,?,?)",
                                (semantic_key, str(output_dir), now))
            self.db.execute("COMMIT")
            _secure_dir(artifact_dir)
            return StartDecision("new", job_id)
        except Exception:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def reserve_diagnostic(self, *, run_id: str, kind: str, semantic_key: str,
                           source_sha256: str, output_dir: Path,
                           credential_id: str, credential_version: int,
                           workspace_id: str, max_cost_microusd: int) -> StartDecision:
        """Reserve one of two fixed, nonpublishing Batch comparison slots.

        The same database transaction guards the application-wide rolling
        weekly cap, the diagnostic group cap and idempotent request identity.
        This does not create a summary consumer or a publication candidate.
        """
        if (not isinstance(run_id, str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{7,127}", run_id) is None
                or kind not in DIAGNOSTIC_KINDS
                or re.fullmatch(r"[a-f0-9]{64}", semantic_key) is None
                or re.fullmatch(r"[a-f0-9]{64}", source_sha256) is None
                or not isinstance(workspace_id, str) or not workspace_id
                or not isinstance(credential_id, str) or not credential_id
                or type(credential_version) is not int or credential_version < 1):
            raise ValueError("invalid diagnostic identity")
        if not 0 < max_cost_microusd <= JOB_CAP_MICROUSD:
            return StartDecision("blocked", None, "job_budget_exceeded")
        output_dir = Path(output_dir)
        billing_group_id = "diagnostic-" + run_id
        now = time.time()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            prior = self.db.execute("SELECT * FROM jobs WHERE semantic_key=?",
                                    (semantic_key,)).fetchone()
            if prior is not None:
                if (prior["kind"] != kind or prior["source_sha256"] != source_sha256
                        or prior["output_dir"] != str(output_dir)
                        or prior["credential_id"] != credential_id
                        or prior["credential_version"] != credential_version
                        or prior["workspace_id"] != workspace_id
                        or prior["billing_group_id"] != billing_group_id
                        or prior["reserved_microusd"] != max_cost_microusd):
                    raise ValueError("diagnostic semantic identity changed")
                self.db.execute("COMMIT")
                return StartDecision("accepted" if prior["status"] == "diagnostic_complete"
                                     else "pending", prior["id"], prior["status"])
            rows = self.db.execute("SELECT * FROM jobs WHERE billing_group_id=?",
                                   (billing_group_id,)).fetchall()
            if (len(rows) >= DIAGNOSTIC_MAX_DISPATCHES
                    or any(row["kind"] == kind or row["kind"] not in DIAGNOSTIC_KINDS
                           or row["source_sha256"] != source_sha256
                           or row["output_dir"] != str(output_dir)
                           or row["credential_id"] != credential_id
                           or row["credential_version"] != credential_version
                           or row["workspace_id"] != workspace_id for row in rows)):
                self.db.execute("ROLLBACK")
                return StartDecision("blocked", None, "diagnostic_group_identity_or_limit")
            group_cost = sum(row["reserved_microusd"] if row["billed_microusd"] is None
                             else row["billed_microusd"] for row in rows
                             if row["status"] not in {"rejected_before_submit", "cancelled_before_submit"})
            if group_cost + max_cost_microusd > DIAGNOSTIC_GROUP_CAP_MICROUSD:
                self.db.execute("ROLLBACK")
                return StartDecision("blocked", None, "diagnostic_group_budget_exceeded")
            if sum(row["dispatches"] for row in rows) >= DIAGNOSTIC_MAX_DISPATCHES:
                self.db.execute("ROLLBACK")
                return StartDecision("blocked", None, "diagnostic_group_dispatch_limit")
            weekly = self.db.execute("""SELECT COALESCE(SUM(CASE WHEN billed_microusd IS NULL
                THEN reserved_microusd ELSE billed_microusd END),0) FROM jobs
                WHERE created_at>=? AND status NOT IN
                ('rejected_before_submit','cancelled_before_submit')""",
                (now - WEEK_SECONDS,)).fetchone()[0]
            if weekly + max_cost_microusd > WEEK_CAP_MICROUSD:
                self.db.execute("ROLLBACK")
                return StartDecision("blocked", None, "weekly_budget_exceeded")
            job_id = uuid.uuid4().hex
            self.db.execute("""INSERT INTO jobs
              (id,semantic_key,source_sha256,output_dir,artifact_dir,status,
               credential_id,credential_version,workspace_id,custom_id,remote_id,
               reserved_microusd,billed_microusd,dispatches,created_at,updated_at,kind,root_job_id,
               billing_group_id,writer_parent_id)
              VALUES (?,?,?,?,?,'reserved',?,?,?,?,NULL,?,NULL,0,?,?,?,?,?,NULL)""",
              (job_id, semantic_key, source_sha256, str(output_dir),
               str(self.root / "jobs" / job_id), credential_id, credential_version,
               workspace_id, kind + "-" + job_id, max_cost_microusd,
               now, now, kind, job_id, billing_group_id))
            self.db.execute("COMMIT")
            return StartDecision("new", job_id)
        except Exception:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def authorize_direct_opus_run(self, *, semantic_key: str, source_sha256: str,
                                  output_dir: Path, authorization_ref: str) -> bool:
        """Pin the new $1.60 ceiling to one future, nonpublishing Opus writer.

        The historical Opus v3 $2 authorization is never reused. This row is
        checked again inside the reservation transaction, against the same
        shared rolling expenditure as every other app job.
        """
        if (re.fullmatch(r"[a-f0-9]{64}", semantic_key or "") is None
                or re.fullmatch(r"[a-f0-9]{64}", source_sha256 or "") is None
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{7,159}",
                                authorization_ref or "") is None):
            raise ValueError("invalid direct Opus authorization")
        output = str(Path(output_dir))
        self.db.execute("BEGIN IMMEDIATE")
        try:
            prior = self.db.execute(
                "SELECT * FROM weekly_spending_authorizations WHERE semantic_key=?",
                (semantic_key,)).fetchone()
            if prior is not None:
                if (prior["source_sha256"] != source_sha256 or prior["output_dir"] != output
                        or prior["quality_policy_version"] != OPUS_DIRECT_POLICY
                        or prior["authorization_ref"] != authorization_ref
                        or prior["cap_microusd"] != OPUS_DIRECT_WEEK_CAP_MICROUSD):
                    raise ValueError("different direct Opus authorization already recorded")
                self.db.execute("COMMIT")
                return False
            if self.db.execute("SELECT 1 FROM jobs WHERE semantic_key=?", (semantic_key,)).fetchone():
                raise ValueError("direct Opus authorization requires a future run")
            if self.db.execute("""SELECT 1 FROM weekly_spending_authorizations
                WHERE quality_policy_version=? LIMIT 1""",
                               (OPUS_DIRECT_POLICY,)).fetchone():
                raise ValueError("direct Opus authorization already assigned to another run")
            self.db.execute("""INSERT INTO weekly_spending_authorizations
                (semantic_key,source_sha256,output_dir,quality_policy_version,
                 authorization_ref,cap_microusd,authorized_at,root_job_id)
                VALUES (?,?,?,?,?,?,?,NULL)""",
                (semantic_key, source_sha256, output, OPUS_DIRECT_POLICY,
                 authorization_ref, OPUS_DIRECT_WEEK_CAP_MICROUSD, time.time()))
            self.db.execute("COMMIT")
            return True
        except Exception:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def reserve_direct_opus(self, *, semantic_key: str, source_sha256: str,
                            output_dir: Path, credential_id: str,
                            credential_version: int, workspace_id: str,
                            max_cost_microusd: int) -> StartDecision:
        """Atomically reserve the one physical request in the app-wide ledger."""
        if (re.fullmatch(r"[a-f0-9]{64}", semantic_key or "") is None
                or re.fullmatch(r"[a-f0-9]{64}", source_sha256 or "") is None
                or not isinstance(credential_id, str) or not credential_id
                or type(credential_version) is not int or credential_version < 1
                or not isinstance(workspace_id, str) or not workspace_id):
            raise ValueError("invalid direct Opus identity")
        if not 0 < max_cost_microusd <= OPUS_DIRECT_CALL_CAP_MICROUSD:
            return StartDecision("blocked", None, "job_budget_exceeded")
        output = str(Path(output_dir))
        now = time.time()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            prior = self.db.execute("SELECT * FROM jobs WHERE semantic_key=?",
                                    (semantic_key,)).fetchone()
            if prior is not None:
                if (prior["kind"] != OPUS_DIRECT_KIND or prior["source_sha256"] != source_sha256
                        or prior["output_dir"] != output or prior["credential_id"] != credential_id
                        or prior["credential_version"] != credential_version
                        or prior["workspace_id"] != workspace_id):
                    raise ValueError("direct Opus semantic identity changed")
                self.db.execute("COMMIT")
                return StartDecision("accepted" if prior["status"] == "direct_complete"
                                     else "pending", prior["id"], prior["status"])
            authorization = self.db.execute(
                "SELECT * FROM weekly_spending_authorizations WHERE semantic_key=?",
                (semantic_key,)).fetchone()
            if (authorization is None or authorization["source_sha256"] != source_sha256
                    or authorization["output_dir"] != output
                    or authorization["quality_policy_version"] != OPUS_DIRECT_POLICY
                    or authorization["cap_microusd"] != OPUS_DIRECT_WEEK_CAP_MICROUSD
                    or authorization["root_job_id"] is not None):
                self.db.execute("ROLLBACK")
                return StartDecision("blocked", None, "direct_opus_authorization_missing")
            spent = self.db.execute("""SELECT COALESCE(SUM(CASE WHEN billed_microusd IS NULL
                THEN reserved_microusd ELSE billed_microusd END),0) FROM jobs
                WHERE created_at>=? AND status NOT IN
                ('rejected_before_submit','cancelled_before_submit')""",
                (now - WEEK_SECONDS,)).fetchone()[0]
            if spent + max_cost_microusd > OPUS_DIRECT_WEEK_CAP_MICROUSD:
                self.db.execute("ROLLBACK")
                return StartDecision("blocked", None, "weekly_budget_exceeded")
            job_id = uuid.uuid4().hex
            self.db.execute("""INSERT INTO jobs
              (id,semantic_key,source_sha256,output_dir,artifact_dir,status,
               credential_id,credential_version,workspace_id,custom_id,remote_id,
               reserved_microusd,billed_microusd,dispatches,created_at,updated_at,kind,root_job_id,
               billing_group_id,writer_parent_id)
              VALUES (?,?,?,?,?,'reserved',?,?,?,?,NULL,?,NULL,0,?,?,?,?,?,NULL)""",
              (job_id, semantic_key, source_sha256, output,
               str(self.root / "jobs" / job_id), credential_id, credential_version,
               workspace_id, "direct-" + job_id, max_cost_microusd,
               now, now, OPUS_DIRECT_KIND, job_id, job_id))
            self.db.execute("UPDATE weekly_spending_authorizations SET root_job_id=? WHERE semantic_key=?",
                            (job_id, semantic_key))
            self.db.execute("COMMIT")
            _secure_dir(self.root / "jobs" / job_id)
            return StartDecision("new", job_id)
        except Exception:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    @staticmethod
    def _check_accepted_inventory_unavailable(parent: sqlite3.Row) -> None:
        """Bind the exception to the prior sealed, published degraded decision."""
        artifacts = Path(parent["artifact_dir"])
        document = Path(parent["accepted_document_path"]) if parent["accepted_document_path"] else None
        generation_id = parent["generation_id"]
        if (document is None or document != artifacts / "candidate_document.json"
                or document.is_symlink() or not document.is_file()
                or not isinstance(generation_id, str)
                or re.fullmatch(r"[0-9]{8}-[0-9]{6}-[0-9a-f]{12}", generation_id) is None):
            raise ValueError("accepted inventory continuation publication unavailable")
        generation = Path(parent["output_dir"]) / "summary_generations" / generation_id
        quality_path = artifacts / "quality_decision.json"
        run_path = generation / "run_manifest.json"
        manifest_path = generation / "generation_manifest.json"
        if (generation.is_symlink() or not generation.is_dir()
                or any(path.is_symlink() or not path.is_file()
                       for path in (quality_path, run_path, manifest_path))):
            raise ValueError("accepted inventory continuation publication unavailable")
        try:
            decision = json.loads(quality_path.read_text(encoding="utf-8"))
            run_bytes = run_path.read_bytes()
            run = json.loads(run_bytes)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError("accepted inventory continuation publication unavailable") from exc
        review = decision.get("quality_review") if isinstance(decision, dict) else None
        digests = manifest.get("artifact_sha256") if isinstance(manifest, dict) else None
        if (not isinstance(review, dict) or review.get("status") != "inventory_unavailable"
                or not isinstance(run, dict) or run.get("quality_review") != review
                or run.get("job_id") != parent["id"]
                or run.get("source_sha256") != parent["source_sha256"]
                or not isinstance(manifest, dict)
                or manifest.get("generation_id") != generation_id
                or manifest.get("source_sha256") != parent["source_sha256"]
                or not isinstance(digests, dict)
                or digests.get("run_manifest.json") != hashlib.sha256(run_bytes).hexdigest()):
            raise ValueError("accepted inventory continuation requires sealed inventory_unavailable")

    def get(self, job_id: str) -> dict | None:
        row = self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def by_semantic_key(self, semantic_key: str) -> dict | None:
        row = self.db.execute("SELECT * FROM jobs WHERE semantic_key=?", (semantic_key,)).fetchone()
        return dict(row) if row else None

    def stage(self, root_job_id: str, kind: str) -> dict | None:
        if kind not in QUALITY_STAGE_KINDS:
            raise ValueError("invalid quality stage")
        row = self.db.execute("SELECT * FROM jobs WHERE root_job_id=? AND kind=? ORDER BY created_at LIMIT 1",
                              (root_job_id, kind)).fetchone()
        return dict(row) if row else None

    def quality_pending(self, *, now: float | None = None) -> list[dict]:
        now = time.time() if now is None else now
        return [dict(row) for row in self.db.execute(
            "SELECT * FROM jobs WHERE kind='summary' AND status='quality_pending' "
            "AND (next_poll_at IS NULL OR next_poll_at<=?) ORDER BY created_at", (now,))]

    def mark_quality_pending(self, job_id: str) -> None:
        changed = self.db.execute("UPDATE jobs SET status='quality_pending',updated_at=? "
                                  "WHERE id=? AND kind='summary' AND status='completed_raw'",
                                  (time.time(), job_id))
        if changed.rowcount != 1:
            raise ValueError("invalid draft transition")

    def mark_continuation_ready(self, job_id: str) -> bool:
        """Expose a sealed audit-only root to the scheduler exactly once.

        The separate preparing state prevents another process from seeing a
        quality_pending row before the caller has pinned its manifest and
        saved draft. Repeating this after a crash is an exact no-op.
        """
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if (row is None or row["kind"] != "summary"
                    or row["writer_parent_id"] is None
                    or row["reserved_microusd"] != 0
                    or row["dispatches"] != 0 or row["remote_id"] is not None
                    or row["status"] not in {"preparing", "quality_pending"}):
                raise ValueError("invalid continuation transition")
            artifacts = Path(row["artifact_dir"])
            for name in ("manifest.json", "draft_document.json"):
                path = artifacts / name
                if path.is_symlink() or not path.is_file():
                    raise ValueError("continuation artifacts not sealed")
            if row["status"] == "quality_pending":
                self.db.execute("COMMIT")
                return False
            changed = self.db.execute(
                "UPDATE jobs SET status='quality_pending',updated_at=? "
                "WHERE id=? AND status='preparing'", (time.time(), job_id))
            if changed.rowcount != 1:
                raise ValueError("continuation transition lost")
            self.db.execute("COMMIT")
            return True
        except Exception:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def stage_completed(self, job_id: str, report_path: Path) -> None:
        changed = self.db.execute("UPDATE jobs SET status='stage_complete',accepted_document_path=?,updated_at=? "
                                  "WHERE id=? AND kind IN ('audit','verify','inventory_1','inventory_2','inventory_3','reconcile',"
                                  "'reconcile_1','reconcile_2',"
                                  "'segment_1','segment_2','segment_3') AND status='completed_raw'",
                                  (str(report_path), time.time(), job_id))
        if changed.rowcount != 1:
            raise ValueError("invalid stage completion transition")

    def defer_quality(self, job_id: str, *, delay_seconds: int, error_code: str) -> None:
        if not 30 <= delay_seconds <= 3600:
            raise ValueError("invalid quality retry delay")
        self.db.execute("UPDATE jobs SET next_poll_at=?,error_code=?,updated_at=? "
                        "WHERE id=? AND kind='summary' AND status='quality_pending'",
                        (time.time() + delay_seconds, error_code[:120], time.time(), job_id))

    def failed_quality(self, job_id: str, code: str) -> None:
        self.db.execute("UPDATE jobs SET status='failed_validation',error_code=?,updated_at=? "
                        "WHERE id=? AND kind='summary' AND status='quality_pending'",
                        (code[:120], time.time(), job_id))

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
            row = self.db.execute("SELECT status,dispatches,billing_group_id,kind FROM jobs WHERE id=?", (job_id,)).fetchone()
            individual_limit = (1 if row and row["kind"] in DIAGNOSTIC_KINDS | {OPUS_DIRECT_KIND}
                                else MAX_DISPATCHES_PER_JOB)
            if not row or row["status"] != "reserved" or row["dispatches"] >= individual_limit:
                self.db.execute("ROLLBACK")
                return False
            group_dispatches = self.db.execute(
                "SELECT COALESCE(SUM(dispatches),0) FROM jobs WHERE billing_group_id=?",
                (row["billing_group_id"],),
            ).fetchone()[0]
            group_limit = (1 if row["kind"] == OPUS_DIRECT_KIND else
                           DIAGNOSTIC_MAX_DISPATCHES if row["kind"] in DIAGNOSTIC_KINDS
                           else MAX_DISPATCHES_PER_JOB)
            if group_dispatches >= group_limit:
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
        self.db.execute("UPDATE jobs SET status='cancelled_before_submit',error_code=?,updated_at=? WHERE id=? AND status IN ('reserved','preparing')",
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
        changed = self.db.execute("UPDATE jobs SET status='accepted',accepted_document_path=?,generation_id=?,updated_at=? WHERE id=? AND kind='summary' AND status IN ('completed_raw','quality_pending')",
            (str(document_path), generation_id, time.time(), job_id))
        if changed.rowcount != 1:
            raise ValueError("invalid accepted transition")

    def accept_recovered_failed(self, job_id: str, *, audit_job_id: str,
                                expected_error_code: str, document_path: Path,
                                generation_id: str) -> bool:
        """Commit one locally recovered, already published summary without another dispatch.

        The original failed result and its error remain in private artifacts and
        the recovery intent.  Jobs and the primary consumer move together in a
        single conditional transaction.  A repeated call is an exact no-op.
        """
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None or row["kind"] != "summary":
                raise ValueError("recovery root is missing")
            consumer = self.db.execute(
                "SELECT * FROM consumers WHERE semantic_key=? AND output_dir=?",
                (row["semantic_key"], row["output_dir"]),
            ).fetchone()
            if consumer is None:
                raise ValueError("recovery consumer is missing")
            if row["status"] == "accepted":
                if (row["accepted_document_path"] != str(document_path)
                        or row["generation_id"] != generation_id
                        or consumer["status"] != "published"
                        or consumer["generation_id"] != generation_id):
                    raise ValueError("a different recovery is already accepted")
                self.db.execute("COMMIT")
                return False
            audit = self.db.execute("SELECT * FROM jobs WHERE id=?", (audit_job_id,)).fetchone()
            verify = self.db.execute(
                "SELECT id FROM jobs WHERE root_job_id=? AND kind='verify'", (job_id,),
            ).fetchone()
            if (row["status"] != "failed_validation"
                    or row["error_code"] != expected_error_code
                    or row["accepted_document_path"] is not None
                    or row["generation_id"] is not None
                    or Path(document_path).parent != Path(row["artifact_dir"])
                    or consumer["status"] != "pending"
                    or audit is None or audit["kind"] != "audit"
                    or audit["root_job_id"] != job_id
                    or audit["status"] != "stage_complete" or verify is not None):
                raise ValueError("recovery state changed before acceptance")
            now = time.time()
            changed = self.db.execute(
                "UPDATE jobs SET status='accepted',accepted_document_path=?,generation_id=?,"
                "error_code=NULL,updated_at=? WHERE id=? AND status='failed_validation' AND error_code=?",
                (str(document_path), generation_id, now, job_id, expected_error_code),
            )
            if changed.rowcount != 1:
                raise ValueError("recovery root transition lost")
            changed = self.db.execute(
                "UPDATE consumers SET status='published',generation_id=?,error_code=NULL,updated_at=? "
                "WHERE semantic_key=? AND output_dir=? AND status='pending'",
                (generation_id, now, row["semantic_key"], row["output_dir"]),
            )
            if changed.rowcount != 1:
                raise ValueError("recovery consumer transition lost")
            self.db.execute("COMMIT")
            return True
        except Exception:
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise

    def failed_validation(self, job_id: str, code: str) -> None:
        self.db.execute("UPDATE jobs SET status='failed_validation',error_code=?,updated_at=? WHERE id=? AND status='completed_raw'",
            (code, time.time(), job_id))

    def diagnostic_completed(self, job_id: str, report_path: Path) -> None:
        changed = self.db.execute("UPDATE jobs SET status='diagnostic_complete',"
                                  "accepted_document_path=?,updated_at=? WHERE id=? "
                                  "AND kind IN ('diagnostic_baseline','diagnostic_compact') "
                                  "AND status='completed_raw'",
                                  (str(report_path), time.time(), job_id))
        if changed.rowcount != 1:
            raise ValueError("invalid diagnostic completion transition")

    def direct_opus_completed(self, job_id: str, document_path: Path) -> None:
        changed = self.db.execute("UPDATE jobs SET status='direct_complete',"
                                  "accepted_document_path=?,updated_at=? WHERE id=? "
                                  "AND kind=? AND status='completed_raw'",
                                  (str(document_path), time.time(), job_id, OPUS_DIRECT_KIND))
        if changed.rowcount != 1:
            raise ValueError("invalid direct Opus completion transition")

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
