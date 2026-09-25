"""Private OpenRouter inference-key store for the summary-only backend.

The Fernet master key comes from a protected process credential, never from the
repository or SQLite. Callers reopen SQLite for every dispatch so a key change
in the dashboard is visible to workers without a process restart.
"""

from __future__ import annotations

import base64
from contextlib import contextmanager
import fcntl
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import sqlite3
import stat
import sys
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path


KEY_INFO_URL = "https://openrouter.ai/api/v1/key"
LABEL_RE = re.compile(r"^[^\x00-\x1f\x7f]{1,80}$")
HASH_RE = re.compile(r"^scrypt\$([0-9]+)\$([0-9]+)\$([0-9]+)\$([A-Za-z0-9_-]+)\$([A-Za-z0-9_-]+)$")


class CredentialError(ValueError):
    """A safe error whose text may be sent to the administrator."""


def _utcnow():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_admin_password_hash(password: str) -> str:
    """Make a scrypt verifier; run interactively, never commit the password."""
    if len(password) < 16:
        raise CredentialError("Пароль администратора должен содержать минимум 16 символов")
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=16384, r=8, p=1)
    return "scrypt$16384$8$1${}${}".format(
        base64.urlsafe_b64encode(salt).decode("ascii").rstrip("="),
        base64.urlsafe_b64encode(digest).decode("ascii").rstrip("="),
    )


def verify_admin_password(password: str, encoded: str | None) -> bool:
    if not encoded or not isinstance(password, str) or len(password) > 4096:
        return False
    match = HASH_RE.fullmatch(encoded)
    if match is None:
        return False
    n, r, p = (int(match.group(i)) for i in (1, 2, 3))
    if (n, r, p) != (16384, 8, 1):
        return False
    try:
        salt = base64.urlsafe_b64decode(match.group(4) + "===")
        expected = base64.urlsafe_b64decode(match.group(5) + "===")
        actual = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p)
    except (ValueError, UnicodeError):
        return False
    return hmac.compare_digest(actual, expected)


def _protected_file(path: Path) -> bytes:
    if not path.is_absolute():
        raise CredentialError("Путь к master key должен быть абсолютным обычным файлом")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise CredentialError("Нельзя открыть защищённый master key") from exc
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
            raise CredentialError("Master key должен принадлежать процессу и иметь права 0600")
        data = os.read(fd, 257).strip()
    finally:
        os.close(fd)
    if len(data) > 256:
        raise CredentialError("Недопустимая длина master key")
    return data


def load_master_key() -> bytes:
    file_name = os.environ.get("TRANSCRI_SUMMARY_MASTER_KEY_FILE")
    inline = os.environ.get("TRANSCRI_SUMMARY_MASTER_KEY")
    if bool(file_name) == bool(inline):
        raise CredentialError("Задайте ровно один защищённый master key для суммаризатора")
    if file_name:
        return _protected_file(Path(file_name))
    try:
        return inline.encode("ascii")
    except UnicodeEncodeError as exc:
        raise CredentialError("Неверный формат master key") from exc


def _fernet(master_key: bytes):
    try:
        from cryptography.fernet import Fernet
    except ImportError as exc:
        raise CredentialError("Для хранилища ключей нужен отдельный runtime с cryptography") from exc
    try:
        return Fernet(master_key)
    except (ValueError, TypeError) as exc:
        raise CredentialError("Неверный формат master key") from exc


def _check_token(value: str) -> str:
    if not isinstance(value, str) or not 16 <= len(value) <= 512 or any(not 33 <= ord(c) <= 126 for c in value):
        raise CredentialError("Неверный формат ключа")
    return value


def _check_label(value: str) -> str:
    if not isinstance(value, str):
        raise CredentialError("Введите название ключа")
    value = value.strip()
    if not LABEL_RE.fullmatch(value):
        raise CredentialError("Название ключа должно быть длиной до 80 символов")
    return value


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "Redirect blocked", headers, fp)


def _safe_key_metadata(data):
    workspace = data.get("workspace_id")
    if not isinstance(workspace, str) or not re.fullmatch(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", workspace):
        workspace = None
    reset = data.get("limit_reset")
    if reset not in ("daily", "weekly", "monthly", "never", None):
        reset = None
    result = {"workspace_id": workspace, "limit_reset": reset}
    for name in ("limit", "limit_remaining", "usage_weekly"):
        value = data.get(name)
        result[name] = float(value) if type(value) in (int, float) and math.isfinite(value) and 0 <= value < 1e9 else None
    return result


@contextmanager
def credential_dispatch_guard(database_path: Path):
    """Serialize a new Batch POST with credential replacement or deletion.

    The dashboard takes this lock before checking active ledger jobs and keeps
    it through the credential RPC. The worker keeps it from fresh selection
    through the durable POST outcome. The lock file contains no secret.
    """
    directory = Path(database_path).parent
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    metadata = directory.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
        raise CredentialError("Каталог ключей должен принадлежать процессу и иметь права 0700")
    lock_path = directory / "credential-dispatch.lock"
    try:
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    except OSError as exc:
        raise CredentialError("Нельзя открыть блокировку ключей") from exc
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
            raise CredentialError("Блокировка ключей должна иметь права 0600")
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


class CredentialStore:
    """SQLite metadata + Fernet ciphertext; never returns secrets from list()."""

    def __init__(self, path: Path, master_key: bytes | None = None):
        self.path = Path(path)
        self.cipher = _fernet(master_key if master_key is not None else load_master_key())
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.path.parent.stat().st_mode & 0o077:
            raise CredentialError("Каталог ключей должен иметь права 0700")
        if not self.path.exists():
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
            os.close(fd)
        metadata = self.path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
            raise CredentialError("Хранилище ключей должно иметь права 0600")
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS credentials (
                    id TEXT PRIMARY KEY, version INTEGER NOT NULL, label TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    ciphertext BLOB NOT NULL, mask TEXT NOT NULL, enabled INTEGER NOT NULL,
                    priority INTEGER NOT NULL, status TEXT NOT NULL, checked_at TEXT,
                    workspace_id TEXT, limit_amount REAL, limit_remaining REAL,
                    limit_reset TEXT, usage_weekly REAL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS credential_events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL, actor TEXT NOT NULL,
                    credential_id TEXT NOT NULL, version INTEGER NOT NULL, operation TEXT NOT NULL
                );
            """)

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        return db

    @staticmethod
    def _public(row, primary_id=None):
        return {
            "id": row["id"], "version": row["version"], "label": row["label"],
            "mask": row["mask"], "enabled": bool(row["enabled"]),
            "priority": row["priority"], "primary": row["id"] == primary_id,
            "status": row["status"], "checked_at": row["checked_at"],
            "workspace_id": row["workspace_id"], "limit": row["limit_amount"],
            "limit_remaining": row["limit_remaining"], "limit_reset": row["limit_reset"],
            "usage_weekly": row["usage_weekly"], "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _event(db, actor, row, operation):
        db.execute(
            "INSERT INTO credential_events(at,actor,credential_id,version,operation) VALUES(?,?,?,?,?)",
            (_utcnow(), actor, row["id"], row["version"], operation),
        )

    def list(self):
        with self._connect() as db:
            rows = db.execute("SELECT * FROM credentials ORDER BY priority,id").fetchall()
            primary = next((r["id"] for r in rows if r["enabled"]), None)
            revision = db.execute("SELECT COALESCE(MAX(seq),0) FROM credential_events").fetchone()[0]
        return {"revision": revision, "keys": [self._public(r, primary) for r in rows]}

    def add(self, label: str, token: str, actor="admin"):
        label, token = _check_label(label), _check_token(token)
        identifier, stamp = uuid.uuid4().hex, _utcnow()
        ciphertext = self.cipher.encrypt(token.encode("ascii"))
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            priority = db.execute("SELECT COALESCE(MAX(priority),-1)+1 FROM credentials").fetchone()[0]
            try:
                db.execute("""INSERT INTO credentials(id,version,label,ciphertext,mask,enabled,priority,status,created_at,updated_at)
                              VALUES(?,1,?,?,?,1,?,'unchecked',?,?)""",
                           (identifier, label, ciphertext, "••••" + token[-4:], priority, stamp, stamp))
            except sqlite3.IntegrityError as exc:
                raise CredentialError("Такое название ключа уже существует") from exc
            row = db.execute("SELECT * FROM credentials WHERE id=?", (identifier,)).fetchone()
            self._event(db, actor, row, "add")
        return self._public(row)

    def replace(self, identifier: str, token: str, active_jobs: int = 0, actor="admin"):
        if active_jobs:
            raise CredentialError("Ключ используется незавершённым запросом; сначала завершите его")
        token = _check_token(token)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._required(db, identifier)
            db.execute("""UPDATE credentials SET version=version+1,ciphertext=?,mask=?,status='unchecked',
                          checked_at=NULL,workspace_id=NULL,limit_amount=NULL,limit_remaining=NULL,
                          limit_reset=NULL,usage_weekly=NULL,updated_at=? WHERE id=?""",
                       (self.cipher.encrypt(token.encode("ascii")), "••••" + token[-4:], _utcnow(), identifier))
            changed = self._required(db, identifier)
            self._event(db, actor, changed, "replace")
        return self._public(changed)


    @staticmethod
    def _required(db, identifier):
        if not isinstance(identifier, str) or not re.fullmatch(r"[a-f0-9]{32}", identifier):
            raise CredentialError("Ключ не найден")
        row = db.execute("SELECT * FROM credentials WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise CredentialError("Ключ не найден")
        return row

    def set_order(self, identifiers: list[str], actor="admin"):
        if not isinstance(identifiers, list) or len(identifiers) != len(set(identifiers)):
            raise CredentialError("Неверный порядок ключей")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            actual = {r[0] for r in db.execute("SELECT id FROM credentials")}
            if set(identifiers) != actual:
                raise CredentialError("Порядок должен включать все ключи")
            for priority, identifier in enumerate(identifiers):
                db.execute("UPDATE credentials SET priority=?,updated_at=? WHERE id=?", (priority, _utcnow(), identifier))
                self._event(db, actor, self._required(db, identifier), "reorder")
        return self.list()

    def set_enabled(self, identifier: str, enabled: bool, actor="admin"):
        if not isinstance(enabled, bool):
            raise CredentialError("Неверное состояние ключа")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._required(db, identifier)
            db.execute("UPDATE credentials SET enabled=?,updated_at=? WHERE id=?", (int(enabled), _utcnow(), identifier))
            row = self._required(db, identifier)
            self._event(db, actor, row, "enable" if enabled else "disable_new_dispatch")
        return self._public(row)

    def delete(self, identifier: str, active_jobs: int, actor="admin"):
        if active_jobs:
            raise CredentialError("Ключ используется незавершённым запросом; сначала завершите его или отключите новые отправки")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._required(db, identifier)
            db.execute("DELETE FROM credentials WHERE id=?", (identifier,))
            self._event(db, actor, row, "delete_local_secret")
        return {"id": identifier, "revoked_upstream": False}

    def dispatch_candidates(self):
        """Fresh metadata each call; caller enforces common budget/policy/scope."""
        with self._connect() as db:
            rows = db.execute("SELECT * FROM credentials WHERE enabled=1 AND status='ready' ORDER BY priority,id").fetchall()
        return [self._public(r) for r in rows]

    def reveal_for_dispatch(self, identifier: str, version: int) -> str:
        with self._connect() as db:
            row = self._required(db, identifier)
        if not row["enabled"] or row["status"] != "ready" or row["version"] != version:
            raise CredentialError("Ключ недоступен для новой отправки")
        try:
            return self.cipher.decrypt(row["ciphertext"]).decode("ascii")
        except Exception as exc:
            raise CredentialError("Не удалось открыть ключ; проверьте master key") from exc

    def reveal_for_existing_job(self, identifier: str, version: int) -> str:
        """Poll an existing remote job even after disabling new submissions."""
        with self._connect() as db:
            row = self._required(db, identifier)
        if row["version"] != version:
            raise CredentialError("Исходная версия ключа недоступна для незавершённого запроса")
        try:
            return self.cipher.decrypt(row["ciphertext"]).decode("ascii")
        except Exception as exc:
            raise CredentialError("Не удалось открыть ключ; проверьте master key") from exc

    def check(self, identifier: str, actor="admin", opener=None):
        """Free key-metadata GET, not a paid inference or a ZDR/allowlist proof."""
        with self._connect() as db:
            row = self._required(db, identifier)
        try:
            token = self.cipher.decrypt(row["ciphertext"]).decode("ascii")
        except Exception as exc:
            raise CredentialError("Не удалось открыть ключ; проверьте master key") from exc
        request = urllib.request.Request(KEY_INFO_URL, headers={"Authorization": "Bearer " + token, "Accept": "application/json"})
        status, values = "error", {}
        try:
            with (opener or urllib.request.build_opener(_NoRedirect()).open)(request, timeout=8) as response:
                raw = response.read(16385)
                if len(raw) > 16384:
                    raise CredentialError("Слишком большой ответ проверки ключа")
                payload = json.loads(raw)
                data = payload.get("data") if isinstance(payload, dict) else None
                if not isinstance(data, dict):
                    raise CredentialError("Неожиданный ответ проверки ключа")
                values = _safe_key_metadata(data)
                remaining = values.get("limit_remaining")
                status = ("error" if values.get("workspace_id") is None else
                          "exhausted" if isinstance(remaining, (int, float)) and remaining <= 0 else "ready")
        except urllib.error.HTTPError as exc:
            status = "invalid" if exc.code == 401 else "rate_limited" if exc.code == 429 else "error"
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, CredentialError):
            status = "error"
        stamp = _utcnow()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            current = self._required(db, identifier)
            if current["version"] != row["version"]:
                raise CredentialError("Ключ изменён во время проверки; повторите проверку новой версии")
            db.execute("""UPDATE credentials SET status=?,checked_at=?,workspace_id=?,limit_amount=?,
                          limit_remaining=?,limit_reset=?,usage_weekly=?,updated_at=? WHERE id=?""",
                       (status, stamp, values.get("workspace_id"), values.get("limit"),
                        values.get("limit_remaining"), values.get("limit_reset"), values.get("usage_weekly"), stamp, identifier))
            changed = self._required(db, identifier)
            self._event(db, actor, changed, "check_metadata_" + status)
        return self._public(changed)

def _rpc_main():
    """One-shot encrypted-store boundary for the dashboard's core Python."""
    try:
        raw = sys.stdin.buffer.read(4097)
        if len(raw) > 4096:
            raise CredentialError("Превышен лимит запроса к хранилищу")
        request = json.loads(raw)
        if not isinstance(request, dict) or not isinstance(request.get("body"), dict):
            raise CredentialError("Неверный запрос к хранилищу")
        path = Path(os.environ.get("TRANSCRI_SUMMARY_CREDENTIAL_DB", ""))
        if not path.is_absolute():
            raise CredentialError("Не настроен путь к хранилищу ключей")
        store = CredentialStore(path)
        operation, body = request.get("operation"), request["body"]
        if operation == "list":
            result = store.list()
        elif operation == "add":
            result = store.add(body.get("label"), body.get("key"))
        elif operation == "check":
            result = store.check(body.get("id"))
        elif operation == "replace":
            result = store.replace(body.get("id"), body.get("key"), active_jobs=body.get("active_jobs", 0))
        elif operation == "order":
            result = store.set_order(body.get("ids"))
        elif operation == "enabled":
            result = store.set_enabled(body.get("id"), body.get("enabled"))
        elif operation == "delete":
            result = store.delete(body.get("id"), active_jobs=body.get("active_jobs", 0))
        else:
            raise CredentialError("Неизвестная операция с ключом")
        print(json.dumps({"ok": True, "result": result}, ensure_ascii=False), flush=True)
        return 0
    except CredentialError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), flush=True)
        return 2
    except Exception:
        print('{"ok":false,"error":"Ошибка защищённого хранилища"}', flush=True)
        return 3


if __name__ == "__main__" and sys.argv[1:] == ["--credential-rpc"]:
    raise SystemExit(_rpc_main())
