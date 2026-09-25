#!/usr/bin/env python3
"""Protected record deletion for the existing dashboard, without editing pipeline.py.

The service invokes this file with the same arguments previously given to
pipeline.py (usually ``watch``).  The wrapper imports that exact release,
patches one HTTP route, and calls its normal ``main()`` once.  Speech stages,
their cache identity, and all other HTTP routes stay in the imported release.

The administrator config is a private JSON file at
``<record root>/state/record_delete_admin.json``::

    {"version": 1, "origin": "https://example.tailnet.ts.net:8443",
     "verifier": "scrypt$16384$8$1$<salt>$<digest>"}

The verifier format is compatible with the existing summary admin password
hash.  Store only the verifier, never the plaintext password.  This module
does not create a verifier or change server credentials.
"""

from __future__ import annotations

import base64
from contextlib import closing
import hashlib
import hmac
import importlib.util
import ipaddress
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import stat
import sys
import threading
import time
import uuid
from urllib.parse import urlparse, urlsplit


ROUTE = "/api/jobs/delete"
REQUEST_HEADER = "TranscriSummaryzator-Admin"
ACTIVE_SUMMARY = frozenset({"queued", "queued_force", "running", "pending_batch",
                            "submission_unknown", "credential_required"})
VERIFIER_RE = re.compile(r"scrypt\$16384\$8\$1\$([A-Za-z0-9_-]{20,24})\$([A-Za-z0-9_-]{80,90})\Z")
AUTH_FAILURES: list[float] = []
AUTH_LOCK = threading.Lock()


class DeleteError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _private_dir(path: Path) -> Path:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
        raise DeleteError(503, "Защищённое хранилище удаления недоступно")
    return path


def _sync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_private_json(path: Path, value: dict) -> None:
    _private_dir(path.parent)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_admin_config(root: Path) -> dict:
    path = root / "state" / "record_delete_admin.json"
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
                raise ValueError("unsafe admin config")
            raw = os.read(descriptor, 4097)
        finally:
            os.close(descriptor)
        if len(raw) > 4096:
            raise ValueError("oversized admin config")
        value = json.loads(raw)
        if value.get("version") != 1 or not isinstance(value.get("verifier"), str):
            raise ValueError("invalid admin config")
        origin = value.get("origin")
        parsed = urlsplit(origin) if isinstance(origin, str) else None
        if (not parsed or parsed.scheme not in {"https", "http"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment
                or origin != f"{parsed.scheme}://{parsed.netloc}"):
            raise ValueError("invalid admin origin")
        if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("insecure admin origin")
        if VERIFIER_RE.fullmatch(value["verifier"]) is None:
            raise ValueError("invalid verifier")
        return value
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise DeleteError(503, "Управление удалением не настроено") from exc


def _check_password(password: str, verifier: str) -> bool:
    if len(password) > 4096:
        return False
    match = VERIFIER_RE.fullmatch(verifier)
    if match is None:
        return False
    try:
        salt = base64.urlsafe_b64decode(match.group(1) + "===")
        expected = base64.urlsafe_b64decode(match.group(2) + "===")
        if len(salt) != 16 or len(expected) != 64:
            return False
        actual = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=16384, r=8, p=1)
    except (ValueError, UnicodeError):
        return False
    return hmac.compare_digest(actual, expected)


def _reply(handler, value: dict, status: int, *, challenge: bool = False) -> None:
    payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Cache-Control", "no-store, private")
    handler.send_header("X-Content-Type-Options", "nosniff")
    if challenge:
        handler.send_header("WWW-Authenticate", 'Basic realm="Record deletion", charset="UTF-8"')
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def _authorize(handler, root: Path) -> None:
    settings = _read_admin_config(root)
    origin = settings["origin"]
    try:
        peer_is_local = ipaddress.ip_address(handler.client_address[0]).is_loopback
    except (ValueError, IndexError, TypeError):
        peer_is_local = False
    if not peer_is_local or handler.headers.get("Host", "").lower() != urlsplit(origin).netloc.lower():
        raise DeleteError(403, "Недопустимый адрес управления")
    if (handler.headers.get("Origin") != origin
            or handler.headers.get("X-Requested-With") != REQUEST_HEADER
            or handler.headers.get("Sec-Fetch-Site", "same-origin") not in {"same-origin", "none"}):
        raise DeleteError(403, "Проверка происхождения запроса не пройдена")
    current = time.monotonic()
    with AUTH_LOCK:
        AUTH_FAILURES[:] = [t for t in AUTH_FAILURES if current - t < 60]
        if len(AUTH_FAILURES) >= 10:
            raise DeleteError(429, "Слишком много попыток входа")
    header = handler.headers.get("Authorization", "")
    try:
        if not header.startswith("Basic ") or len(header) > 4096:
            raise ValueError("bad header")
        decoded = base64.b64decode(header[6:], validate=True).decode("utf-8")
        username, password = decoded.split(":", 1)
        accepted = username == "admin" and _check_password(password, settings["verifier"])
    except (ValueError, UnicodeError):
        accepted = False
    if not accepted:
        with AUTH_LOCK:
            AUTH_FAILURES.append(current)
        raise DeleteError(401, "Требуется доступ администратора")


def _request(handler) -> tuple[int, str, str]:
    if handler.headers.get("Content-Type", "").split(";", 1)[0].lower().strip() != "application/json":
        raise DeleteError(415, "Требуется application/json")
    if handler.headers.get("Transfer-Encoding"):
        raise DeleteError(413, "Потоковый запрос не поддерживается")
    try:
        length = int(handler.headers.get("Content-Length", ""))
    except ValueError as exc:
        raise DeleteError(411, "Укажите размер запроса") from exc
    if not 0 < length <= 4096:
        raise DeleteError(413, "Превышен лимит размера запроса")
    try:
        connection = getattr(handler, "connection", None)
        if connection is not None:
            connection.settimeout(10)
        payload = handler.rfile.read(length)
        if len(payload) != length:
            raise ValueError("incomplete body")
        value = json.loads(payload)
    except (OSError, ValueError, UnicodeError) as exc:
        raise DeleteError(400, "Неверный JSON") from exc
    if not isinstance(value, dict) or set(value) != {"id", "name", "created_at"}:
        raise DeleteError(400, "Укажите id, имя и время создания записи")
    job_id, name, created_at = value["id"], value["name"], value["created_at"]
    if (type(job_id) is not int or not 0 < job_id < 2**63 or not isinstance(name, str)
            or not 0 < len(name) <= 255 or not isinstance(created_at, str)
            or not 0 < len(created_at) <= 64):
        raise DeleteError(400, "Неверные поля записи")
    return job_id, name, created_at


def _owned_child(raw: str | None, parent: Path, *, kind: str) -> Path | None:
    if not raw:
        return None
    candidate = Path(raw)
    if (not candidate.is_absolute() or candidate.parent.resolve() != parent.resolve()
            or candidate == parent or candidate.is_symlink()):
        raise DeleteError(409, f"{kind}: путь вне хранилища записи")
    if candidate.exists():
        mode = candidate.lstat().st_mode
        if kind == "source" and not stat.S_ISREG(mode):
            raise DeleteError(409, "Исходный файл имеет неверный тип")
        if kind != "source" and not stat.S_ISDIR(mode):
            raise DeleteError(409, f"{kind}: каталог имеет неверный тип")
    return candidate


def _ledger_has_linked_output(pipeline, output_dir: Path | None) -> bool:
    if output_dir is None:
        return False
    path = pipeline.STATE / "summary_private" / "luna.sqlite3"
    if not path.exists():
        return False
    try:
        ledger = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
        try:
            query = """SELECT COUNT(*) FROM jobs AS j
                        LEFT JOIN consumers AS c ON c.semantic_key=j.semantic_key
                        WHERE j.output_dir=? OR c.output_dir=?"""
            count = ledger.execute(query, (str(output_dir), str(output_dir))).fetchone()[0]
            return bool(count)
        finally:
            ledger.close()
    except sqlite3.Error as exc:
        raise DeleteError(409, "Состояние внешнего Batch неизвестно") from exc


def _manifest_paths(manifest: dict, staging: Path) -> list[tuple[Path, Path]]:
    return [(Path(item["original"]), staging / item["slot"]) for item in manifest["paths"]]


def _receipt_path(root: Path, manifest: dict) -> Path:
    receipts = _private_dir(root / "state" / "record_delete" / "receipts")
    return receipts / f"{manifest['job_id']}-{manifest['transaction_id']}.json"


def _record_receipt(root: Path, manifest: dict, status: str) -> None:
    _write_private_json(_receipt_path(root, manifest), {
        "version": 1, "transaction_id": manifest["transaction_id"], "job_id": manifest["job_id"],
        "fingerprint": manifest["fingerprint"], "name": manifest["name"],
        "created_at": manifest["created_at"], "status": status,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "paths": manifest["paths"],
    })


def _restore(staging: Path, manifest: dict) -> None:
    for original, staged in reversed(_manifest_paths(manifest, staging)):
        if not staged.exists():
            continue
        if original.exists() or original.is_symlink():
            raise DeleteError(500, "Восстановление удаления требует ручной проверки")
        os.replace(staged, original)
        _sync_dir(original.parent)
        _sync_dir(staging)


def recover_staged_deletions(pipeline) -> list[str]:
    """Finish committed deletions or restore files when SQLite rolled back."""
    root = Path(pipeline.ROOT)
    staging_root = root / "state" / "record_delete" / "staging"
    if not staging_root.exists():
        return []
    _private_dir(staging_root)
    outcomes = []
    for staging in sorted(staging_root.iterdir()):
        if not staging.is_dir() or staging.is_symlink():
            continue
        try:
            manifest = json.loads((staging / "manifest.json").read_text(encoding="utf-8"))
            if (manifest.get("version") != 1 or manifest.get("transaction_id") != staging.name
                    or not isinstance(manifest.get("job_id"), int)
                    or not isinstance(manifest.get("paths"), list)):
                raise ValueError("invalid deletion manifest")
            with closing(sqlite3.connect(pipeline.DB_PATH, timeout=30)) as db:
                has_tombstones = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='record_deletions'").fetchone()
                tombstone = (db.execute("SELECT job_id, fingerprint, created_at FROM record_deletions WHERE transaction_id=?", (staging.name,)).fetchone()
                             if has_tombstones else None)
                row = db.execute("SELECT fingerprint, created_at, original_name FROM jobs WHERE id=?", (manifest["job_id"],)).fetchone()
            if tombstone is not None:
                if tuple(tombstone) != (manifest["job_id"], manifest["fingerprint"], manifest["created_at"]):
                    raise ValueError("deletion tombstone identity changed")
                _record_receipt(root, manifest, "deleted")
                outcome = "purged"
            elif row is not None and tuple(row) == (manifest["fingerprint"], manifest["created_at"], manifest["name"]):
                _restore(staging, manifest)
                _record_receipt(root, manifest, "rolled_back")
                outcome = "restored"
            else:
                raise ValueError("deletion outcome cannot be established")
            shutil.rmtree(staging)
            _sync_dir(staging_root)
            outcomes.append(f"{manifest['job_id']}:{outcome}")
        except (OSError, ValueError, KeyError, TypeError, DeleteError, sqlite3.Error) as exc:
            # A failed recovery never silently discards the source or receipt.
            outcomes.append(f"{staging.name}:recovery_required:{type(exc).__name__}")
    return outcomes


def delete_record(pipeline, job_id: int, name: str, created_at: str) -> dict:
    """Delete one terminal queue row and only its owned filesystem objects."""
    root = Path(pipeline.ROOT)
    db_path = Path(pipeline.DB_PATH)
    if not db_path.is_file():
        raise DeleteError(503, "Очередь записей недоступна")
    private = _private_dir(root / "state" / "record_delete")
    staging_root = _private_dir(private / "staging")
    db = sqlite3.connect(db_path, timeout=30, isolation_level=None)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA busy_timeout=30000")
    staging = None
    manifest = None
    committed = False
    try:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise DeleteError(404, "Запись не найдена")
        if row["original_name"] != name or row["created_at"] != created_at:
            raise DeleteError(409, "Список изменился; обновите страницу")
        if row["status"] not in {"done", "failed"} or row["summary_status"] in ACTIVE_SUMMARY:
            raise DeleteError(409, "Идёт обработка или внешняя отправка; удаление закрыто")
        if row["summary_status"] not in {"done", "failed", "not_started", "blocked"}:
            raise DeleteError(409, "Статус саммари не завершён")
        paths = []
        for slot, column, parent, kind in (
            ("source", "source_path", pipeline.INBOX, "source"),
            ("work", "job_dir", pipeline.JOBS, "work"),
            ("output", "output_dir", pipeline.OUTPUTS, "output"),
        ):
            path = _owned_child(row[column], Path(parent), kind=kind)
            if path is None and slot != "output":
                raise DeleteError(409, "Нет пути исходного или рабочего файла")
            if path is not None:
                refs = db.execute(f"SELECT COUNT(*) FROM jobs WHERE {column}=? AND id<>?",
                                  (str(path), job_id)).fetchone()[0]
                if refs:
                    raise DeleteError(409, "Файлы используются другой записью")
                paths.append({"slot": slot, "original": str(path), "existed": path.exists()})
        output = _owned_child(row["output_dir"], Path(pipeline.OUTPUTS), kind="output")
        if _ledger_has_linked_output(pipeline, output):
            raise DeleteError(409, "Запись связана с журналом Luna; требуется согласованное удаление")
        transaction_id = uuid.uuid4().hex
        staging = staging_root / transaction_id
        staging.mkdir(mode=0o700)
        _sync_dir(staging_root)
        manifest = {"version": 1, "transaction_id": transaction_id, "job_id": job_id,
                    "fingerprint": row["fingerprint"], "name": name, "created_at": created_at,
                    "paths": paths}
        _write_private_json(staging / "manifest.json", manifest)
        for item in paths:
            if item["existed"]:
                original = Path(item["original"])
                os.replace(original, staging / item["slot"])
                _sync_dir(original.parent)
                _sync_dir(staging)
        removed = db.execute("DELETE FROM jobs WHERE id=? AND fingerprint=?", (job_id, row["fingerprint"]))
        if removed.rowcount != 1:
            raise DeleteError(409, "Запись изменилась")
        db.execute("""CREATE TABLE IF NOT EXISTS record_deletions (
            transaction_id TEXT PRIMARY KEY, job_id INTEGER NOT NULL, fingerprint TEXT NOT NULL,
            created_at TEXT NOT NULL, committed_at TEXT NOT NULL)""")
        db.execute("INSERT INTO record_deletions VALUES (?,?,?,?,?)",
                   (transaction_id, job_id, row["fingerprint"], created_at,
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
        db.execute("COMMIT")
        committed = True
        try:
            pipeline.write_status_snapshot(db)
        except Exception:
            pass  # /api/status reads the authoritative database directly.
        cleanup_pending = False
        try:
            _record_receipt(root, manifest, "deleted")
            shutil.rmtree(staging)
            _sync_dir(staging_root)
        except (OSError, DeleteError):
            cleanup_pending = True
            try:
                _record_receipt(root, manifest, "deleted_cleanup_pending")
            except (OSError, DeleteError):
                pass
        return {"ok": True, "job_id": job_id, "cleanup_pending": cleanup_pending}
    except Exception:
        if not committed:
            if db.in_transaction:
                db.execute("ROLLBACK")
            if staging is not None and manifest is not None:
                recover_staged_deletions(pipeline)  # Tombstone distinguishes a committed-but-ambiguous COMMIT.
        raise
    finally:
        db.close()


def handle_delete(handler, pipeline) -> None:
    try:
        _authorize(handler, Path(pipeline.ROOT))
        job_id, name, created_at = _request(handler)
        result = delete_record(pipeline, job_id, name, created_at)
        _reply(handler, result, 200)
    except DeleteError as exc:
        _reply(handler, {"error": exc.message}, exc.status, challenge=exc.status == 401)
    except (OSError, sqlite3.Error):
        _reply(handler, {"error": "Удаление не завершено; проверьте состояние записи"}, 500)


def install_handler(pipeline) -> None:
    original = pipeline.DashboardHandler.do_POST

    # The old dashboard capped history at 50, making older records impossible to delete.
    def all_status_rows(db):
        return db.execute(
            "SELECT id, original_name, status, stage, progress, detail, error, output_dir, speaker_count, created_at, started_at, finished_at, updated_at, summary_status, summary_stage, summary_progress, summary_detail, summary_error, summary_started_at, summary_finished_at FROM jobs ORDER BY id DESC"
        ).fetchall()

    def patched(handler):
        if urlparse(handler.path).path == ROUTE:
            return handle_delete(handler, pipeline)
        return original(handler)

    pipeline.DashboardHandler.do_POST = patched
    pipeline.status_rows = all_status_rows


def load_pipeline(root: Path):
    root = root.resolve(strict=True)
    source = root / "pipeline.py"
    if not source.is_file():
        raise RuntimeError("Production pipeline.py was not found")
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("record_delete_production_pipeline", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("Production pipeline.py could not be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    root = Path(os.environ.get("TRANSCRI_RECORD_ROOT", "/srv/meeting-transcript"))
    pipeline = load_pipeline(root)
    install_handler(pipeline)
    for outcome in recover_staged_deletions(pipeline):
        print("record-delete recovery: " + outcome, file=sys.stderr, flush=True)
    pipeline.main()


if __name__ == "__main__":
    main()
