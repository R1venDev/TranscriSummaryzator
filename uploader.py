#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sqlite3
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INBOX = ROOT / "inbox"
STATE = ROOT / "state"
CONFIG = ROOT / "server.json"
DB = STATE / "uploads.sqlite3"
MEDIA = {".mkv", ".mp4", ".mov", ".m4v", ".webm", ".wav", ".mp3", ".m4a", ".flac", ".ogg"}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def cfg():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def db():
    STATE.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE IF NOT EXISTS uploads (path TEXT PRIMARY KEY, size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL, fingerprint TEXT, remote_name TEXT, status TEXT NOT NULL, progress REAL NOT NULL DEFAULT 0, detail TEXT, updated_at TEXT NOT NULL)")
    conn.commit()
    return conn


def snapshot(conn):
    rows = [dict(row) for row in conn.execute("SELECT * FROM uploads ORDER BY updated_at DESC LIMIT 50")]
    target = STATE / "upload-progress.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps({"uploads": rows, "updated_at": now()}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def update(conn, path, **values):
    values["updated_at"] = now()
    conn.execute("UPDATE uploads SET {} WHERE path = ?".format(", ".join(f"{key} = ?" for key in values)), tuple(values.values()) + (str(path),))
    conn.commit()
    snapshot(conn)


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def remote_name(path, digest):
    stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in path.stem).strip("._")[:120] or "recording"
    return f"{stem}-{digest[:8]}{path.suffix.lower()}"


def ssh_base(settings):
    return ["ssh", "-i", settings["identity_file"], "-p", str(settings.get("port", 22)), "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", f'{settings["user"]}@{settings["host"]}']


def upload(conn, path, settings):
    update(conn, path, status="hashing", progress=1, detail="Проверяю запись")
    digest = fingerprint(path)
    name = remote_name(path, digest)
    partial = f'{settings["remote_inbox"]}/.{name}.partial'
    final = f'{settings["remote_inbox"]}/{name}'
    update(conn, path, fingerprint=digest, remote_name=name, status="uploading", progress=3, detail="Передаю на сервер")
    command = ["rsync", "-a", "--partial", "--progress", "-e", " ".join(shlex.quote(item) for item in ssh_base(settings)[:-1]), str(path), f'{settings["user"]}@{settings["host"]}:{partial}']
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert process.stdout is not None
    for line in process.stdout:
        if "%" in line:
            try:
                percent = float(line.split("%", 1)[0].split()[-1])
                update(conn, path, progress=min(98, max(3, percent)), detail=f"Передаю на сервер — {percent:.0f}%")
            except ValueError:
                pass
    if process.wait():
        raise RuntimeError("Передача не завершилась; следующая попытка продолжит с остановленного места")
    remote_command = "mv -- {} {}".format(shlex.quote(partial), shlex.quote(final))
    subprocess.run(ssh_base(settings) + [remote_command], check=True)
    update(conn, path, status="done", progress=100, detail="Передано; сервер добавит запись в очередь")


def discover(conn, initialize=False):
    current = time.time()
    for path in sorted(INBOX.iterdir()):
        if not path.is_file() or path.suffix.casefold() not in MEDIA or path.name.endswith(".partial"):
            continue
        stat = path.stat()
        row = conn.execute("SELECT * FROM uploads WHERE path = ?", (str(path),)).fetchone()
        if row is None:
            status = "skipped" if initialize else "waiting"
            detail = "Уже была в папке до включения сервера" if initialize else "Жду завершения записи"
            conn.execute("INSERT INTO uploads(path,size,mtime_ns,status,progress,detail,updated_at) VALUES(?,?,?,?,?,?,?)", (str(path), stat.st_size, stat.st_mtime_ns, status, 0, detail, now()))
            conn.commit(); snapshot(conn)
            continue
        if row["status"] in ("done", "skipped", "uploading", "hashing"):
            continue
        if row["size"] != stat.st_size or row["mtime_ns"] != stat.st_mtime_ns:
            conn.execute("UPDATE uploads SET size=?, mtime_ns=?, updated_at=? WHERE path=?", (stat.st_size, stat.st_mtime_ns, now(), str(path)))
            conn.commit(); continue
        if current - stat.st_mtime >= cfg().get("stable_seconds", 30):
            yield path


def run_once(initialize=False):
    conn, settings = db(), cfg()
    for path in discover(conn, initialize=initialize):
        try:
            upload(conn, path, settings)
        except Exception as exc:
            update(conn, path, status="waiting", detail=str(exc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("watch", "once", "init", "status"))
    args = parser.parse_args()
    if args.command == "init":
        run_once(initialize=True); return
    if args.command == "status":
        print(json.dumps({"uploads": [dict(row) for row in db().execute("SELECT * FROM uploads ORDER BY updated_at DESC")]}, ensure_ascii=False, indent=2)); return
    while True:
        run_once()
        if args.command == "once": return
        time.sleep(cfg().get("poll_seconds", 10))


if __name__ == "__main__":
    main()
