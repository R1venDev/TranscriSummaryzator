#!/usr/bin/env python3
"""Container bootstrap and exec, without model downloads or shell secrets."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import secrets
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summary_credentials import _protected_file, make_admin_password_hash


def locations():
    data = Path(os.environ.get("TRANSCRI_DATA_DIR", "/data"))
    private = Path(os.environ.get("TRANSCRI_SECRETS_DIR", "/secrets"))
    config = Path(os.environ.get("TRANSCRI_CONFIG_FILE", str(data / "config.json")))
    return data, private, config


def write_secret(path: Path, value: str, *, replace=False):
    """Publish a complete private file. Existing keys are never replaced."""
    temporary = path.with_name("." + path.name + "." + secrets.token_hex(8))
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def bootstrap(*, reset_admin=False, enable_speech=False):
    from cryptography.fernet import Fernet
    from config_schema import PipelineConfig

    data, private, config = locations()
    for directory in (data, private):
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (private / ".bootstrap.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        master = private / "summary-master.key"
        if not master.exists():
            stored_secrets = data / "state" / "summary_private"
            if any(stored_secrets.rglob("credentials.sqlite3")) or any(stored_secrets.rglob("plane.sqlite3")):
                raise RuntimeError("Secret volume is missing; restore its master key before starting")
            write_secret(master, Fernet.generate_key().decode("ascii"))
        # Validate ownership and permissions even on repeated bootstrap.
        Fernet(_protected_file(master))
        admin = private / "admin.scrypt"
        password = None
        if not admin.exists() or reset_admin:
            password = secrets.token_urlsafe(24)
            write_secret(admin, make_admin_password_hash(password), replace=reset_admin)
        _protected_file(admin)
        for relative in ("state/summary_private", "inbox", "outputs", "voice_profiles",
                         "work/jobs", "work/stage-cache", "work/cache", "work/pycache"):
            (data / relative).mkdir(mode=0o700, parents=True, exist_ok=True)
        vocabulary = data / "vocabulary.json"
        if not vocabulary.exists() and (ROOT / "vocabulary.json").is_file():
            write_secret(vocabulary, (ROOT / "vocabulary.json").read_text(encoding="utf-8").rstrip())
        if not config.exists():
            values = json.loads((ROOT / "config.example.json").read_text(encoding="utf-8"))
            values.update(summary_backend="luna_batch",
                          processing_enabled=os.environ.get("TRANSCRI_IMAGE_VARIANT") == "speech")
            PipelineConfig.model_validate(values)
            config.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            write_secret(config, json.dumps(values, ensure_ascii=False, indent=2))
        elif enable_speech:
            values = json.loads(config.read_text(encoding="utf-8"))
            values["processing_enabled"] = True
            PipelineConfig.model_validate(values)
            write_secret(config, json.dumps(values, ensure_ascii=False, indent=2), replace=True)
        else:
            PipelineConfig.model_validate_json(config.read_text(encoding="utf-8"))
        print("Persistent data and encrypted credential storage are ready.", flush=True)
        if password:
            print("Administrator: admin\nPassword (shown once): " + password, flush=True)
            print("Save this password now. Only its scrypt verifier is stored.", flush=True)
        else:
            print("Existing administrator and encryption key preserved.", flush=True)


def main():
    os.umask(0o077)
    arguments = sys.argv[1:] or ["dashboard"]
    command = arguments[0]
    if command in {"bootstrap", "reset-admin"}:
        if any(value != "--enable-speech" for value in arguments[1:]):
            raise RuntimeError("Unsupported bootstrap option")
        bootstrap(reset_admin=command == "reset-admin", enable_speech="--enable-speech" in arguments)
        return
    allowed = {"dashboard", "watch", "summary-scheduler", "summary-scheduler-once", "doctor", "status"}
    if command not in allowed or len(arguments) != 1:
        raise RuntimeError("Unsupported container command")
    data, private, config = locations()
    if not config.is_file():
        raise RuntimeError("Run scripts/docker-install.sh to initialize the persistent volumes")
    _protected_file(private / "summary-master.key")
    os.environ["TRANSCRI_SUMMARY_ADMIN_PASSWORD_HASH"] = _protected_file(private / "admin.scrypt").decode("ascii")
    if command == "watch" and os.environ.get("TRANSCRI_IMAGE_VARIANT") != "speech":
        raise RuntimeError("Audio processing requires the speech image; see docs/DOCKER.md")
    os.execv(sys.executable, [sys.executable, str(ROOT / "pipeline.py"), command])


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("Container startup failed: " + str(exc), file=sys.stderr, flush=True)
        raise SystemExit(1) from None
