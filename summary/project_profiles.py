"""Application project namespaces. Ledger and budgets remain app-wide."""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import re
import sqlite3
import time
import uuid

DEFAULT = "default"
CURRENT = ContextVar("transcri_project", default=DEFAULT)


def validate_id(value):
    if not isinstance(value, str) or not (value == DEFAULT or re.fullmatch(r"[a-f0-9]{32}", value)):
        raise ValueError("Профиль проекта не найден")
    return value


@contextmanager
def scope(identifier):
    token = CURRENT.set(validate_id(identifier))
    try:
        yield
    finally:
        CURRENT.reset(token)


def private_path(root, filename, identifier=None):
    identifier = validate_id(CURRENT.get() if identifier is None else identifier)
    root = Path(root)
    return (root if identifier == DEFAULT else root / "projects" / identifier) / filename


class Projects:
    def __init__(self, state):
        self.path = Path(state) / "projects.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise ValueError("Недопустимое хранилище проектов")
        with self.db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS projects(id TEXT PRIMARY KEY,name TEXT NOT NULL UNIQUE COLLATE NOCASE,revision INTEGER NOT NULL,created REAL NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS project_events(seq INTEGER PRIMARY KEY,at REAL NOT NULL,project_id TEXT NOT NULL,operation TEXT NOT NULL,name TEXT NOT NULL)")
            db.execute("INSERT OR IGNORE INTO projects VALUES(?,?,1,?)", (DEFAULT, "Aurion", time.time()))

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def list(self):
        with self.db() as db:
            return [dict(row) for row in db.execute("SELECT * FROM projects ORDER BY created,id")]

    def require(self, identifier):
        validate_id(identifier)
        with self.db() as db:
            row = db.execute("SELECT * FROM projects WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise ValueError("Профиль проекта не найден")
        return dict(row)

    @staticmethod
    def name(value):
        if not isinstance(value, str) or not value.strip() or len(value.strip()) > 80 or any(ord(c) < 32 for c in value):
            raise ValueError("Введите название до 80 символов")
        return value.strip()

    def create(self, name):
        identifier, name = uuid.uuid4().hex, self.name(name)
        try:
            with self.db() as db:
                db.execute("INSERT INTO projects VALUES(?,?,1,?)", (identifier, name, time.time()))
                db.execute("INSERT INTO project_events(at,project_id,operation,name) VALUES(?,?,?,?)",(time.time(),identifier,"create",name))
        except sqlite3.IntegrityError:
            raise ValueError("Профиль с таким названием уже существует") from None
        return self.require(identifier)

    def rename(self, identifier, name, revision):
        self.require(identifier)
        if type(revision) is not int:
            raise ValueError("Обновите страницу перед сохранением")
        try:
            with self.db() as db:
                changed = db.execute("UPDATE projects SET name=?,revision=revision+1 WHERE id=? AND revision=?", (self.name(name), identifier, revision))
                if changed.rowcount != 1:
                    raise ValueError("Название изменилось. Обновите страницу")
                db.execute("INSERT INTO project_events(at,project_id,operation,name) VALUES(?,?,?,?)",(time.time(),identifier,"rename",self.name(name)))
        except sqlite3.IntegrityError:
            raise ValueError("Профиль с таким названием уже существует") from None
        return self.require(identifier)
