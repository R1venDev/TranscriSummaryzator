"""Plane delivery from published summaries; no inference or speech dependencies.

REST v1 is deliberately used: the installed Plane 3.1.0 exposes these routes.
A durable intent precedes POST. An interrupted POST is reconciled by the app's
external identity; absence is never interpreted as permission to send again.
Remote cards are preserved on regeneration (including human edits).
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
from html import escape
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from scripts.summary_credentials import CredentialError, _fernet, load_master_key

EXTERNAL_SOURCE = "transcrisummaryzator"
# Plane returns HTML, JSON, stripped text and binary projections together.
MAX_PLANE_RESPONSE_BYTES = 16 * 1024 * 1024
DEFAULTS = dict(base_url="", workspace_slug="", project_id="", collection_id="",
                parent_page_id="", task_type_id="", hypothesis_type_id="",
                auto_tasks=False, auto_hypotheses=False,
                auto_meeting_page=True)
DESTINATION_FIELDS = ("base_url", "workspace_slug", "project_id", "collection_id", "parent_page_id")


class PlaneError(ValueError):
    """Safe, credential-free message suitable for the protected settings UI."""


class PlanePreflightError(PlaneError):
    """Known failure before any mutating request was sent."""


class PlaneHTTPError(PlaneError):
    def __init__(self, status, retry_after=60):
        self.status = status
        self.retry_after = max(30, min(3600, retry_after))
        super().__init__(f"Plane HTTP {status}. Проверьте права ключа и выбранный раздел.")


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _destination(settings):
    return {key: settings.get(key, "") for key in DESTINATION_FIELDS}


def _delivery_destination(settings, kind):
    names = ("base_url", "workspace_slug") + (() if kind == "page" else ("project_id",))
    return {key: settings.get(key, "") for key in names}


def _external_id(job_id, source_id, kind, item_id):
    return "ts-" + _hash([str(job_id) if kind == "page" else source_id, kind, item_id])[:40]


def _id(value):
    if not isinstance(value, str) or not value or len(value) > 256 or any(ord(c) < 32 for c in value):
        raise PlaneError("Недопустимый идентификатор")
    return value


class _SafeHTML(HTMLParser):
    tags = {"p", "br", "strong", "b", "em", "i", "u", "s", "h1", "h2", "h3", "h4", "h5", "h6",
            "ul", "ol", "li", "blockquote", "pre", "code", "table", "thead", "tbody", "tr", "th", "td",
            "a", "details", "summary", "hr", "div", "mention-component"}
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.hidden = [], 0
    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "iframe", "object"}:
            self.hidden += 1
        if self.hidden or tag not in self.tags:
            return
        safe = ""
        if tag == "a":
            href = dict(attrs).get("href", "")
            if urllib.parse.urlsplit(href).scheme in {"https", "http"}:
                safe = ' href="' + escape(href, quote=True) + '"'
        attributes = dict(attrs)
        if tag == "p" and "data-transcri-participants" in attributes:
            try:
                names = json.loads(attributes["data-transcri-participants"])
                if not isinstance(names, list) or any(not isinstance(n, str) for n in names):
                    raise ValueError()
                safe = ' data-transcri-participants="' + escape(json.dumps(names, ensure_ascii=False), quote=True) + '"'
            except (ValueError, TypeError):
                raise PlaneError("Неверный список участников Wiki") from None
        elif tag == "details":
            identifier = attributes.get("data-id", "")
            if re.fullmatch(r"transcri-transcript-[0-9a-f]{32}", identifier):
                safe = ' class="editor-details-block" data-id="' + identifier + '"'
        elif tag == "summary":
            safe = ' class="editor-details-summary"'
        elif tag == "div" and attributes.get("data-type") == "detailsContent":
            safe = ' class="editor-details-content" data-type="detailsContent"'
        elif tag == "mention-component":
            try:
                identifier = str(uuid.UUID(attributes["entity_identifier"]))
                node_id = str(uuid.UUID(attributes["id"]))
            except (KeyError, ValueError, TypeError, AttributeError):
                return
            if attributes.get("entity_name") != "user_mention":
                return
            safe = f' id="{node_id}" entity_identifier="{identifier}" entity_name="user_mention"'
        self.out.append("<" + tag + safe + ">")
    def handle_endtag(self, tag):
        if tag in {"script", "style", "iframe", "object"} and self.hidden:
            self.hidden -= 1
            return
        if not self.hidden and tag in self.tags and tag not in {"br", "hr"}:
            self.out.append("</" + tag + ">")
    def handle_data(self, data):
        if not self.hidden:
            self.out.append(escape(data))


def safe_html(value):
    if not isinstance(value, str) or len(value.encode()) > 1024 * 1024:
        raise PlaneError("Описание Plane отсутствует или слишком большое")
    parser = _SafeHTML()
    parser.feed(value)
    parser.close()
    return "".join(parser.out)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "Redirect blocked", headers, fp)


class PlaneClient:
    def __init__(self, settings, api_key):
        self.base = settings["base_url"]
        self.key = api_key
        self.prefix = "/api/v1/workspaces/" + urllib.parse.quote(settings["workspace_slug"], safe="") + "/"
        self.project = settings.get("project_id", "")
        self.opener = urllib.request.build_opener(_NoRedirect())

    def request(self, method, path, body=None, allowed_statuses=()):
        if not path.startswith(self.prefix) or path.startswith("//"):
            raise PlaneError("Недопустимый путь Plane API")
        encoded = _json(body).encode() if body is not None else None
        request = urllib.request.Request(self.base + path, data=encoded, method=method,
            headers={"X-API-Key": self.key, "Content-Type": "application/json", "Accept": "application/json"})
        try:
            with self.opener.open(request, timeout=25) as response:
                data = response.read(MAX_PLANE_RESPONSE_BYTES + 1)
                if len(data) > MAX_PLANE_RESPONSE_BYTES:
                    raise PlaneError("Ответ Plane превышает допустимый размер")
                return json.loads(data) if data else {}
        except urllib.error.HTTPError as exc:
            if exc.code in allowed_statuses:
                data = exc.read(MAX_PLANE_RESPONSE_BYTES + 1)
                try:
                    if len(data) <= MAX_PLANE_RESPONSE_BYTES:
                        return json.loads(data)
                except (ValueError, TypeError):
                    pass
                raise PlaneError("Неизвестный формат ответа Plane") from None
            try:
                retry = int(exc.headers.get("Retry-After", "60"))
            except (ValueError, TypeError, AttributeError):
                retry = 60
            raise PlaneHTTPError(exc.code, retry) from None
        except PlaneError:
            raise
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            raise PlaneError("Не удалось получить достоверный ответ Plane") from None

    def list_all(self, path, max_pages=10):
        result, cursor = [], None
        for _ in range(max_pages):
            params = {"per_page": 100}
            if cursor:
                params["cursor"] = cursor
            response = self.request("GET", path + ("&" if "?" in path else "?") + urllib.parse.urlencode(params))
            if isinstance(response, list):
                return result + response
            if not isinstance(response, dict) or not isinstance(response.get("results"), list):
                raise PlaneError("Неизвестный формат списка Plane")
            result.extend(response["results"])
            if not response.get("next_page_results"):
                return result
            next_cursor = response.get("next_cursor")
            if not isinstance(next_cursor, str) or next_cursor == cursor:
                raise PlaneError("Plane не вернул следующий курсор")
            cursor = next_cursor
        raise PlaneError("Список Plane слишком большой для безопасной сверки")

    def check(self):
        projects = self.list_all(self.prefix + "projects/")
        collections = self.list_all(self.prefix + "collections/")
        # Read permission does not prove create permission; no probe writes.
        types = self.list_all(self.prefix + "projects/" + self.project + "/work-item-types/") if self.project else []
        return {"projects": [{"id": p["id"], "name": p.get("name", "")} for p in projects],
                "types": [{"id": t["id"], "name": t.get("name", ""),
                           "is_active": t.get("is_active") is True, "is_epic": t.get("is_epic") is True}
                          for t in types], "types_project_id": self.project,
                "collections": [{"id": p["id"], "name": p.get("name", "")} for p in collections]}

    def recover(self, row):
        if row["kind"] != "page":
            path = self.prefix + "projects/" + self.project + "/work-items/?" + urllib.parse.urlencode(
                {"external_source": EXTERNAL_SOURCE, "external_id": row["external_id"]})
            try:
                item = self.request("GET", path)
            except PlaneHTTPError as exc:
                if exc.status == 404:
                    return None
                raise
            if isinstance(item, dict) and item.get("id"):
                return item
            raise PlaneError("Plane не подтвердил external ID карточки")
        # This Plane version does not expose page external_id in GET serializers.
        # Use an exact application marker inside the body, never fuzzy title match.
        pages = self.list_all(self.prefix + "pages/?type=all&" + urllib.parse.urlencode({"search": row["title"]}))
        marker = "transcrisummaryzator-id:" + row["external_id"]
        found = []
        for page in pages:
            detail = self.request("GET", self.prefix + "pages/" + str(uuid.UUID(page["id"])) + "/")
            if marker in detail.get("description_html", ""):
                found.append(detail)
        if len(found) > 1:
            raise PlaneError("Найдены несколько Wiki-страниц с одной identity; автоматическая отправка остановлена")
        return found[0] if found else None

    def create(self, row, settings):
        payload = dict(json.loads(row["payload"]))
        if row["kind"] == "page":
            path = self.prefix + "pages/"
            from summary.plane_wiki import PARTICIPANTS, resolve_mentions
            if PARTICIPANTS in payload.get("description_html", ""):
                try:
                    members = self.list_all(self.prefix + "members/")
                    payload["description_html"] = resolve_mentions(payload["description_html"], members)
                except PlaneError as exc:
                    raise PlanePreflightError("Не удалось проверить участников Plane; страница не отправлена") from exc
        else:
            try:
                project = self.request("GET", self.prefix + "projects/" + self.project + "/")
            except PlaneHTTPError as exc:
                if exc.status == 429:
                    raise  # Known rejection; the normal bounded backoff applies.
                raise PlanePreflightError(str(exc)) from None
            except PlaneError as exc:
                raise PlanePreflightError(str(exc)) from None
            # Plane 3.1.0 v1 AND v2 apply a project default even for an empty list.
            # Never accidentally assign or notify a person not named by the source.
            if not isinstance(project, dict) or "default_assignee" not in project or project["default_assignee"] is not None:
                raise PlanePreflightError("Для неназначенных карточек отключите исполнителя по умолчанию в выбранном проекте Plane")
            if payload.get("type_id"):
                try:
                    types = self.list_all(self.prefix + "projects/" + self.project + "/work-item-types/")
                except PlaneHTTPError as exc:
                    if exc.status == 429:
                        raise
                    raise PlanePreflightError(str(exc)) from None
                except PlaneError as exc:
                    raise PlanePreflightError(str(exc)) from None
                matches = [t for t in types if t.get("id") == payload["type_id"]]
                if len(matches) != 1 or matches[0].get("is_active") is not True or matches[0].get("is_epic") is not False:
                    raise PlanePreflightError("Выбранный тип карточки отсутствует в проекте, отключён или является Epic. Выберите действующий тип в настройках Plane")
            path = self.prefix + "projects/" + self.project + "/work-items/"
        return self.request("POST", path, payload)


class PlaneStore:
    def __init__(self, path, master_key=None, client_factory=None):
        self.path = Path(path)
        self.master_key = master_key
        self.client_factory = client_factory or PlaneClient
        self.path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        if self.path.is_symlink():
            raise PlaneError("Хранилище Plane не должно быть символьной ссылкой")
        self.lock_path = self.path.with_suffix(".lock")
        with self._db() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS settings(id INTEGER PRIMARY KEY CHECK(id=1), revision INTEGER NOT NULL,
                body TEXT NOT NULL, encrypted_key BLOB, status TEXT NOT NULL, options TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS meetings(job_id TEXT PRIMARY KEY, generation_id TEXT NOT NULL, source_id TEXT NOT NULL, auto_seen TEXT);
            CREATE TABLE IF NOT EXISTS desired(job_id TEXT NOT NULL, kind TEXT NOT NULL, item_id TEXT NOT NULL,
                title TEXT NOT NULL, description TEXT NOT NULL, html TEXT NOT NULL, generation_id TEXT NOT NULL,
                source_id TEXT NOT NULL, ambiguous INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(job_id,kind,item_id));
            CREATE TABLE IF NOT EXISTS deliveries(id TEXT PRIMARY KEY, job_id TEXT NOT NULL, kind TEXT NOT NULL,
                item_id TEXT NOT NULL, generation_id TEXT NOT NULL, external_id TEXT NOT NULL, title TEXT NOT NULL,
                destination TEXT NOT NULL, payload TEXT NOT NULL, state TEXT NOT NULL, remote_id TEXT, remote_url TEXT,
                error TEXT NOT NULL DEFAULT '', next_attempt REAL NOT NULL DEFAULT 0, attempts INTEGER NOT NULL DEFAULT 0,
                created REAL NOT NULL, updated REAL NOT NULL, automatic INTEGER NOT NULL DEFAULT 0, UNIQUE(external_id,destination));
            CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT, created REAL NOT NULL,
                delivery_id TEXT, event TEXT NOT NULL, details TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS hypotheses(source_id TEXT NOT NULL, item_id TEXT PRIMARY KEY,
                text TEXT NOT NULL, anchors TEXT NOT NULL, ambiguous INTEGER NOT NULL DEFAULT 0);
            """)
            from summary.plane_media import init_table
            init_table(db)
            db.execute("INSERT OR IGNORE INTO settings VALUES(1,0,?,NULL,?,?)",
                       (_json(DEFAULTS), _json({"state": "not_configured", "message": "Plane не настроен"}), _json({"projects": [], "collections": []})))
        os.chmod(self.path, 0o600)

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @contextmanager
    def _guard(self):
        fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def baseline_complete(self):
        with self._db() as db:
            return db.execute("SELECT 1 FROM metadata WHERE key='baseline' AND value='1'").fetchone() is not None

    def mark_baseline_complete(self):
        with self._guard(), self._db() as db:
            db.execute("INSERT OR REPLACE INTO metadata VALUES('baseline','1')")
            db.execute("UPDATE meetings SET auto_seen=generation_id")

    def known_generation(self, job_id):
        with self._db() as db:
            row = db.execute("SELECT auto_seen FROM meetings WHERE job_id=?", (str(job_id),)).fetchone()
            return row[0] if row else None

    def _row(self, db):
        return dict(db.execute("SELECT * FROM settings WHERE id=1").fetchone())

    def _settings(self, row):
        return {**DEFAULTS, **json.loads(row["body"])}

    def _key(self, row):
        if not row["encrypted_key"]:
            raise PlaneError("Добавьте API-ключ Plane в настройках")
        try:
            return _fernet(self.master_key or load_master_key()).decrypt(row["encrypted_key"]).decode()
        except Exception:
            raise PlaneError("Не удалось открыть защищённый ключ Plane") from None

    def public_settings(self):
        with self._db() as db:
            row = self._row(db)
        settings = self._settings(row)
        settings.update(revision=row["revision"], key_mask="••••••••" if row["encrypted_key"] else "",
                        configured=bool(row["encrypted_key"] and settings["base_url"] and settings["workspace_slug"]))
        return {"settings": settings, "status": json.loads(row["status"]), "options": json.loads(row["options"])}

    def save_settings(self, data):
        with self._guard(), self._db() as db:
            row = self._row(db)
            if type(data.get("expected_revision")) is not int or data["expected_revision"] != row["revision"]:
                raise PlaneError("Настройки изменились. Обновите страницу перед сохранением")
            old = self._settings(row)
            settings = dict(old)
            for name in DEFAULTS:
                if name not in data:
                    continue
                value = data[name]
                if name.startswith("auto_"):
                    if type(value) is not bool:
                        raise PlaneError("Переключатель должен быть логическим значением")
                elif not isinstance(value, str):
                    raise PlaneError("Недопустимое значение настройки")
                else:
                    value = value.strip()
                settings[name] = value
            settings["base_url"] = settings["base_url"].rstrip("/")
            if settings["base_url"]:
                url = urllib.parse.urlsplit(settings["base_url"])
                insecure_allowed = os.getenv("TRANSCRI_PLANE_ALLOW_HTTP") == "1"
                if (url.scheme not in ({"https", "http"} if insecure_allowed else {"https"}) or not url.hostname
                        or url.username or url.password or url.path or url.query or url.fragment):
                    raise PlaneError("Укажите HTTPS-адрес Plane без пути, пароля и параметров")
                try:
                    url.port
                except ValueError:
                    raise PlaneError("Недопустимый порт Plane") from None
            if settings["workspace_slug"] and not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", settings["workspace_slug"]):
                raise PlaneError("Недопустимый workspace slug")
            for name in ("project_id", "collection_id", "parent_page_id", "task_type_id", "hypothesis_type_id"):
                if settings[name]:
                    try:
                        settings[name] = str(uuid.UUID(settings[name]))
                    except ValueError:
                        raise PlaneError("Идентификатор проекта, коллекции, страницы или типа должен быть UUID") from None
            if old["project_id"] != settings["project_id"]:
                for name in ("task_type_id", "hypothesis_type_id"):
                    if name not in data:
                        settings[name] = ""
            if settings["collection_id"] and settings["parent_page_id"]:
                raise PlaneError("Выберите коллекцию или родительскую страницу, не оба сразу")
            token = data.get("api_key", "")
            encrypted = row["encrypted_key"]
            if token:
                if not isinstance(token, str) or not 16 <= len(token) <= 1024 or any(not 33 <= ord(c) <= 126 for c in token):
                    raise PlaneError("Недопустимый API-ключ Plane")
                try:
                    encrypted = _fernet(self.master_key or load_master_key()).encrypt(token.encode())
                except CredentialError as exc:
                    raise PlaneError(str(exc)) from None
            if old["base_url"] != settings["base_url"] and encrypted and not token:
                raise PlaneError("Для другого адреса Plane введите его ключ заново")
            if data.get("remove_key") is True:
                encrypted = None
            for kind, name in (("task", "auto_tasks"), ("hypothesis", "auto_hypotheses"), ("page", "auto_meeting_page")):
                if old[name] and not settings[name]:
                    db.execute("UPDATE deliveries SET state='auto_disabled',error='Автосоздание отключено',updated=? WHERE kind=? AND automatic=1 AND state='queued'", (time.time(), kind))
            if token and old["base_url"] == settings["base_url"]:
                db.execute("UPDATE deliveries SET state='queued',next_attempt=0,error='' WHERE state='credential_required'")
            changed_dest = _destination(old) != _destination(settings)
            db.execute("UPDATE settings SET revision=revision+1,body=?,encrypted_key=?,status=?,options=? WHERE id=1",
                (_json(settings), encrypted, _json({"state": "unchecked", "message": "Настройки сохранены; проверка без создания карточек"}),
                 _json({"projects": [], "collections": []}) if changed_dest else row["options"]))
            self._event(db, None, "settings_changed", {"revision": row["revision"] + 1})
        return self.public_settings()

    def check(self):
        with self._guard():
            with self._db() as db:
                row = self._row(db)
            settings = self._settings(row)
            if not settings["base_url"] or not settings["workspace_slug"]:
                raise PlaneError("Сначала сохраните адрес Plane, workspace и ключ")
            try:
                options = self.client_factory(settings, self._key(row)).check()
                if options.get("types_project_id") == settings["project_id"]:
                    for name in ("task_type_id", "hypothesis_type_id"):
                        if settings[name] and not any(t.get("id") == settings[name] and t.get("is_active") is True and t.get("is_epic") is False for t in options.get("types", [])):
                            raise PlaneError("Выбранный тип отсутствует или недоступен в проекте Plane")
                status = {"state": "ready", "message": "Чтение доступно. Права создания проверяются при отправке", "checked_at": time.time()}
            except PlaneError as exc:
                options = {"projects": [], "collections": []}
                status = {"state": "error", "message": str(exc), "checked_at": time.time()}
            with self._db() as db:
                db.execute("UPDATE settings SET status=?,options=? WHERE id=1", (_json(status), _json(options)))
        return self.public_settings()

    def _event(self, db, delivery_id, event, details=None):
        db.execute("INSERT INTO events(created,delivery_id,event,details) VALUES(?,?,?,?)",
                   (time.time(), delivery_id, event, _json(details or {})))

    def hypothesis_ids(self, source_id, ideas):
        source_id = _id(str(source_id))
        normalized = [(" ".join(str(i["text"]).split()), _json(sorted(set(i.get("source_ids", []))))) for i in ideas]
        result = []
        with self._guard(), self._db() as db:
            old = [dict(r) for r in db.execute("SELECT * FROM hypotheses WHERE source_id=?", (source_id,))]
            for text, anchors in normalized:
                exact = [r for r in old if r["text"] == text and r["anchors"] == anchors]
                same = [r for r in old if r["anchors"] == anchors]
                if len(exact) == 1:
                    item_id = exact[0]["item_id"]
                elif len(same) == 1 and sum(a == anchors for _, a in normalized) == 1:
                    item_id = same[0]["item_id"]
                    db.execute("UPDATE hypotheses SET text=? WHERE item_id=?", (text, item_id))
                else:
                    item_id = str(uuid.uuid4())
                    ambiguous = bool(same)
                    db.execute("INSERT INTO hypotheses VALUES(?,?,?,?,?)", (source_id, item_id, text, anchors, ambiguous))
                result.append(item_id)
        return result

    def observe_generation(self, job_id, generation_id, items, page=None, source_id="", allow_auto=False):
        job_id, generation_id = _id(str(job_id)), _id(generation_id)
        source_id = _id(source_id or job_id)
        prepared = []
        for item in items:
            if item.get("kind") not in {"task", "hypothesis"}:
                raise PlaneError("Недопустимый вид карточки")
            prepared.append(dict(item))
        if page:
            prepared.append({**page, "kind": "page", "item_id": "meeting"})
        identities = [(i["kind"], i["item_id"]) for i in prepared]
        if len(identities) != len(set(identities)):
            raise PlaneError("Дублирующиеся локальные identity")
        with self._guard(), self._db() as db:
            previous = db.execute("SELECT * FROM meetings WHERE job_id=?", (job_id,)).fetchone()
            fresh = previous is None or previous["auto_seen"] != generation_id
            db.execute("INSERT INTO meetings(job_id,generation_id,source_id) VALUES(?,?,?) ON CONFLICT(job_id) DO UPDATE SET generation_id=excluded.generation_id,source_id=excluded.source_id",
                       (job_id, generation_id, source_id))
            db.execute("DELETE FROM desired WHERE job_id=?", (job_id,))
            for item in prepared:
                title = str(item.get("title", "")).strip()
                if not title or len(title) > 255:
                    raise PlaneError("Название карточки должно содержать от 1 до 255 символов")
                item_id = _id(item["item_id"])
                html = safe_html(item.get("description_html", ""))
                if not html.strip():
                    raise PlaneError("Описание карточки пустое")
                known = db.execute("SELECT ambiguous FROM hypotheses WHERE item_id=?", (item_id,)).fetchone()
                ambiguous = int(bool(known and known[0]))
                db.execute("INSERT INTO desired VALUES(?,?,?,?,?,?,?,?,?)",
                           (job_id, item["kind"], item_id, title, str(item.get("description", "")), html, generation_id, source_id, ambiguous))
            # Refresh only known-unsent intents. A submitted/unknown payload remains
            # immutable so recovery still refers to exactly the request sent.
            queued = list(db.execute("SELECT * FROM deliveries WHERE job_id=? AND state='queued'", (job_id,)))
            for pending in queued:
                current = db.execute("SELECT * FROM desired WHERE job_id=? AND kind=? AND item_id=?", (job_id, pending["kind"], pending["item_id"])).fetchone()
                still_current = current is not None and not current["ambiguous"] and _external_id(job_id, current["source_id"], pending["kind"], pending["item_id"]) == pending["external_id"]
                if not still_current:
                    self._finish(db, dict(pending), "cancelled_obsolete", "Карточка больше не входит в текущий конспект")
                elif pending["destination"] == _json(_delivery_destination(self._settings(self._row(db)), pending["kind"])):
                    self._enqueue(db, job_id, pending["kind"], pending["item_id"], generation_id, automatic=bool(pending["automatic"]))
            if allow_auto:
                db.execute("UPDATE meetings SET auto_seen=? WHERE job_id=?", (generation_id, job_id))
            settings_row = self._row(db)
            settings = self._settings(settings_row)
            # Report drift for every exported card, including manually exported
            # cards with automation off. Comparison never updates a remote body.
            for item in prepared:
                self._enqueue(db, job_id, item["kind"], item["item_id"], generation_id,
                              automatic=True, compare_only=True)
            if allow_auto and fresh and settings_row["encrypted_key"] and settings["base_url"] and settings["workspace_slug"]:
                enabled = {"task": settings["auto_tasks"], "hypothesis": settings["auto_hypotheses"], "page": settings["auto_meeting_page"]}
                for item in prepared:
                    if enabled[item["kind"]] and (item["kind"] == "page" or settings["project_id"]):
                        self._enqueue(db, job_id, item["kind"], item["item_id"], generation_id, automatic=True)
        return self.items(job_id)

    def _enqueue(self, db, job_id, kind, item_id, generation_id, automatic=False, compare_only=False):
        row = db.execute("SELECT * FROM desired WHERE job_id=? AND kind=? AND item_id=?", (str(job_id), kind, item_id)).fetchone()
        if row is None or row["generation_id"] != generation_id:
            raise PlaneError("Саммари изменилось. Обновите страницу")
        if row["ambiguous"]:
            if automatic:
                return
            raise PlaneError("Неоднозначная identity гипотезы; проверьте существующую карточку перед созданием")
        setting_row = self._row(db)
        settings = self._settings(setting_row)
        if not compare_only and (not setting_row["encrypted_key"] or not settings["base_url"] or not settings["workspace_slug"] or (kind != "page" and not settings["project_id"])):
            if automatic:
                return
            raise PlaneError("Сначала настройте подключение и выберите проект Plane")
        external_id = _external_id(job_id, row["source_id"], kind, item_id)
        destination = _json(_delivery_destination(settings, kind))
        old = db.execute("SELECT * FROM deliveries WHERE external_id=? AND destination=?", (external_id, destination)).fetchone()
        payload = {"name": ("Гипотеза: " if kind == "hypothesis" and not row["title"].lower().startswith("гипотеза:") else "") + row["title"],
                   "description_html": row["html"], "external_id": external_id, "external_source": EXTERNAL_SOURCE}
        payload["name"] = payload["name"][:255]
        if kind == "page":
            payload["access"] = 0
            payload["description_html"] += "<p>transcrisummaryzator-id:" + external_id + "</p>"
            if settings["parent_page_id"]:
                payload["parent_id"] = settings["parent_page_id"]
            elif settings["collection_id"]:
                payload["collection_id"] = settings["collection_id"]
        else:
            payload["assignees"] = []
            payload["priority"] = "none"
            type_id = settings["task_type_id" if kind == "task" else "hypothesis_type_id"]
            if type_id:
                payload["type_id"] = type_id
        encoded = _json(payload)
        if old and old["state"] in {"created", "update_available"}:
            changed = old["payload"] != encoded
            state = "update_available" if changed else "created"
            if old["state"] != state:
                db.execute("UPDATE deliveries SET state=?,error=?,updated=? WHERE id=?",
                    (state, "В саммари есть изменения. Карточка Plane сохранена без перезаписи ручных правок" if changed else "", time.time(), old["id"]))
                self._event(db, old["id"], "local_projection_changed" if changed else "local_projection_restored")
            return
        if compare_only:
            return
        if old:
            if old["state"] == "cancelled_obsolete" or (not automatic and old["state"] in {"failed", "auto_disabled", "credential_required", "destination_changed"}):
                db.execute("UPDATE deliveries SET state='queued',automatic=?,error='',next_attempt=0,payload=?,generation_id=?,title=?,updated=? WHERE id=?",
                           (int(automatic), encoded, generation_id, row["title"], time.time(), old["id"]))
                self._event(db, old["id"], "automatic_revival" if automatic else "manual_retry_known_failure")
            elif old["state"] == "queued":
                db.execute("UPDATE deliveries SET payload=?,generation_id=?,title=?,updated=? WHERE id=?",
                           (encoded, generation_id, row["title"], time.time(), old["id"]))
            return
        delivery_id = str(uuid.uuid4())
        db.execute("INSERT INTO deliveries(id,job_id,kind,item_id,generation_id,external_id,title,destination,payload,state,created,updated,automatic) VALUES(?,?,?,?,?,?,?,?,?,'queued',?,?,?)",
                   (delivery_id, str(job_id), kind, item_id, generation_id, external_id, row["title"], destination, encoded, time.time(), time.time(), int(automatic)))
        self._event(db, delivery_id, "enqueued", {"automatic": automatic, "generation_id": generation_id})

    def enqueue(self, job_id, kind, item_id, generation_id):
        with self._guard(), self._db() as db:
            self._enqueue(db, str(job_id), kind, item_id, generation_id)
        return self.items(job_id)

    def items(self, job_id):
        with self._db() as db:
            meeting = db.execute("SELECT * FROM meetings WHERE job_id=?", (str(job_id),)).fetchone()
            desired = list(db.execute("SELECT * FROM desired WHERE job_id=? ORDER BY rowid", (str(job_id),)))
            settings_row = self._row(db)
            settings = self._settings(settings_row)
            items, page = [], None
            for item in desired:
                external_id = _external_id(job_id, item["source_id"], item["kind"], item["item_id"])
                destination = _json(_delivery_destination(settings, item["kind"]))
                sent = db.execute("SELECT * FROM deliveries WHERE external_id=? AND destination=?", (external_id, destination)).fetchone()
                state = "identity_ambiguous" if item["ambiguous"] else (sent["state"] if sent else "not_created")
                configured = bool(settings_row["encrypted_key"] and settings["base_url"] and settings["workspace_slug"] and (item["kind"] == "page" or settings["project_id"]))
                if not configured and not sent:
                    state = "unconfigured"
                out = {"kind": item["kind"], "item_id": item["item_id"], "title": item["title"], "description": item["description"],
                       "state": state, "remote_url": sent["remote_url"] if sent else None, "error": sent["error"] if sent else ("Настройте подключение к Plane" if not configured else "")}
                if item["kind"] == "page":
                    media = db.execute("SELECT state,error FROM page_media WHERE delivery_id=?", (sent["id"],)).fetchone() if sent else None
                    out["media_state"] = media["state"] if media else "unavailable"
                    out["media_error"] = media["error"] if media else ""
                    page = out
                else:
                    items.append(out)
        return {"generation_id": meeting["generation_id"] if meeting else None, "items": items, "meeting_page": page}

    def drain_one(self):
        result = self._drain_delivery_one()
        if result is None:
            from summary.plane_media import drain
            return drain(self)
        return result

    def _drain_delivery_one(self):
        with self._guard():
            with self._db() as db:
                row = db.execute("SELECT * FROM deliveries WHERE state IN ('queued','sending','submission_unknown') AND next_attempt<=? ORDER BY created LIMIT 1", (time.time(),)).fetchone()
                if row is None:
                    return None
                row = dict(row)
                settings_row = self._row(db)
                settings = self._settings(settings_row)
                recovering = row["state"] in {"sending", "submission_unknown"}
                if row["destination"] != _json(_delivery_destination(settings, row["kind"])):
                    db.execute("UPDATE deliveries SET error=?,next_attempt=? WHERE id=?", ("Подключение изменилось; старая отправка не перенаправлена", time.time() + 300, row["id"]))
                    return {"state": "destination_changed"}
                setting_name = {"task": "auto_tasks", "hypothesis": "auto_hypotheses", "page": "auto_meeting_page"}[row["kind"]]
                if row["automatic"] and not recovering and not settings[setting_name]:
                    self._finish(db, row, "auto_disabled", "Автосоздание отключено")
                    return {"state": "auto_disabled"}
                try:
                    client = self.client_factory(settings, self._key(settings_row))
                except PlaneError as exc:
                    self._finish(db, row, "submission_unknown" if recovering else "credential_required", str(exc), 300)
                    return {"state": "credential_required"}
                if not recovering:
                    # Only unsent intents follow the current type selection. Type
                    # is not part of external identity, so rotation cannot duplicate a card.
                    if row["kind"] != "page":
                        payload = json.loads(row["payload"])
                        type_id = settings["task_type_id" if row["kind"] == "task" else "hypothesis_type_id"]
                        if type_id:
                            payload["type_id"] = type_id
                        else:
                            payload.pop("type_id", None)
                        encoded = _json(payload)
                        if encoded != row["payload"]:
                            row["payload"] = encoded
                            db.execute("UPDATE deliveries SET payload=? WHERE id=?", (encoded, row["id"]))
                            self._event(db, row["id"], "unsent_type_changed", {"type_id": type_id})
                    # COMMIT before network. Crash after this point means recovery only.
                    db.execute("UPDATE deliveries SET state='sending',attempts=attempts+1,updated=? WHERE id=?", (time.time(), row["id"]))
                    self._event(db, row["id"], "submission_intent")
            try:
                response = client.recover(row) if recovering else client.create(row, settings)
                if not isinstance(response, dict) or not response.get("id"):
                    raise PlaneError("Исход отправки неизвестен; повторное создание отключено")
                remote_id = str(uuid.UUID(response["id"]))
                suffix = ("wiki/" + remote_id if row["kind"] == "page" else "projects/" + settings["project_id"] + "/issues/" + remote_id)
                url = settings["base_url"] + "/" + settings["workspace_slug"] + "/" + suffix
                with self._db() as db:
                    db.execute("UPDATE deliveries SET state='created',remote_id=?,remote_url=?,error='',updated=? WHERE id=?", (remote_id, url, time.time(), row["id"]))
                    self._event(db, row["id"], "recovered" if recovering else "created", {"remote_id": remote_id})
                return {"state": "created", "remote_url": url}
            except PlanePreflightError as exc:
                state, delay, error = "failed", 0, str(exc)
            except PlaneHTTPError as exc:
                if exc.status == 429 and not recovering:
                    state, delay = "queued", exc.retry_after
                elif not recovering and exc.status in {400, 401, 403, 404, 405, 413, 422}:
                    state, delay = "failed", 0
                else:
                    state, delay = "submission_unknown", 300
                error = str(exc)
            except PlaneError as exc:
                state, delay, error = "submission_unknown", 300, str(exc)
            except Exception:
                # Untrusted response/transport exceptions never expose remote data or credentials.
                state, delay, error = "submission_unknown", 300, "Исход отправки неизвестен; выполняется только сверка"
            with self._db() as db:
                self._finish(db, row, state, error, delay)
            return {"state": state, "error": error}

    def _finish(self, db, row, state, error, delay=0):
        db.execute("UPDATE deliveries SET state=?,error=?,next_attempt=?,updated=? WHERE id=?",
                   (state, error, time.time() + delay, time.time(), row["id"]))
        self._event(db, row["id"], state)
