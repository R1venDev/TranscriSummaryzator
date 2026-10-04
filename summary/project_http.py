"""Thin namespace adapter for the existing HTTP server and speech worker."""
from functools import wraps
from html import escape
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import hashlib
import json
import os
import sqlite3
from summary.project_profiles import CURRENT, DEFAULT, Projects, scope


def install(pipeline, *, registry_state=None, native=False):
    registry = lambda: Projects(registry_state if registry_state is not None else pipeline.STATE)
    handler = pipeline.DashboardHandler
    if getattr(handler, '_projects_installed', False):
        return
    handler._projects_installed = True

    def selected(h):
        values = parse_qs(urlparse(h.path).query).get('project', [])
        header = h.headers.get('X-Transcri-Project')
        if len(values) > 1 or header and values and header != values[0]:
            raise ValueError('Противоречивый профиль проекта')
        identifier = header or (values[0] if values else DEFAULT)
        registry().require(identifier)
        return identifier

    def inject(body, content_type):
        if not content_type.startswith('text/html'):
            return body
        marker = ('<meta name="transcri-project" content="' + escape(CURRENT.get(), quote=True)
                  + '"><script src="/project-scope.js" ></script>')
        return body.replace(b'</head>', marker.encode() + b'</head>', 1)

    for name in ('send_bytes', 'send_summary_admin_bytes'):
        if not hasattr(handler, name):
            continue
        original = getattr(handler, name)
        def wrap_send(original):
            @wraps(original)
            def send(h, body, content_type, *a, **kw):
                return original(h, inject(body, content_type), content_type, *a, **kw)
            return send
        setattr(handler, name, wrap_send(original))

    def wrap(original, write):
        @wraps(original)
        def request(h):
            try:
                identifier = selected(h)
                with scope(identifier):
                    path = urlparse(h.path).path
                    if path == '/project-scope.js':
                        return h.send_bytes((Path(__file__).resolve().parents[1] / 'project-scope.js').read_bytes(), 'text/javascript; charset=utf-8')
                    if path == '/project-folders':
                        if not h.require_summary_admin():
                            return
                        return h.send_summary_admin_bytes((Path(pipeline.ROOT) / 'project_folders.html').read_bytes(), 'text/html; charset=utf-8')
                    if path == '/api/projects' and not write:
                        return h.send_json({'projects':registry().list(), 'selected':identifier})
                    if path in ('/api/projects/create', '/api/projects/rename') and write:
                        if not h.require_summary_admin(write=True):
                            return
                        body = h.summary_admin_request_json()
                        if body is None:
                            return
                        result = (registry().create(body.get('name')) if path.endswith('/create')
                                  else registry().rename(identifier, body.get('name'), body.get('expected_revision')))
                        return h.send_summary_admin_json({'project':result}, 201 if path.endswith('/create') else 200)
                    # Every recording view/export/mutation is fenced before its
                    # actual existing handler. Voice/sample IDs have own paths.
                    if identifier != DEFAULT and path in ('/summary-download','/api/summary-test','/summary-test','/summary-test.html'):
                        return h.send_json({'error':'Архив недоступен в этом профиле'},404)
                    params = parse_qs(urlparse(h.path).query)
                    if not path.startswith('/api/profiles'):
                        job_id = params.get('job_id', params.get('id', ['']))[0]
                        if job_id:
                            if not job_id.isdecimal():
                                raise ValueError('Неверный номер записи')
                            with pipeline.connect() as db:
                                row = db.execute('SELECT project_id FROM jobs WHERE id=?', (int(job_id),)).fetchone()
                            if row is None or row[0] != identifier:
                                return h.send_json({'error':'Запись не найдена в этом профиле'},404)
                    if write and not native and os.environ.get('TRANSCRI_NATIVE_SPEECH_ROOT') and path in ('/api/apply-profiles','/api/speakers'):
                        if not h.require_summary_admin(write=True):
                            return
                        from summary.speech_bridge import request as request_speech
                        count=pipeline.normalize_speaker_count(params.get('count',['auto'])[0]) if path == '/api/speakers' else None
                        return h.send_json(request_speech(pipeline,int(job_id),'speakers' if path == '/api/speakers' else 'apply',count))
                    return original(h)
            except ValueError as exc:
                return h.send_json({'error':str(exc)},400)
            except (OSError, sqlite3.Error):
                return h.send_json({'error':'Профили проектов временно недоступны'},503)
        return request
    handler.do_GET = wrap(handler.do_GET, False)
    handler.do_POST = wrap(handler.do_POST, True)

    # Context is per thread and per job; a UI switch cannot mutate a running job.
    for name in ('process_job', 'process_summary'):
        if not hasattr(pipeline, name):
            continue
        original = getattr(pipeline, name)
        def wrap_job(original):
            @wraps(original)
            def run(job_id, *args, **kwargs):
                with pipeline.connect() as db:
                    row = db.execute('SELECT project_id FROM jobs WHERE id=?', (job_id,)).fetchone()
                identifier = row[0] if row else DEFAULT
                registry().require(identifier)
                with scope(identifier):
                    return original(job_id, *args, **kwargs)
            return run
        setattr(pipeline, name, wrap_job(original))
