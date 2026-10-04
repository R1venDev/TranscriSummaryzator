"""Namespace adapter for an existing native speech service; no model changes.

Native speech remains the single speech runner. Docker owns summary jobs and
credentials. Only explicitly submitted app jobs are bridged; no backfill.
"""
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import hashlib
import os
import sqlite3
import threading
from summary.project_profiles import CURRENT, DEFAULT, Projects, scope


class ScopedConnection:
    def __init__(self, db): self.db = db
    def __getattr__(self, name): return getattr(self.db, name)
    def __enter__(self): self.db.__enter__(); return self
    def __exit__(self,*args):
        try: return self.db.__exit__(*args)
        finally: self.db.close()
    def execute(self, sql, params=()):
        if 'INSERT INTO jobs' in sql and 'project_id' not in sql:
            before = 'created_at, updated_at)'
            after = "VALUES (?, ?, ?, ?, 'queued', 'queued', ?, ?, ?, ?)"
            if before not in sql or after not in sql:
                raise ValueError('Unsupported native enqueue contract')
            sql = sql.replace(before,'created_at, updated_at, project_id)').replace(after,"VALUES (?, ?, ?, ?, 'queued', 'queued', ?, ?, ?, ?, ?)")
            params = tuple(params) + (CURRENT.get(),)
        return self.db.execute(sql,params)


class VoiceRoot:
    def __init__(self, root):
        self.root = Path(root)
    def current(self):
        return self.root if CURRENT.get() == DEFAULT else self.root / 'projects' / CURRENT.get()
    def __truediv__(self, value):
        return self.current() / value
    def exists(self):
        return self.current().exists()
    def glob(self, pattern):
        return self.current().glob(pattern)


def install(pipeline, app_data):
    app_data = Path(app_data)
    projects = Projects(app_data / 'state')
    original_connect = pipeline.connect
    migration_lock = threading.Lock()
    def connect():
        with migration_lock:
            db = original_connect()
            columns = {row[1] for row in db.execute('PRAGMA table_info(jobs)')}
            if 'project_id' not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN project_id TEXT NOT NULL DEFAULT 'default'")
                db.commit()
            return ScopedConnection(db)
    pipeline.connect = connect
    pipeline.VOICE_PROFILES = VoiceRoot(pipeline.VOICE_PROFILES)
    fingerprint = pipeline.submission_fingerprint
    pipeline.submission_fingerprint = lambda sha, name: fingerprint(sha, name) if CURRENT.get() == DEFAULT else hashlib.sha256((fingerprint(sha,name)+'\0'+CURRENT.get()).encode()).hexdigest()
    def existing(db, sha, name, source_path=None):
        return db.execute('''SELECT * FROM jobs WHERE project_id=? AND
            (fingerprint=? OR (content_sha256=? AND original_name=?) OR
             (? IS NOT NULL AND content_sha256=? AND source_path=?)) ORDER BY id LIMIT 1''',
            (CURRENT.get(),pipeline.submission_fingerprint(sha,name),sha,Path(name).name,
             str(source_path) if source_path else None,sha,str(source_path))).fetchone()
    pipeline.find_existing_job = existing
    # Ownership is inserted in the same transaction through ScopedConnection.
    enqueue = pipeline.enqueue
    def project_enqueue(*a, **kw):
        # A watcher cannot re-enqueue another project's already-owned path.
        if not kw and len(a) == 1:
            with connect() as db:
                prior = db.execute("SELECT id FROM jobs WHERE source_path=?", (str(Path(a[0]).resolve()),)).fetchone()
            if prior is not None:
                return prior[0]
        projects.require(CURRENT.get())
        return enqueue(*a, **kw)
    pipeline.enqueue = project_enqueue
    config = pipeline.config
    # Summary belongs to the existing Docker scheduler, never legacy inference.
    pipeline.config = lambda: {**config(), 'summary_enabled':False, 'open_dashboard_on_job':False}
    original_rows = pipeline.status_rows
    pipeline.status_rows = lambda db: [row for row in original_rows(db) if db.execute('SELECT project_id FROM jobs WHERE id=?',(row['id'],)).fetchone()[0] == CURRENT.get()]
    scan = pipeline.scan_inbox
    def scan_with_app(seen, cfg):
        bridge_dispatch(pipeline, app_data)
        return scan(seen, cfg)
    pipeline.scan_inbox = scan_with_app
    from summary.project_http import install as install_http
    install_http(pipeline, registry_state=app_data / 'state', native=True)


def bridge_dispatch(pipeline, app_data):
    """Restart-safe enqueue of app-owned uploads into the existing speech queue."""
    db = sqlite3.connect(Path(app_data) / 'state/queue.sqlite3', timeout=15)
    db.row_factory = sqlite3.Row
    try:
        bridge_operations(pipeline, db)
        rows = db.execute("SELECT * FROM jobs WHERE status='queued' AND native_job_id IS NULL ORDER BY id").fetchall()
        for row in rows:
            Projects(Path(app_data)/'state').require(row['project_id'])
            relative = Path(row['source_path']).relative_to('/data')
            source = Path(app_data) / relative
            if source.resolve().parent != (Path(app_data)/'inbox').resolve() or not source.is_file():
                continue
            with scope(row['project_id']):
                # Recover an INSERT whose app mapping was not saved after crash.
                with pipeline.connect() as speech:
                    old = speech.execute('SELECT id,project_id FROM jobs WHERE source_path=? AND content_sha256=?', (str(source.resolve()),row['content_sha256'])).fetchone()
                if old:
                    identifier = old['id']
                    if old['project_id'] != row['project_id']:
                        raise ValueError('Speech bridge ownership mismatch')
                else:
                    identifier = pipeline.enqueue(source,known_fingerprint=row['content_sha256'],speaker_count=row['speaker_count'],original_name=row['original_name'])
            db.execute("UPDATE jobs SET native_job_id=?,stage='speech_queued',detail='Ожидает обработки речи' WHERE id=? AND native_job_id IS NULL", (identifier,row['id']))
            db.commit()
    finally:
        db.close()


def bridge_operations(pipeline, db):
    """Native receipt fences side effects; uncertain work is never replayed."""
    import json
    with pipeline.connect() as native:
        native.execute("CREATE TABLE IF NOT EXISTS project_operation_receipts (id TEXT PRIMARY KEY, job_id INTEGER NOT NULL, project_id TEXT NOT NULL, status TEXT NOT NULL)")
    for row in db.execute('SELECT * FROM jobs WHERE native_operation IS NOT NULL AND native_job_id IS NOT NULL ORDER BY id').fetchall():
        operation={'status':'failed'}
        try:
            operation=json.loads(row['native_operation'])
            if operation['status'] != 'requested':
                continue
            with scope(row['project_id']), pipeline.connect() as speech:
                job=speech.execute('SELECT * FROM jobs WHERE id=?',(row['native_job_id'],)).fetchone()
                if not job or job['project_id'] != row['project_id'] or job['content_sha256'] != row['content_sha256']:
                    raise ValueError('native_operation_identity_mismatch')
                receipt=speech.execute('SELECT * FROM project_operation_receipts WHERE id=?',(operation['id'],)).fetchone()
                if receipt:
                    if receipt['job_id'] != job['id'] or receipt['project_id'] != row['project_id']:
                        raise ValueError('native_receipt_identity_mismatch')
                    operation['status']='applied' if receipt['status']=='applied' else 'unknown'
                elif job['status'] == 'running':
                    continue
                else:
                    speech.execute('INSERT INTO project_operation_receipts VALUES(?,?,?,?)',(operation['id'],job['id'],row['project_id'],'started'))
                    speech.commit()  # durable before an external/local speech side effect
                    if operation['kind'] == 'speakers':
                        job_dir=Path(job['job_dir'])
                        for name in ('diarization.json','diarization.rttm'):
                            (job_dir/name).unlink(missing_ok=True)
                        pipeline.update_job(speech,job['id'],speaker_count=operation['speaker_count'],status='queued',stage='queued',progress=12,detail='Изменено количество участников',error=None,started_at=None,finished_at=None,summary_status='waiting')
                    elif operation['kind'] == 'apply':
                        if job['status'] != 'done':
                            raise ValueError('native_operation_source_not_ready')
                        job_dir=Path(job['job_dir'])
                        report=pipeline.identify_speakers(Path(job['output_dir']),pipeline.config(),job_dir/'processing.log',audio=job_dir/'audio.wav',asr=pipeline.load_json(job_dir/'asr.json'),diarization=pipeline.consensus_diarization(job_dir,pipeline.load_json(job_dir/'diarization.json')),cache_dir=job_dir,consensus=pipeline.load_json(job_dir/'consensus.json').get('intervals',[]) if (job_dir/'consensus.json').exists() else None)
                        pipeline.rename_export(job['id'],report.get('diarization'))
                    else:
                        raise ValueError('native_operation_unknown')
                    speech.execute("UPDATE project_operation_receipts SET status='applied' WHERE id=?",(operation['id'],))
                    speech.commit()
                    operation['status']='applied'
            db.execute("UPDATE jobs SET native_operation=?,status='queued',stage='speech_queued',speaker_count=COALESCE(?,speaker_count) WHERE id=? AND native_operation=?",(json.dumps(operation),operation.get('speaker_count'),row['id'],row['native_operation']))
        except Exception as exc:
            # Unknown started effects require an operator decision, never a poll retry.
            operation.update(status='unknown',error=type(exc).__name__)
            db.execute("UPDATE jobs SET native_operation=?,detail='Изменение голосов требует восстановления' WHERE id=? AND native_operation=?",(json.dumps(operation),row['id'],row['native_operation']))
        db.commit()
