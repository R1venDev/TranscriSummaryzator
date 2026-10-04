"""Offline checks on real namespace, HTTP, ledger, and store paths."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import threading
import unittest
from unittest.mock import patch
from urllib.request import urlopen, Request
from urllib.error import HTTPError
from http.server import ThreadingHTTPServer
from cryptography.fernet import Fernet
import pipeline
from summary.project_profiles import Projects, CURRENT, DEFAULT, scope, private_path
from summary.luna_v1.ledger import Ledger
from summary_credentials import CredentialStore
from summary.plane import PlaneStore
from scripts.project_speech_adapter import ScopedConnection, VoiceRoot


class ProjectIsolationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.state=self.root/'state'; self.state.mkdir()
        values={'STATE':self.state,'DB_PATH':self.state/'queue.sqlite3','DATA_ROOT':self.root,
                'INBOX':self.root/'inbox','OUTPUTS':self.root/'outputs','JOBS':self.root/'work/jobs',
                'VOICE_PROFILES':self.root/'voice_profiles','SUMMARY_CREDENTIAL_DB':self.state/'summary_private/credentials.sqlite3'}
        for key,value in values.items():
            p=patch.object(pipeline,key,value);p.start();self.addCleanup(p.stop)
        self.env=patch.dict(os.environ,{'TRANSCRI_SUMMARY_MASTER_KEY':Fernet.generate_key().decode()})
        self.env.start();self.addCleanup(self.env.stop)
        unset=patch.dict(os.environ);unset.start();self.addCleanup(unset.stop);os.environ.pop('TRANSCRI_SUMMARY_MASTER_KEY_FILE',None)
        c=patch.object(pipeline,'config',return_value={'processing_enabled':False,'open_dashboard_on_job':False});c.start();self.addCleanup(c.stop)
        self.projects=Projects(self.state);self.other=self.projects.create('ExLand')['id']
        with pipeline.connect(): pass

    def job(self, identifier, number):
        with pipeline.connect() as db:
            cursor=db.execute('''INSERT INTO jobs(fingerprint,source_path,original_name,status,stage,job_dir,created_at,updated_at,project_id)
              VALUES(?,?,?,'done','done',?,?,?,?)''', (str(number)*64,'/unused',str(number),'/unused',pipeline.now(),pipeline.now(),identifier))
            return cursor.lastrowid

    def test_existing_default_migration_and_rename_cas(self):
        self.assertEqual(self.projects.require(DEFAULT)['name'],'Aurion')
        self.projects.rename(self.other,'ExLand 2',1)
        with self.assertRaises(ValueError): self.projects.rename(self.other,'Lost edit',1)
        self.assertEqual(Projects(self.state).require(self.other)['name'],'ExLand 2')
        with self.assertRaises(ValueError): self.projects.create('AURION')
        with self.assertRaises(ValueError): self.projects.require('../inbox')

    def test_credential_and_plane_stores_are_separate_no_inheritance(self):
        default=CredentialStore(private_path(self.state/'summary_private','credentials.sqlite3'))
        key=default.add('Primary','sk-or-v1-'+'a'*40)
        other=CredentialStore(private_path(self.state/'summary_private','credentials.sqlite3',self.other))
        self.assertEqual(other.list()['keys'],[])
        other.add('Primary','sk-or-v1-'+'b'*40)
        self.assertNotEqual(default.list()['keys'][0]['id'],other.list()['keys'][0]['id'])
        with self.assertRaises(ValueError): other.reveal_for_existing_job(key['id'],1)
        a=PlaneStore(private_path(self.state/'summary_private','plane.sqlite3'))
        b=PlaneStore(private_path(self.state/'summary_private','plane.sqlite3',self.other))
        a.save_settings({'expected_revision':a.public_settings()['settings']['revision'],'workspace_slug':'aurion'})
        self.assertEqual(b.public_settings()['settings']['workspace_slug'],'')
        self.assertEqual(a.public_settings()['settings']['workspace_slug'],'aurion')

    def test_voice_scope_and_thread_context_cannot_leak(self):
        root=VoiceRoot(self.root/'voice_profiles')
        old=root/'a'; nested=[]
        with scope(self.other):
            self.assertEqual(root/'a',self.root/'voice_profiles/projects'/self.other/'a')
            t=threading.Thread(target=lambda:nested.append(CURRENT.get()));t.start();t.join()
        self.assertEqual(root/'a',old);self.assertEqual(nested,[DEFAULT])
        self.assertEqual(CURRENT.get(),DEFAULT)

    def test_dedup_and_status_fenced_before_history_limit(self):
        a=self.job(DEFAULT,1);b=self.job(self.other,2)
        with scope(self.other):
            self.assertEqual([r['id'] for r in pipeline.status_rows(pipeline.connect())],[b])
            foreign_fp=pipeline.submission_fingerprint('a'*64,'meeting.mkv')
        self.assertNotEqual(foreign_fp,pipeline.submission_fingerprint('a'*64,'meeting.mkv'))
        with pipeline.connect() as db:
            self.assertEqual(db.execute('SELECT project_id FROM jobs WHERE id=?',(a,)).fetchone()[0],DEFAULT)

    def test_http_cross_profile_views_and_mutations_are_rejected(self):
        foreign=self.job(self.other,3)
        server=ThreadingHTTPServer(('127.0.0.1',0),pipeline.DashboardHandler)
        t=threading.Thread(target=server.serve_forever,daemon=True);t.start()
        self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
        origin='http://127.0.0.1:'+str(server.server_port)
        for path in ['/download?id='+str(foreign)+'&file=transcript.json','/api/summary?id='+str(foreign),'/summary-tasks?id='+str(foreign)]:
            with self.assertRaises(HTTPError) as caught:
                urlopen(Request(origin+path,method='POST' if path.startswith('/api/summary?') else 'GET'))
            self.assertEqual(caught.exception.code,404)
        with scope(self.other): pipeline.dashboard_payload()
        with urlopen(origin+'/api/status?project='+self.other) as r:
            self.assertEqual([j['id'] for j in json.load(r)['jobs']],[foreign])
        with self.assertRaises(HTTPError) as caught: urlopen(origin+'/api/projects?project='+'c'*32)
        self.assertEqual(caught.exception.code,400)

    def test_html_profile_binds_before_inline_fetch(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),pipeline.DashboardHandler)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
        with urlopen('http://127.0.0.1:'+str(server.server_port)+'/?project='+self.other) as r: body=r.read().decode()
        self.assertIn('content="'+self.other+'"',body)
        self.assertLess(body.index('/project-scope.js'),body.index('</head>'))
        self.assertNotIn('/project-scope.js" defer',body)

    def test_global_ledger_persists_project_and_shares_weekly_budget(self):
        ledger=Ledger(self.state/'summary_private');self.addCleanup(ledger.close)
        ids=[]
        for n,project in [(1,DEFAULT),(2,self.other)]:
            raw=json.dumps({'planned_capacity_microusd':1000,'capacity_basis':{'offline':True}}).encode();f=self.root/('m'+str(n));f.write_bytes(raw)
            d=ledger.create_source_first_job(semantic_key=str(n)*64,source_sha256='f'*64,output_dir=self.root/str(n),manifest_path=f,manifest_sha256=hashlib.sha256(raw).hexdigest(),credential_id='key'+str(n),credential_version=1,workspace_id='workspace',project_id=project)
            ids.append(d.job_id)
            self.assertTrue(ledger.reserve_source_first_plan(d.job_id,plan_sha256=hashlib.sha256(raw).hexdigest(),reserve_microusd=1000))
        self.assertEqual(ledger.get_batch_workflow(ids[1])['project_id'],self.other)
        # Same ledger identity and shared accumulator, no per-project budget reset.
        self.assertEqual(ledger._rolling_spent_microusd(time.time()),2000)

    def test_native_ownership_written_with_insert(self):
        db=pipeline.connect();wrapped=ScopedConnection(db)
        with scope(self.other):
            cursor=wrapped.execute("""INSERT INTO jobs
            (fingerprint, content_sha256, source_path, original_name, status, stage, job_dir, speaker_count, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'queued', 'queued', ?, ?, ?, ?)""", ('b'*64,'b'*64,'/unused','b','/unused',None,pipeline.now(),pipeline.now()))
        self.assertEqual(db.execute('SELECT project_id FROM jobs WHERE id=?',(cursor.lastrowid,)).fetchone()[0],self.other)
        db.close()


class ProjectWatcherTests(unittest.TestCase):
    setUp = ProjectIsolationTests.setUp
    job = ProjectIsolationTests.job
    def test_watcher_does_not_duplicate_owned_upload(self):
        pipeline.INBOX.mkdir()
        media=pipeline.INBOX/'meeting.wav';media.write_bytes(b'synthetic audio bytes only')
        with scope(self.other):
            original=pipeline.enqueue(media,known_fingerprint='d'*64,original_name='meeting.wav')
        seen={};cfg={'stable_seconds':0}
        pipeline.scan_inbox(seen,cfg);pipeline.scan_inbox(seen,cfg)
        with pipeline.connect() as db:
            rows=db.execute('SELECT id,project_id FROM jobs WHERE source_path=?',(str(media.resolve()),)).fetchall()
        self.assertEqual([(r[0],r[1]) for r in rows],[(original,self.other)])

    def test_watcher_recovers_profile_from_published_upload_metadata(self):
        pipeline.INBOX.mkdir();media=pipeline.INBOX/'meeting.wav';media.write_bytes(b'synthetic')
        (pipeline.INBOX/'.web-upload-1.upload.json').write_text(json.dumps({'storage_key':media.name,'project_id':self.other}))
        seen={};pipeline.scan_inbox(seen,{'stable_seconds':0});pipeline.scan_inbox(seen,{'stable_seconds':0})
        with pipeline.connect() as db:
            row=db.execute('SELECT project_id FROM jobs WHERE source_path=?',(str(media.resolve()),)).fetchone()
        self.assertEqual(row[0],self.other)

    def test_speech_failure_does_not_skip_summary_tick(self):
        from unittest.mock import Mock
        with patch('summary.speech_bridge.sync',side_effect=ValueError('bad export')),patch.object(pipeline,'luna_scheduler_tick') as luna,patch.object(pipeline,'plane_scheduler_tick'):
            pipeline.summary_scheduler(once=True)
        luna.assert_called_once()

    def test_delete_transaction_rejects_json_id_from_other_project(self):
        import record_dashboard_wrapper as records
        job=self.job(self.other,5)
        from types import SimpleNamespace
        facade=SimpleNamespace(ROOT=self.root,DB_PATH=pipeline.DB_PATH)
        with self.assertRaises(records.DeleteError) as caught:
            records.delete_record(facade,job,'5',pipeline.now())
        self.assertEqual(caught.exception.status,404)
        with pipeline.connect() as db: self.assertIsNotNone(db.execute('SELECT id FROM jobs WHERE id=?',(job,)).fetchone())

class ProjectOperationTests(unittest.TestCase):
    setUp = ProjectIsolationTests.setUp
    job = ProjectIsolationTests.job

    def ready(self):
        job=self.job(DEFAULT,8)
        with pipeline.connect() as db:
            pipeline.update_job(db,job,native_job_id=job,output_dir=str(self.root/'out'),summary_status='done')
        return job

    def test_source_change_fences_summary_scheduler(self):
        from summary.speech_bridge import request
        job=self.ready()
        with patch.object(pipeline,'_luna_output_has_active_batch',return_value=False): request(pipeline,job,'speakers',3)
        with pipeline.connect() as db:
            pipeline.update_job(db,job,summary_status='queued_force')
        with patch.object(pipeline,'process_summary') as run:
            self.assertFalse(pipeline.run_next_summary());run.assert_not_called()

    def test_credentials_pending_and_active_ledger_prevent_source_change(self):
        from summary.speech_bridge import request
        job=self.ready()
        with pipeline.connect() as db: pipeline.update_job(db,job,summary_status='credential_required')
        with self.assertRaises(ValueError): request(pipeline,job,'apply')
        with pipeline.connect() as db: pipeline.update_job(db,job,summary_status='done')
        with patch.object(pipeline,'_luna_output_has_active_batch',return_value=True):
            with self.assertRaises(ValueError): request(pipeline,job,'apply')

    def test_native_started_receipt_never_replays_and_runtime_failure_isolated(self):
        from scripts.project_speech_adapter import bridge_operations
        job=self.ready();operation={'id':'one-operation','kind':'apply','status':'requested','speaker_count':None}
        with pipeline.connect() as db:
            pipeline.update_job(db,job,native_operation=json.dumps(operation),content_sha256='a'*64)
            db.execute('CREATE TABLE project_operation_receipts (id TEXT PRIMARY KEY,job_id INTEGER,project_id TEXT,status TEXT)')
            db.execute('INSERT INTO project_operation_receipts VALUES(?,?,?,?)',(operation['id'],job,DEFAULT,'started'))
        with pipeline.connect() as db,patch.object(pipeline,'identify_speakers') as identify:
            bridge_operations(pipeline,db);bridge_operations(pipeline,db);identify.assert_not_called()
        with pipeline.connect() as db:
            row=db.execute('SELECT native_operation FROM jobs WHERE id=?',(job,)).fetchone()
            self.assertEqual(json.loads(row[0])['status'],'unknown')
            db.execute('DELETE FROM project_operation_receipts');pipeline.update_job(db,job,native_operation=json.dumps(operation),status='done')
        with pipeline.connect() as db,patch.object(pipeline,'load_json',return_value={}),patch.object(pipeline,'consensus_diarization',return_value={}),patch.object(pipeline,'identify_speakers',side_effect=RuntimeError('synthetic worker exit')) as identify:
            bridge_operations(pipeline,db);bridge_operations(pipeline,db);self.assertEqual(identify.call_count,1)

    def test_import_waits_for_native_operation_receipt(self):
        from summary.speech_bridge import _sync_row
        job=self.ready()
        with pipeline.connect() as db:
            pipeline.update_job(db,job,native_operation=json.dumps({'status':'requested'}))
            row=db.execute('SELECT * FROM jobs WHERE id=?',(job,)).fetchone()
            from unittest.mock import Mock
            remote=Mock();_sync_row(pipeline,db,remote,self.root,row);remote.execute.assert_not_called()
