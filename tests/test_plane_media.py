"""Real outbox/native node paths, synthetic media and credentials only."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import uuid

from cryptography.fernet import Fernet
from summary.plane import PlaneClient, PlaneError, PlaneStore, _external_id
from summary.plane_media import media_spec, native_blocks, enqueue, drain, upload_file


class MediaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'video.mp4'
        self.source.write_bytes(b'synthetic-video')
        self.spec = {'path': str(self.source), 'sha256': hashlib.sha256(self.source.read_bytes()).hexdigest(),
                     'name': 'video <test>.mkv', 'source_sha256': 'source-1',
                     'chapters': [{'seconds': 3, 'label': '<script>danger</script>', 'start_id': 'U00001', 'end_id': 'U00002'}]}
        self.remote = str(uuid.uuid4())
        self.body = '<p>Manual edit</p><p>transcrisummaryzator-id:' + _external_id(1, 'source-1', 'page', 'meeting') + '</p>'
        self.ids, self.metadata, self.calls = {}, {}, []
        self.fail_put = False
        self.fail_allocate = False
        self.store = PlaneStore(self.root / 'private' / 'plane.sqlite3', Fernet.generate_key(), lambda *a: self)
        self.store.save_settings({'expected_revision': 0, 'base_url': 'https://plane.test', 'workspace_slug': 'test',
                                 'api_key': 'synthetic-plane-api-secret'})
        self.store.observe_generation(1, 'g1', [], {'title': 'Meeting', 'description_html': '<p>Text</p>'}, 'source-1')
        self.store.enqueue(1, 'page', 'meeting', 'g1')
        with self.store._db() as db:
            db.execute("UPDATE deliveries SET state='created',remote_id=?", (self.remote,))
        self.prefix = '/api/v1/workspaces/test/'
        self.base = 'https://plane.test'
        self.key = 'synthetic-plane-api-secret'
    def tearDown(self):
        self.temp.cleanup()
    def request(self, method, path, body=None, allowed_statuses=()):
        self.calls.append((method, path, body))
        if path.endswith('/assets/'):
            key = body['external_id']
            exists = key in self.ids
            asset_id = self.ids.setdefault(key, str(uuid.uuid4()))
            self.metadata.setdefault(asset_id, {'is_uploaded': False})
            if self.fail_allocate:
                self.fail_allocate = False
                raise PlaneError('Не удалось получить достоверный ответ Plane')
            return {'asset_id': asset_id, **({} if exists else {'upload_data': {'url': self.base + '/storage/', 'fields': {'credential': 'synthetic-presigned-secret'}}})}
        if '/attachments/' in path:
            return self.metadata[path.rstrip('/').split('/')[-1]]
        if '/assets/' in path:
            self.metadata[path.rstrip('/').split('/')[-1]]['is_uploaded'] = True
            return {}
        if method == 'PUT':
            self.body = body['description_html']
            if self.fail_put:
                self.fail_put = False
                raise PlaneError('Не удалось получить достоверный ответ Plane')
        return {'description_html': self.body}
    def prepared(self, *a):
        return [{'role': 'preview', 'path': str(self.source), 'name': self.spec['name'], 'type': 'video/mp4',
                 'sha256': self.spec['sha256'], 'size': self.source.stat().st_size}]
    def run_media(self):
        with patch('summary.plane_media.prepare', side_effect=self.prepared), patch('summary.plane_media.upload_file') as upload:
            result = drain(self.store)
            return result, upload.call_count
    def list_all(self, path):
        return [{'id': str(uuid.uuid5(uuid.NAMESPACE_URL, 'user')), 'display_name': 'User'}]
    def test_layout_migration_reuses_assets_full_source_and_lost_put(self):
        from summary.plane_media import refresh_layout
        enqueue(self.store, 1, 'g1', self.spec)
        self.assertEqual(self.run_media()[0]['state'], 'media_published')
        with self.store._db() as db:
            row = db.execute('SELECT id,payload FROM deliveries').fetchone()
            payload = json.loads(row['payload']); payload['description_html'] = '<h1>Meeting</h1>' + payload['description_html']
            db.execute('UPDATE deliveries SET payload=? WHERE id=?', (json.dumps(payload), row['id']))
        self.body += '<h1>Meeting</h1><p><strong>Участники по транскрипции:</strong> @User</p><h2>Таймкоды</h2><ul><li>Duplicate chapter</li></ul>'
        index = {'source_sha256': 'source-1', 'participants': ['@User'], 'by_id': {'U00001': {'start_ms': 3, 'speaker': '@User', 'text': 'Exact original text'}}}
        self.fail_put = True
        with self.assertRaises(PlaneError):
            refresh_layout(self.store, 1, index, 'https://app.test/result?id=1')
        put_count = sum(m == 'PUT' for m, _, _ in self.calls)
        self.assertEqual(refresh_layout(self.store, 1, index, 'https://app.test/result?id=1')['state'], 'layout_current')
        self.assertEqual(sum(m == 'PUT' for m, _, _ in self.calls), put_count)
        self.assertEqual(len(self.ids), 1)
        self.assertEqual(self.body.count('<attachment-component'), 1)
        self.assertIn('Exact original text', self.body)
        self.assertIn('<p>Manual edit</p>', self.body)
        self.assertIn('<mention-component', self.body)
        self.assertNotIn('Duplicate chapter', self.body)
        self.assertNotIn('<h1>', self.body)
    def test_layout_migration_denies_wrong_source_before_remote_write(self):
        from summary.plane_media import refresh_layout
        enqueue(self.store, 1, 'g1', self.spec)
        with self.assertRaises(PlaneError):
            refresh_layout(self.store, 1, {'source_sha256': 'other'}, 'https://app.test')
        self.assertEqual(self.calls, [])
    def test_native_blocks_and_verified_chapters_escape_text(self):
        asset = {**self.prepared()[0], 'asset_id': str(uuid.uuid4())}
        rendered = native_blocks('identity', [asset], self.spec['chapters'])
        self.assertIn('<attachment-component', rendered)
        self.assertIn('data-block-type="callout-component"', rendered)
        self.assertIn('plane-timecodes-identity', rendered)
        self.assertIn('00:03 &lt;script&gt;', rendered)
        self.assertNotIn('<script>', rendered)
        self.assertNotIn(str(self.source), rendered)
    def test_durable_upload_append_manual_content_restart_no_duplicates(self):
        enqueue(self.store, 1, 'g1', self.spec)
        result, uploads = self.run_media()
        self.assertEqual(result['state'], 'media_published')
        self.assertEqual(uploads, 1)
        self.assertIn('<p>Manual edit</p>', self.body)
        before = self.body
        enqueue(self.store, 1, 'g1', self.spec)
        result, uploads = self.run_media()
        self.assertIsNone(result)
        self.assertEqual(uploads, 0)
        self.assertEqual(before, self.body)
        public = json.dumps(self.store.items(1))
        self.assertNotIn('synthetic-presigned-secret', public)
        self.assertNotIn(str(self.source), public)
        self.assertEqual(self.store.items(1)['meeting_page']['media_state'], 'published')
        self.assertNotIn(b'synthetic-presigned-secret', self.store.path.read_bytes())
    def test_lost_put_is_recovered_without_second_put_or_upload(self):
        enqueue(self.store, 1, 'g1', self.spec)
        self.fail_put = True
        self.assertEqual(self.run_media()[0]['state'], 'media_pending')
        with self.store._db() as db:
            db.execute('UPDATE page_media SET next_attempt=0')
        self.assertEqual(self.run_media()[0]['state'], 'media_published')
        self.assertEqual(sum(m == 'PUT' for m, _, _ in self.calls), 1)
        self.assertEqual(len(self.ids), 1)
    def test_unknown_allocation_recovers_identity_and_blocks_missing_upload_url(self):
        enqueue(self.store, 1, 'g1', self.spec)
        self.fail_allocate = True
        self.assertEqual(self.run_media()[0]['state'], 'media_pending')
        with self.store._db() as db:
            db.execute('UPDATE page_media SET next_attempt=0')
        self.assertEqual(self.run_media()[0]['state'], 'media_blocked')
        self.assertEqual(len(self.ids), 1)
        self.assertNotIn('<attachment-component', self.body)
    def test_wrong_page_marker_and_destination_never_upload(self):
        enqueue(self.store, 1, 'g1', self.spec)
        self.body = '<p>Another meeting</p>'
        result, uploads = self.run_media()
        self.assertEqual(result['state'], 'media_blocked')
        self.assertEqual(uploads, 0)
        self.assertEqual(len(self.ids), 0)
    def test_source_binding_no_filename_guess_and_source_identity(self):
        view = SimpleNamespace(source_sha256='source-1', rendered={'summary.json': {'timecodes': [
            {'start_id': 'U00001', 'end_id': 'U00002', 'topic': 'Topic'}]}})
        index = {'by_id': {'U00001': {'start_ms': 3050}, 'U00002': {'end_ms': 10000}}}
        job = {'source_path': str(self.root / 'transcript.json'), 'content_sha256': 'f' * 64, 'original_name': 'Meeting.mkv'}
        self.assertIsNone(media_spec(job, self.root, view, index, self.root))
        (self.root / 'plane_media_source.json').write_text(json.dumps({'source_sha256': 'source-1', 'path': str(self.source), 'media_sha256': self.spec['sha256']}))
        spec = media_spec(job, self.root, view, index, self.root)
        self.assertEqual(spec['chapters'][0]['seconds'], 3)
        view.source_sha256 = 'different-source'
        with self.assertRaises(ValueError):
            media_spec(job, self.root, view, index, self.root)
    def test_stream_upload_denies_unapproved_origin_before_connection(self):
        with patch('http.client.HTTPSConnection') as connection:
            with self.assertRaises(PlaneError):
                upload_file(self, {'url': 'https://untrusted.test/upload', 'fields': {}}, self.source, 'video/mp4')
            connection.assert_not_called()
    def test_real_multipart_stream_has_no_plane_key_and_no_redirect(self):
        connection = unittest.mock.MagicMock()
        connection.getresponse.return_value.status = 302
        with patch('http.client.HTTPSConnection', return_value=connection):
            with self.assertRaises(PlaneError):
                upload_file(self, {'url': self.base + '/storage/', 'fields': {'policy': 'synthetic'}}, self.source, 'video/mp4')
        transmitted = repr(connection.mock_calls)
        self.assertNotIn(self.key, transmitted)
        self.assertNotIn('X-API-Key', transmitted)
        self.assertIn('synthetic-video', transmitted)

    def test_explicit_minio_https_proxy_mapping_preserves_path_and_fields(self):
        connection = unittest.mock.MagicMock()
        connection.getresponse.return_value.status = 204
        with patch.dict('os.environ', {'TRANSCRI_PLANE_STORAGE_ORIGIN_MAP': json.dumps({'http://minio:9000': self.base})}), patch('http.client.HTTPSConnection', return_value=connection) as create:
            upload_file(self, {'url': 'http://minio:9000/uploads', 'fields': {'policy': 'signed-policy'}}, self.source, 'video/mp4')
        self.assertEqual(create.call_args.args[0], 'plane.test')
        connection.putrequest.assert_called_once_with('POST', '/uploads')
        self.assertIn('signed-policy', repr(connection.mock_calls))
        self.assertNotIn(self.key, repr(connection.mock_calls))
    def test_two_workers_and_disabled_wiki_do_not_upload(self):
        enqueue(self.store, 1, 'g1', self.spec)
        self.store.save_settings({'expected_revision': 1, 'auto_meeting_page': False})
        self.assertIsNone(self.run_media()[0])
        self.assertEqual(self.calls, [])

    def test_changed_spec_never_relabels_allocated_assets(self):
        enqueue(self.store, 1, 'g1', self.spec)
        changed = {**self.spec, 'sha256': 'f' * 64}
        enqueue(self.store, 1, 'g1', changed)
        with self.store._db() as db:
            stored = json.loads(db.execute('SELECT spec FROM page_media').fetchone()[0])
        self.assertEqual(stored['sha256'], self.spec['sha256'])

    def test_stale_destination_does_not_starve_current_page(self):
        enqueue(self.store, 1, 'g1', self.spec)
        self.store.save_settings({'expected_revision': 1, 'workspace_slug': 'other'})
        self.store.enqueue(1, 'page', 'meeting', 'g1')
        with self.store._db() as db:
            db.execute("UPDATE deliveries SET state='created',remote_id=?", (self.remote,))
        enqueue(self.store, 1, 'g1', self.spec)
        self.assertEqual(self.run_media()[0]['state'], 'media_published')
        with self.store._db() as db:
            self.assertEqual(sorted(r[0] for r in db.execute('SELECT state FROM page_media')), ['published', 'queued'])

    def test_missing_source_record_does_not_change_task_or_page_state(self):
        from summary.plane_media import unavailable
        unavailable(self.store, 1, 'source-1')
        self.assertEqual(self.store.items(1)['meeting_page']['state'], 'created')
        enqueue(self.store, 1, 'g1', self.spec)
        self.assertEqual(self.run_media()[0]['state'], 'media_published')

if __name__ == '__main__':
    unittest.main()
