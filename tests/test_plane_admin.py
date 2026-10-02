"""Plane endpoints use the real dashboard authorization and storage."""
from unittest import TestCase
from unittest.mock import patch
import json
from test_summary_credentials_admin import AdminHttpTests


class PlaneAdminTests(TestCase):
    setUp = AdminHttpTests.setUp
    tearDown = AdminHttpTests.tearDown
    request = AdminHttpTests.request

    def test_unauthorized_read_and_csrf_write_rejected(self):
        status, _, _ = self.request('GET', '/api/summary/plane/settings', auth=False)
        self.assertEqual(status,401)
        status, _, _ = self.request('POST','/api/summary/plane/settings', {'expected_revision':0},origin=False)
        self.assertEqual(status,403)
        status, _, _ = self.request('GET','/api/summary/plane/settings',host='attacker.example')
        self.assertEqual(status,403)

    def test_settings_key_mask_and_revision(self):
        secret = 'plane-test-only-secret-123456'
        body = {'expected_revision':0,'base_url':'https://plane.example.test', 'workspace_slug':'team','api_key':secret}
        with patch('pipeline.plane_scheduler_tick'):
            status, payload, _ = self.request('POST','/api/summary/plane/settings',body)
        self.assertEqual(status,200,payload)
        self.assertNotIn(secret.encode(),payload)
        data=json.loads(payload)
        self.assertEqual(data['settings']['revision'],1)
        status,payload,_=self.request('GET','/api/summary/plane/settings')
        self.assertEqual(status,200)
        self.assertNotIn(secret.encode(),payload)
        self.assertNotIn(b'encrypted_key',payload)
        with patch('pipeline.plane_scheduler_tick'):
            status,_,_=self.request('POST','/api/summary/plane/settings',body)
        self.assertEqual(status,409)

    def test_write_never_calls_inference_or_speech(self):
        with patch('pipeline.plane_scheduler_tick'), patch('pipeline.run_next_summary') as run, patch('pipeline.run_command') as cmd:
            status,_,_=self.request('POST','/api/summary/plane/settings',{'expected_revision':0,'auto_tasks':True,'auto_hypotheses':False})
        self.assertEqual(status,200)
        run.assert_not_called();cmd.assert_not_called()

    def test_routes_for_assets(self):
        for path in ['/plane-settings','/plane-settings.js','/plane-actions.js']:
            status,body,headers=self.request('GET',path)
            self.assertEqual(status,200)
            self.assertTrue(body)
            self.assertIn('no-store',headers['Cache-Control'])

    def test_scheduler_baselines_without_backfill_and_never_runs_inference(self):
        from unittest.mock import MagicMock
        import pipeline
        fake_store = MagicMock()
        fake_store.baseline_complete.return_value=False
        fake_store.known_generation.return_value=None
        fake_db = MagicMock()
        fake_db.execute.return_value.fetchall.return_value=[{'id':7,'output_dir':str(self.root)}]
        with patch('pipeline.plane_store',return_value=fake_store), patch('pipeline.connect',return_value=fake_db), \
             patch('pipeline.current_summary_generation_id',return_value='g1'), patch('pipeline.plane_selected') as observe, \
             patch('pipeline.run_command') as external:
            pipeline.plane_scheduler_tick(deliver=False)
        observe.assert_called_once_with(7,allow_auto=False)
        fake_store.mark_baseline_complete.assert_called_once()
        fake_store.drain_one.assert_not_called();external.assert_not_called()

    def test_summary_speech_functions_default_equivalent(self):
        import ast
        import subprocess
        from pathlib import Path
        import pipeline
        # Container storage root changes only path resolution in speech helpers.
        # No models, arguments, thresholds or algorithms change.
        try:
            original=subprocess.check_output(['git','show','762df04:pipeline.py'],cwd=pipeline.ROOT,stderr=subprocess.DEVNULL,text=True)
        except (OSError,subprocess.CalledProcessError):
            self.skipTest('Git baseline unavailable in source archive')
        old=ast.parse(original);new=ast.parse(Path(pipeline.__file__).read_text())
        class Normalize(ast.NodeTransformer):
            def visit_Name(self,node):
                if node.id=='DATA_ROOT': node.id='ROOT'
                return node
        expected={n.name:ast.dump(n,include_attributes=False) for n in old.body if isinstance(n,ast.FunctionDef)}
        for node in new.body:
            if isinstance(node,ast.FunctionDef) and node.name in ('voice_embedding_model_path','extract_redimnet_embeddings','process_job'):
                self.assertEqual(ast.dump(Normalize().visit(node),include_attributes=False),expected[node.name])
