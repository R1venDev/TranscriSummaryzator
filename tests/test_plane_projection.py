"""Plane exports share accepted task overrides and preserve source evidence."""
import unittest
from types import SimpleNamespace
from summary.plane_projection import project_view, add_action_controls
from summary.luna_v1.render import render_document, HEADINGS
from test_luna_tasks import document, task, INDEX


class ProjectionTests(unittest.TestCase):
    def view(self):
        doc = document([task('Проверить X или Y', 'Сделать один из двух вариантов, не оба.', 'U00001')])
        doc['ideas'] = [{'text':'Возможно, тест улучшит результат <script>alert(1)</script>', 'source_ids':['U00002']}]
        doc['chapters'] = []
        exports = render_document(doc, INDEX)
        return SimpleNamespace(rendered=exports, tasks=exports['tasks.json'], generation_id='g', source_sha256=INDEX['source_sha256'])

    def test_effective_content_optional_actor_hypothesis_and_all_sections(self):
        view = self.view()
        view.tasks[0]['description'] = 'Ручное уточнение: проверить только X.'
        items, page = project_view(view, 7, 'https://example.test', ['H-safe'])
        self.assertIn('Ручное уточнение', items[0]['description_html'])
        self.assertIn('Не назначен', items[0]['description_html'])
        self.assertIn('Не принятое обязательство', items[1]['description_html'])
        self.assertIn('&lt;script&gt;', items[1]['description_html'])
        self.assertNotIn('<script>', page['description_html'])
        self.assertIn('https://example.test/result?id=7#u-U00001', page['description_html'])
        for heading in HEADINGS:
            self.assertIn(heading, page['description_html'])
        self.assertEqual(items[0]['item_id'], view.tasks[0]['action_id'])

    def test_controls_only_live_fragment_and_no_mutation(self):
        view = self.view()
        before = view.rendered['summary.fragment.html']
        changed = add_action_controls(before, view.tasks, ['H-safe'])
        self.assertEqual(changed.count('class="plane-action"'), 2)
        self.assertNotIn('plane-action', before)
        self.assertIn('data-plane-item-id="H-safe"', changed)

    def test_origin_rejects_credential_url(self):
        with self.assertRaises(ValueError):
            project_view(self.view(), 7, 'https://key@example.test', ['H-safe'])
