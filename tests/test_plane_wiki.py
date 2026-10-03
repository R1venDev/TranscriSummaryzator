import unittest
import uuid
from summary.plane_wiki import *
from summary.plane import safe_html
from summary.plane_media import native_blocks
from summary.plane_projection import project_view
from test_plane_projection import ProjectionTests
from test_luna_tasks import INDEX


class WikiTests(unittest.TestCase):
    def test_native_source_and_no_summary_duplicates(self):
        view = ProjectionTests().view()
        _, page = project_view(view, 1, 'https://app.test', ['H1'], INDEX)
        body = safe_html(page['description_html'])
        self.assertNotIn('<h1>', body)
        self.assertEqual(body.count(PARTICIPANTS), 1)
        self.assertNotIn('<h2>Таймкоды</h2>', body)
        self.assertIn('<details class="editor-details-block"', body)
        self.assertNotIn(' open', body)
        for uid, unit in INDEX['by_id'].items():
            self.assertIn(unit['text'], body)
            self.assertIn('#u-' + uid, body)
        self.assertIn('data-type="detailsContent"', body)
    def test_unique_exact_real_mentions_ambiguous_and_unknown_plain(self):
        a, b = str(uuid.uuid4()), str(uuid.uuid4())
        value = participant_header({'participants': ['@User', 'Missing', 'Duplicate'], 'unattributed_speech': True})
        result = resolve_mentions(value, [{'id': a, 'display_name': 'user'}, {'id': a, 'display_name': 'Duplicate'}, {'id': b, 'display_name': 'duplicate'}])
        self.assertEqual(result.count('<mention-component'), 1)
        self.assertIn('entity_identifier="' + a, result)
        self.assertIn('Missing, Duplicate, Участник не определён', result)
        self.assertIn('<mention-component', safe_html(result))
        self.assertNotIn('onclick', safe_html(result.replace('entity_name=', 'onclick="evil" entity_name=')))
    def test_commas_in_names_never_mention_another_person(self):
        full, other = str(uuid.uuid4()), str(uuid.uuid4())
        body = safe_html(participant_header({'participants': ['Smith, Alice']}))
        resolved = resolve_mentions(body, [{'id': full, 'display_name': 'Smith, Alice'}, {'id': other, 'display_name': 'Alice'}])
        self.assertIn(full, resolved)
        self.assertNotIn(other, resolved)
    def test_roundtrip_missing_text_not_accepted(self):
        value = transcript_block(INDEX, 'https://app.test')
        self.assertTrue(verify_transcript(value, INDEX, 'https://app.test'))
        self.assertFalse(verify_transcript(value.replace('Подготовим данные.', ''), INDEX, 'https://app.test'))
    def test_only_preview_and_media_between_header_and_transcript(self):
        original, preview = str(uuid.uuid4()), str(uuid.uuid4())
        assets = [{'role': role, 'asset_id': uid, 'name': 'video', 'type': 'video/mp4', 'size': 123} for role, uid in [('original', original), ('preview', preview)]]
        blocks = native_blocks('test', assets, [{'seconds': 3, 'label': 'Topic'}])
        self.assertNotIn(original, blocks)
        self.assertEqual(blocks.count('<attachment-component'), 1)
        value = insert_media(participant_header(INDEX) + transcript_block(INDEX, 'https://app.test/result?id=1') + '<p>Summary</p>', blocks)
        self.assertLess(value.index(PARTICIPANTS), value.index('<attachment-component'))
        self.assertLess(value.index('plane-timecodes-test'), value.index('<details'))
    def test_migration_preserves_other_body_and_details(self):
        value = '<h1>Generated title</h1><p><strong>' + PARTICIPANTS + '</strong> User</p><h2>Таймкоды</h2><ul><li>Duplicate</li></ul><h2>Главное</h2><p>Manual edit</p><details><summary>History</summary><p>Old facts</p></details>'
        result = clean_summary(value + '<h1>Manual section</h1>', meeting_title='Generated title')
        self.assertIn('<h1>Manual section</h1>', result)
        self.assertNotIn('Duplicate', result)
        self.assertIn('<p>Manual edit</p>', result)
        self.assertIn('<details><summary>History</summary><p>Old facts</p></details>', result)
    def test_source_injection_escaped_and_wrong_revision_rejected(self):
        index = {**INDEX, 'by_id': {'U00001': {**INDEX['by_id']['U00001'], 'text': '<script>evil</script>'}}}
        self.assertIn('&lt;script&gt;', transcript_block(index, 'https://app.test'))
        with self.assertRaises(ValueError):
            project_view(ProjectionTests().view(), 1, 'https://app.test', ['H1'], {**INDEX, 'source_sha256': 'other'})
