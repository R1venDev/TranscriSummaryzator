import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import asr_worker

def w(text,a,b):return {'text':text,'start':a,'end':b}
class BoundaryTests(unittest.TestCase):
 def test_repetition_inside_chunk_preserved(self):
  words=[w('Да',1,1.1),w('да',1.12,1.22)]
  self.assertTrue(asr_worker.same_word(*words))
  self.assertEqual(asr_worker.merge_chunk_words([],words,0),words)
 def test_repetition_across_nonoverlapping_chunks_preserved(self):
  a=[w('Да',1,1.1)];b=[w('да',1.12,1.22)]
  self.assertEqual(len(asr_worker.merge_chunk_words(a,b,1.11,1.1)),2)
 def test_duplicate_in_actual_overlap_removed(self):
  a=[w('Проверим',21.7,21.9)];b=[w('Проверим',21.72,21.92),w('потом',22.1,22.5)]
  merged=asr_worker.merge_chunk_words(a,b,21.55,22)
  self.assertEqual([word['text'] for word in merged],['Проверим','потом'])
  self.assertIn('asr_boundary',merged[0]['flags'])
 def test_unmatched_overlap_word_is_marked(self):
  merged=asr_worker.merge_chunk_words([w('один',21.7,21.9)],[w('два',21.72,21.92)],21.55,22)
  self.assertIn('asr_boundary',merged[-1]['flags'])
