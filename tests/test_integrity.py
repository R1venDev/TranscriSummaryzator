import copy, importlib.util, io, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from test_summary_worker import summary, fact, utterance
spec=importlib.util.spec_from_file_location('consensus',Path(__file__).resolve().parents[1]/'scripts/consensus.py')
consensus=importlib.util.module_from_spec(spec);spec.loader.exec_module(consensus)

class IntegrityTests(unittest.TestCase):
    def test_same_utterance_can_support_two_different_facts(self):
        a=fact(statement='Обучить порог размера имбаланса')
        b=dict(fact(statement='Маленькие имбалансы создают шум'),fact_id='F00002')
        self.assertEqual(len(summary.deduplicate([a,b])),2)
    def test_identical_claim_same_scope_merges(self):
        a=fact();b=copy.deepcopy(a);b['fact_id']='F00002'
        self.assertEqual(len(summary.deduplicate([a,b])),1)
    def test_different_speakers_not_merged(self):
        a=fact();b=copy.deepcopy(a);b['speaker_refs']=['@Misha']
        self.assertEqual(len(summary.deduplicate([a,b])),2)
    def test_negation_not_merged(self):
        a=fact(statement='Задержка не решена');b=fact(statement='Задержка решена')
        self.assertEqual(len(summary.deduplicate([a,b])),2)
    def test_invalid_verdict_rejected(self):
        kept,denied=summary.apply_reviews([fact()],[{'fact_id':'F00001','verdict':'maybe'}],strict=True)
        self.assertEqual(kept,[]);self.assertEqual(len(denied),1)
    def test_missing_review_rejected(self):
        self.assertEqual(summary.apply_reviews([fact()],[],strict=True)[0],[])
    def test_truncated_ollama_stream_rejected(self):
        raw=io.BytesIO(b'{"message":{"content":"{}"}}\n')
        with patch('urllib.request.urlopen',return_value=raw):
            with self.assertRaisesRegex(ValueError,'оборван'):summary.Ollama().chat('test','','')
    def test_length_limited_ollama_stream_rejected(self):
        raw=io.BytesIO(b'{"message":{"content":"{}"},"done":true,"done_reason":"length"}\n')
        with patch('urllib.request.urlopen',return_value=raw):
            with self.assertRaises(ValueError):summary.Ollama().chat('test','','')
    def test_complete_ollama_stream_accepted(self):
        raw=io.BytesIO(b'{"message":{"content":"{}"},"done":true,"done_reason":"stop"}\n')
        with patch('urllib.request.urlopen',return_value=raw):self.assertEqual(summary.Ollama().chat('test','','')[0],'{}')
    def test_short_confirmation_not_erased_by_boundary_snap(self):
        intervals=[{'start':1.,'end':1.15,'speaker':'A'}]
        _,result=consensus.consensus(intervals,intervals,tolerance=.3)
        self.assertTrue(result);self.assertAlmostEqual(sum(i['end']-i['start'] for i in result),.15)
    def test_final_auditor_retypes_supported_content_instead_of_deleting_it(self):
        class Client:
            def chat(self,*a,**kw):return json.dumps({'reviews':[{'fact_id':'F00001','verdict':'reject','reason':'Вопрос не подтверждает завершение'}]}),{}
        with tempfile.TemporaryDirectory() as d:
            accepted,rejected=summary.audit_final_facts(Client(),'auditor',[fact()],Path(d))
            self.assertEqual(len(accepted),1);self.assertEqual(rejected,[])
            self.assertEqual(accepted[0]['type'],'proposal')
            self.assertEqual(accepted[0]['interpretation_status'],'retyped_after_review')
    def test_technical_token_not_invented(self):
        item=fact(statement='Обсудили алгоритм')
        d={'topics':[{'title':'Тема','items':[{'text':'Использовать M4','fact_ids':['F00001']}]}]}
        clean,rejected=summary.sanitize_structured(d,[item]);self.assertEqual(clean['topics'],[])
    def test_final_coverage_is_not_extraction_coverage(self):
        us=[utterance(1,0,1,text='Нужно проверить алгоритм и сохранить результат'),utterance(2,2,3,text='Надо сделать разметчик и проверить результат')]
        f=fact(evidence=[us[0]])
        self.assertEqual(summary.evidence_coverage(us,[f])['material_coverage_ratio'],.5)
