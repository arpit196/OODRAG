import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from report_insurance_results import report
from ui.insurance_app import create_app, document_path, presentation
from fastapi.testclient import TestClient


class ReportingTests(unittest.TestCase):
    def test_denominators_unknown_cost_and_manual_review_preserved(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'raw_results.jsonl'
            records = []
            for i, (variant, action, score) in enumerate([('guarded', 'generate', 1), ('guarded', 'abstain', None), ('standard', 'generate', None)]):
                records.append({'case': {'id': str(i)}, 'row': {'query_id': str(i), 'query': 'q', 'variant': variant, 'repeat': 1,
                    'category': 'id' if score else 'ood', 'expected_actions': ['generate'], 'error': None,
                    'request_latency_ms': 100 * (i + 1), 'estimated_cost': None, 'evidence_recall@5': score},
                    'result': {'action': action, 'answer': 'I cannot answer.'}})
            path.write_text('\n'.join(json.dumps(r) for r in records), encoding='utf-8')
            rows, metrics = report(path)
            get = lambda name: next(m for m in metrics if m['group'] == 'all' and m['metric'] == name)
            self.assertEqual(get('evidence_recall@5')['measured_count'], 1)
            self.assertEqual(get('evidence_recall@5')['value'], 1)
            self.assertEqual(get('request_latency_ms_p50')['value'], 200)
            self.assertIsNone(get('estimated_cost_complete_total')['value'])
            self.assertIsNone(rows[2]['manual_actually_answered'])
            review = Path(folder) / 'manual_review.csv'
            with review.open(newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                fields, data = reader.fieldnames, list(reader)
            data[2]['actually_answered'] = '0'
            with review.open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(data)
            rows, _ = report(path)
            self.assertEqual(rows[2]['manual_actually_answered'], 0)
            self.assertEqual(len(json.loads((Path(folder) / 'simple_results.json').read_text())), 3)


class UITests(unittest.TestCase):
    def result(self, action='generate'):
        return {'action': action, 'answer': 'Answer [1]', 'citations': [{'id': 'c', 'n': 1}],
                'evidence_chunks': [{'id': 'c', 'text': 'Source', 'metadata': {'doc_id': 'd', 'page': 3, 'title': 'Policy'}}]}

    def test_sources_require_cited_nonfailed_response_and_real_page(self):
        docs = {'d': {'path': Path('source.pdf')}}
        self.assertIn('#page=3', presentation(self.result(), docs)['sources'][0]['viewer_url'])
        self.assertEqual(presentation(self.result('abstain'), docs)['sources'], [])
        result = self.result(); result['decision_support'] = {'technical_failure': True}
        self.assertEqual(presentation(result, docs)['sources'], [])
        result = self.result(); result['citations'] = []
        self.assertEqual(presentation(result, docs)['sources'], [])
        result = self.result(); result['evidence_chunks'][0]['metadata']['page'] = 0
        self.assertIsNone(presentation(result, docs)['sources'][0]['viewer_url'])

    def test_api_context_validation_and_document_confinement(self):
        with TemporaryDirectory() as folder:
            root = Path(folder); (root / 'raw').mkdir(); (root / 'raw/source.pdf').write_bytes(b'%PDF-1.4\n')
            meta = {'doc_id': 'd', 'raw_path': '/mnt/c/project/raw/source.pdf'}
            self.assertEqual(document_path(meta, root), root / 'raw/source.pdf')
            self.assertIsNone(document_path({'raw_path': '../outside.pdf'}, root))
            calls = []
            def answer(query, **kwargs):
                calls.append(kwargs); return self.result()
            agent = SimpleNamespace(answer=answer, evidence_pipeline=SimpleNamespace(retriever=SimpleNamespace(chunks=[{'metadata': meta}])))
            with TestClient(create_app(lambda: agent, root)) as client:
                self.assertEqual(client.get('/healthz').json(), {'status': 'ok'})
                self.assertEqual(client.get('/readyz').json(), {'status': 'ready'})
                self.assertEqual(client.get('/').status_code, 200)
                response = client.post('/api/chat', json={'query': 'Question'})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(calls[0]['customer_context']['request_type'], 'general')
                self.assertTrue(calls[0]['include_evidence'])
                self.assertEqual(client.get('/api/documents/d').status_code, 200)
                self.assertEqual(client.get('/api/documents/unknown').status_code, 404)
                self.assertEqual(client.post('/api/chat', json={'query': '  '}).status_code, 422)
                self.assertEqual(client.post('/api/chat', json={'query': 'q', 'top_k': 99}).status_code, 422)


if __name__ == '__main__':
    unittest.main()
