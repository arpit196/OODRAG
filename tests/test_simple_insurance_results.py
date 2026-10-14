import unittest
from simplify_insurance_results import simplify


class SimpleResultsTests(unittest.TestCase):
    def record(self):
        return {'result': {'query': 'Question', 'answer': 'Recorded answer', 'action': 'generate',
            'reason': 'all_parts_supported', 'citations': [{'id': 'a'}],
            'evidence_chunks': [{'id': 'a', 'text': 'Source text',
                                 'metadata': {'doc_id': 'policy', 'title': 'Policy', 'page': 3}}],
            'grounding_metrics': {'assessment': {'parts': [{'question_part': 'Question',
                'status': 'supported', 'evidence': [{'chunk_id': 'a', 'quote': 'Source text'}]}]}}}}

    def test_preserves_answer_and_merges_citation_with_assessment_reference(self):
        result = simplify(self.record())
        self.assertEqual(result['answer'], 'Recorded answer')
        self.assertEqual(len(result['supporting_chunks']), 1)
        chunk = result['supporting_chunks'][0]
        self.assertEqual(chunk['page'], 3)
        self.assertEqual(chunk['role'], ['answer_citation', 'assessment_reference'])

    def test_preserves_case_identity_for_comparison(self):
        record = self.record()
        record.update(case={'id': 'Q005'}, row={'variant': 'guarded', 'repeat': 2})
        result = simplify(record)
        self.assertEqual(result['query_id'], 'Q005')
        self.assertEqual(result['variant'], 'guarded')
        self.assertEqual(result['repeat'], 2)

    def test_html_page_unknown_and_uncited_retrieval_not_called_support(self):
        record = self.record()
        record['result']['evidence_chunks'][0]['metadata']['page'] = 0
        record['result']['evidence_chunks'].append({'id': 'unselected', 'text': 'Noise'})
        result = simplify(record)
        self.assertIsNone(result['supporting_chunks'][0]['page'])
        self.assertEqual(len(result['supporting_chunks']), 1)

    def test_failed_assessment_is_not_exported_as_support(self):
        record = self.record()
        record['result'].update(citations=[], decision_support={'technical_failure': True})
        result = simplify(record)
        self.assertEqual(result['supporting_chunks'], [])
        self.assertEqual(result['reasoning']['question_parts'], [])


if __name__ == '__main__':
    unittest.main()
