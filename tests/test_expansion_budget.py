"""Budget admission uses whole chunks and preserves mandatory evidence."""
import json
import unittest

from src.evidence_gate import CustomerContext, InsuranceEvidencePipeline, assessment_payload


def chunk(cid, size=1800, record=False):
    return {'id': cid, 'text': 'x' * size, 'metadata': {
        'doc_id': 'policy', **({'evidence_type': 'customer_policy_record'} if record else {})}}


class ExpansionBudgetTests(unittest.TestCase):
    def pipeline(self, limit=32000):
        pipeline = InsuranceEvidencePipeline.__new__(InsuranceEvidencePipeline)
        pipeline.max_evidence_chars = limit
        return pipeline

    def state(self, chunks):
        return {'action': None, 'query': 'What is covered?', 'context': CustomerContext(request_type='general'),
                'chunks': chunks, 'primary_chunk_ids': ['primary'], 'retrieval_diagnostics': {}}

    def test_whole_expansion_is_capped_and_primary_preserved(self):
        chunks = [chunk('primary')] + [chunk(str(i)) for i in range(30)]
        state = self.state(chunks)
        pipeline = self.pipeline()
        pipeline._fit_expansion_budget(state)
        pipeline._check_evidence_budget(state, 'test')
        self.assertIsNone(state['action'])
        self.assertEqual(len(state['chunks']), 16)
        self.assertIn('primary', [c['id'] for c in state['chunks']])
        self.assertTrue(all(c['text'] == 'x' * 1800 for c in state['chunks']))
        self.assertFalse(state['retrieval_diagnostics']['expansion_budget_selection']['source_text_truncated'])

    def test_serialized_metadata_and_unicode_count_toward_budget(self):
        state = self.state([chunk('primary', 20), chunk('optional', 10)])
        state['chunks'][1]['metadata']['section'] = '£' * 600
        pipeline = self.pipeline(400)
        pipeline._fit_expansion_budget(state)
        pipeline._check_evidence_budget(state, 'test')
        self.assertIsNone(state['action'])
        self.assertEqual([c['id'] for c in state['chunks']], ['primary'])
        self.assertEqual(state['evidence_budget']['payload_chars'], len(json.dumps(
            assessment_payload(state['query'], state['context'], state['chunks']), ensure_ascii=False)))

    def test_record_is_protected_and_oversized_primary_still_fails(self):
        state = self.state([chunk('primary', 30), chunk('optional', 500), chunk('record', 30, True)])
        pipeline = self.pipeline(700)
        pipeline._fit_expansion_budget(state)
        self.assertEqual([c['id'] for c in state['chunks']], ['primary', 'record'])
        pipeline._check_evidence_budget(state, 'test')
        self.assertIsNone(state['action'])
        state = self.state([chunk('primary', 2000), chunk('optional', 30)])
        pipeline._fit_expansion_budget(state)
        pipeline._check_evidence_budget(state, 'test')
        self.assertEqual(state['reason'], 'evidence_budget_exceeded')
        self.assertEqual(state['chunks'][0]['text'], 'x' * 2000)


if __name__ == '__main__':
    unittest.main()
