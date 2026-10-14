import copy
import unittest
from unittest.mock import MagicMock

from src.evidence_gate import InsuranceEvidencePipeline
from src.guardrails import insurance_gate_response


class RequestedPolicyTests(unittest.TestCase):
    def setUp(self):
        self.retriever = MagicMock()
        self.retriever.chunks = [
            {'id': 'axa_chunk', 'text': 'AXA home wording', 'metadata': {
                'doc_id': 'axa', 'source_type': 'policy', 'provider': 'AXA', 'title': 'AXA Home Insurance'}},
            {'id': 'lv_chunk', 'text': 'LV home wording', 'metadata': {
                'doc_id': 'lv', 'source_type': 'policy', 'provider': 'LV=', 'title': 'LV= Home Insurance'}}]
        self.assessor = MagicMock()
        self.pipeline = InsuranceEvidencePipeline(self.retriever, self.assessor, ['axa', 'lv'])

    def test_missing_insurer_and_product_reject_before_retrieval_or_assessment(self):
        for query in ('What does Direct Line Home Insurance cover for subsidence?',
                      'What does Admiral Gold Home Insurance say about leaking pipes?',
                      'Does travel insurance reimburse a cancelled international flight?',
                      'What does AXA motor insurance cover?'):
            with self.subTest(query=query):
                result = self.pipeline.invoke(query, {'request_type': 'general'})
                self.assertEqual(result['action'], 'abstain')
                self.assertEqual(result['reason'], 'requested_policy_unavailable')
                self.assertFalse(result['decision_support']['technical_failure'])
        self.retriever.search_chunks.assert_not_called()
        self.assessor.assess.assert_not_called()
        self.assertIn('unavailable', insurance_gate_response('requested_policy_unavailable'))

    def test_present_insurer_scopes_retrieval_to_its_own_policy(self):
        self.retriever.search_chunks.return_value = []
        result = self.pipeline.invoke('What does AXA Home Insurance cover?', {'request_type': 'general'})
        self.assertEqual(result['reason'], 'no_evidence')
        self.assertEqual(self.retriever.search_chunks.call_args.kwargs['allowed_document_ids'], ['axa'])

    def test_context_for_other_insurer_cannot_override_named_insurer(self):
        result = self.pipeline.invoke('What does AXA Home Insurance cover?',
            {'request_type': 'general', 'policy_document_id': 'lv'})
        self.assertEqual(result['reason'], 'requested_policy_unavailable')
        self.retriever.search_chunks.assert_not_called()

    def test_comparison_requires_both_requested_insurers(self):
        result = self.pipeline.invoke('Compare AXA and Direct Line home insurance.', {'request_type': 'general'})
        self.assertEqual(result['reason'], 'requested_policy_unavailable')

    def test_generic_question_still_uses_existing_allowlist(self):
        self.retriever.search_chunks.return_value = []
        self.pipeline.invoke('Where is the excess specified?', {'request_type': 'general'})
        self.assertEqual(self.retriever.search_chunks.call_args.kwargs['allowed_document_ids'], ['axa', 'lv'])


if __name__ == '__main__':
    unittest.main()
