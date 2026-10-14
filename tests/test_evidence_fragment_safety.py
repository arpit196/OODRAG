import unittest
from src.evidence_gate import EvidenceAssessment, InsuranceEvidencePipeline


class EvidenceFragmentSafetyTests(unittest.TestCase):
    def decide(self, query, text):
        report = EvidenceAssessment.model_validate({'assessed_question': query,
            'all_question_parts_identified': True, 'parts': [{
                'question_part': query, 'status': 'supported',
                'evidence': [{'chunk_id': 'source', 'quote': text}],
                'missing_evidence': [], 'missing_customer_facts': [], 'conflicts': []}]})
        state = {'query': query, 'action': None, 'steps': [], 'assessment': report,
                 'chunks': [{'id': 'source', 'text': text}]}
        return InsuranceEvidencePipeline._decide(None, state)

    def test_unrelated_truncated_amount_does_not_override_supported_part(self):
        result = self.decide('What is the limit for contents in the garden?',
                             'Buildings £1,000,\nContents in the garden £1,500')
        self.assertEqual(result['action'], 'generate')
        self.assertTrue(result['assessment_valid'])
        self.assertEqual(result['assessment'].parts[0].missing_evidence, [])

    def test_intact_amount_and_normal_punctuation_still_pass(self):
        for text in ('Buildings £1,000,000.', 'Pay £500, subject to the stated conditions.'):
            self.assertEqual(self.decide('How much cover?', text)['action'], 'generate')

    def test_real_settlement_clause_passes(self):
        result = self.decide('When must a claim be settled?', 'Settle claims promptly once settlement terms are agreed.')
        self.assertEqual(result['action'], 'generate')


if __name__ == '__main__':
    unittest.main()
