"""Whitespace tolerance must restore provenance without tolerating changed claims."""
import copy
import json
import unittest

from src.evidence_gate import InsuranceEvidencePipeline, source_quote_span
from test_evidence_gate import FakeRetriever, FakeAssessor

QUERY = 'Are motor vehicles included in contents?'
SOURCE = ('Contents are not:\n• motor vehicles (including motorbikes, quad bikes and motorised scooters),\n'
          'caravans, trailers, watercraft, aircraft and all their accessories;')
QUOTE = SOURCE.replace('),\ncaravans', '), caravans')


class QuoteWhitespaceTests(unittest.TestCase):
    def setUp(self):
        self.retriever, self.assessor = FakeRetriever(), FakeAssessor()
        self.retriever.chunks[1]['text'] = SOURCE + '\nThe excess is £250.'
        self.assessor.report = {'assessed_question': QUERY, 'all_question_parts_identified': True,
            'parts': [{'question_part': QUERY, 'status': 'supported',
                       'evidence': [{'chunk_id': 'seed', 'quote': QUOTE}],
                       'missing_evidence': [], 'missing_customer_facts': [], 'conflicts': []}]}
        self.pipeline = InsuranceEvidencePipeline(self.retriever, self.assessor, ['policy'])
        self.context = {'request_type': 'general', 'policy_document_id': 'policy'}

    def invoke(self):
        return self.pipeline.invoke(QUERY, self.context)

    def test_pdf_line_wrapping_passes_and_restores_original_quote(self):
        with self.assertLogs('rag_agent.evidence_gate', level='INFO') as captured:
            result = self.invoke()
        self.assertEqual(result['action'], 'generate')
        self.assertTrue(result['assessment_valid'])
        restored = result['assessment'].parts[0].evidence[0].quote
        self.assertEqual(restored, SOURCE)
        self.assertIn(restored, self.retriever.chunks[1]['text'])
        self.assertEqual(len(self.assessor.calls), 1)
        events = [json.loads(record.getMessage()) for record in captured.records]
        self.assertEqual(events[0]['event'], 'evidence_quote_whitespace_restored')
        self.assertEqual(events[-1]['whitespace_restored_count'], 1)
        # Diagnostics go to logging, never into response/state fields.
        self.assertNotIn('evidence_quote_whitespace_restored', json.dumps(result, default=str))
        self.assertTrue(all('quote' not in event and 'query' not in event for event in events))

    def test_unicode_spaces_tabs_crlf_and_outer_whitespace_restore_contiguous_span(self):
        original = 'Contents\u00a0are\t not:\r\n• motor\u202fvehicles;'
        for quote in ('Contents are not: • motor vehicles;', '  Contents are not: • motor vehicles;\n'):
            with self.subTest(quote=quote):
                self.assertEqual(source_quote_span(quote, 'Intro\n' + original + '\nEnd'), original)

    def test_exact_quotes_are_unchanged(self):
        self.assessor.report['parts'][0]['evidence'][0]['quote'] = SOURCE
        with self.assertLogs('rag_agent.evidence_gate', level='INFO') as captured:
            result = self.invoke()
        self.assertEqual(result['assessment'].parts[0].evidence[0].quote, SOURCE)
        self.assertEqual(json.loads(captured.records[-1].getMessage())['whitespace_restored_count'], 0)

    def test_changed_wording_numbers_punctuation_and_negation_still_fail(self):
        references = [
            ('changed_amount', 'seed', 'The excess is £500.'),
            ('removed_not', 'seed', QUOTE.replace('Contents are not:', 'Contents are:')),
            ('paraphrase', 'seed', 'Motor vehicles are excluded from contents.'),
            ('changed_punctuation', 'seed', QUOTE.replace('Contents are not:', 'Contents are not;')),
            ('changed_case', 'seed', QUOTE.replace('Contents', 'contents')),
            ('unknown_id', 'absent', QUOTE),
            ('different_chunk', 'before', QUOTE),
            ('empty_quote', 'seed', '\n\t '),
        ]
        baseline = copy.deepcopy(self.assessor.report)
        for name, chunk_id, quote in references:
            with self.subTest(case=name):
                self.assessor.report = copy.deepcopy(baseline)
                self.assessor.report['parts'][0]['evidence'] = [{'chunk_id': chunk_id, 'quote': quote}]
                with self.assertLogs('rag_agent.evidence_gate', level='INFO') as captured:
                    result = self.invoke()
                self.assertEqual(result['reason'], 'invalid_or_incomplete_assessment')
                self.assertTrue(result['decision_support']['technical_failure'])
                self.assertIn('invalid_evidence_reference', result['validation_issues'])
                self.assertEqual(result['decision_support']['question_parts'], [])
                event = json.loads(captured.records[0].getMessage())
                expected = ('unknown_chunk_id' if name == 'unknown_id' else
                            'empty_quote' if name == 'empty_quote' else 'quote_text_mismatch')
                self.assertEqual(event['failure'], expected)

    def test_missing_chunk_id_field_is_schema_error(self):
        del self.assessor.report['parts'][0]['evidence'][0]['chunk_id']
        with self.assertLogs('rag_agent.evidence_gate', level='WARNING') as captured:
            result = self.invoke()
        self.assertEqual(result['reason'], 'stage2_error')
        self.assertEqual(result['error_type'], 'ValidationError')
        self.assertTrue(result['decision_support']['technical_failure'])
        event = json.loads(captured.records[-1].getMessage())
        self.assertEqual(event['event'], 'evidence_pipeline_error')
        self.assertEqual(event['failed_step'], 'part_assessment')

    def test_matching_reference_does_not_rescue_an_invalid_second_reference(self):
        self.assessor.report['parts'][0]['evidence'].append({'chunk_id': 'absent', 'quote': QUOTE})
        result = self.invoke()
        self.assertEqual(result['action'], 'abstain')
        self.assertEqual(result['reason'], 'invalid_or_incomplete_assessment')
        self.assertEqual(result['assessment'].parts[0].evidence[0].quote, SOURCE)

    def test_partial_or_noncontiguous_quotes_are_not_stitched(self):
        self.assertIsNone(source_quote_span('Contents are not: aircraft;', SOURCE))
        self.assertIsNone(source_quote_span('motorvehicles', SOURCE))
        self.assertIsNone(source_quote_span('motor\u200b vehicles', SOURCE))


if __name__ == '__main__':
    unittest.main()
