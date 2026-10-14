"""Source selection must preserve evidence, not repair generated quotations."""
import copy
import json
import unittest
from unittest.mock import MagicMock

from langchain_core.messages import AIMessageChunk
from src.evidence_gate import (CustomerContext, LangChainEvidenceAssessor,
                               InsuranceEvidencePipeline, evidence_spans)


class SourceSpanTests(unittest.TestCase):
    def assess(self, references, duplicate=False, duplicate_status='supported'):
        self.text = 'Y our cover:\n• Damage is not covered.\n• The excess is £500.'
        part = {'question_part': 'What are the terms?', 'status': 'supported',
                'evidence': references, 'missing_evidence': [],
                'missing_customer_facts': [], 'conflicts': []}
        wire = {'assessed_question': 'Terms?', 'all_question_parts_identified': True,
                'parts': [part]}
        if duplicate:
            wire['parts'].append(copy.deepcopy(part))
            wire['parts'][-1]['status'] = duplicate_status
            if duplicate_status == 'insufficient_evidence':
                wire['parts'][-1]['missing_evidence'] = ['Missing restrictions']
        llm = MagicMock()
        self.llm = llm
        llm.bind.return_value.stream.return_value = [AIMessageChunk(content=json.dumps(wire))]
        return LangChainEvidenceAssessor(llm=llm).assess(
            'Terms?', CustomerContext(request_type='general'),
            [{'id': 'source', 'text': self.text, 'metadata': {}}])[0]

    def test_lossless_source_segmentation(self):
        for text in ('Y our cover:\n• Damage is not covered.\n• The excess is £500.',
                     'Wrapped\nwording;  figures £1,000.00 remain unchanged.', ''):
            self.assertEqual(''.join(evidence_spans(text)), text)

    def test_source_wording_numbers_negation_and_bullets_restored(self):
        report = self.assess([{'chunk_id': 'source', 'span_index': 0},
                              {'chunk_id': 'source', 'span_index': 2}])
        self.assertEqual(report.parts[0].evidence[0].quote, 'Y our cover:')
        self.assertEqual(report.parts[0].evidence[1].quote, '• Damage is not covered.')

    def test_invalid_ids_indices_and_free_form_quotes_fail(self):
        for ref in ({'chunk_id': 'wrong', 'span_index': 0},
                    {'chunk_id': 'source', 'span_index': -1},
                    {'chunk_id': 'source', 'span_index': 1000},
                    {'chunk_id': 'source', 'span_index': True},
                    {'chunk_id': 'source', 'span_index': 1},  # whitespace-only
                    {'chunk_id': 'source', 'quote': 'Damage is covered.'}):
            with self.subTest(ref=ref), self.assertRaises(ValueError):
                self.assess([ref])

    def test_identical_parts_merge_without_duplicate_references(self):
        report = self.assess([{'chunk_id': 'source', 'span_index': 2}], duplicate=True)
        self.assertEqual(len(report.parts), 1)
        self.assertEqual(len(report.parts[0].evidence), 1)

    def test_different_duplicate_status_preserves_missing_evidence(self):
        report = self.assess([{'chunk_id': 'source', 'span_index': 2}],
                             duplicate=True, duplicate_status='insufficient_evidence')
        self.assertEqual(len(report.parts), 1)
        self.assertEqual(report.parts[0].missing_evidence, ['Missing restrictions'])
        state = {'query': 'Terms?', 'action': None, 'assessment': report, 'steps': [],
                 'chunks': [{'id': 'source', 'text': self.text}]}
        InsuranceEvidencePipeline._decide(None, state)
        self.assertEqual(state['reason'], 'insufficient_evidence')
        self.assertTrue(state['assessment_valid'])

    def test_conflicting_duplicate_cannot_be_promoted_to_supported(self):
        report = self.assess([{'chunk_id': 'source', 'span_index': 2}],
                             duplicate=True, duplicate_status='conflicting_evidence')
        state = {'query': 'Terms?', 'action': None, 'assessment': report, 'steps': [],
                 'chunks': [{'id': 'source', 'text': self.text}]}
        InsuranceEvidencePipeline._decide(None, state)
        self.assertEqual(state['reason'], 'conflicting_evidence')
        self.assertTrue(state['assessment_valid'])

    def test_schema_binds_nonempty_indices_to_each_chunk_and_bounds_output(self):
        self.assess([{'chunk_id': 'source', 'span_index': 2}])
        schema = self.llm.bind.return_value.stream.call_args.kwargs['response_format']['json_schema']['schema']
        props = schema['$defs']['EvidenceReference']['anyOf'][0]['properties']
        self.assertEqual(props['chunk_id']['enum'], ['source'])
        self.assertNotIn(1, props['span_index']['enum'])
        self.assertEqual(schema['properties']['parts']['maxItems'], 8)
        self.assertEqual(schema['$defs']['QuestionPart']['properties']['evidence']['maxItems'], 12)


if __name__ == '__main__':
    unittest.main()
