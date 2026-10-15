"""Source selection must preserve evidence, not repair generated quotations."""
import copy
import json
import unittest
from unittest.mock import MagicMock

from langchain_core.messages import AIMessageChunk
from src.evidence_gate import (CustomerContext, LangChainEvidenceAssessor,
                               InsuranceEvidencePipeline, evidence_spans, is_single_cause_wording_question)


class SourceSpanTests(unittest.TestCase):
    def test_cause_wording_schema_cannot_invent_additional_requests(self):
        query = 'What does Aviva Signature Home say about escape of water caused by failed sealant or grout?'
        llm = MagicMock()
        llm.bind.return_value.stream.return_value = [AIMessageChunk(content=json.dumps({
            'assessed_question': query, 'all_question_parts_identified': True,
            'parts': [{'question_part': query, 'status': 'insufficient_evidence', 'evidence': [],
                       'missing_evidence': ['Applicable clause'], 'missing_customer_facts': [], 'conflicts': []}]}))]
        report, _ = LangChainEvidenceAssessor(llm=llm).assess(query, CustomerContext(request_type='general'), [])
        schema = llm.bind.return_value.stream.call_args.kwargs['response_format']['json_schema']['schema']
        self.assertEqual(schema['properties']['parts']['maxItems'], 1)
        self.assertEqual(schema['$defs']['QuestionPart']['properties']['question_part']['enum'], [query])
        self.assertEqual(report.parts[0].status, 'insufficient_evidence')
        for other in ('What does Aviva say about wear and tear and maintenance?',
                      query[:-1] + ' and what is my excess?',
                      'What does my policy say about water damage caused by grout?',
                      'Is damage caused by grout covered and what is the limit?'):
            self.assertFalse(is_single_cause_wording_question(other), other)

    def test_explicit_exclusion_supports_negative_answer_without_claim_inference(self):
        text = 'We do not cover water damage caused by failed grout or sealant.'
        query = 'What does this policy say about water damage caused by failed grout or sealant?'
        part = {'question_part': query, 'status': 'supported',
                'evidence': [{'chunk_id': 'exclusion', 'span_index': 0}],
                'missing_evidence': [], 'missing_customer_facts': [], 'conflicts': []}
        llm = MagicMock()
        llm.bind.return_value.stream.return_value = [AIMessageChunk(content=json.dumps({
            'assessed_question': query, 'all_question_parts_identified': True, 'parts': [part]}))]
        chunks = [{'id': 'exclusion', 'text': text, 'metadata': {}}]
        report, _ = LangChainEvidenceAssessor(llm=llm).assess(
            query, CustomerContext(request_type='general'), chunks)
        prompt = llm.bind.return_value.stream.call_args.args[0][0].content
        self.assertIn('missing merely because the answer is negative', prompt)
        state = dict(query=query, action=None, assessment=report, steps=[], chunks=chunks)
        InsuranceEvidencePipeline._decide(None, state)
        self.assertEqual((state['action'], state['reason']), ('generate', 'all_parts_supported'))
        # An unknown cause in a customer's incident still needs clarification.
        report.parts[0].status = 'needs_clarification'
        report.parts[0].missing_customer_facts = ['Was the leak caused by failed grout or sealant?']
        state.update(action=None)
        InsuranceEvidencePipeline._decide(None, state)
        self.assertEqual(state['action'], 'clarify')

    def limit_assessment(self, query, include_exclusions=False):
        text = 'Trace and access: We pay up to £5,000 to find the source of a leak.'
        part = {'question_part': query, 'status': 'supported',
                'evidence': [{'chunk_id': 'limit', 'span_index': 1}],
                'missing_evidence': [], 'missing_customer_facts': [], 'conflicts': []}
        parts = [part]
        if include_exclusions:
            parts.append({'question_part': 'What exclusions apply?', 'status': 'insufficient_evidence',
                          'evidence': [], 'missing_evidence': ['Applicable exclusions'],
                          'missing_customer_facts': [], 'conflicts': []})
        wire = {'assessed_question': query, 'all_question_parts_identified': True, 'parts': parts}
        llm = MagicMock()
        llm.bind.return_value.stream.return_value = [AIMessageChunk(content=json.dumps(wire))]
        chunks = [{'id': 'limit', 'text': text, 'metadata': {}}]
        report, _ = LangChainEvidenceAssessor(llm=llm).assess(
            query, CustomerContext(request_type='general'), chunks)
        state = {'query': query, 'action': None, 'assessment': report, 'steps': [], 'chunks': chunks}
        InsuranceEvidencePipeline._decide(None, state)
        return state, llm

    def test_limit_only_supported_part_generates_and_receives_scoped_prompt(self):
        state, llm = self.limit_assessment('What is the trace and access limit under LV= buildings insurance?')
        self.assertEqual((state['action'], state['reason']), ('generate', 'all_parts_supported'))
        messages = llm.bind.return_value.stream.call_args.args[0]
        self.assertIn('For a limit-only question, assess the stated limit in ONE part', messages[0].content)

    def test_explicitly_requested_exclusions_remain_unresolved(self):
        state, _ = self.limit_assessment('What is the trace and access limit and what exclusions apply?', True)
        self.assertNotEqual(state['action'], 'generate')
        self.assertEqual(state['assessment'].parts[1].missing_evidence, ['Applicable exclusions'])

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
