"""Dependency recall, scope, and partial support; no model downloads or API calls."""
import copy
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from insurance_retrieval_fixed import InsuranceHybridRetriever
from src.evidence_gate import InsuranceEvidencePipeline, assessment_payload
from test_insurance_grounding_retrieval import FakeCollection
from evaluate_insurance_boundary import load_cases, validate_source_anchors, source_anchor_checks, make_record

QUERY = ('According to this LV policy wording, where is my excess specified, and '
         'what is my exact escape-of-water excess? My personal details document is unavailable.')
QUOTE = 'If your personal details shows you have to pay an excess, this is the amount you must pay as the first part of any claim.'


def chunk(cid, text, doc='policy', index=0, source='policy'):
    return {'id': cid, 'text': text, 'embedding': [1., 0.], 'metadata': {
        'doc_id': doc, 'source_type': source, 'chunk_index': index,
        'title': 'LV= Home Insurance policy wording', 'provider': 'LV=',
        'page': 19, 'source_url': 'https://example.test/policy.pdf'}}


class Assessor:
    def __init__(self):
        self.calls = []

    def assess(self, query, context, chunks):
        self.calls.append(copy.deepcopy(chunks))
        return {'assessed_question': query, 'all_question_parts_identified': True,
                'parts': [{'question_part': 'Where is my excess specified?', 'status': 'supported',
                    'evidence': [{'chunk_id': 'clause', 'quote': QUOTE}],
                    'missing_evidence': [], 'missing_customer_facts': [], 'conflicts': []},
                    {'question_part': 'What is my exact escape-of-water excess?',
                     'status': 'insufficient_evidence', 'evidence': [],
                     'missing_evidence': ['Applicable customer personal-details document with the exact excess'],
                     'missing_customer_facts': [], 'conflicts': []}]}, {}


class ExcessRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [chunk('noise', 'Privacy information about personal data.', index=0),
                       chunk('clause', 'Excesses that apply ' + QUOTE, index=20),
                       chunk('other', 'Excesses that apply ' + QUOTE, doc='other', index=20),
                       chunk('regulator', 'Excesses that apply ' + QUOTE, source='fca', index=20),
                       chunk('number', 'A policy limit is £500 and an excess may apply.', index=40)]
        self.model = MagicMock()
        self.model.encode.side_effect = lambda texts, **kw: np.tile([1., 0.], (len(texts), 1))
        self.collection = FakeCollection(self.chunks, ['noise'])
        self.retriever = InsuranceHybridRetriever(self.collection, self.model)
        self.retriever.bm25 = MagicMock()
        self.retriever.bm25.get_scores.return_value = np.array([100., 1., 1., 1., 1.])

    def test_scoped_clause_lookup_preserves_provenance_without_encoding(self):
        refs = self.retriever.search_excess_references(QUERY, ['policy'])
        self.assertEqual([c['id'] for c in refs], ['clause'])
        self.assertEqual(refs[0]['metadata']['page'], 19)
        self.assertEqual(refs[0]['text'], self.chunks[1]['text'])
        refs[0]['metadata']['page'] = 99
        self.assertEqual(self.retriever.chunks[1]['metadata']['page'], 19)
        self.model.encode.assert_not_called()
        self.assertEqual(self.collection.embedding_reads, 0)

    def test_no_scope_no_hit_and_non_excess_question_no_hit(self):
        self.assertEqual(self.retriever.search_excess_references(QUERY, []), [])
        self.assertEqual(self.retriever.search_excess_references(QUERY, ['missing']), [])
        self.assertEqual(self.retriever.search_excess_references('Is theft covered?', ['policy']), [])
        self.model.encode.assert_not_called()

    def test_recovers_primary_miss_but_clarifies_for_customer_amount(self):
        assessor = Assessor()
        pipeline = InsuranceEvidencePipeline(self.retriever, assessor, ['policy'], top_k=1)
        result = pipeline.invoke(QUERY, {'request_type': 'general', 'policy_document_id': 'policy'})
        details = result['decision_support']
        lookup = details['retrieval_diagnostics']['excess_clause_lookup']
        self.assertEqual(lookup['status'], 'retrieval_miss_recovered')
        self.assertEqual(lookup['initial_chunk_ids'], ['noise'])
        self.assertEqual(lookup['recovered_chunk_ids'], ['clause'])
        self.assertEqual(result['reason'], 'partial_support')
        self.assertTrue(result['assessment_valid'])
        self.assertEqual(details['question_parts'][0]['status'], 'supported')
        self.assertEqual(details['question_parts'][1]['status'], 'insufficient_evidence')
        self.assertFalse(details['customer_schedule_evidence']['verified_record_provided'])
        self.assertEqual(len(assessor.calls), 1)
        self.assertEqual(len({c['id'] for c in result['chunks']}), len(result['chunks']))
        # Focused route keeps the reference clause, without an expansion batch.
        self.assertEqual(self.model.encode.call_count, 1)
        self.assertEqual([c['id'] for c in result['chunks']], ['clause'])
        self.assertEqual(details['retrieval_diagnostics']['evidence_selection']['supplementary_searches'], 0)

    def test_existing_clause_is_not_duplicated(self):
        pipeline = InsuranceEvidencePipeline(self.retriever, Assessor(), ['policy'])
        self.retriever.search_chunks = MagicMock(return_value=[copy.deepcopy(self.retriever.chunks[1])])
        result = pipeline.invoke(QUERY, {'request_type': 'general', 'policy_document_id': 'policy'})
        self.assertEqual(result['decision_support']['retrieval_diagnostics']['excess_clause_lookup']['status'], 'already_retrieved')
        self.assertEqual(sum(c['id'] == 'clause' for c in result['chunks']), 1)

    def test_no_matching_clause_is_not_called_corpus_absence(self):
        self.retriever.excess_reference_chunks = {}
        pipeline = InsuranceEvidencePipeline(self.retriever, Assessor(), ['policy'])
        result = pipeline.invoke(QUERY, {'request_type': 'general', 'policy_document_id': 'policy'})
        self.assertEqual(result['decision_support']['retrieval_diagnostics']['excess_clause_lookup']['status'], 'not_found_by_targeted_lookup')

    def test_expected_source_check_flags_retrieval_miss(self):
        case = {'source_refs': [{'doc_id': 'policy', 'anchor': QUOTE}]}
        missing = source_anchor_checks(case, {'evidence_chunks': [self.chunks[0]]})
        found = source_anchor_checks(case, {'evidence_chunks': [self.chunks[1]]})
        self.assertFalse(missing[0]['retrieved'])
        self.assertTrue(found[0]['retrieved'])
        self.assertEqual(found[0]['matching_chunks'], [{'id': 'clause', 'page': 19}])

    def test_invalid_assessment_is_not_successful_abstention(self):
        case = load_cases(Path('data/insurance_excess_retrieval_cases.jsonl'))[0]
        result = {'action': 'abstain', 'reason': 'invalid_or_incomplete_assessment',
                  'decision_support': {'technical_failure': True}}
        row = make_record(case, 'guarded', 1, result)
        self.assertEqual(row['status'], 'error')
        self.assertFalse(row['guarded_action_matches'])

    def test_real_case_anchors_exist(self):
        cases = load_cases(Path('data/insurance_excess_retrieval_cases.jsonl'))
        self.assertEqual(len(cases), 2)
        validate_source_anchors(cases)

    def test_unused_bookkeeping_does_not_exhaust_assessment_budget(self):
        for c in self.retriever.chunks:
            c['metadata'].update(raw_path='x' * 18000, sha256='y' * 18000)
        assessor = Assessor()
        pipeline = InsuranceEvidencePipeline(self.retriever, assessor, ['policy'], top_k=1)
        result = pipeline.invoke(QUERY, {'request_type': 'general', 'policy_document_id': 'policy'})
        import json
        self.assertGreater(len(json.dumps(result['chunks'], ensure_ascii=False)), 32000)
        budget = result['decision_support']['evidence_budget']
        expected = len(json.dumps(assessment_payload(QUERY, result['context'], result['chunks']), ensure_ascii=False))
        self.assertEqual(budget['payload_chars'], expected)
        self.assertFalse(budget['exceeded'])
        self.assertEqual(result['reason'], 'partial_support')
        self.assertTrue(result['assessment_valid'])
        self.assertEqual(len(assessor.calls), 1)
        self.assertIn(QUOTE, next(c['text'] for c in result['chunks'] if c['id'] == 'clause'))
        self.assertEqual(next(c for c in result['chunks'] if c['id'] == 'clause')['metadata']['raw_path'], 'x' * 18000)

    def test_genuinely_large_source_text_still_rejects_without_truncating(self):
        complete_text = QUOTE + ' Full source text ' * 3000
        self.retriever.chunks[1]['text'] = complete_text
        self.collection.chunks[1]['text'] = complete_text
        assessor = Assessor()
        pipeline = InsuranceEvidencePipeline(self.retriever, assessor, ['policy'], top_k=1)
        result = pipeline.invoke(QUERY, {'request_type': 'general', 'policy_document_id': 'policy'})
        self.assertEqual(result['reason'], 'evidence_budget_exceeded')
        self.assertEqual(result['error_code'], 'assessment_input_budget_exceeded')
        self.assertTrue(result['decision_support']['evidence_budget']['exceeded'])
        self.assertEqual(next(c['text'] for c in result['chunks'] if c['id'] == 'clause'), complete_text)
        self.assertFalse(assessor.calls)

    def test_focused_selection_removes_noise_and_keeps_contract_context(self):
        primary = [copy.deepcopy(self.chunks[0]),
                   chunk('contract', 'Your document of home insurance and personal details make up your contract.'),
                   chunk('phone', 'For legal advice call our telephone helpline.'),
                   chunk('conflicting_amount', 'The escape-of-water excess is £750.')]
        self.retriever.search_chunks = MagicMock(return_value=primary)
        result = InsuranceEvidencePipeline(self.retriever, Assessor(), ['policy']).invoke(
            QUERY, {'request_type': 'general', 'policy_document_id': 'policy'})
        self.assertEqual({c['id'] for c in result['chunks']}, {'contract', 'clause', 'conflicting_amount'})
        selection = result['decision_support']['retrieval_diagnostics']['evidence_selection']
        self.assertEqual(selection['selected_count'], 3)
        self.assertEqual(set(selection['removed_chunk_ids']), {'noise', 'phone'})
        self.model.encode.assert_not_called()
        self.retriever.search_chunks.assert_called_once()

    def test_mixed_requests_retain_broad_expansion(self):
        queries = [
            'Where is the excess specified and is this leak covered?',
            'Explain trace-and-access cover and its limit, then tell me my exact escape-of-water excess.',
            'What is my excess and how much alternative accommodation compensation can I claim?',
            'What is the excess if my house is unoccupied?',
        ]
        for query in queries:
            with self.subTest(query=query):
                self.model.encode.reset_mock()
                result = InsuranceEvidencePipeline(self.retriever, Assessor(), ['policy']).invoke(
                    query, {'request_type': 'general', 'policy_document_id': 'policy'})
                self.assertNotIn('evidence_selection', result['decision_support']['retrieval_diagnostics'])
                self.assertEqual(self.model.encode.call_count, 2)

    def test_without_reference_clauses_or_pinned_policy_use_broad_route(self):
        for context, cache in (({'request_type': 'general'}, self.retriever.excess_reference_chunks),
                               ({'request_type': 'general', 'policy_document_id': 'policy'}, {})):
            with self.subTest(context=context):
                self.retriever.excess_reference_chunks = cache
                self.model.encode.reset_mock()
                result = InsuranceEvidencePipeline(self.retriever, Assessor(), ['policy']).invoke(QUERY, context)
                self.assertNotIn('evidence_selection', result['decision_support']['retrieval_diagnostics'])
                self.assertEqual(self.model.encode.call_count, 2)

    def test_customer_context_counts_toward_input_budget(self):
        assessor = Assessor()
        pipeline = InsuranceEvidencePipeline(self.retriever, assessor, ['policy'], top_k=1)
        result = pipeline.invoke(QUERY, {'request_type': 'general', 'policy_document_id': 'policy',
                                       'facts': {'large_fact': 'x' * 33000}})
        self.assertEqual(result['reason'], 'evidence_budget_exceeded')
        self.assertGreater(result['decision_support']['evidence_budget']['payload_chars'], 32000)
        self.assertFalse(assessor.calls)
