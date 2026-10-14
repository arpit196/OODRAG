"""Actual indexed clauses plus deterministic customer-document clarification."""
import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from langchain_core.messages import AIMessageChunk

from evaluate_insurance_benchmark import indexed_chunks
from src.evidence_gate import (CustomerContext, InsuranceEvidencePipeline,
                               LangChainEvidenceAssessor, is_definition_comparison)
from src.guardrails import insurance_gate_response


class ReferenceRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        index = Path('insurance_chroma_index_v3')
        if not index.exists():
            raise unittest.SkipTest('Local insurance index unavailable')
        cls.chunks = indexed_chunks(index, 'insurance_chunks')

    def retrieve(self, query, ids):
        retriever = MagicMock()
        retriever.chunks = self.chunks
        retriever.search_chunks.return_value = []
        pipeline = InsuranceEvidencePipeline(retriever, MagicMock(), ids)
        state = {'query': query, 'action': None, 'steps': [], 'top_k': 10,
                 'query_embedding': None, 'context': CustomerContext(request_type='general'),
                 'allowed_ids': ids, 'retrieval_diagnostics': {}}
        return pipeline, pipeline._retrieve(state)

    def test_g002_recovers_the_actual_rule_without_neighbouring_rule_expansion(self):
        pipeline, state = self.retrieve(
            'Under ICOBS 8.1.1, when must an insurer settle a claim after settlement terms are agreed?',
            ['financial_conduct_authority_a7473f38887e'])
        self.assertTrue(any('settle claims promptly once settlement terms are agreed.' in c['text']
                            for c in state['chunks']))
        ids = [c['id'] for c in state['chunks']]
        pipeline._expand(state)
        self.assertEqual(ids, [c['id'] for c in state['chunks']])
        pipeline.retriever.model.encode.assert_not_called()

    def test_unknown_or_out_of_scope_rule_does_not_borrow_a_different_rule(self):
        for query, ids in [('Under ICOBS 8.1.10, when must claims be settled?', ['financial_conduct_authority_a7473f38887e']),
                           ('Under ICOBS 8.1.1, when must claims be settled?', ['lv_03904e7e51ee'])]:
            _, state = self.retrieve(query, ids)
            self.assertEqual(state['reason'], 'no_evidence')
            self.assertNotIn('rule_lookup', state['retrieval_diagnostics'])

    def test_v001_recovers_both_versions_without_other_axa_products(self):
        ids = sorted({c['metadata']['doc_id'] for c in self.chunks
                      if c['metadata']['doc_id'].startswith('axa_')})
        pipeline, state = self.retrieve('Compare the unoccupied definitions in the AXA Premier 2023 file and 2025 file.', ids)
        self.assertEqual({c['metadata']['version_label'] for c in state['chunks']}, {'2023 file', '2025 file'})
        self.assertEqual(len(state['chunks']), 2)
        for c in state['chunks']:
            self.assertIn('Premier', c['metadata']['title'])
            self.assertIn('Unoccupied Not lived in and not occupied overnight', c['text'])
        self.assertEqual({c['metadata']['page'] for c in state['chunks']}, {12, 56})
        pipeline._expand(state)
        self.assertEqual(len(state['chunks']), 2)
        pipeline.retriever.model.encode.assert_not_called()


class CustomerClarificationTests(unittest.TestCase):
    def test_exact_excess_requests_customer_document_without_contract_boilerplate(self):
        report = {'parts': [{'missing_evidence': ['Personal details document with the exact excess'],
                             'missing_customer_facts': []}]}
        text = insurance_gate_response('partial_support', report,
            query='What is my exact LV escape-of-water excess? My personal details document is unavailable.')
        self.assertIn('cannot determine your exact excess', text)
        self.assertIn('verified customer-policy record', text)
        self.assertNotIn('applicable wording states', text)

    def test_optional_cover_selection_requests_schedule(self):
        text = insurance_gate_response('verified_policy_required',
            query='Is accidental damage selected on my LV customer policy?')
        self.assertIn('whether this optional cover is selected', text)
        self.assertIn('schedule', text)

    def test_mixed_excess_question_keeps_supported_location_quote(self):
        report = {'parts': [{'status': 'supported', 'evidence': [{'chunk_id': 'c', 'quote': 'See your personal details.'}],
                             'missing_evidence': [], 'missing_customer_facts': []},
                            {'status': 'insufficient_evidence', 'evidence': [],
                             'missing_evidence': ['Personal details document'], 'missing_customer_facts': []}]}
        text = insurance_gate_response('partial_support', report, [{'id': 'c', 'cite_id': 1}],
                                      query='Where is my excess specified and what is my exact excess?')
        self.assertIn('See your personal details.', text)
        self.assertIn('[1]', text)


class ComparisonAssessmentTests(unittest.TestCase):
    query = 'Compare the unoccupied definitions in the AXA Premier 2023 file and 2025 file.'

    def assess(self, ids, status='supported'):
        wire = {'assessed_question': self.query, 'all_question_parts_identified': True,
                'parts': [{'question_part': self.query, 'status': status,
                           'evidence': [{'chunk_id': cid, 'span_index': 0} for cid in ids],
                           'missing_evidence': [], 'missing_customer_facts': [], 'conflicts': []}]}
        llm = MagicMock()
        llm.bind.return_value.stream.return_value = [AIMessageChunk(content=json.dumps(wire))]
        chunks = [{'id': year, 'text': 'Definition in this fixture.',
                   'metadata': {'version_label': year + ' file'}} for year in ('2023', '2025')]
        report, _ = LangChainEvidenceAssessor(llm=llm).assess(
            self.query, CustomerContext(request_type='general'), chunks)
        return report, llm.bind.return_value.stream.call_args.kwargs['response_format']['json_schema']['schema']

    def test_supported_comparison_cannot_cover_only_one_version(self):
        report, _ = self.assess(['2023'])
        self.assertFalse(report.all_question_parts_identified)

    def test_both_version_references_preserved_in_one_comparison(self):
        report, schema = self.assess(['2023', '2025'])
        self.assertTrue(report.all_question_parts_identified)
        self.assertEqual(len(report.parts[0].evidence), 2)
        self.assertEqual(schema['$defs']['QuestionPart']['properties']['question_part']['enum'], [self.query])

    def test_insufficient_comparison_may_report_missing_evidence_without_citations(self):
        report, _ = self.assess([], 'insufficient_evidence')
        self.assertTrue(report.all_question_parts_identified)

    def test_compound_request_is_not_squeezed_into_single_comparison_part(self):
        self.assertFalse(is_definition_comparison(self.query.rstrip('.') + ' and explain my cover.'))


if __name__ == '__main__':
    unittest.main()
