"""Synthetic arithmetic fixtures only; benchmark labels come from real documents."""
import unittest
from pathlib import Path
import json
from unittest.mock import MagicMock

from evaluate_insurance_benchmark import (retrieval_metrics, metadata_mismatches, percentile,
    aggregate, usage_and_cost, RetrievalObserver, indexed_chunks, validate_cases)


def hit(cid, doc):
    return {'id': cid, 'text': 'Unit-test fixture', 'metadata': {'doc_id': doc}}


class BenchmarkMetricTests(unittest.TestCase):
    def setUp(self):
        self.case = {'gold_document_ids': ['a', 'b'], 'gold_evidence': [
            {'evidence_id': 'span1', 'matching_chunk_ids': ['a1', 'a1_overlap']},
            {'evidence_id': 'span2', 'matching_chunk_ids': ['b1']}]}

    def test_document_and_evidence_relevance_are_different(self):
        m = retrieval_metrics(self.case, [hit('not_evidence', 'a'), hit('a1', 'a')], [1, 2])
        self.assertEqual(m['document_recall@1'], .5)
        self.assertEqual(m['evidence_recall@1'], 0)
        self.assertEqual(m['document_precision@1'], 1)
        self.assertEqual(m['evidence_precision@1'], 0)
        self.assertEqual(m['mrr'], 1)

    def test_duplicate_chunks_do_not_inflate_document_or_anchor_recall(self):
        m = retrieval_metrics(self.case, [hit('a1', 'a'), hit('a1_overlap', 'a'), hit('b1', 'b')], [2, 3])
        self.assertEqual(m['document_recall@2'], .5)
        self.assertEqual(m['evidence_recall@2'], .5)
        self.assertEqual(m['evidence_recall@3'], 1)

    def test_mrr_and_empty_retrieval(self):
        self.assertEqual(retrieval_metrics(self.case, [hit('x', 'x'), hit('a1', 'a')], [2])['mrr'], .5)
        m = retrieval_metrics(self.case, [], [5])
        self.assertEqual(m['mrr'], 0)
        self.assertEqual(m['hit@5'], 0)
        self.assertEqual(m['document_recall@5'], 0)

    def test_unannotated_evidence_and_ood_gold_stay_blank(self):
        case = {'gold_document_ids': ['a'], 'gold_evidence': None}
        self.assertIsNone(retrieval_metrics(case, [hit('a1', 'a')], [1])['evidence_recall@1'])
        case['gold_document_ids'] = []
        self.assertTrue(all(value is None for value in retrieval_metrics(case, [hit('a1', 'a')], [1]).values()))

    def test_short_result_list_precision_uses_declared_k(self):
        self.assertEqual(retrieval_metrics(self.case, [hit('a1', 'a')], [5])['document_precision@5'], .2)

    def test_wrong_insurer_product_and_version_are_separate(self):
        case = {'gold_documents': [{'provider': 'AXA', 'product': 'Premier', 'version_label': '2023 file'}]}
        def flags(provider='AXA', product='Premier', version='2023 file'):
            return metadata_mismatches(case, {'source_type': 'policy', 'provider': provider,
                                             'product': product, 'version_label': version})
        self.assertEqual(flags(version='2025 file'), {'wrong_insurer': 0, 'wrong_product': 0, 'wrong_version': 1})
        self.assertEqual(flags(product='Plus')['wrong_product'], 1)
        self.assertIsNone(flags(product='Plus')['wrong_version'])
        self.assertEqual(flags(provider='LV')['wrong_insurer'], 1)
        self.assertIsNone(flags(version=None)['wrong_version'])
        self.assertTrue(all(v is None for v in metadata_mismatches(case, {'source_type': 'fca'}).values()))

    def test_percentiles_are_interpolated_and_unknown_is_not_zero(self):
        self.assertEqual(percentile([10, 20, 30, 40], .5), 25)
        self.assertAlmostEqual(percentile([10, 20, 30, 40], .95), 38.5)
        self.assertAlmostEqual(percentile([10, 20, 30, 40], .99), 39.7)
        self.assertIsNone(percentile([], .95))

    def test_aggregate_denominators_and_incomplete_cost(self):
        rows = [{'variant': 'retrieval', 'category': 'supported', 'expected_actions': ['generate'],
                 'error': None, 'mrr': 1, 'request_latency_ms': 10, 'retrieval_latency_ms': 9, 'estimated_cost': .1},
                {'variant': 'retrieval', 'category': 'supported', 'expected_actions': ['generate'],
                 'error': 'timeout', 'mrr': None, 'request_latency_ms': 20, 'retrieval_latency_ms': None, 'estimated_cost': None}]
        values = {r['metric']: r for r in aggregate(rows, ['mrr']) if r['group'] == 'all'}
        self.assertEqual(values['mrr']['value'], 1)
        self.assertEqual(values['mrr']['measured_count'], 1)
        self.assertEqual(values['error_rate']['value'], .5)
        self.assertIsNone(values['total_estimated_cost']['value'])
        self.assertEqual(values['total_estimated_cost']['measured_count'], 1)

    def test_cost_uses_reported_usage_and_supplied_rates_only(self):
        result = {'repro': {'generation_model': 'openai:test-model'},
                  'generation_usage': {'input_tokens': 100, 'output_tokens': 10, 'cached_tokens': 20}}
        prices = {'currency': 'TEST', 'models': {'test-model': {
            'input_per_million': 1, 'output_per_million': 2, 'cached_input_per_million': .5}}}
        self.assertAlmostEqual(usage_and_cost(result, prices)['estimated_cost'], .00011)
        self.assertIsNone(usage_and_cost(result, None)['estimated_cost'])
        result['decision_support'] = {'technical_failure': True}
        self.assertIsNone(usage_and_cost(result, prices)['estimated_cost'])
        self.assertIsNone(usage_and_cost({}, None)['input_tokens'])

    def test_assessment_and_aggregated_generation_usage_counted_once(self):
        result = {'grounding_metrics': {'assessor_model': 'test', 'assessment_usage': {
            'input_tokens': 100, 'output_tokens': 10, 'input_token_details': {'cache_read': 20}}},
            'repro': {'generation_model': 'openai:test'},
            'generation_usage': {'input_tokens': 200, 'output_tokens': 30, 'cached_tokens': 0}}
        u = usage_and_cost(result, None)
        self.assertEqual(u['input_tokens'], 300)
        self.assertEqual(u['output_tokens'], 40)
        self.assertEqual(u['cached_input_tokens'], 20)

    def test_observer_delegates_without_mutating_arguments_or_ranking(self):
        retriever = MagicMock()
        hits = [hit('a1', 'a')]
        retriever.search_chunks.return_value = hits
        observer = RetrievalObserver(retriever)
        self.assertIs(observer.search_chunks('question', top_k=5), hits)
        retriever.search_chunks.assert_called_once_with('question', top_k=5)
        self.assertEqual(observer.calls[0]['hits'], hits)
        self.assertIsNot(observer.calls[0]['hits'], hits)

    def test_real_dataset_has_at_least_30_queries_and_checked_ground_truth(self):
        root = Path(__file__).resolve().parents[1]
        cases = [json.loads(line) for line in (root / 'data/insurance_retrieval_cases_v2.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertGreaterEqual(len(cases), 30)
        self.assertTrue({'fca', 'fos', 'near_ood', 'far_ood', 'version_comparison'} <= {c['category'] for c in cases})
        self.assertEqual({a for c in cases for a in c['expected_actions']}, {'generate', 'abstain', 'clarify'})
        validate_cases(cases, indexed_chunks(root / 'insurance_chroma_index_v3', 'insurance_chunks'))

    def test_every_answerable_case_has_source_passages_without_inventing_private_facts(self):
        root = Path(__file__).resolve().parents[1]
        cases = [json.loads(line) for line in (root / 'data/insurance_retrieval_cases_v2.jsonl').read_text(encoding='utf-8').splitlines()]
        supported = [c for c in cases if c['expected_actions'] == ['generate']]
        self.assertEqual(len(supported), 44)
        self.assertTrue(all(c['gold_evidence'] for c in supported))
        self.assertEqual(sum(not c['gold_evidence'] for c in cases), 10)
        versions = next(c for c in cases if c['id'] == 'V001')
        self.assertEqual({e['doc_id'] for e in versions['gold_evidence']}, set(versions['gold_document_ids']))

    def test_real_source_index_gaps_score_as_misses_instead_of_being_dropped(self):
        root = Path(__file__).resolve().parents[1]
        cases = [json.loads(line) for line in (root / 'data/insurance_retrieval_cases_v2.jsonl').read_text(encoding='utf-8').splitlines()]
        case = next(c for c in cases if c['id'] == 'Q003')
        self.assertIn('1,000,000', case['gold_evidence'][0]['quote'])
        self.assertEqual(case['gold_evidence'][0]['index_status'], 'source_passage_missing_from_index')
        self.assertEqual(case['gold_evidence'][0]['matching_chunk_ids'], [])
        self.assertEqual(retrieval_metrics(case, [], [10])['evidence_recall@10'], 0)


if __name__ == '__main__':
    unittest.main()
