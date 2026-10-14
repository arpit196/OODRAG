import copy
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from evaluate_insurance_benchmark import indexed_chunks
from src.evidence_gate import CustomerContext, InsuranceEvidencePipeline


class DefinitionLookupTests(unittest.TestCase):
    def setUp(self):
        self.definition = {'id': 'definition', 'text': 'Accidental damage means sudden damage.',
                           'metadata': {'doc_id': 'lv', 'source_type': 'policy', 'section': 'Definitions'}}
        self.noise = {'id': 'coverage', 'text': 'A flattened list of cover and exclusions.',
                      'metadata': {'doc_id': 'lv', 'source_type': 'policy', 'section': 'Cover'}}
        self.other = copy.deepcopy(self.definition)
        self.other.update(id='wrong_policy')
        self.other['metadata']['doc_id'] = 'other'
        self.retriever = MagicMock()
        self.retriever.chunks = [self.noise, self.definition, self.other]
        self.retriever.search_chunks.return_value = [self.noise]
        self.pipeline = InsuranceEvidencePipeline(self.retriever, MagicMock(), ['lv'])

    def retrieve(self, query):
        return self.pipeline._retrieve({'query': query, 'action': None, 'steps': [],
            'top_k': 5, 'query_embedding': None, 'allowed_ids': ['lv'],
            'context': CustomerContext(request_type='general'), 'retrieval_diagnostics': {}})

    def test_recovers_definition_first_without_wrong_policy_or_extra_search(self):
        state = self.retrieve('How does LV define accidental damage?')
        self.assertEqual([c['id'] for c in state['chunks']], ['definition', 'coverage'])
        self.retriever.search_chunks.assert_called_once()

    def test_non_definition_question_keeps_existing_retrieval(self):
        state = self.retrieve('What accidental damage is covered?')
        self.assertEqual([c['id'] for c in state['chunks']], ['coverage'])

    def test_q005_recovers_actual_page_seven_definition(self):
        index = Path('insurance_chroma_index_v3')
        if not index.exists():
            self.skipTest('Local insurance index unavailable')
        self.retriever.chunks = indexed_chunks(index, 'insurance_chunks')
        state = {'query': 'How does LV= define accidental damage?', 'action': None, 'steps': [],
                 'top_k': 10, 'query_embedding': None, 'allowed_ids': ['lv_03904e7e51ee'],
                 'context': CustomerContext(request_type='general'), 'retrieval_diagnostics': {}}
        # Simulate the recorded miss: coverage is retrieved instead of the glossary.
        self.retriever.search_chunks.return_value = []
        self.pipeline._retrieve(state)
        first = state['chunks'][0]
        self.assertEqual(first['id'], 'lv_03904e7e51ee__476fba66a133__12')
        self.assertEqual(first['metadata']['page'], 7)
        self.assertIn('damage caused suddenly by external means which is not expected and not deliberate.', first['text'])


if __name__ == '__main__':
    unittest.main()
