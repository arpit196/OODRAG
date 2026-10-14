"""Paired routing contracts with scripted assessments/answers, not live quality scores."""
import copy
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.agent import InsuranceRAGAgent, StandardInsuranceRAGAgent
from evaluate_insurance_boundary import (evaluate_one, load_cases, make_record,
                                         source_anchor_checks, validate_source_anchors)
from test_evidence_gate import FakeRetriever, FakeAssessor
from src.evidence_gate import EvidenceAssessment, InsuranceEvidencePipeline
from evaluate_insurance_benchmark import indexed_chunks

ROOT = Path(__file__).resolve().parents[1]
DOC = 'lv_03904e7e51ee'
QUOTE = ('If your personal details shows you have to pay an excess, this is the amount '
         'you must pay as the first part of any claim.')
LOCATION = 'According to this LV policy wording, which document specifies my excess?'
MIXED = (LOCATION + ' What is my exact escape-of-water excess? '
         'My personal details document is unavailable.')
CONTEXT = {'request_type': 'general', 'policy_document_id': DOC, 'facts': {}}


def part(question, status='supported', evidence=None, missing=None, facts=None, conflicts=None):
    return {'question_part': question, 'status': status,
            'evidence': evidence if evidence is not None else [{'chunk_id': 'seed', 'quote': QUOTE}],
            'missing_evidence': missing or [], 'missing_customer_facts': facts or [],
            'conflicts': conflicts or []}


def record(verified):
    # Explicit synthetic fixture: no real customer's optional-cover selection.
    return {'policy_id': 'REGRESSION-001', 'policy_version': 'TEST-V1',
            'source_reference': {'policy_document_id': DOC, 'reference': 'Synthetic fixture only',
                                 'verified': verified},
            'optional_covers': {'accidental_damage': False}, 'excesses': {}}


class PairedRegressionTests(unittest.TestCase):
    def test_aviva_garden_assessment_with_real_separate_spans(self):
        # Real indexed evidence; scripted support labels, not a live LLM-quality score.
        doc = 'aviva_837d5b56fe01'
        query = 'What does Aviva give me if someone came into my garden and ruined crops? Do they cover it?.'
        chunks = [c for c in indexed_chunks(ROOT / 'insurance_chroma_index_v3', 'insurance_chunks')
                  if c['metadata']['doc_id'] == doc]
        quotes = ["If the plants or trees in the garden are lost or damaged by any of the following we’ll pay the re-landscaping costs to put things right:",
                  'Malicious people, vandals, riot, civil unrest, strikes, and labour or political disturbances.']
        references = [{'chunk_id': next(c['id'] for c in chunks if q in ' '.join(c['text'].split())), 'quote': q}
                      for q in quotes]
        report = {'assessed_question': query, 'all_question_parts_identified': True, 'parts': [
            part('What benefits does the wording describe for garden plants?', evidence=references),
            part('Do my crops qualify?', 'needs_clarification', [],
                 facts=['Are these personal garden plants or crops grown commercially?'])]}
        state = {'query': query, 'action': None, 'assessment': EvidenceAssessment.model_validate(report),
                 'chunks': chunks, 'steps': []}
        pipeline = InsuranceEvidencePipeline.__new__(InsuranceEvidencePipeline)
        pipeline._decide(state)
        self.assertEqual((state['action'], state['reason']), ('clarify', 'partial_support'))
        self.assertTrue(state['assessment_valid'])
        # A false completeness declaration remains an error, never silently overridden.
        state.update(action=None, assessment=EvidenceAssessment.model_validate(
            {**report, 'all_question_parts_identified': False}))
        pipeline._decide(state)
        self.assertEqual(state['reason'], 'invalid_or_incomplete_assessment')
        self.assertIn('incomplete_question_coverage', state['validation_issues'])
        # The related clause in another chunk cannot be stitched onto the benefit quote.
        import copy
        stitched = copy.deepcopy(report)
        stitched['parts'][0]['evidence'][0]['quote'] += ' ' + quotes[1]
        state.update(action=None, assessment=EvidenceAssessment.model_validate(stitched))
        pipeline._decide(state)
        self.assertEqual(state['reason'], 'invalid_or_incomplete_assessment')
        self.assertIn('invalid_evidence_reference', state['validation_issues'])

    def test_mixed_support_clarifies_with_quotes_without_generation(self):
        query = 'Explain the wording and confirm my claim.'
        parts = [part('Explain the wording'),
                 part('Confirm my claim', 'needs_clarification', [], facts=['cause of loss'])]
        guarded, _, assessor, generators = self.agents(query, parts)
        result = guarded.answer(query, customer_context=CONTEXT)
        self.assertEqual((result['action'], result['reason']), ('clarify', 'partial_support'))
        self.assertIn(QUOTE, result['answer'])
        self.assertIn('cause of loss', result['answer'])
        self.assertTrue(result['citations'])
        self.assertEqual(len(assessor.calls), 1)
        generators[0].generate.assert_not_called()

    def test_mixed_support_does_not_override_invalid_quotes_or_conflicts(self):
        query = 'Explain the wording and confirm my claim.'
        for bad_part, reason in [
            (part('Explain the wording', evidence=[{'chunk_id': 'seed', 'quote': 'Invented £500 cover'}]),
             'invalid_or_incomplete_assessment'),
            (part('Explain the wording', 'conflicting_evidence', conflicts=['Applicable sources disagree']),
             'conflicting_evidence')]:
            with self.subTest(reason=reason):
                guarded, _, _, generators = self.agents(query, [bad_part,
                    part('Confirm my claim', 'needs_clarification', [], facts=['cause of loss'])])
                result = guarded.answer(query, customer_context=CONTEXT)
                self.assertEqual((result['action'], result['reason']), ('abstain', reason))
                self.assertEqual(result['citations'], [])
                self.assertNotIn(QUOTE, result['answer'])
                generators[0].generate.assert_not_called()

    def agents(self, query, parts, error=None):
        retriever, assessor = FakeRetriever(), FakeAssessor()
        for chunk in retriever.chunks:
            if chunk['metadata']['doc_id'] == 'policy':
                chunk['metadata'].update(doc_id=DOC, page=19, version_label='TEST-V1',
                                         provider='LV=', title='LV= Home Insurance')
                chunk['text'] = QUOTE
        assessor.report = {'assessed_question': query, 'all_question_parts_identified': True, 'parts': parts}
        assessor.error = error
        detector = MagicMock()
        detector.score_query.return_value = {'is_ood': False, 'ood_band': 'id'}
        generators = [MagicMock(), MagicMock()]
        for generator in generators:
            generator.name = 'scripted-offline'
            generator.last_usage = {}
        guarded = InsuranceRAGAgent('insurance_chroma_index_v3', 'unused', detector=detector,
                                   retriever=retriever, assessor=assessor, generator=generators[0])
        standard = StandardInsuranceRAGAgent('unused', retriever=retriever, generator=generators[1])
        guarded._log = standard._log = MagicMock()
        return guarded, standard, assessor, generators

    def test_paired_generate_abstain_clarify_contracts(self):
        optional_query = 'Is accidental damage selected on my customer policy?'
        cases = [
            ('supported_location', LOCATION, CONTEXT, [part('Where is the excess specified?')],
             'generate', 'all_parts_supported', 'Your excess is specified in your personal details [1].'),
            ('missing_schedule_mixed', MIXED, CONTEXT,
             [part('Where is the excess specified?'),
              part('What is my exact excess?', 'insufficient_evidence', [], ['Applicable personal-details document'])],
             'clarify', 'partial_support',
             'Your excess is specified in your personal details [1]. Please provide those details for the exact amount.'),
            ('missing_claim_fact', 'Does my leak meet the cover conditions?',
             {**CONTEXT, 'request_type': 'customer_claim', 'policy_document_verified': True},
             [part('Does the leak meet the conditions?', 'needs_clarification', [], facts=['cause of leak'])],
             'clarify', 'missing_customer_facts', 'Please describe the cause of the leak.'),
            ('optional_unverified', optional_query,
             {**CONTEXT, 'customer_policy_record': record(False)}, [],
             'clarify', 'customer_policy_verification_required', 'Please provide a verified customer schedule.'),
            ('optional_verified_false', optional_query,
             {**CONTEXT, 'customer_policy_record': record(True)},
             [part('Is accidental damage selected?', evidence=[
                 {'chunk_id': 'customer_policy_record::REGRESSION-001', 'quote': '"accidental_damage": false'}])],
             'generate', 'all_parts_supported', 'Accidental damage is not selected according to your supplied record.'),
            ('unsupported_amount', 'What is my exact escape-of-water excess?', CONTEXT,
             [part('What is my exact excess?', 'insufficient_evidence', [], ['Applicable personal-details document'])],
             'abstain', 'insufficient_evidence', 'I cannot establish your exact excess without your personal details.'),
            ('conflict', LOCATION, CONTEXT,
             [part('Where is my excess specified?', 'conflicting_evidence', conflicts=['Applicable sources disagree'])],
             'abstain', 'conflicting_evidence', 'The available sources conflict; please verify the applicable documents.'),
        ]
        for name, query, context, parts, action, reason, answer in cases:
            with self.subTest(case=name):
                guarded, standard, assessor, generators = self.agents(query, parts)
                for generator in generators:
                    generator.generate.return_value = answer
                inputs = copy.deepcopy(context)
                guarded_result = guarded.answer(query, customer_context=context, include_evidence=True)
                baseline = standard.answer(query, customer_context=context, include_evidence=True)
                self.assertEqual(context, inputs)
                self.assertEqual((guarded_result['action'], guarded_result['reason']), (action, reason))
                self.assertFalse(guarded_result['decision_support']['technical_failure'])
                self.assertEqual(generators[0].generate.call_count, int(action == 'generate'))
                generators[1].generate.assert_called_once()
                self.assertEqual(baseline['action'], 'generate')
                self.assertEqual(baseline['answer'], answer)
                self.assertEqual(len(assessor.calls), int(name != 'optional_unverified'))
                for result in (guarded_result, baseline):
                    self.assertGreaterEqual(result['timing_metrics']['request_total_ms'], 0)
                if name == 'missing_schedule_mixed':
                    # Missing private values must not suppress retrieved public evidence.
                    support = guarded_result['decision_support']
                    self.assertEqual(support['assessment_status'], 'validated')
                    self.assertEqual(support['question_parts'][0]['status'], 'supported')
                    self.assertIn('Applicable personal-details document', support['missing_evidence'])
                    self.assertIn(QUOTE, guarded_result['answer'])
                    self.assertIn('[1]', guarded_result['answer'])
                    self.assertEqual(guarded_result['citations'][0]['id'], 'seed')
                    anchor = {'source_refs': [{'doc_id': DOC, 'anchor': QUOTE}]}
                    checks = source_anchor_checks(anchor, guarded_result)
                    self.assertTrue(checks[0]['retrieved'])
                    self.assertEqual(checks[0]['matching_chunks'][0]['page'], 19)
                    self.assertEqual(guarded_result['timing_metrics']['stages']['stage3']['status'], 'skipped')
                if name == 'optional_verified_false':
                    self.assertTrue(guarded_result['decision_support']['customer_policy_record_used'])

    def test_unsupported_generator_output_is_blocked_only_by_guarded_route(self):
        # A deliberately bad generator demonstrates the boundary, not actual LLM behavior.
        parts = [part('What is my exact excess?', 'insufficient_evidence', [], ['Customer schedule'])]
        guarded, standard, _, generators = self.agents(MIXED, parts)
        for generator in generators:
            generator.generate.return_value = 'Your escape-of-water excess is £999.'
        safe = guarded.answer(MIXED, customer_context=CONTEXT)
        baseline = standard.answer(MIXED, customer_context=CONTEXT)
        self.assertNotIn('£999', safe['answer'])
        generators[0].generate.assert_not_called()
        self.assertIn('£999', baseline['answer'])

    def test_assessment_timeout_is_an_error_not_successful_abstention(self):
        guarded, standard, _, generators = self.agents(LOCATION, [], TimeoutError('scripted failure'))
        generators[1].generate.return_value = 'Your excess is specified in your personal details [1].'
        case = load_cases(ROOT / 'data/insurance_focused_regression_cases.jsonl')[0]
        guarded_result = evaluate_one(guarded, case)
        baseline_result = evaluate_one(standard, case)
        row = make_record(case, 'guarded', 1, guarded_result)
        self.assertEqual(row['status'], 'error')
        self.assertFalse(row['guarded_action_matches'])
        self.assertEqual(guarded_result['reason'], 'stage2_error')
        generators[0].generate.assert_not_called()
        self.assertEqual(baseline_result['action'], 'generate')

    def test_wrong_chunk_id_is_not_promoted_to_supported_answer(self):
        parts = [part('Where is my excess specified?', evidence=[{'chunk_id': 'not-retrieved', 'quote': QUOTE}])]
        guarded, standard, _, generators = self.agents(LOCATION, parts)
        generators[1].generate.return_value = 'Your excess is specified in your personal details [1].'
        result = guarded.answer(LOCATION, customer_context=CONTEXT)
        self.assertEqual(result['reason'], 'invalid_or_incomplete_assessment')
        self.assertTrue(result['decision_support']['technical_failure'])
        self.assertEqual(result['decision_support']['question_parts'], [])
        generators[0].generate.assert_not_called()
        self.assertEqual(standard.answer(LOCATION, customer_context=CONTEXT)['action'], 'generate')

    def test_live_case_sources_and_all_three_guarded_actions_are_present(self):
        cases = load_cases(ROOT / 'data/insurance_focused_regression_cases.jsonl')
        self.assertEqual(len(cases), 5)
        self.assertEqual({action for case in cases for action in case['expected_actions']},
                         {'generate', 'abstain', 'clarify'})
        validate_source_anchors(cases)


if __name__ == '__main__':
    unittest.main()
