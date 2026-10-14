"""Customer records are caller input, not model-generated policy facts."""
import copy
import json
import unittest
from unittest.mock import MagicMock

from langchain_core.messages import AIMessageChunk
from pydantic import ValidationError
from src.evidence_gate import (CustomerPolicyRecord,
    InsuranceEvidencePipeline, LangChainEvidenceAssessor)
from test_evidence_gate import FakeRetriever, FakeAssessor, QUERY
import test_evidence_gate as fixtures


def record():
    return {"policy_id": "TEST-001", "policy_version": "2025-01",
            "source_reference": {"policy_document_id": "policy",
                                 "reference": "verified schedule page 1", "verified": True},
            "optional_covers": {"personal_possessions": True, "accidental_damage": False,
                                "home_emergency": None},
            "excesses": {"escape_of_water": {"amount": 500, "currency": "GBP"}}}


class CustomerPolicyRecordTests(unittest.TestCase):
    def setUp(self):
        self.retriever, self.assessor = FakeRetriever(), FakeAssessor()
        for c in self.retriever.chunks:
            c['metadata']['version_label'] = '2025-01'
        self.pipeline = InsuranceEvidencePipeline(self.retriever, self.assessor, ['policy'])
        self.context = {'request_type': 'customer_claim', 'customer_policy_record': record()}

    def test_verified_record_resolves_scope_and_is_citable(self):
        self.assessor.report['parts'][1]['evidence'] = [
            {'chunk_id': 'customer_policy_record::TEST-001',
             'quote': '"escape_of_water": {"amount": 500.0, "currency": "GBP"}'}]
        result = self.pipeline.invoke(QUERY, self.context)
        self.assertEqual(result['action'], 'generate')
        self.assertTrue(result['assessment_valid'])
        self.assertEqual(result['customer_policy_record']['excesses']['escape_of_water']['amount'], 500)
        self.assertTrue(result['decision_support']['customer_policy_record_used'])
        self.assertEqual(result['context'].policy_document_id, 'policy')
        self.assertTrue(result['context'].policy_document_verified)
        self.assertEqual(len(self.assessor.calls), 1)
        self.assertEqual(self.context['customer_policy_record'], record())
        self.assertEqual(result['chunks'][-1]['metadata']['source_reference'], 'verified schedule page 1')

    def test_unknown_and_unselected_covers_are_distinct(self):
        result = self.pipeline.invoke(QUERY, self.context)
        covers = result['customer_policy_record']['optional_covers']
        self.assertIs(covers['personal_possessions'], True)
        self.assertIs(covers['accidental_damage'], False)
        self.assertIsNone(covers['home_emergency'])
        self.assertNotIn('bicycle_cover', covers)

    def test_unverified_record_cannot_be_overridden_by_legacy_flag(self):
        self.context.update(policy_document_verified=True, policy_document_id='policy')
        self.context['customer_policy_record']['source_reference']['verified'] = False
        result = self.pipeline.invoke(QUERY, self.context)
        self.assertEqual(result['reason'], 'customer_policy_verification_required')
        self.assertEqual(result['action'], 'clarify')
        self.assertFalse(result['decision_support']['customer_policy_record_used'])
        self.assertFalse(self.retriever.calls)
        self.assertFalse(self.assessor.calls)

    def test_document_and_version_mismatches_stop_before_retrieval(self):
        for change in ('document', 'version'):
            with self.subTest(change=change):
                context = copy.deepcopy(self.context)
                if change == 'document':
                    context['policy_document_id'] = 'different_policy'
                else:
                    context['customer_policy_record']['policy_version'] = '2024-01'
                result = self.pipeline.invoke(QUERY, context)
                self.assertEqual(result['reason'], 'customer_policy_record_mismatch')
                self.assertEqual(result['action'], 'abstain')
                self.assertFalse(self.assessor.calls)
                self.assertFalse(self.retriever.calls)
                self.assertEqual(result['decision_support']['next_step'], 'verify_matching_customer_policy_record')

    def test_untrusted_document_cannot_be_authorized_by_record(self):
        self.context['customer_policy_record']['source_reference']['policy_document_id'] = 'different_policy'
        result = self.pipeline.invoke(QUERY, self.context)
        self.assertEqual(result['reason'], 'applicable_document_unavailable')
        self.assertFalse(self.assessor.calls)

    def test_invalid_values_are_rejected(self):
        for amount in (-1, float('nan'), float('inf'), '500', True):
            with self.subTest(amount=amount):
                bad = record()
                bad['excesses']['escape_of_water']['amount'] = amount
                with self.assertRaises(ValidationError):
                    CustomerPolicyRecord.model_validate(bad)
        for value in ('yes', 1):
            bad = record()
            bad['optional_covers']['personal_possessions'] = value
            with self.assertRaises(ValidationError):
                CustomerPolicyRecord.model_validate(bad)
        bad = record()
        bad['policy_version'] = ' '
        with self.assertRaises(ValidationError):
            CustomerPolicyRecord.model_validate(bad)

    def test_zero_excess_is_not_missing(self):
        self.context['customer_policy_record']['excesses']['escape_of_water']['amount'] = 0
        result = self.pipeline.invoke(QUERY, self.context)
        self.assertEqual(result['customer_policy_record']['excesses']['escape_of_water']['amount'], 0)

    def test_missing_excess_does_not_become_zero_and_clarifies(self):
        self.context['customer_policy_record']['excesses'] = {}
        part = self.assessor.report['parts'][1]
        part.update(status='needs_clarification', evidence=[], missing_customer_facts=['escape-of-water excess'])
        result = self.pipeline.invoke(QUERY, self.context)
        self.assertEqual(result['action'], 'clarify')
        self.assertEqual(result['customer_policy_record']['excesses'], {})
        self.assertEqual(result['decision_support']['missing_customer_facts'], ['escape-of-water excess'])

    def test_forged_quote_from_customer_record_fails_validation(self):
        self.assessor.report['parts'][1]['evidence'] = [
            {'chunk_id': 'customer_policy_record::TEST-001', 'quote': '"amount": 999'}]
        result = self.pipeline.invoke(QUERY, self.context)
        self.assertEqual(result['reason'], 'invalid_or_incomplete_assessment')
        self.assertIn('invalid_evidence_reference', result['decision_support']['validation_issues'])

    def test_record_reaches_live_assessor_payload(self):
        llm = MagicMock()
        llm.bind.return_value.stream.return_value = iter([
            AIMessageChunk(content=json.dumps(self.assessor.report))])
        self.pipeline.assessor = LangChainEvidenceAssessor(llm=llm)
        self.pipeline.invoke(QUERY, self.context)
        messages = llm.bind.return_value.stream.call_args.args[0]
        payload = json.loads(messages[1].content)
        self.assertEqual(payload['customer_context']['customer_policy_record']['policy_id'], 'TEST-001')
        self.assertEqual(payload['evidence'][-1]['chunk_id'], 'customer_policy_record::TEST-001')

    def test_legacy_context_still_works_without_record(self):
        result = self.pipeline.invoke(QUERY, {'request_type': 'general'})
        self.assertEqual(result['action'], 'generate')
        self.assertIsNone(result['customer_policy_record'])

    def test_record_preserved_when_assessor_fails(self):
        self.assessor.error = TimeoutError('offline timeout')
        result = self.pipeline.invoke(QUERY, self.context)
        self.assertEqual(result['reason'], 'stage2_error')
        self.assertEqual(result['customer_policy_record']['policy_id'], 'TEST-001')
        self.assertEqual(result['decision_support']['question_parts'], [])
        self.assertEqual(result['error_details']['failed_step'], 'part_assessment')

    def test_conflicting_record_and_wording_is_reported(self):
        part = self.assessor.report['parts'][1]
        part.update(status='conflicting_evidence', conflicts=['Schedule and wording give different excesses'])
        part['evidence'].append({'chunk_id': 'customer_policy_record::TEST-001', 'quote': '"amount": 500.0'})
        result = self.pipeline.invoke(QUERY, self.context)
        self.assertEqual(result['reason'], 'conflicting_evidence')
        self.assertEqual(result['decision_support']['conflicts'], part['conflicts'])

    def test_invalid_record_input_returns_validation_diagnostics(self):
        self.context['customer_policy_record']['excesses']['escape_of_water']['amount'] = -1
        result = self.pipeline.invoke(QUERY, self.context)
        self.assertEqual(result['reason'], 'stage2_error')
        self.assertEqual(result['error_details']['failed_step'], 'context_validation')
        self.assertIsNone(result['customer_policy_record'])
        self.assertFalse(self.assessor.calls)


class CustomerPolicyAgentTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.InsuranceAgentTests()
        fixture.setUp()
        self.agent = fixture.agent
        self.generator = fixture.generator

    def test_record_returned_on_generation_and_sent_to_generator(self):
        value = record()
        value['source_reference']['policy_document_id'] = 'lv_03904e7e51ee'
        result = self.agent.answer(QUERY, customer_context={
            'request_type': 'customer_claim', 'customer_policy_record': value})
        self.assertEqual(result['customer_policy_record']['policy_id'], 'TEST-001')
        self.assertIn('TEST-001', self.generator.generate.call_args.args[0])
        self.assertTrue(any(c['metadata'].get('evidence_type') == 'customer_policy_record'
                            for c in self.generator.generate.call_args.kwargs['chunks']))

    def test_record_returned_on_early_clarification_without_generation(self):
        value = record()
        value['source_reference']['verified'] = False
        result = self.agent.answer(QUERY, customer_context={'customer_policy_record': value})
        self.assertEqual(result['action'], 'clarify')
        self.assertEqual(result['customer_policy_record']['policy_id'], 'TEST-001')
        self.assertIn('verified customer-policy source reference', result['answer'])
        self.generator.generate.assert_not_called()

    def test_stage1_rejection_does_not_claim_record_was_checked(self):
        self.agent.detector.score_query.return_value = {'is_ood': True}
        result = self.agent.answer(QUERY, customer_context={'customer_policy_record': record()})
        self.assertIsNone(result['customer_policy_record'])
        self.assertEqual(result['decision_support']['applicability']['status'], 'not_checked')
