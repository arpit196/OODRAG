import unittest
from evaluate_insurance_boundary import direct_query_case, make_record, summarize


class DirectQueryTests(unittest.TestCase):
    def test_defaults_do_not_fabricate_gold_or_customer_verification(self):
        case = direct_query_case(' Question? ', 'policy')
        self.assertEqual(case['query'], 'Question?')
        self.assertEqual(case['source_refs'], [])
        self.assertEqual(case['expected_actions'], [])
        self.assertEqual(case['customer_context']['request_type'], 'general')
        self.assertNotIn('policy_document_verified', case['customer_context'])
        with self.assertRaises(ValueError):
            direct_query_case('  ')

    def test_ungraded_query_is_excluded_from_action_agreement(self):
        result = dict(action='generate', answer='Answer', execution_path=[], evaluation_wall_ms=10)
        ungraded = make_record(direct_query_case('q'), 'guarded', 1, result)
        self.assertIsNone(ungraded['guarded_action_matches'])
        entry = summarize([ungraded])['variants']['guarded']
        self.assertIsNone(entry['action_agreement_rate'])
        graded = make_record(direct_query_case('q', expected_action='generate'), 'guarded', 1, result)
        entry = summarize([ungraded, graded])['variants']['guarded']
        self.assertEqual(entry['action_agreement_rate'], 1)
        self.assertEqual(entry['action_agreement_measured_count'], 1)


if __name__ == '__main__':
    unittest.main()
