"""Batching preserves independent citation checks and rejects incomplete results."""
import json
import unittest
from unittest.mock import MagicMock

from src.guardrails import verify_and_correct_citations, verify_claim_entailments_batch


class BatchedCitationTests(unittest.TestCase):
    def test_same_decisions_as_individual_checks(self):
        chunks = [{"text": "Supports claim", "cite_id": 1}, {"text": "Unrelated", "cite_id": 2}]
        scalar = MagicMock()
        scalar.generate.side_effect = ["YES", "NO"]
        batched = MagicMock()
        batched.generate.return_value = json.dumps({"checks": [
            {"check_id": 0, "supported": True}, {"check_id": 1, "supported": False}]})
        original = verify_and_correct_citations("Claim [1] [2].", chunks, scalar, True)
        optimized = verify_and_correct_citations("Claim [1] [2].", chunks, batched, True, True)
        self.assertEqual(original, optimized)
        self.assertEqual(scalar.generate.call_count, 2)
        self.assertEqual(batched.generate.call_count, 1)

    def test_duplicate_pairs_are_checked_once_and_sources_reused(self):
        evaluator = MagicMock()
        evaluator.generate.return_value = '{"checks":[{"check_id":0,"supported":true},{"check_id":1,"supported":false}]}'
        result = verify_claim_entailments_batch([("A", "source"), ("A", "source"), ("B", "source")], evaluator)
        self.assertEqual(result, {("A", "source"): True, ("B", "source"): False})
        payload = json.loads(evaluator.generate.call_args.args[0].split("\n", 1)[1])
        self.assertEqual(len(payload["sources"]), 1)
        self.assertEqual(len(payload["checks"]), 2)

    def test_unknown_citations_do_not_trigger_an_api_call(self):
        evaluator = MagicMock()
        result = verify_and_correct_citations("Claim [99].", [], evaluator, True, True)
        evaluator.generate.assert_not_called()
        self.assertEqual(result["corrected_answer"], "")

    def test_uncited_claims_are_checked_when_strict(self):
        evaluator = MagicMock()
        evaluator.generate.return_value = '{"checks":[{"check_id":0,"supported":false}]}'
        result = verify_and_correct_citations("Unsupported claim.", [{"text": "source", "cite_id": 1}],
                                             evaluator, True, True)
        self.assertEqual(result["corrected_answer"], "")

    def test_invalid_or_missing_verdicts_raise_instead_of_skipping_checks(self):
        invalid = ["YES", '{"checks":[]}', '{"checks":[{"check_id":0,"supported":"true"}]}',
                   '{"checks":[{"check_id":2,"supported":true}]}',
                   '{"checks":[{"check_id":0,"supported":true},{"check_id":0,"supported":false}]}']
        for response in invalid:
            with self.subTest(response=response):
                evaluator = MagicMock()
                evaluator.generate.return_value = response
                with self.assertRaises(ValueError):
                    verify_claim_entailments_batch([("A", "source")], evaluator)

    def test_large_check_sets_use_bounded_batches(self):
        evaluator = MagicMock()
        def respond(prompt, **kwargs):
            data = json.loads(prompt.split("\n", 1)[1])
            return json.dumps({"checks": [{"check_id": c["check_id"], "supported": True}
                                           for c in data["checks"]]})
        evaluator.generate.side_effect = respond
        result = verify_claim_entailments_batch([(str(i), "source") for i in range(17)], evaluator)
        self.assertEqual(len(result), 17)
        self.assertEqual(evaluator.generate.call_count, 2)


if __name__ == "__main__":
    unittest.main()
