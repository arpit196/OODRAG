"""Offline comparison tests; no API calls, model downloads or Chroma needed."""
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from evaluate_insurance_boundary import (load_cases, validate_source_anchors, evaluate_one,
                                         make_record, summarize)

ROOT = Path(__file__).resolve().parents[1]


class BoundaryDatasetTests(unittest.TestCase):
    def test_case_counts_sources_and_label_distinction(self):
        cases = load_cases(ROOT / "data/insurance_boundary_cases.jsonl")
        self.assertEqual(len(cases), 28)
        self.assertEqual(sum(c["domain_label"] == "near_ood" for c in cases), 24)
        self.assertEqual(sum(c["domain_label"] == "mixed" for c in cases), 2)
        self.assertEqual(sum(c["domain_label"] == "id" for c in cases), 2)
        validate_source_anchors(cases)
        self.assertTrue(all(not c["customer_context"].get("policy_document_verified", False) for c in cases))

    def test_positive_controls_are_supported_id_targets(self):
        cases = load_cases(ROOT / "data/insurance_boundary_positive_controls.jsonl")
        self.assertEqual(len(cases), 4)
        self.assertTrue(all(c["expected_actions"] == ["generate"] for c in cases))
        validate_source_anchors(cases)

    def test_errors_are_not_counted_as_successful_abstentions(self):
        case = load_cases(ROOT / "data/insurance_boundary_cases.jsonl")[0]
        agent = MagicMock()
        agent.answer.side_effect = TimeoutError()
        result = evaluate_one(agent, case)
        row = make_record(case, "guarded", 1, result)
        self.assertEqual(row["status"], "error")
        self.assertFalse(row["guarded_action_matches"])
        summary = summarize([row])["variants"]["guarded"]
        self.assertEqual(summary["errors"], 1)
        self.assertIsNone(summary["action_agreement_rate"])

    def test_standard_generation_action_is_not_graded_as_hallucination(self):
        case = load_cases(ROOT / "data/insurance_boundary_cases.jsonl")[0]
        result = {"action": "generate", "answer": "The documents do not answer this.", "evaluation_wall_ms": 10}
        row = make_record(case, "standard", 1, result)
        self.assertIsNone(row["guarded_action_matches"])
        self.assertEqual(row["review_verdict"], "unreviewed")
        self.assertNotIn("action_agreement_rate", summarize([row])["variants"]["standard"])

    def test_stage2_diagnostic_bypasses_agent_answer_and_serializes_evidence(self):
        case = load_cases(ROOT / "data/insurance_boundary_cases.jsonl")[0]
        agent = MagicMock()
        agent.evidence_pipeline.invoke.return_value = {"action": "abstain", "reason": "insufficient_evidence",
            "assessment": None, "chunks": [], "assessment_usage": {}, "steps": [], "timing_metrics": {}}
        result = evaluate_one(agent, case, stage2_only=True)
        agent.answer.assert_not_called()
        self.assertEqual(result["execution_path"], ["stage2_diagnostic"])
        self.assertEqual(result["action"], "abstain")


class StandardInsuranceTests(unittest.TestCase):
    def setUp(self):
        from src.agent import StandardInsuranceRAGAgent
        self.retriever = MagicMock()
        self.chunk = {"id": "lv_chunk", "text": "A home policy clause.",
                      "metadata": {"doc_id": "lv_03904e7e51ee", "source_type": "policy"}}
        self.retriever.search_chunks.return_value = [self.chunk]
        self.generator = MagicMock()
        self.generator.name = "offline"
        self.generator.last_usage = {}
        self.generator.generate.return_value = "I cannot answer this from the supplied documents."
        self.agent = StandardInsuranceRAGAgent("unused", generator=self.generator, retriever=self.retriever)

    def test_standard_retrieves_same_selected_document_and_allows_textual_abstention(self):
        result = self.agent.answer("What does my motor policy cover?", customer_context={
            "request_type": "general", "policy_document_id": "lv_03904e7e51ee"})
        self.assertEqual(result["action"], "generate")
        self.assertIn("cannot answer", result["answer"])
        self.assertEqual(self.retriever.search_chunks.call_args.kwargs["allowed_document_ids"], ["lv_03904e7e51ee"])
        self.assertFalse(result["grounding_metrics"]["enabled"])
        self.assertTrue(self.generator.generate.call_args.kwargs["batch_verification"])
        self.assertNotIn("domain generalization", self.generator.generate.call_args.args[0])

    def test_matched_evidence_skips_retrieval_and_does_not_mutate_input(self):
        result = self.agent.answer("Question", customer_context={"request_type": "general"},
                                   evidence_chunks=[self.chunk], include_evidence=True)
        self.retriever.search_chunks.assert_not_called()
        self.assertEqual(result["baseline_evidence_mode"], "matched_guarded_evidence")
        self.assertNotIn("cite_id", self.chunk)
        self.assertEqual(result["evidence_chunks"][0]["cite_id"], 1)

    def test_api_failure_is_an_error_not_extractive_fallback(self):
        self.generator.generate.side_effect = TimeoutError()
        with self.assertRaises(TimeoutError):
            self.agent.answer("Question", customer_context={"request_type": "general"})


if __name__ == "__main__":
    unittest.main()
