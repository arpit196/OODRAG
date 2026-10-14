"""Stage 2 regression tests use real LCEL, fake retrieval and no paid calls."""
import copy
import json
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.evidence_gate import (CustomerContext, EvidenceAssessment,
                               InsuranceEvidencePipeline, LangChainEvidenceAssessor)
from langchain_core.messages import AIMessageChunk


QUERY = "Is the leak covered and what is the excess?"


def chunk(cid, index=0, doc_id="policy", text="Sudden leaks are covered. The excess is £250."):
    return {"id": cid, "text": text, "metadata": {
        "doc_id": doc_id, "source_type": "policy", "chunk_index": index,
        "title": "Policy", "section": "Escape of water"}}


def supported_report():
    return {"assessed_question": QUERY, "all_question_parts_identified": True,
            "parts": [{"question_part": part, "status": "supported",
                       "evidence": [{"chunk_id": "seed", "quote": quote}],
                       "missing_evidence": [], "missing_customer_facts": [], "conflicts": []}
                      for part, quote in (("Is the leak covered?", "Sudden leaks are covered."),
                                          ("What is the excess?", "The excess is £250."))]}


def span_report():
    report = supported_report()
    for index, part in enumerate(report["parts"]):
        part["evidence"] = [{"chunk_id": "seed", "span_index": index}]
    return report


class FakeRetriever:
    def __init__(self):
        self.model = MagicMock()
        self.model.encode.side_effect = lambda texts, **kwargs: np.tile([1., 0.], (len(texts), 1))
        self.chunks = [chunk("before", 0), chunk("seed", 1), chunk("after", 2),
                       chunk("other", 2, "different_policy")]
        self.calls = []

    def search_chunks(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return [copy.deepcopy(self.chunks[1])]


class FakeAssessor:
    model = "offline"

    def __init__(self):
        self.report = supported_report()
        self.calls = []
        self.error = None

    def assess(self, query, context, chunks):
        self.calls.append(chunks)
        if self.error:
            raise self.error
        return self.report, {"input_tokens": 100, "output_tokens": 20}


class EvidenceGateTests(unittest.TestCase):
    def setUp(self):
        self.retriever, self.assessor = FakeRetriever(), FakeAssessor()
        self.pipeline = InsuranceEvidencePipeline(self.retriever, self.assessor, ["policy"])
        self.context = {"request_type": "customer_claim", "policy_document_id": "policy",
                        "policy_document_verified": True, "facts": {"cause": "sudden pipe leak"}}

    def run_gate(self):
        return self.pipeline.invoke(QUERY, self.context)

    def test_full_support_expands_same_document_and_allows_generation(self):
        result = self.run_gate()
        self.assertEqual(result["action"], "generate")
        self.assertEqual(result["steps"], ["applicability", "hybrid_retrieval", "evidence_expansion",
                                           "part_assessment", "deterministic_decision"])
        self.assertEqual({c["id"] for c in result["chunks"]}, {"before", "seed", "after"})
        self.assertEqual(len(self.assessor.calls), 1)
        self.assertTrue(all(kwargs["allowed_document_ids"] == ["policy"]
                            for _, kwargs in self.retriever.calls))
        self.assertTrue(all(entry["status"] == "completed" and entry["duration_ms"] >= 0
                            for entry in result["timing_metrics"].values()))

    def test_missing_policy_clarifies_without_retrieval_or_llm(self):
        result = self.pipeline.invoke(QUERY)
        self.assertEqual(result["reason"], "verified_policy_required")
        self.assertEqual(result["action"], "clarify")
        self.assertFalse(self.retriever.calls)
        self.assertFalse(self.assessor.calls)
        self.assertEqual(result["timing_metrics"]["applicability"]["status"], "completed")
        self.assertEqual(result["timing_metrics"]["hybrid_retrieval"],
                         {"duration_ms": None, "status": "skipped"})

    def test_unverified_public_wording_does_not_establish_customer_cover(self):
        self.context["policy_document_verified"] = False
        self.assertEqual(self.run_gate()["action"], "clarify")

    def test_missing_document_abstains(self):
        self.context["policy_document_id"] = "unknown"
        self.assertEqual(self.run_gate()["reason"], "applicable_document_unavailable")
        self.assertFalse(self.retriever.calls)

    def test_general_question_does_not_require_customer_policy(self):
        self.assertEqual(self.pipeline.invoke(QUERY, {"request_type": "general"})["action"], "generate")

    def test_partial_support_clarifies(self):
        self.assessor.report["parts"][1].update(status="insufficient_evidence", missing_evidence=["excess"])
        self.assertEqual(self.run_gate()["action"], "clarify")

    def test_missing_fact_clarifies(self):
        self.assessor.report["parts"][0].update(status="needs_clarification", missing_customer_facts=["leak cause"])
        self.assertEqual(self.run_gate()["reason"], "partial_support")

    def test_missing_evidence_takes_priority_over_clarification(self):
        self.assessor.report["parts"][0].update(status="needs_clarification", missing_customer_facts=["cause"])
        self.assessor.report["parts"][1]["missing_evidence"] = ["excess"]
        self.assertEqual(self.run_gate()["action"], "abstain")

    def test_conflicts_reject_even_if_model_claims_support(self):
        self.assessor.report["parts"][0]["conflicts"] = ["Contradictory clauses"]
        self.assertEqual(self.run_gate()["reason"], "conflicting_evidence")

    def test_unknown_citation_and_fabricated_quote_reject(self):
        for ref in ({"chunk_id": "other", "quote": "Sudden leaks are covered."},
                    {"chunk_id": "seed", "quote": "All leaks are covered."},
                    {"chunk_id": "seed", "quote": ""}):
            with self.subTest(ref=ref):
                self.assessor.report["parts"][0]["evidence"] = [ref]
                self.assertEqual(self.run_gate()["reason"], "invalid_or_incomplete_assessment")

    def test_no_support_references_reject(self):
        self.assessor.report["parts"][0]["evidence"] = []
        self.assertEqual(self.run_gate()["action"], "abstain")

    def test_unassessed_question_or_empty_parts_reject(self):
        for change in ({"assessed_question": "different"}, {"parts": []},
                       {"all_question_parts_identified": False}):
            with self.subTest(change=change):
                self.assessor.report = {**supported_report(), **change}
                self.assertEqual(self.run_gate()["action"], "abstain")

    def test_timeout_and_malformed_assessment_fail_closed(self):
        self.assessor.error = TimeoutError()
        result = self.run_gate()
        self.assertEqual(result["reason"], "stage2_error")
        self.assertEqual(result["timing_metrics"]["part_assessment"]["status"], "error")
        self.assertIsNotNone(result["timing_metrics"]["part_assessment"]["duration_ms"])
        self.assertEqual(result["timing_metrics"]["deterministic_decision"]["status"], "skipped")
        self.assessor.error = None
        self.assessor.report = {"parts": "bad"}
        self.assertEqual(self.run_gate()["action"], "abstain")

    def test_retrieval_error_fail_closed(self):
        self.retriever.search_chunks = MagicMock(side_effect=RuntimeError("retrieval down"))
        self.assertEqual(self.run_gate()["reason"], "stage2_error")
        self.assertFalse(self.assessor.calls)

    def test_invalid_context_and_retrieval_limit_fail_closed(self):
        self.assertEqual(self.pipeline.invoke(QUERY, {"request_type": "invented"})["action"], "abstain")
        self.assertEqual(self.pipeline.invoke(QUERY, self.context, top_k=0)["action"], "abstain")
        self.pipeline.invoke(QUERY, self.context, top_k=3)
        self.assertEqual(self.retriever.calls[0][1]["top_k"], 3)

    def test_budget_rejects_without_silent_truncation(self):
        self.pipeline.max_evidence_chars = 10
        self.assertEqual(self.run_gate()["reason"], "evidence_budget_exceeded")
        self.assertFalse(self.assessor.calls)

    def test_structured_assessor_parsing_failure_and_usage(self):
        llm = MagicMock()
        llm.bind.return_value.stream.return_value = [AIMessageChunk(
            content=json.dumps(span_report()),
            usage_metadata={"input_tokens": 42, "output_tokens": 20, "total_tokens": 62})]
        assessor = LangChainEvidenceAssessor(llm=llm)
        report, usage = assessor.assess(QUERY, CustomerContext(request_type="general"), [chunk("seed")])
        self.assertEqual(usage["input_tokens"], 42)
        self.assertEqual(len(report.parts), 2)
        llm.bind.return_value.stream.return_value = [AIMessageChunk(content="invalid JSON")]
        with self.assertRaises(ValueError):
            assessor.assess(QUERY, CustomerContext(), [chunk("seed")])

    def test_assessment_retains_evidence_and_policy_metadata_without_ingestion_paths(self):
        llm = MagicMock()
        llm.bind.return_value.stream.return_value = [AIMessageChunk(content=json.dumps(span_report()))]
        assessor = LangChainEvidenceAssessor(llm=llm)
        evidence = chunk("seed")
        evidence["metadata"].update(raw_path="raw.pdf", extracted_sha256="hash", version_label="2025")
        assessor.assess(QUERY, CustomerContext(request_type="general"), [evidence])
        schema = llm.bind.return_value.stream.call_args.kwargs['response_format']['json_schema']['schema']
        self.assertEqual(schema['properties']['assessed_question']['enum'], [QUERY])
        reference = schema['$defs']['EvidenceReference']['anyOf'][0]['properties']
        self.assertNotIn('quote', reference)
        self.assertEqual(reference['chunk_id']['enum'], ['seed'])
        self.assertEqual(reference['span_index']['enum'], [0, 1])
        self.assertNotIn('enum', assessor.response_schema['properties']['assessed_question'])
        messages = llm.bind.return_value.stream.call_args.args[0]
        payload = json.loads(messages[1].content)
        self.assertEqual("".join(span["text"] for span in payload["evidence"][0]["spans"]), evidence["text"])
        meta = payload["evidence"][0]["metadata"]
        self.assertEqual(meta["version_label"], "2025")
        self.assertNotIn("raw_path", meta)
        self.assertNotIn("extracted_sha256", meta)
        self.assertIn("raw_path", evidence["metadata"])

    def test_expansion_queries_are_encoded_once_as_a_batch(self):
        self.run_gate()
        self.retriever.model.encode.assert_called_once()
        queries = self.retriever.model.encode.call_args.args[0]
        self.assertEqual(queries, [f"{QUERY} {topic}" for topic in
                                  ("definitions", "general exclusions", "policy conditions")])
        self.assertEqual(len(self.retriever.calls), 4)

    def test_seed_embedding_is_forwarded_to_retriever(self):
        vector = np.array([1., 0.])
        self.pipeline.invoke(QUERY, self.context, query_embedding=vector)
        self.assertIs(self.retriever.calls[0][1]["query_embedding"], vector)

    def test_decision_support_explains_partial_support_without_extra_calls(self):
        self.assessor.report["parts"][1]["missing_evidence"] = ["customer policy schedule excess"]
        result = self.run_gate()
        details = result["decision_support"]
        self.assertEqual(details["category"], "partial_support")
        self.assertEqual(details["assessment_status"], "validated")
        self.assertEqual(details["question_parts"][0]["status"], "supported")
        self.assertEqual(details["question_parts"][1]["status"], "insufficient_evidence")
        self.assertEqual(details["missing_evidence"], ["customer policy schedule excess"])
        self.assertEqual(details["documents"][0]["doc_id"], "policy")
        self.assertEqual(details["next_step"], "provide_missing_evidence_or_customer_facts")
        self.assertEqual(len(self.assessor.calls), 1)
        self.assertEqual(len(self.retriever.calls), 4)

    def test_policy_clarification_lists_requirements_without_assessment(self):
        details = self.pipeline.invoke(QUERY)["decision_support"]
        self.assertEqual(details["category"], "missing_customer_context")
        self.assertIn("policy_document_id", details["applicability"]["missing_requirements"])
        self.assertEqual(details["assessment_status"], "not_run")
        self.assertFalse(self.assessor.calls)

    def test_customer_claim_cannot_use_regulatory_document_as_policy(self):
        for c in self.retriever.chunks:
            c["metadata"]["source_type"] = "regulator"
        pipeline = InsuranceEvidencePipeline(self.retriever, self.assessor, ["policy"])
        result = pipeline.invoke(QUERY, self.context)
        self.assertEqual(result["reason"], "applicable_document_unavailable")
        self.assertFalse(self.retriever.calls)
        self.assertFalse(self.assessor.calls)

    def test_out_of_scope_retrieval_fails_before_assessment(self):
        self.retriever.search_chunks = MagicMock(return_value=[chunk("wrong", doc_id="other")])
        result = self.run_gate()
        self.assertEqual(result["reason"], "evidence_scope_mismatch")
        self.assertTrue(result["decision_support"]["technical_failure"])
        self.assertFalse(self.assessor.calls)
        self.assertEqual(result["decision_support"]["documents"], [])

    def test_invalid_assessment_is_not_reported_as_missing_evidence(self):
        self.assessor.report["parts"][0]["evidence"][0]["quote"] = "Invented clause"
        details = self.run_gate()["decision_support"]
        self.assertEqual(details["category"], "assessment_error")
        self.assertTrue(details["technical_failure"])
        self.assertIn("invalid_evidence_reference", details["validation_issues"])
        self.assertEqual(details["question_parts"], [])

    def test_duplicate_question_parts_fail_closed(self):
        self.assessor.report["parts"][1]["question_part"] = "  IS THE LEAK COVERED?  "
        self.assertIn("empty_or_duplicate_question_parts",
                      self.run_gate()["decision_support"]["validation_issues"])

    def test_timeout_is_assessment_error_but_retrieval_failure_is_pipeline_error(self):
        self.assessor.error = TimeoutError()
        self.assertEqual(self.run_gate()["decision_support"]["category"], "assessment_error")
        self.retriever.search_chunks = MagicMock(side_effect=RuntimeError())
        self.assertEqual(self.run_gate()["decision_support"]["category"], "pipeline_error")

    def test_assessor_has_output_headroom_without_retries(self):
        with patch("langchain_openai.ChatOpenAI") as chat:
            assessor = LangChainEvidenceAssessor()
        self.assertEqual(chat.call_args.kwargs["max_tokens"], 3072)
        self.assertEqual(chat.call_args.kwargs["max_retries"], 0)
        self.assertEqual(assessor.max_output_tokens, 3072)
        with self.assertRaises(ValueError):
            LangChainEvidenceAssessor(llm=MagicMock(), max_output_tokens=0)

    def test_sdk_length_error_remains_identifiable_before_and_during_stream(self):
        from openai import LengthFinishReasonError
        from types import SimpleNamespace
        error = LengthFinishReasonError(completion=SimpleNamespace(usage=None))
        for during_stream in (False, True):
            with self.subTest(during_stream=during_stream):
                llm = MagicMock()
                stream = llm.bind.return_value.stream
                if during_stream:
                    def broken_stream():
                        yield AIMessageChunk(content='{"parts":[')
                        raise error
                    stream.return_value = broken_stream()
                else:
                    stream.side_effect = error
                assessor = LangChainEvidenceAssessor(llm=llm)
                pipeline = InsuranceEvidencePipeline(self.retriever, assessor, ["policy"])
                result = pipeline.invoke(QUERY, self.context)
                self.assertEqual(result["reason"], "stage2_error")
                self.assertEqual(result["error_type"], "LengthFinishReasonError")
                self.assertEqual(result["error_code"], "assessment_output_limit")
                self.assertEqual(result["assessment_output_token_limit"], 3072)
                self.assertEqual(result["decision_support"]["category"], "assessment_error")
                self.assertEqual(result["decision_support"]["question_parts"], [])
                stream.assert_called_once()

    def test_length_finish_reason_never_accepts_partial_structured_output(self):
        llm = MagicMock()
        llm.bind.return_value.stream.return_value = [
            AIMessageChunk(content="partial", response_metadata={"finish_reason": "length"})]
        pipeline = InsuranceEvidencePipeline(self.retriever, LangChainEvidenceAssessor(llm=llm), ["policy"])
        result = pipeline.invoke(QUERY, self.context)
        self.assertEqual(result["action"], "abstain")
        self.assertEqual(result["error_code"], "assessment_output_limit")
        self.assertIsNone(result["assessment"])
        self.assertEqual(result["error_details"]["partial_output_head"], "partial")
        self.assertEqual(result["error_details"]["finish_reason"], "length")

    def test_length_exception_captures_message_completion_and_usage(self):
        import json
        from openai import LengthFinishReasonError
        from types import SimpleNamespace
        completion = SimpleNamespace(
            id="completion-test", model="gpt-4.1-mini",
            usage=SimpleNamespace(prompt_tokens=5000, completion_tokens=3072, total_tokens=8072),
            choices=[SimpleNamespace(finish_reason="length",
                                     message=SimpleNamespace(content='{"parts":[' + "x" * 10000))])
        self.assessor.error = LengthFinishReasonError(completion=completion)
        result = self.run_gate()
        details = result["error_details"]
        self.assertIn("length limit was reached", details["error_message"])
        self.assertEqual(details["failed_step"], "part_assessment")
        self.assertEqual(details["response_id"], "completion-test")
        self.assertEqual(details["token_usage"]["output_tokens"], 3072)
        self.assertEqual(result["assessment_usage"], details["token_usage"])
        self.assertEqual(len(details["partial_output_head"]), 6000)
        self.assertEqual(len(details["partial_output_tail"]), 2000)
        self.assertTrue(details["partial_output_excerpted"])
        self.assertTrue(details["partial_output_untrusted"])
        self.assertEqual(result["action"], "abstain")
        json.dumps(result["error_details"])

    def test_hard_01_complete_assessment_rejects_missing_personal_excess(self):
        query = ("Explain trace-and-access cover and its limit, then tell me my exact "
                 "escape-of-water excess. My personal details document is unavailable.")
        text = "We’ll pay up to £5,000 towards finding the source of a water or oil leak."
        self.retriever.chunks[1]["text"] = text
        self.assessor.report = {
            "assessed_question": query, "all_question_parts_identified": True,
            "parts": [
                {"question_part": "Explain trace-and-access cover and its limit", "status": "supported",
                 "evidence": [{"chunk_id": "seed", "quote": text}],
                 "missing_evidence": [], "missing_customer_facts": [], "conflicts": []},
                {"question_part": "Tell me my exact escape-of-water excess", "status": "insufficient_evidence",
                 "evidence": [], "missing_evidence": ["Personal details document showing the applicable excess"],
                 "missing_customer_facts": [], "conflicts": []}]}
        result = self.pipeline.invoke(query, {"request_type": "general", "policy_document_id": "policy"})
        self.assertEqual(result["reason"], "partial_support")
        self.assertEqual(result["decision_support"]["assessment_status"], "validated")
        self.assertFalse(result["decision_support"]["technical_failure"])
        self.assertEqual(len(self.assessor.calls), 1)


class InsuranceAgentTests(unittest.TestCase):
    def setUp(self):
        from src.agent import InsuranceRAGAgent
        self.detector = MagicMock()
        self.detector.score_query.return_value = {"is_ood": False, "ood_band": "id"}
        self.assessor, self.retriever = FakeAssessor(), FakeRetriever()
        # Fake doc ID matches the existing trusted scope without changing files.
        for c in self.retriever.chunks:
            if c["metadata"]["doc_id"] == "policy":
                c["metadata"]["doc_id"] = "lv_03904e7e51ee"
        self.generator = MagicMock(name="generator")
        self.generator.name = "fake"
        self.generator.last_usage = {}
        self.generator.generate.return_value = "Sudden leaks are covered [1]. The excess is £250 [1]."
        self.agent = InsuranceRAGAgent("insurance_chroma_index_v3", "unused",
                                       detector=self.detector, retriever=self.retriever,
                                       assessor=self.assessor, generator=self.generator)

    def test_stage1_short_circuit(self):
        self.detector.score_query.return_value = {"is_ood": True}
        result = self.agent.answer(QUERY, customer_context={"request_type": "general"})
        self.assertEqual(result["execution_path"], ["stage1"])
        self.assertFalse(self.retriever.calls)
        self.generator.generate.assert_not_called()
        timings = result["timing_metrics"]
        self.assertEqual(timings["stages"]["stage2"]["status"], "skipped")
        self.assertEqual(timings["stages"]["stage3"]["duration_ms"], None)
        self.assertEqual(timings["slowest_stage"], "stage1")

    def test_stage2_rejection_never_calls_generator(self):
        self.assessor.report["parts"][0]["missing_evidence"] = ["cover conditions"]
        self.assessor.report["parts"][1]["missing_evidence"] = ["excess"]
        result = self.agent.answer(QUERY, customer_context={"request_type": "general"})
        self.assertEqual(result["execution_path"], ["stage1", "stage2"])
        self.assertEqual(result["action"], "abstain")
        self.generator.generate.assert_not_called()
        self.assertEqual(result["decision_support"]["category"], "missing_evidence")
        self.assertIn("excess", result["answer"])

    def test_assessed_evidence_and_explicit_facts_reach_generation(self):
        result = self.agent.answer(QUERY, customer_context={"request_type": "general", "facts": {"cause": "sudden"}})
        self.assertEqual(result["action"], "generate")
        self.assertEqual(result["execution_path"], ["stage1", "stage2", "stage3"])
        args, kwargs = self.generator.generate.call_args
        self.assertEqual({c["id"] for c in kwargs["chunks"]}, {"seed", "before", "after"})
        self.assertIn('"cause": "sudden"', args[0])
        self.assertNotIn("domain generalization", args[0])
        timings = result["timing_metrics"]
        self.assertTrue(all(e["status"] == "completed" for e in timings["stages"].values()))
        self.assertGreaterEqual(timings["request_total_ms"],
                                sum(e["duration_ms"] for e in timings["stages"].values()) - .01)
        self.assertEqual(result["repro"]["latency_seconds"], round(timings["request_total_ms"] / 1000, 3))

    def test_generation_error_abstains_without_extractive_fallback(self):
        self.generator.generate.side_effect = RuntimeError("API down")
        result = self.agent.answer(QUERY, customer_context={"request_type": "general"})
        self.assertEqual(result["action"], "abstain")
        self.assertEqual(result["reason"], "generation_error")
        self.assertFalse(result["citations"])
        self.assertEqual(result["timing_metrics"]["stages"]["stage3"]["status"], "error")
        self.assertEqual(result["decision_support"]["action"], "abstain")
        self.assertEqual(result["decision_support"]["category"], "generation_error")

    def test_failed_stage2_reports_timing_and_no_generation(self):
        self.assessor.error = TimeoutError()
        result = self.agent.answer(QUERY, customer_context={"request_type": "general"})
        self.assertEqual(result["timing_metrics"]["stages"]["stage2"]["status"], "error")
        self.assertEqual(result["timing_metrics"]["stage2_steps"]["part_assessment"]["status"], "error")
        self.generator.generate.assert_not_called()
        self.assertEqual(result["grounding_metrics"]["error_details"]["error_type"], "TimeoutError")

    def test_repetitive_assessment_stream_abstains_without_generation(self):
        llm = MagicMock()
        llm.bind.return_value.stream.return_value = iter([
            AIMessageChunk(content='{"quote":"Trace and access '),
            AIMessageChunk(content="\\u0000" * 8),
            AIMessageChunk(content="NEVER NEEDED")])
        self.agent.evidence_pipeline.assessor = LangChainEvidenceAssessor(llm=llm)
        result = self.agent.answer(QUERY, customer_context={"request_type": "general"})
        metrics = result["grounding_metrics"]
        self.assertEqual(result["action"], "abstain")
        self.assertEqual(result["reason"], "stage2_error")
        self.assertEqual(metrics["error_code"], "repetitive_assessment_output")
        self.assertTrue(metrics["error_details"]["stream_aborted"])
        self.assertEqual(metrics["error_details"]["failed_step"], "part_assessment")
        self.assertNotIn("NEVER NEEDED", metrics["error_details"]["partial_output_head"])
        self.assertEqual(result["decision_support"]["category"], "assessment_error")
        self.generator.generate.assert_not_called()
        llm.bind.return_value.stream.assert_called_once()

    def test_timing_event_is_logged_and_response_record_contains_metrics(self):
        from src.agent import logger
        self.agent._log = MagicMock()
        with patch.object(logger, "info") as info:
            result = self.agent.answer(QUERY, customer_context={"request_type": "general"})
        import json
        event = json.loads(info.call_args.args[0])
        self.assertEqual(event["event"], "insurance_stage_timings")
        self.assertEqual(event["request_total_ms"], result["timing_metrics"]["request_total_ms"])
        self.agent._log.assert_called_once_with(result)


class GeneratorConfigurationTests(unittest.TestCase):
    def test_batched_verification_uses_one_eval_for_three_sources(self):
        from langchain_core.messages import AIMessage
        from src.generator import OpenAICompatibleGenerator
        with patch("src.generator.ChatOpenAI") as chat:
            generator = OpenAICompatibleGenerator(api_key="offline-test")
        chat.return_value.bind.return_value.invoke.side_effect = [
            AIMessage(content="Cover applies [1] [2] [3]."),
            AIMessage(content='{"checks":[{"check_id":0,"supported":true},{"check_id":1,"supported":true},{"check_id":2,"supported":false}]}')]
        answer = generator.generate("prompt", chunks=[{"text": f"Source {i}", "cite_id": i}
                                                       for i in range(1, 4)], batch_verification=True)
        self.assertIn("[1]", answer)
        self.assertIn("[2]", answer)
        self.assertNotIn("[3]", answer)
        self.assertEqual([c["action"] for c in generator.last_timing_metrics["llm_calls"]], ["generate", "eval"])
        self.assertEqual(chat.return_value.bind.call_args.kwargs["response_format"]["type"], "json_schema")

    def test_generation_and_verification_timings_are_separate(self):
        from langchain_core.messages import AIMessage
        from src.generator import OpenAICompatibleGenerator
        with patch("src.generator.ChatOpenAI") as chat:
            generator = OpenAICompatibleGenerator(api_key="offline-test")
        chat.return_value.bind.return_value.invoke.side_effect = [
            AIMessage(content="Cover applies [1]."), AIMessage(content="YES")]
        with patch("src.timing.perf_counter", side_effect=[0., 1., 2., 3., 4., 5.]):
            generator.generate("prompt", chunks=[{"text": "Cover applies.", "cite_id": 1}])
        metrics = generator.last_timing_metrics
        self.assertEqual(metrics["answer_generation_llm"]["duration_ms"], 1000.)
        self.assertEqual(metrics["citation_verification"]["duration_ms"], 3000.)
        self.assertEqual([c["action"] for c in metrics["llm_calls"]], ["generate", "eval"])
        self.assertEqual(metrics["llm_calls"][1]["request"]["duration_ms"], 1000.)
        generator.reset_usage()
        self.assertEqual(generator.last_timing_metrics["llm_calls"], [])

    def test_failed_llm_call_retains_elapsed_time(self):
        from src.generator import OpenAICompatibleGenerator
        with patch("src.generator.ChatOpenAI") as chat:
            generator = OpenAICompatibleGenerator(api_key="offline-test")
        chat.return_value.bind.return_value.invoke.side_effect = TimeoutError()
        with patch("src.timing.perf_counter", side_effect=[0., 2.]):
            with self.assertRaises(TimeoutError):
                generator.generate("prompt")
        self.assertEqual(generator.last_timing_metrics["answer_generation_llm"],
                         {"duration_ms": 2000., "status": "error"})
        self.assertEqual(generator.last_timing_metrics["citation_verification"]["status"], "skipped")

    def test_api_key_is_not_confused_with_base_url_and_usage_resets(self):
        from src.generator import OpenAICompatibleGenerator
        with patch("src.generator.ChatOpenAI") as chat:
            generator = OpenAICompatibleGenerator(api_key="offline-test", base_url="https://example.invalid/v1")
        self.assertEqual(chat.call_args.kwargs["api_key"], "offline-test")
        self.assertEqual(chat.call_args.kwargs["base_url"], "https://example.invalid/v1")
        generator.last_usage["input_tokens"] = 17
        generator.reset_usage()
        self.assertEqual(generator.last_usage["input_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
