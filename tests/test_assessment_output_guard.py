"""No network: guard tests plus real LangChain/SDK SSE transport integration."""
import json
import unittest
from unittest.mock import MagicMock

from langchain_core.messages import AIMessageChunk
from src.assessment_output_guard import AssessmentOutputGuard, UnsafeAssessmentOutputError
from src.evidence_gate import CustomerContext, EvidenceReference, LangChainEvidenceAssessor


def report():
    return {"assessed_question": "What is the limit?", "all_question_parts_identified": True,
            "parts": [{"question_part": "What is the limit?", "status": "supported",
                       "evidence": [{"chunk_id": "seed", "quote": "The limit is £5,000."}],
                       "missing_evidence": [], "missing_customer_facts": [], "conflicts": []}]}


class OutputGuardTests(unittest.TestCase):
    def test_repeated_escapes_detected_even_one_character_per_delta(self):
        guard = AssessmentOutputGuard()
        text = "\\u0000" * 8
        for character in text[:-1]:
            guard.feed(character)
        with self.assertRaises(UnsafeAssessmentOutputError):
            guard.feed(text[-1])

    def test_controls_rejected_unicode_and_normal_whitespace_preserved(self):
        guard = AssessmentOutputGuard()
        guard.feed("£5,000 • café\n\r\t" + "\\n" * 20)
        guard.feed("\\u0000")  # A single JSON escape is checked again after decoding.
        for control in ("\x00", "\x01", "\x1b", "\x7f", "\x85"):
            with self.subTest(control=repr(control)), self.assertRaises(UnsafeAssessmentOutputError):
                AssessmentOutputGuard().feed(control)

    def test_stream_stops_at_pattern_and_closes_before_remaining_output(self):
        llm = MagicMock()
        consumed, closed = [], []
        def deltas():
            try:
                for text in ('{"quote":"', "\\u0000" * 7, "\\u0", "000", "NEVER CONSUMED"):
                    consumed.append(text)
                    yield AIMessageChunk(content=text)
            finally:
                closed.append(True)
        llm.bind.return_value.stream.return_value = deltas()
        assessor = LangChainEvidenceAssessor(llm=llm)
        with self.assertRaises(UnsafeAssessmentOutputError) as caught:
            assessor.assess("query", CustomerContext(request_type="general"), [])
        self.assertNotIn("NEVER CONSUMED", consumed)
        self.assertEqual(closed, [True])
        self.assertIn("\\u0000", caught.exception.raw_response.content)
        llm.bind.return_value.stream.assert_called_once()

    def test_single_escaped_control_rejected_after_decoding_in_any_field(self):
        for field in ("quote", "missing_evidence"):
            with self.subTest(field=field):
                data = report()
                part = data["parts"][0]
                if field == "quote":
                    part["evidence"][0]["quote"] += "\x00"
                else:
                    part[field] = ["missing\x00schedule"]
                llm = MagicMock()
                llm.bind.return_value.stream.return_value = [AIMessageChunk(content=json.dumps(data))]
                with self.assertRaises(UnsafeAssessmentOutputError):
                    LangChainEvidenceAssessor(llm=llm).assess("query", CustomerContext(), [])

    def test_injected_assessment_quote_also_rejects_controls(self):
        with self.assertRaises(ValueError):
            EvidenceReference(chunk_id="seed", quote="limit\x00")


class RealStreamingTransportTests(unittest.TestCase):
    def exercise(self, fragments):
        import httpx
        from langchain_openai import ChatOpenAI
        consumed, closed, requests = [], [], []
        class Events(httpx.SyncByteStream):
            def __iter__(self):
                for index, fragment in enumerate(fragments):
                    consumed.append(fragment)
                    chunk = {"id": "chatcmpl-offline", "object": "chat.completion.chunk", "created": 0,
                             "model": "gpt-4.1-mini", "choices": [{"index": 0,
                             "delta": {"content": fragment, **({"role": "assistant"} if index == 0 else {})},
                             "finish_reason": None}]}
                    yield ("data: " + json.dumps(chunk) + "\n\n").encode()
                final = {"id": "chatcmpl-offline", "object": "chat.completion.chunk", "created": 0,
                         "model": "gpt-4.1-mini", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                yield ("data: " + json.dumps(final) + "\n\n").encode()
                usage = {**final, "choices": [], "usage": {"prompt_tokens": 100,
                         "completion_tokens": 50, "total_tokens": 150}}
                yield ("data: " + json.dumps(usage) + "\n\ndata: [DONE]\n\n").encode()
            def close(self):
                closed.append(True)
        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Events())
        client = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(client.close)
        llm = ChatOpenAI(model="gpt-4.1-mini", api_key="offline-test", base_url="https://offline.invalid/v1",
                         http_client=client, max_retries=0)
        assessor = LangChainEvidenceAssessor(llm=llm)
        return assessor, consumed, closed, requests

    def test_real_sdk_valid_schema_stream_preserves_output_and_usage(self):
        wire = report()
        wire["parts"][0]["evidence"] = [{"chunk_id": "seed", "span_index": 0}]
        text = json.dumps(wire)
        assessor, consumed, closed, requests = self.exercise([text[:90], text[90:]])
        actual, usage = assessor.assess("What is the limit?", CustomerContext(request_type="general"),
                                       [{"id": "seed", "text": "The limit is £5,000.", "metadata": {}}])
        self.assertEqual(actual.model_dump(), report())
        self.assertEqual(usage["output_tokens"], 50)
        self.assertEqual(len(requests), 1)
        self.assertTrue(requests[0]["response_format"]["json_schema"]["strict"])
        self.assertTrue(closed)

    def test_real_sdk_runaway_stream_closes_before_completion(self):
        assessor, consumed, closed, requests = self.exercise(
            ['{"quote":"', "\\u0000" * 7, "\\u0", "000", "NEVER CONSUMED"])
        with self.assertRaises(UnsafeAssessmentOutputError):
            assessor.assess("query", CustomerContext(), [])
        self.assertNotIn("NEVER CONSUMED", consumed)
        self.assertTrue(closed)
        self.assertEqual(len(requests), 1)


if __name__ == "__main__":
    unittest.main()
