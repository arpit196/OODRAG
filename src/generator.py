"""Pluggable text generators for the RAG agent.

Swap backends by implementing ``generate(prompt, **kwargs) -> str``.  The agent
never imports a vendor SDK directly — only this module does.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Optional, Protocol
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from src.timing import measure, skipped_metrics

load_dotenv()


class Generator(Protocol):
    name: str

    def generate(self, prompt: str, **kwargs) -> str:
        ...


class ExtractiveGenerator:
    """No-API fallback: first sentences of retrieved chunks plus [n] cites.

    Lets the rest of the pipeline run without an LLM key.  Replace this in
    production via ``load_generator('openai')``.
    """

    name = "extractive"

    def generate(self, prompt: str, chunks=None, action: str = "generate", **kwargs) -> str:
        from src.guardrails import fallback_answer

        if action in ("abstain", "chitchat") or not chunks:
            return fallback_answer(action)
        parts = []
        for c in chunks[:3]:
            text = (c.get("text") or "").replace("\n", " ").strip()
            snippet = text[:360].rsplit(" ", 1)[0] if len(text) > 360 else text
            n = c.get("cite_id", 1)
            parts.append(f"{snippet} [{n}]")
        return " ".join(parts)


class OpenAICompatibleGenerator:
    """OpenAI Chat Completions, also works with Ollama / vLLM / Groq if you
    set OPENAI_BASE_URL.  Requires OPENAI_API_KEY (Ollama often accepts any string)."""

    def __init__(self, model: str = "gpt-4o-mini",
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 temperature: float = 0.2,
                 max_tokens: int = 512):
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")
        
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "key")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
            base_url=self.base_url,
        )
        self.name = f"openai:{model}"
        self.last_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
        }
        self.last_timing_metrics = skipped_metrics(("answer_generation_llm", "citation_verification"))
        self.last_timing_metrics["llm_calls"] = []

    def reset_usage(self):
        for key in self.last_usage:
            self.last_usage[key] = 0
        self.last_timing_metrics = skipped_metrics(("answer_generation_llm", "citation_verification"))
        self.last_timing_metrics["llm_calls"] = []

    def generate(self, prompt: str, chunks=None, action: str = "generate", **kwargs) -> str:
        from src.guardrails import fallback_answer, verify_and_correct_citations
        #print(f"Generating with: {prompt}...")
        #if action in ("abstain", "chitchat"):
        #   return fallback_answer(action)

        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        '''body = json.dumps({
            "model": self.model,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "messages": [{"role": "user", "content": prompt}],
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )'''

        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        llm = self.llm.bind(
            temperature=temperature,
            max_tokens=max_tokens,
            **({"response_format": kwargs["response_format"]} if "response_format" in kwargs else {}),
        )

        # One entry per invocation, including verification calls and failures.
        # eval calls recur through this generator, so retain the outer metrics.
        call = {"action": action}
        self.last_timing_metrics["llm_calls"].append(call)
        try:
            with measure(call, "request"):
                response = llm.invoke([HumanMessage(content=prompt)])
        finally:
            if action == "generate":
                self.last_timing_metrics["answer_generation_llm"] = dict(call["request"])
        raw_output = response.content
        if not isinstance(raw_output, str):
            raw_output = str(raw_output)

        raw_output = raw_output.strip()
        usage = response.usage_metadata or {}
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)
        total_tokens = usage.get(
            "total_tokens",
            prompt_tokens + completion_tokens
        )

        cached_tokens = (
            usage.get("prompt_tokens_details", {})
                .get("cached_tokens", 0)
        )

        self.last_usage["input_tokens"] += prompt_tokens
        self.last_usage["output_tokens"] += completion_tokens
        self.last_usage["total_tokens"] += total_tokens
        self.last_usage["cached_tokens"] += cached_tokens
        

        if chunks and action == "generate":
            self.last_timing_metrics["verification_mode"] = (
                "batch" if kwargs.get("batch_verification", False) else "per_pair")
            # Pass self as the evaluator to perform lightweight NLI verification checks
            with measure(self.last_timing_metrics, "citation_verification"):
                verification_result = verify_and_correct_citations(
                    answer=raw_output,
                    chunks=chunks,
                    evaluator_generator=self,
                    strict_ungrounded_drop=kwargs.get("strict_ungrounded_drop", False),
                    batch_verification=kwargs.get("batch_verification", False),
                )
            return verification_result["corrected_answer"]
        return raw_output


def load_generator(kind: str = "extractive", **kwargs) -> Generator:
    """Factory: ``extractive`` | ``openai``.  Add new names here as backends grow."""
    kind = (kind or "extractive").lower()
    if kind in ("extractive", "echo", "stub"):
        return ExtractiveGenerator()
    if kind in ("openai", "openai-compatible"):
        return OpenAICompatibleGenerator(**kwargs)
    raise ValueError(f"unknown generator {kind!r}")
