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
        from guardrails import fallback_answer

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
        self.name = f"openai:{model}"

    def generate(self, prompt: str, chunks=None, action: str = "generate", **kwargs) -> str:
        from guardrails import fallback_answer

        if action in ("abstain", "chitchat"):
            return fallback_answer(action)

        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        body = json.dumps({
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
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:400]
            raise RuntimeError(f"generation HTTP {error.code}: {detail}") from error
        return payload["choices"][0]["message"]["content"].strip()


def load_generator(kind: str = "extractive", **kwargs) -> Generator:
    """Factory: ``extractive`` | ``openai``.  Add new names here as backends grow."""
    kind = (kind or "extractive").lower()
    if kind in ("extractive", "echo", "stub"):
        return ExtractiveGenerator()
    if kind in ("openai", "openai-compatible"):
        return OpenAICompatibleGenerator(**kwargs)
    raise ValueError(f"unknown generator {kind!r}")
