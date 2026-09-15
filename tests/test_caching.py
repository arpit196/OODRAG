"""Unit tests for cache policy without models, FastAPI, or filesystem access."""

from __future__ import annotations

import unittest

from caching import AnswerCache, AnswerRequest, CacheKeyFactory, CacheSettings, ConversationTurn, InMemoryTTLCache


class FakeAnswerer:
    """Deterministic answerer used to verify cache behaviour."""

    def __init__(self) -> None:
        self.calls = 0

    def answer(self, query: str, history: list[dict[str, str]] | None = None, top_k_final: int = 5) -> dict[str, object]:
        self.calls += 1
        return {"answer": query, "history_size": len(history or []), "top_k": top_k_final}


class CacheTests(unittest.TestCase):
    """Validate identity, expiry, and mutation isolation."""

    def test_request_history_and_top_k_are_part_of_identity(self) -> None:
        settings = CacheSettings(capacity=2, ttl_seconds=30)
        cache = AnswerCache(InMemoryTTLCache(settings), CacheKeyFactory(settings.namespace))
        agent = FakeAnswerer()
        request = AnswerRequest("What is OOD?", (), 5)

        self.assertFalse(cache.answer(agent, request).cache_hit)
        self.assertTrue(cache.answer(agent, request).cache_hit)
        self.assertFalse(cache.answer(agent, AnswerRequest("What is OOD?", (), 10)).cache_hit)
        self.assertFalse(cache.answer(agent, AnswerRequest(
            "What is OOD?", (ConversationTurn("user", "Earlier question"),), 5
        )).cache_hit)
        self.assertEqual(agent.calls, 3)

    def test_expired_entries_are_not_returned(self) -> None:
        now = [0.0]
        settings = CacheSettings(capacity=2, ttl_seconds=10)
        backend = InMemoryTTLCache(settings, clock=lambda: now[0])
        backend.set("key", {"answer": "cached"})
        now[0] = 10.0
        self.assertIsNone(backend.get("key"))


if __name__ == "__main__":
    unittest.main()
