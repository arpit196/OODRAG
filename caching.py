"""Cache and conversation primitives for the OOD-Aware RAG application.

The module deliberately contains no FastAPI, terminal, model, or filesystem
code. Applications inject an ``Answerer`` and a ``CacheBackend`` instead.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Mapping, MutableSequence, Protocol, Sequence

ResponseData = dict[str, Any]


def _positive_int(value: str, variable: str) -> int:
    """Parse a strictly positive integer environment variable."""
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{variable} must be an integer, got {value!r}") from error
    if parsed <= 0:
        raise ValueError(f"{variable} must be greater than zero")
    return parsed


@dataclass(frozen=True, slots=True)
class CacheSettings:
    """Configuration for a local TTL/LRU cache.

    ``namespace`` is part of every key. Change it when a corpus, prompt,
    model, or response policy changes and existing entries must be ignored.
    """

    capacity: int = 256
    ttl_seconds: int = 900
    namespace: str = "ood-rag-v1"
    max_history_turns: int = 12

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "CacheSettings":
        """Load settings from ``RAG_CACHE_*`` variables."""
        values = os.environ if env is None else env
        return cls(
            capacity=_positive_int(values.get("RAG_CACHE_CAPACITY", "256"), "RAG_CACHE_CAPACITY"),
            ttl_seconds=_positive_int(values.get("RAG_CACHE_TTL_SECONDS", "900"), "RAG_CACHE_TTL_SECONDS"),
            namespace=values.get("RAG_CACHE_NAMESPACE", "ood-rag-v1"),
            max_history_turns=_positive_int(values.get("RAG_MAX_HISTORY_TURNS", "12"), "RAG_MAX_HISTORY_TURNS"),
        )


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """An immutable, serialisable conversation turn used in cache identity."""

    role: str
    content: str

    def as_dict(self) -> dict[str, str]:
        """Return the format expected by the current RAG agent."""
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class AnswerRequest:
    """All inputs that can change an answer and must affect its cache key."""

    query: str
    history: tuple[ConversationTurn, ...]
    top_k: int


@dataclass(frozen=True, slots=True)
class CachedResponse:
    """A response plus explicit cache provenance for callers and telemetry."""

    payload: ResponseData
    cache_hit: bool


class Answerer(Protocol):
    """Port implemented by ``RAGAgent`` or any compatible answer service."""

    def answer(self, query: str, history: Sequence[dict[str, str]] | None = None, top_k_final: int = 5) -> ResponseData:
        """Generate a response for a query and its preceding conversation."""


class CacheBackend(ABC):
    """Storage port for cached RAG responses."""

    @abstractmethod
    def get(self, key: str) -> ResponseData | None:
        """Return an unexpired value, or ``None`` when no value exists."""

    @abstractmethod
    def set(self, key: str, value: ResponseData) -> None:
        """Store a value using backend-specific expiration and eviction."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all entries from the backend."""


@dataclass(slots=True)
class _CacheEntry:
    expires_at: float
    value: ResponseData


class InMemoryTTLCache(CacheBackend):
    """Thread-safe bounded LRU cache with per-entry TTL.

    This backend is process-local. Replace ``CacheBackend`` with Redis when
    running multiple application workers.
    """

    def __init__(self, settings: CacheSettings, clock: Callable[[], float] = time.monotonic) -> None:
        self._capacity = settings.capacity
        self._ttl_seconds = settings.ttl_seconds
        self._clock = clock
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str) -> ResponseData | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at <= self._clock():
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return copy.deepcopy(entry.value)

    def set(self, key: str, value: ResponseData) -> None:
        with self._lock:
            self._entries[key] = _CacheEntry(self._clock() + self._ttl_seconds, copy.deepcopy(value))
            self._entries.move_to_end(key)
            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


class CacheKeyFactory:
    """Creates stable, versioned keys from answer-affecting request inputs."""

    def __init__(self, namespace: str) -> None:
        self._namespace = namespace

    def make(self, request: AnswerRequest) -> str:
        """Hash canonical JSON so formatting differences do not fragment keys."""
        document = {
            "namespace": self._namespace,
            "query": request.query.strip(),
            "history": [turn.as_dict() for turn in request.history],
            "top_k": request.top_k,
        }
        canonical = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class AnswerCache:
    """Cache-aware application service around an injected ``Answerer``.

    Per-key locks coalesce concurrent identical misses, avoiding duplicate
    expensive retrieval and generation calls.
    """

    def __init__(self, backend: CacheBackend, key_factory: CacheKeyFactory) -> None:
        self._backend = backend
        self._key_factory = key_factory
        self._inflight: dict[str, threading.Lock] = {}
        self._inflight_guard = threading.Lock()

    def answer(self, answerer: Answerer, request: AnswerRequest) -> CachedResponse:
        """Return a cached response or compute and store one once per key."""
        key = self._key_factory.make(request)
        cached = self._backend.get(key)
        if cached is not None:
            return CachedResponse(payload=cached, cache_hit=True)
        with self._lock_for(key):
            cached = self._backend.get(key)
            if cached is not None:
                return CachedResponse(payload=cached, cache_hit=True)
            payload = answerer.answer(
                query=request.query,
                history=[turn.as_dict() for turn in request.history],
                top_k_final=request.top_k,
            )
            self._backend.set(key, payload)
            return CachedResponse(payload=copy.deepcopy(payload), cache_hit=False)

    def _lock_for(self, key: str) -> threading.Lock:
        with self._inflight_guard:
            return self._inflight.setdefault(key, threading.Lock())


class Conversation:
    """Bounded mutable conversation state for terminal or session adapters."""

    def __init__(self, max_turns: int) -> None:
        self._max_messages = max_turns * 2
        self._turns: MutableSequence[ConversationTurn] = []

    def snapshot(self) -> tuple[ConversationTurn, ...]:
        """Return immutable history suitable for an ``AnswerRequest``."""
        return tuple(self._turns)

    def append_exchange(self, user_message: str, assistant_message: str) -> None:
        """Record an exchange and discard the oldest complete messages."""
        self._turns.extend((ConversationTurn("user", user_message), ConversationTurn("assistant", assistant_message)))
        overflow = len(self._turns) - self._max_messages
        if overflow > 0:
            del self._turns[:overflow]
