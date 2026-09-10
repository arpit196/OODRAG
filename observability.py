"""Framework-independent retrieval telemetry and JSONL monitoring storage."""

from __future__ import annotations

import json
import math
import threading
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Iterable, Mapping, Protocol, Sequence


@dataclass(slots=True)
class RetrievalTrace:
    """Per-request retrieval timings and dense/BM25 agreement measurements."""

    stage_latency_ms: dict[str, float] = field(default_factory=dict)
    dense_ids: list[str] = field(default_factory=list)
    bm25_ids: list[str] = field(default_factory=list)
    agreement: dict[str, float | int | bool] = field(default_factory=dict)

    def time_stage(self, name: str, operation: Any) -> Any:
        """Run ``operation`` and record its wall-clock duration in milliseconds."""
        start = perf_counter()
        try:
            return operation()
        finally:
            self.stage_latency_ms[name] = round((perf_counter() - start) * 1000, 3)

    def record_rankings(self, dense_ids: Sequence[str], bm25_ids: Sequence[str]) -> None:
        """Record overlap and rank agreement for two independently ranked lists."""
        self.dense_ids, self.bm25_ids = list(dense_ids), list(bm25_ids)
        dense_set, bm25_set = set(dense_ids), set(bm25_ids)
        shared = dense_set & bm25_set
        union = dense_set | bm25_set
        shared_rank_deltas = [
            abs(dense_ids.index(identifier) - bm25_ids.index(identifier))
            for identifier in shared
        ]
        self.agreement = {
            "top_k_jaccard": round(len(shared) / len(union), 4) if union else 1.0,
            "shared_top_k": len(shared),
            "top_1_agrees": bool(dense_ids and bm25_ids and dense_ids[0] == bm25_ids[0]),
            "mean_shared_rank_delta": round(mean(shared_rank_deltas), 3) if shared_rank_deltas else None,
        }


class TelemetryStore(Protocol):
    """Port for persistence of request-level monitoring events."""

    def append(self, event: Mapping[str, Any]) -> None:
        """Persist one JSON-serialisable event."""

    def read_recent(self, limit: int) -> list[dict[str, Any]]:
        """Return at most ``limit`` most-recent events."""


class JsonlTelemetryStore:
    """Small, append-only process-local JSONL telemetry store.

    Replace this implementation with a database, OpenTelemetry exporter, or
    metrics backend by keeping the ``TelemetryStore`` interface unchanged.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.RLock()

    def append(self, event: Mapping[str, Any]) -> None:
        """Append an event without allowing observability failures to affect RAG."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(dict(event), separators=(",", ":"), default=str)
            with self._lock, self._path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            return

    def read_recent(self, limit: int) -> list[dict[str, Any]]:
        """Read a bounded monitoring window; malformed lines are ignored."""
        with self._lock:
            try:
                lines = self._path.read_text(encoding="utf-8").splitlines()[-limit:]
            except FileNotFoundError:
                return []
        events: list[dict[str, Any]] = []
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
        return events


def request_event(response: Mapping[str, Any], cache_hit: bool) -> dict[str, Any]:
    """Normalise an agent response into one privacy-conscious telemetry event."""
    repro = response.get("repro") if isinstance(response.get("repro"), Mapping) else {}
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cache_hit": cache_hit,
        "action": response.get("action", "unknown"),
        "route": (response.get("route") or {}).get("decision", "unknown"),
        "latency_ms": (repro or {}).get("latency_seconds", 0.0) * 1000,
        "retrieval": (repro or {}).get("retrieval", {}),
        "query_ood": response.get("query_metrics", {}),
        "chunk_ood": response.get("chunk_metrics", []),
    }


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1)
    return round(ordered[index], 3)


def _histogram(values: Sequence[float], bins: int = 10) -> list[dict[str, float | int]]:
    if not values:
        return []
    lower, upper = min(values), max(values)
    if lower == upper:
        return [{"from": round(lower, 4), "to": round(upper, 4), "count": len(values)}]
    width = (upper - lower) / bins
    counts = [0] * bins
    for value in values:
        counts[min(bins - 1, int((value - lower) / width))] += 1
    return [
        {"from": round(lower + i * width, 4), "to": round(lower + (i + 1) * width, 4), "count": count}
        for i, count in enumerate(counts)
    ]


def _numeric_values(events: Iterable[Mapping[str, Any]], path: tuple[str, ...]) -> list[float]:
    values: list[float] = []
    for event in events:
        current: Any = event
        for key in path:
            if not isinstance(current, Mapping):
                break
            current = current.get(key)
        if isinstance(current, (int, float)) and math.isfinite(float(current)):
            values.append(float(current))
    return values


def monitoring_summary(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate a recent event window for the operational dashboard."""
    actions = Counter(str(event.get("action", "unknown")) for event in events)
    total = len(events)
    uncached = [event for event in events if not event.get("cache_hit")]
    stage_names = ("dense", "bm25", "fusion", "rerank")
    latencies = {
        name: _latency_summary(_numeric_values(uncached, ("retrieval", "stage_latency_ms", name)))
        for name in stage_names
    }
    agreement_values = _numeric_values(uncached, ("retrieval", "agreement", "top_k_jaccard"))
    top_one_values = _numeric_values(uncached, ("retrieval", "agreement", "top_1_agrees"))
    ood = {
        "query": _ood_summary(event.get("query_ood", {}) for event in events),
        "chunks": _ood_summary(
            metric for event in events for metric in event.get("chunk_ood", []) if isinstance(metric, Mapping)
        ),
    }
    return {
        "requests": total,
        "cache_hit_rate": round(sum(bool(event.get("cache_hit")) for event in events) / total, 4) if total else 0.0,
        "action_distribution": {
            action: {"count": count, "rate": round(count / total, 4) if total else 0.0}
            for action, count in sorted(actions.items())
        },
        "stage_latency_ms": latencies,
        "dense_bm25_agreement": {
            "samples": len(agreement_values),
            "mean_top_k_jaccard": round(mean(agreement_values), 4) if agreement_values else None,
            "top_1_agreement_rate": round(mean(top_one_values), 4) if top_one_values else None,
        },
        "ood_distributions": ood,
    }


def _latency_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    return {
        "samples": len(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
    }


def _ood_summary(metrics: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, float | int]]]:
    rows = list(metrics)
    names = ("ood_probability", "energy", "mahalanobis", "knn_distance")
    return {name: _histogram(_numeric_values(rows, (name,))) for name in names}
