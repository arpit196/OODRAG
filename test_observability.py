"""Focused tests for framework-independent monitoring aggregation."""

from __future__ import annotations

import unittest

from observability import monitoring_summary


class MonitoringSummaryTests(unittest.TestCase):
    """Verify health metrics are computed from recorded live-traffic events."""

    def test_aggregates_actions_stages_agreement_and_ood_signals(self) -> None:
        events = [{
            "cache_hit": False,
            "action": "generate",
            "retrieval": {"stage_latency_ms": {"dense": 2.0, "bm25": 3.0, "fusion": 1.0, "rerank": 8.0},
                          "agreement": {"top_k_jaccard": 0.5, "top_1_agrees": True}},
            "query_ood": {"ood_probability": 0.1, "energy": 2.0, "mahalanobis": 1.0, "knn_distance": 0.2},
            "chunk_ood": [{"ood_probability": 0.2, "energy": 3.0, "mahalanobis": 2.0, "knn_distance": 0.3}],
        }, {
            "cache_hit": True,
            "action": "abstain",
            "retrieval": {},
            "query_ood": {"ood_probability": 0.9, "energy": 9.0, "mahalanobis": 8.0, "knn_distance": 0.9},
            "chunk_ood": [],
        }]
        summary = monitoring_summary(events)

        self.assertEqual(summary["requests"], 2)
        self.assertEqual(summary["action_distribution"]["abstain"]["rate"], 0.5)
        self.assertEqual(summary["stage_latency_ms"]["rerank"]["p99"], 8.0)
        self.assertEqual(summary["dense_bm25_agreement"]["mean_top_k_jaccard"], 0.5)
        self.assertTrue(summary["ood_distributions"]["query"]["energy"])


if __name__ == "__main__":
    unittest.main()
