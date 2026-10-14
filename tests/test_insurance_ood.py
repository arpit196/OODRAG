"""Domain adaptation tests without model downloads or a live Chroma process."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np

from src.ood_scoring import (EmbeddingOOD, evaluate_insurance_queries, fit_insurance_ood,
                             insurance_reference, load_insurance_queries)

ROOT = Path(__file__).resolve().parent.parent


class InsuranceOODTests(unittest.TestCase):
    def setUp(self):
        self.rows = load_insurance_queries(ROOT / "data/home_insurance_queries.jsonl")
        self.scope = {"corpus_fingerprint": "snapshot", "trusted_document_ids": ["home"]}
        self.collection = MagicMock()
        self.collection.name = "insurance_chunks"
        self.collection.metadata = {"corpus_fingerprint": "snapshot", "embedding_model": "all-MiniLM-L6-v2",
                                    "hnsw:space": "cosine", "chunk_words": 220, "chunk_overlap": 40}
        rng = np.random.default_rng(3)
        vectors = rng.normal(loc=[2., 0., 0., 0.], scale=.1, size=(20, 4))
        self.collection.get.return_value = {
            "ids": [f"c{i}" for i in range(21)],
            "embeddings": [*vectors, np.array([0., 1., 0., 0.])],
            "metadatas": [{"doc_id": "home", "title": "Home policy", "source_type": "policy"}] * 20
                         + [{"doc_id": "travel", "title": "Travel case", "source_type": "fos"}],
        }

    def test_example_labels_keep_coverage_gaps_in_domain(self):
        self.assertEqual(len(self.rows), 80)
        gaps = [row for row in self.rows if row["answerability_note"] != "not_assessed"]
        self.assertEqual(len(gaps), 14)
        self.assertTrue(all(row["group"] == "id" for row in gaps))
        train_families = {row["family"] for row in self.rows if row["split"] == "train"}
        test_families = {row["family"] for row in self.rows if row["split"] == "test"}
        self.assertFalse(train_families & test_families)

    def test_loader_rejects_duplicates_and_family_leakage(self):
        for mutate in (lambda rows: rows.append(rows[0]),
                       lambda rows: rows[-1].update(family=rows[0]["family"])):
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                rows = [dict(row) for row in self.rows]
                mutate(rows)
                path = Path(directory) / "queries.jsonl"
                path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_insurance_queries(path)

    def test_reference_uses_allowlist_without_research_tiers(self):
        vectors, provenance = insurance_reference(self.collection, self.scope, "all-MiniLM-L6-v2")
        self.assertEqual(len(vectors), 20)
        self.assertEqual(provenance["embedding_dimension"], 4)
        self.assertEqual(set(provenance["trusted_documents"]), {"home"})
        self.assertEqual(set(provenance["excluded_documents"]), {"travel"})
        self.assertEqual(provenance["collection_metadata"]["chunk_words"], 220)

    def test_reference_rejects_drift_missing_documents_and_model_mismatch(self):
        for scope, model in (({**self.scope, "corpus_fingerprint": "changed"}, "all-MiniLM-L6-v2"),
                             ({**self.scope, "trusted_document_ids": ["absent"]}, "all-MiniLM-L6-v2"),
                             (self.scope, "different-model")):
            with self.subTest(scope=scope, model=model), self.assertRaises(ValueError):
                insurance_reference(self.collection, scope, model)

    def test_reference_rejects_invalid_vectors(self):
        self.collection.get.return_value["embeddings"][0] = np.zeros(4)
        with self.assertRaisesRegex(ValueError, "Invalid reference embedding"):
            insurance_reference(self.collection, self.scope, "all-MiniLM-L6-v2")

    def test_fit_excludes_test_queries_and_roundtrips_provenance(self):
        rng = np.random.default_rng(4)
        encoded = np.asarray([rng.normal(loc=[2., 0., 0., 0.], scale=.1, size=4)
                              if row["group"] == "id" else rng.normal(loc=[0., 2., 0., 0.], scale=.1, size=4)
                              for row in self.rows])
        model = MagicMock()
        model.encode.return_value = encoded
        detector, report = fit_insurance_ood(self.collection, self.scope, self.rows, model)
        train_id = encoded[[row["split"] == "train" and row["group"] == "id" for row in self.rows]]
        np.testing.assert_allclose(detector.query_reference_embeddings_, detector._normalize(train_id))
        self.assertEqual(len(detector.query_reference_embeddings_), 24)
        self.assertEqual(len(report["test"]["predictions"]), 32)
        self.assertEqual(report["test"]["groups"]["far_ood"]["count"], 8)
        self.assertEqual(report["training_query_counts"], {"id": 24, "near_ood": 12, "far_ood": 12})
        self.assertIn("auroc_ood_probability", report["test"]["comparisons"]["id_vs_far_ood"])
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "insurance.npz"
            detector.save(path)
            loaded = EmbeddingOOD.load(path)
            self.assertEqual(loaded.artifact_metadata_, detector.artifact_metadata_)
            before, after = detector.score_query(encoded[-1]), loaded.score_query(encoded[-1])
            self.assertAlmostEqual(before["ood_probability"], after["ood_probability"])
            self.assertEqual(before["ood_band"], after["ood_band"])

    def test_save_without_query_calibrator_is_supported(self):
        detector = EmbeddingOOD().fit(self.collection.get.return_value["embeddings"][:20])
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "chunks_only.npz"
            detector.save(path)
            self.assertEqual(EmbeddingOOD.load(path).artifact_metadata_, {})

    def test_evaluation_reports_actual_rejection_not_probability_cutoff(self):
        detector = MagicMock()
        detector.score_query.return_value = {"energy": 1., "knn_distance": .1,
                                            "ood_probability": .99, "is_ood": False, "is_near_ood": False}
        rows = [{"group": group} for group in ("id", "near_ood", "far_ood")]
        report = evaluate_insurance_queries(detector, rows, [[1.]] * 3)
        self.assertEqual(report["groups"]["id"]["reject_rate"], 0.)
        self.assertEqual(report["groups"]["far_ood"]["reject_rate"], 0.)
        with self.assertRaises(ValueError):
            evaluate_insurance_queries(detector, rows, [])


if __name__ == "__main__":
    unittest.main()
