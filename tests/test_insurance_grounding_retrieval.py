"""Grounding retrieval contract tests; no downloads, database or generation."""

import unittest
from unittest.mock import MagicMock

import numpy as np

from insurance_retrieval_fixed import InsuranceHybridRetriever, tokenize
from src.grounding import extract_grounding_features


def evidence(cid, provider="AXA", vector=(1., 0.)):
    return {"id": cid, "text": f"Original clause {cid}", "embedding": vector,
            "metadata": {"doc_id": provider, "title": f"{provider} wording",
                         "provider": provider, "source_type": "policy",
                         "product": "Premier", "version_label": "2025 file",
                         "page": 12, "section": "Escape of water",
                         "effective_from": "unknown", "sha256": "document-hash",
                         "source_url": "", "raw_path": "raw/policy.pdf",
                         "extracted_path": "extracted/policy.txt"}}


class FakeCollection:
    def __init__(self, chunks, dense_ids):
        self.chunks = chunks
        self.dense_ids = dense_ids
        self.last_where = None
        self.embedding_reads = 0

    def get(self, ids=None, include=None):
        if ids is not None:
            self.embedding_reads += 1
            # Return a different order to ensure scores are matched by ID.
            selected = [chunk for chunk in reversed(self.chunks) if chunk["id"] in ids]
            return {"ids": [chunk["id"] for chunk in selected],
                    "embeddings": [chunk["embedding"] for chunk in selected]}
        return {"ids": [chunk["id"] for chunk in self.chunks],
                "documents": [chunk["text"] for chunk in self.chunks],
                "metadatas": [chunk["metadata"] for chunk in self.chunks]}

    def query(self, where=None, n_results=5, **kwargs):
        self.last_where = where
        conditions = where.get("$and", [where]) if where else []
        selected = [chunk for chunk in self.chunks if chunk["id"] in self.dense_ids
                    and all(all(chunk["metadata"].get(key) in value["$in"] if isinstance(value, dict)
                                else chunk["metadata"].get(key) == value for key, value in condition.items())
                            for condition in conditions)][:n_results]
        return {"ids": [[chunk["id"] for chunk in selected]],
                "documents": [[chunk["text"] for chunk in selected]],
                "metadatas": [[chunk["metadata"] for chunk in selected]],
                "distances": [[.25] * len(selected)]}


class GroundingRetrievalTests(unittest.TestCase):
    def test_reused_embedding_preserves_hits_and_scores_without_encoding(self):
        retriever = self.make_retriever()
        original = retriever.search_chunks("water", allowed_document_ids=["AXA"])
        retriever.model.encode.reset_mock()
        reused = retriever.search_chunks("water", allowed_document_ids=["AXA"], query_embedding=[2., 0.])
        retriever.model.encode.assert_not_called()
        self.assertEqual([c["id"] for c in original], [c["id"] for c in reused])
        np.testing.assert_allclose([c["cosine_similarity"] for c in original],
                                   [c["cosine_similarity"] for c in reused])
        self.assertEqual([c["bm25_score"] for c in original], [c["bm25_score"] for c in reused])

    def test_invalid_reused_embeddings_are_rejected(self):
        retriever = self.make_retriever()
        for vector in ([0., 0.], [float("nan"), 0.], [[1., 0.]]):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                retriever.search_chunks("water", query_embedding=vector)

    def test_document_allowlist_filters_dense_and_sparse_before_ranking(self):
        retriever = self.make_retriever()
        chunks = retriever.search_chunks("water", allowed_document_ids=["AXA"])
        self.assertEqual({c["id"] for c in chunks}, {"dense", "sparse"})
        self.assertEqual(retriever.collection.last_where, {"doc_id": {"$in": ["AXA"]}})
        chunks = retriever.search_chunks("water", allowed_document_ids=["LV="], source_type="policy")
        self.assertEqual({c["id"] for c in chunks}, {"other"})
        self.assertEqual(retriever.collection.last_where,
                         {"$and": [{"source_type": "policy"}, {"doc_id": {"$in": ["LV="]}}]})
        reads = retriever.collection.embedding_reads
        self.assertEqual(retriever.search_chunks("water", allowed_document_ids=[]), [])
        self.assertEqual(retriever.collection.embedding_reads, reads)

    def make_retriever(self):
        chunks = [evidence("dense", vector=(3., 4.)),
                  evidence("sparse", vector=(0., 2.)),
                  evidence("other", provider="LV=")]
        collection = FakeCollection(chunks, ["dense", "other"])
        model = MagicMock()
        model.encode.return_value = np.array([[2., 0.]])
        retriever = InsuranceHybridRetriever(collection, model)
        retriever.bm25 = MagicMock()
        # The dense-only hit has a real negative BM25 score, not missing/zero.
        retriever.bm25.get_scores.return_value = np.array([-.2, 3., 8.])
        return retriever

    def test_branch_only_hits_have_real_scores_and_work_with_grounding(self):
        retriever = self.make_retriever()
        chunks = retriever.search_chunks("water's damage", provider="AXA", source_type="policy")
        by_id = {chunk["id"]: chunk for chunk in chunks}
        self.assertEqual(set(by_id), {"dense", "sparse"})
        self.assertAlmostEqual(by_id["dense"]["cosine_similarity"], .6)
        self.assertEqual(by_id["dense"]["bm25_score"], -.2)
        self.assertEqual(by_id["sparse"]["cosine_similarity"], 0.)
        self.assertEqual(by_id["sparse"]["bm25_score"], 3.)
        self.assertIsNone(by_id["sparse"]["dense_distance"])
        self.assertEqual(by_id["dense"]["dense_distance"], .25)
        vector = extract_grounding_features("water's damage", chunks, {})
        np.testing.assert_allclose(vector[5:11], [.6, 0., 0., 0., .75, 0.])
        retriever.bm25.get_scores.assert_called_once_with(tokenize("water's damage"))
        self.assertEqual(retriever.collection.last_where,
                         {"$and": [{"source_type": "policy"}, {"provider": "AXA"}]})

    def test_provenance_preserved_without_assigning_support_or_ood_labels(self):
        retriever = self.make_retriever()
        chunk = retriever.search_chunks("water", provider="AXA", top_k=1)[0]
        self.assertEqual(chunk["text"], "Original clause dense")
        self.assertEqual(chunk["metadata"], retriever.chunks[0]["metadata"])
        self.assertEqual(chunk["title"], "AXA wording")
        for key in ("ood", "grounded", "tier", "ood_probability"):
            self.assertNotIn(key, chunk)
        self.assertNotIn("cosine_similarity", retriever.chunks[0])

    def test_file_search_keeps_grouping_and_chunk_search_limits_chunks(self):
        retriever = self.make_retriever()
        files = retriever.search("water", provider="AXA", top_k=1)
        self.assertEqual(len(files), 1)
        self.assertEqual([chunk["id"] for chunk in files[0].chunks], ["dense", "sparse"])
        self.assertEqual(retriever.collection.embedding_reads, 0)
        chunks = retriever.search_chunks("water", provider="AXA", top_k=1)
        self.assertEqual([chunk["id"] for chunk in chunks], ["dense"])
        self.assertEqual(chunks[0]["fused_score"], files[0].score)

    def test_missing_provider_and_empty_collection_return_no_evidence(self):
        retriever = self.make_retriever()
        self.assertEqual(retriever.search_chunks("water", provider="Missing"), [])
        retriever.model.encode.assert_not_called()
        self.assertEqual(retriever.collection.embedding_reads, 0)
        empty = InsuranceHybridRetriever(FakeCollection([], []), MagicMock())
        self.assertEqual(empty.search_chunks("water"), [])

    def test_missing_embedding_is_an_error_not_a_synthetic_score(self):
        retriever = self.make_retriever()
        retriever.collection.get = MagicMock(return_value={"ids": [], "embeddings": []})
        with self.assertRaisesRegex(ValueError, "Missing stored embedding"):
            retriever.search_chunks("water", provider="AXA")

    def test_invalid_embeddings_are_rejected(self):
        for vector in ((0., 0.), (float("nan"), 0.), (1., 0., 0.)):
            with self.subTest(vector=vector):
                retriever = self.make_retriever()
                retriever.collection.chunks[0]["embedding"] = vector
                with self.assertRaises(ValueError):
                    retriever.search_chunks("water", provider="AXA")
        retriever = self.make_retriever()
        retriever.model.encode.return_value = np.array([[0., 0.]])
        with self.assertRaisesRegex(ValueError, "Query embedding"):
            retriever.search_chunks("water", provider="AXA")

    def test_invalid_search_arguments(self):
        retriever = self.make_retriever()
        for kwargs in ({"top_k": 0}, {"candidates": 0}, {"source_type": "unknown"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                retriever.search_chunks("water", **kwargs)
        with self.assertRaises(ValueError):
            retriever.search_chunks(" ")


if __name__ == "__main__":
    unittest.main()
