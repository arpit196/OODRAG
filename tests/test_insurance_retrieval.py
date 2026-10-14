"""Metadata, chunking and filtered hybrid retrieval tests; no model downloads."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from insurance_retrieval import InsuranceHybridRetriever, iter_chunks, load_chunks, normalize_metadata, resolve_corpus_path


class FakeModel:
    def encode(self, queries, **kwargs):
        return np.array([[1., 0.]] * len(queries))


class FakeCollection:
    def __init__(self, chunks):
        self.chunks = chunks
        self.last_where = None

    def get(self, **kwargs):
        return {"ids": [c["id"] for c in self.chunks], "documents": [c["text"] for c in self.chunks],
                "metadatas": [c["metadata"] for c in self.chunks]}

    def query(self, where=None, n_results=5, **kwargs):
        self.last_where = where
        conditions = where.get("$and", [where]) if where else []
        chunks = [c for c in self.chunks if all(all(c["metadata"].get(k) == v for k, v in condition.items())
                                               for condition in conditions)][:n_results]
        return {"ids": [[c["id"] for c in chunks]], "documents": [[c["text"] for c in chunks]],
                "metadatas": [[c["metadata"] for c in chunks]], "distances": [[.1] * len(chunks)]}


def chunk(cid, doc_id, source, provider, text):
    return {"id": cid, "text": text, "metadata": {"doc_id": doc_id, "title": doc_id,
            "source_type": source, "provider": provider, "source_url": "https://example.org",
            "raw_path": "raw/paper.pdf", "extracted_path": "extracted/paper.txt"}}


class InsuranceRetrievalTests(unittest.TestCase):
    def test_windows_manifest_paths_and_no_benchmark_ingestion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metadata").mkdir()
            (root / "extracted/policies").mkdir(parents=True)
            (root / "extracted/policies/policy.txt").write_text("Escape of water cover", encoding="utf-8")
            record = {"doc_id": "lv", "source_type": "policies", "provider": "LV=", "title": "Policy",
                      "extracted_path": "extracted\\policies\\policy.txt", "raw_path": "raw\\policies\\policy.pdf"}
            benchmark = {"doc_id": "synthetic", "source_type": "benchmark", "extracted_path": "benchmark/fake.txt"}
            (root / "metadata/documents.jsonl").write_text("\n".join(json.dumps(r) for r in (record, record, benchmark)))
            chunks = load_chunks(root)
            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0]["metadata"]["source_type"], "policy")
            self.assertNotIn("effective_from", chunks[0]["metadata"])
            with self.assertRaises(ValueError):
                resolve_corpus_path(root, "../outside.txt")

    def test_page_and_overlap_preservation(self):
        windows = list(iter_chunks("--- PAGE 2 ---\none two three four five six\n--- PAGE 3 ---\nlast page", 4, 1))
        self.assertEqual([(page, text.split("\n\n", 1)[1]) for page, section, text in windows],
                         [(2, "one two three four"), (2, "four five six"), (3, "last page")])
        self.assertTrue(all(section == "unclassified" and text.startswith(f"Section: {section}\n\n")
                            for page, section, text in windows))
        with self.assertRaises(ValueError):
            list(iter_chunks("text", 4, 4))

    def test_regulatory_and_fos_metadata(self):
        base = {"doc_id": "x", "provider": "FCA", "raw_path": "raw/x.html", "extracted_path": "extracted/x.txt"}
        fca = normalize_metadata({**base, "source_type": "fca", "title": "ICOBS 8 - Claims handling"}, Path.cwd())
        self.assertEqual(fca["rule"], "ICOBS 8")
        law = normalize_metadata({**base, "source_type": "legislation", "title": "Insurance Act 2015"}, Path.cwd())
        self.assertEqual(law["source_type"], "legislation")
        fos = normalize_metadata({**base, "source_type": "fos_decisions", "title": "DRN-1234567 Escape of water"}, Path.cwd())
        self.assertEqual(fos["decision_id"], "DRN-1234567")
        self.assertEqual(fos["issue"], "escape_of_water")

    def test_both_searches_filter_and_return_distinct_files_with_metadata(self):
        chunks = [chunk("a1", "axa", "policy", "AXA", "Water leak coverage"),
                  chunk("a2", "axa", "policy", "AXA", "Escape of water excess"),
                  chunk("lv", "lv", "policy", "LV=", "Water exclusion"),
                  chunk("fos", "fos", "fos", "FOS", "Ombudsman water claim"),
                  chunk("fca", "fca", "fca", "FCA", "Claims handling regulations")]
        collection = FakeCollection(chunks)
        retriever = InsuranceHybridRetriever(collection, FakeModel())
        hits = retriever.search("water", source_type="policy", provider="AXA", top_k=5)
        self.assertEqual([hit.doc_id for hit in hits], ["axa"])
        self.assertEqual(len(hits[0].chunks), 2)
        self.assertEqual(collection.last_where, {"$and": [{"source_type": "policy"}, {"provider": "AXA"}]})
        self.assertTrue(all(c["metadata"]["provider"] == "AXA" for c in hits[0].chunks))
        self.assertTrue(any(c["bm25_score"] is not None for c in hits[0].chunks))
        self.assertEqual(retriever.search("water", provider="Missing"), [])

    def test_empty_collection(self):
        self.assertEqual(InsuranceHybridRetriever(FakeCollection([]), FakeModel()).search("water"), [])


if __name__ == "__main__":
    unittest.main()
