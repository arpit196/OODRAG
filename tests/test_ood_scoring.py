import unittest

import numpy as np

from ood_scoring import EmbeddingOOD, auroc, holdout_split


class FakeCollection:
    def __init__(self, ids, embeddings, tiers):
        self.ids, self.embeddings, self.tiers = ids, embeddings, tiers

    def get(self, ids=None, include=None):
        chosen = range(len(self.ids)) if ids is None else [self.ids.index(i) for i in ids]
        return {"ids": [self.ids[i] for i in chosen],
                "embeddings": [self.embeddings[i] for i in chosen],
                "metadatas": [{"tier": self.tiers[i]} for i in chosen]}


class EmbeddingOODTests(unittest.TestCase):
    def test_scores_and_annotates_retrieved_chunks(self):
        rng = np.random.default_rng(7)
        reference = rng.normal(size=(100, 6))
        detector = EmbeddingOOD(threshold=.95).fit(reference)
        id_emb = reference.mean(axis=0)
        collection = FakeCollection(["id", "far"], [id_emb, np.full(6, 8.0)],
                                    ["id_core_text", "far_ood_text"])

        chunks = detector.score_chunks([{"id": "id"}, {"id": "far"}], collection)
        self.assertFalse(chunks[0]["ood"]["is_ood"])
        self.assertEqual(chunks[0]["ood"]["ood_band"], "id")
        self.assertTrue(chunks[1]["ood"]["is_ood"])
        self.assertEqual(chunks[1]["ood"]["ood_band"], "ood")
        self.assertGreater(chunks[1]["ood"]["energy"], chunks[0]["ood"]["energy"])

    def test_ood_query_with_id_chunks_flags_the_request(self):
        rng = np.random.default_rng(7)
        reference = rng.normal(size=(100, 6))
        detector = EmbeddingOOD(threshold=.95).fit(reference)
        id_emb = reference.mean(axis=0)
        collection = FakeCollection(["id"], [id_emb], ["id_core_text"])

        chunks = detector.score_retrieval(np.full(6, 8.0), [{"id": "id"}], collection)
        self.assertFalse(chunks[0]["ood"]["is_ood"])
        self.assertTrue(chunks[0]["query_ood"]["is_ood"])
        self.assertTrue(chunks[0]["request_is_ood"])

    def test_near_ood_band_on_id_tail(self):
        rng = np.random.default_rng(7)
        reference = rng.normal(size=(200, 6))
        detector = EmbeddingOOD(threshold=.95, near_threshold=.80).fit(reference)
        order = np.argsort(detector.reference_energy_)
        near_emb = reference[order[int(0.90 * len(order))]]
        band = detector.score_embeddings([near_emb])[0]["ood_band"]
        self.assertEqual(band, "near_ood")

    def test_holdout_fraction_and_auroc(self):
        rng = np.random.default_rng(0)
        id_emb = rng.normal(size=(100, 6))
        ood_emb = rng.normal(loc=4.0, size=(100, 6))
        rows = [(id_emb[i], 0, "id", "id_core_text") for i in range(100)]
        rows += [(ood_emb[i], 1, "far_ood", "far_ood_text") for i in range(100)]

        train, eval_rows = holdout_split(rows, eval_fraction=0.10, seed=0)
        self.assertEqual(len(eval_rows), 20)
        self.assertEqual(len(train), 180)

        detector = EmbeddingOOD().fit([r[0] for r in train if r[1] == 0])
        report = detector.evaluate(eval_rows)
        self.assertGreater(report["auroc_energy"], 0.95)
        self.assertIn("auroc_knn", report)

        chance = auroc(np.array([0, 0, 1, 1]), np.array([0.1, 0.9, 0.2, 0.8]))
        self.assertAlmostEqual(chance, 0.5)

    def test_fit_from_collection_eval_fraction(self):
        rng = np.random.default_rng(1)
        n = 50
        ids = [f"i{i}" for i in range(n)] + [f"o{i}" for i in range(n)]
        embs = list(rng.normal(size=(n, 6))) + list(rng.normal(loc=5.0, size=(n, 6)))
        tiers = ["id_core_text"] * n + ["far_ood_text"] * n
        detector = EmbeddingOOD.fit_from_collection(
            FakeCollection(ids, embs, tiers), eval_fraction=0.20, eval_seed=0)
        self.assertEqual(len(detector.reference_energy_), 40)
        self.assertEqual(detector.eval_report_["n"], 20)
        self.assertGreater(detector.eval_report_["auroc_energy"], 0.95)


if __name__ == "__main__":
    unittest.main()
