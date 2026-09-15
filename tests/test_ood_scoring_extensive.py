"""
test_ood_scoring.py
---------------------
Two tiers of tests, run at different cadences:

  UNIT (fast, synthetic, no corpus/models needed)
      Isolates "is the scoring math correct" from "is the real corpus
      separable enough" — these are different failure modes. Includes
      permanent regression tests for the two real bugs already found and
      fixed in this codebase, so neither can silently reappear.

  INTEGRATION (slow, needs the real built index + OOD reference)
      Real held-out AUROC, reported PER TIER (never averaged — near-OOD and
      far-OOD are different difficulty levels, and a blended number hides
      exactly the failure that matters), plus a bootstrap confidence
      interval since a point-estimate AUROC on a small held-out set
      overstates what you actually know.

Run:
    pytest test_ood_scoring.py -v                    # unit tests only (default)
    pytest test_ood_scoring.py -v -m integration      # integration tests only
    pytest test_ood_scoring.py -v -m "" -o addopts=""  # everything
"""

import numpy as np
import pytest

from ood_scoring import EmbeddingOOD


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: slow, needs a real built index")


# ---------------------------------------------------------------------------
# UNIT TESTS — synthetic, deterministic, isolate the math from the data
# ---------------------------------------------------------------------------

def _make_synthetic_clusters(n_per_cluster=200, dim=16, seed=0):
    """Three well-separated Gaussian blobs standing in for id/near/far.
    Dimension deliberately small relative to N so the fit is never in the
    underdetermined regime — this test isolates correctness of the math,
    not whether there's enough data, which gets tested separately in
    integration tests against the real corpus."""
    rng = np.random.RandomState(seed)
    id_embeddings = rng.normal(loc=0.0, scale=1.0, size=(n_per_cluster, dim))
    near_embeddings = rng.normal(loc=4.0, scale=1.0, size=(n_per_cluster, dim))
    far_embeddings = rng.normal(loc=15.0, scale=1.0, size=(n_per_cluster, dim))
    return id_embeddings, near_embeddings, far_embeddings


def test_id_points_score_low_false_positive_rate():
    id_emb, _, _ = _make_synthetic_clusters()
    detector = EmbeddingOOD(shrinkage=0.1, threshold=0.95, n_clusters=1).fit(id_emb)
    scores = detector.score_embeddings(id_emb[:20])
    flagged_rate = sum(s["is_ood"] for s in scores) / len(scores)
    assert flagged_rate < 0.15, f"too many in-distribution points flagged as OOD: {flagged_rate:.2%}"


def test_far_ood_strictly_separates_from_id():
    id_emb, _, far_emb = _make_synthetic_clusters()
    detector = EmbeddingOOD(shrinkage=0.1, n_clusters=1).fit(id_emb)
    id_energy = detector.energy(id_emb[:50])
    far_energy = detector.energy(far_emb[:50])
    # Every far point should beat the WORST id point, since these clusters
    # are synthetically far apart. Real embeddings won't be this clean —
    # that's exactly the point: this isolates whether the scoring direction
    # is correct at all, separate from real-world separability.
    assert far_energy.min() > id_energy.max()


def test_multi_cluster_mean_update_regression():
    """Regression test for the resp[:, k:] vs resp[:, k:k+1] bug: with
    n_clusters=2 on two well-separated blobs, each fitted mean should land
    near ONE true center, not drift toward the wrong shape entirely — this
    is exactly the failure the original indexing bug caused for any
    n_clusters > 1 (it only ever worked by accident at n_clusters=1)."""
    rng = np.random.RandomState(1)
    blob_a = rng.normal(loc=-5.0, scale=0.5, size=(150, 8))
    blob_b = rng.normal(loc=5.0, scale=0.5, size=(150, 8))
    combined = np.vstack([blob_a, blob_b])

    detector = EmbeddingOOD(shrinkage=0.05, n_clusters=2).fit(combined)
    true_centers = sorted([-5.0, 5.0])
    fitted_centers = sorted(detector.means_[:, 0].tolist())
    for true_c, fitted_c in zip(true_centers, fitted_centers):
        assert abs(true_c - fitted_c) < 1.0, (
            f"fitted cluster center {fitted_c:.2f} far from true center {true_c:.2f} — "
            "check the M-step mean update uses resp[:, k:k+1], not resp[:, k:]"
        )


def test_tier_prefix_matching_regression():
    """Regression test for the exact-match-vs-prefix bug: a fake collection
    with tier names matching this PROJECT's real naming (near_ood_text_nlp,
    far_ood_text_news), not the bare strings the original buggy default
    ("near_ood_text", "far_ood_text") required to match anything."""
    class FakeCollection:
        def get(self, include):
            embeddings = (
                [[0.0, 0.0]] * 30
                + [[3.0, 3.0]] * 10
                + [[10.0, 10.0]] * 10
            )
            metadatas = (
                [{"tier": "id_core_text"}] * 30
                + [{"tier": "near_ood_text_nlp"}] * 10
                + [{"tier": "far_ood_text_news"}] * 10
            )
            return {"embeddings": embeddings, "metadatas": metadatas}

    detector = EmbeddingOOD.fit_from_collection(FakeCollection(), shrinkage=0.1)
    assert hasattr(detector, "calibration_weight_"), (
        "calibrator never ran — tier matching silently found zero OOD examples again"
    )


def test_save_load_round_trip(tmp_path):
    id_emb, _, _ = _make_synthetic_clusters()
    detector = EmbeddingOOD(shrinkage=0.1, n_clusters=2).fit(id_emb)
    path = tmp_path / "ood_ref.npz"
    detector.save(str(path))
    loaded = EmbeddingOOD.load(str(path))
    np.testing.assert_allclose(loaded.means_, detector.means_)
    np.testing.assert_allclose(
        loaded.score_embeddings(id_emb[:5])[0]["energy"],
        detector.score_embeddings(id_emb[:5])[0]["energy"],
    )


# ---------------------------------------------------------------------------
# INTEGRATION TESTS — real corpus, real index, real held-out AUROC
# ---------------------------------------------------------------------------

# Set these from YOUR first real measurement, then treat them as regression
# gates going forward — not guesses. A drop from 0.80 to 0.60 on near-OOD
# should fail a test, not get noticed three weeks later by accident.
NEAR_OOD_AUROC_FLOOR = 0.65
FAR_OOD_AUROC_FLOOR = 0.90


@pytest.mark.integration
def test_held_out_auroc_meets_floor_per_tier():
    """The real rigor test. Reported PER TIER deliberately — an aggregate
    AUROC would hide exactly the near-OOD weakness this whole exercise
    exists to catch."""
    import chromadb
    client = chromadb.PersistentClient(path="./chroma_index")
    collection = client.get_collection("text_chunks")

    detector = EmbeddingOOD.fit_from_collection(collection, shrinkage=0.10, threshold=0.95)
    report = getattr(detector, "eval_report_", None)
    assert report is not None, "fit_from_collection produced no held-out eval report"

    near_auroc = report.get("near_ood_auroc") or report.get("auroc_near_ood")
    far_auroc = report.get("far_ood_auroc") or report.get("auroc_far_ood")
    assert near_auroc is not None and far_auroc is not None, (
        "eval report has no per-tier AUROC breakdown — check ood_scoring.py's "
        "eval_report_ field names and adjust the keys above to match"
    )
    assert near_auroc >= NEAR_OOD_AUROC_FLOOR, f"near-OOD AUROC regressed to {near_auroc:.3f}"
    assert far_auroc >= FAR_OOD_AUROC_FLOOR, f"far-OOD AUROC regressed to {far_auroc:.3f}"


@pytest.mark.integration
def test_near_ood_auroc_bootstrap_confidence_interval():
    """A point-estimate AUROC on a small held-out set overstates confidence.
    Bootstrap resampling gives the interval you should actually trust and
    report, not the single number."""
    import chromadb
    from sklearn.metrics import roc_auc_score

    client = chromadb.PersistentClient(path="./chroma_index")
    collection = client.get_collection("text_chunks")
    data = collection.get(include=["embeddings", "metadatas"])

    id_emb = np.array([e for e, m in zip(data["embeddings"], data["metadatas"])
                       if m["tier"] == "id_core_text"])
    near_emb = np.array([e for e, m in zip(data["embeddings"], data["metadatas"])
                         if m["tier"].startswith("near_ood")])
    assert len(id_emb) > 10 and len(near_emb) > 10, "not enough data for a meaningful bootstrap"

    rng = np.random.RandomState(0)
    aurocs = []
    for _ in range(200):
        id_sample = id_emb[rng.choice(len(id_emb), len(id_emb), replace=True)]
        near_sample = near_emb[rng.choice(len(near_emb), len(near_emb), replace=True)]
        detector = EmbeddingOOD(shrinkage=0.1).fit(id_sample)
        scores = np.concatenate([detector.energy(id_sample), detector.energy(near_sample)])
        labels = np.concatenate([np.zeros(len(id_sample)), np.ones(len(near_sample))])
        aurocs.append(roc_auc_score(labels, scores))

    low, high = np.percentile(aurocs, [2.5, 97.5])
    print(f"\nNear-OOD AUROC: {np.mean(aurocs):.3f}  95% bootstrap CI: [{low:.3f}, {high:.3f}]")
    assert high - low < 0.35, (
        f"CI is very wide ([{low:.3f}, {high:.3f}]) — the held-out set is probably too "
        "small to trust a point-estimate AUROC here; that's a sample-size fact, not a bug"
    )


@pytest.mark.integration
@pytest.mark.parametrize("shrinkage", [0.05, 0.10, 0.25, 0.50, 0.75])
def test_shrinkage_sensitivity_on_near_ood_auroc(shrinkage):
    """Validates the shrinkage hyperparameter against evidence instead of
    trusting the 0.10 default — given N is likely close to or below D for
    this corpus (discussed earlier), higher shrinkage may genuinely score
    better. Run this once, read the printed table, then hardcode the winner
    as the new default rather than leaving it at an unvalidated guess."""
    import chromadb
    from sklearn.metrics import roc_auc_score

    client = chromadb.PersistentClient(path="./chroma_index")
    collection = client.get_collection("text_chunks")
    data = collection.get(include=["embeddings", "metadatas"])

    id_emb = np.array([e for e, m in zip(data["embeddings"], data["metadatas"])
                       if m["tier"] == "id_core_text"])
    near_emb = np.array([e for e, m in zip(data["embeddings"], data["metadatas"])
                         if m["tier"].startswith("near_ood")])

    detector = EmbeddingOOD(shrinkage=shrinkage).fit(id_emb)
    scores = np.concatenate([detector.energy(id_emb), detector.energy(near_emb)])
    labels = np.concatenate([np.zeros(len(id_emb)), np.ones(len(near_emb))])
    auroc = roc_auc_score(labels, scores)
    print(f"\nshrinkage={shrinkage:.2f}  near-OOD AUROC={auroc:.3f}")
