"""Embedding-space OOD detection for RAG queries and retrieved chunks.

Fit this once on trusted ``id_core_text`` chunks, save the reference, then
call ``score_retrieval`` after retrieval.  Score the query as well as the
chunks: an off-topic question can retrieve plausible ID snippets.

``fit_from_collection`` holds out ``eval_fraction`` (default 10%) of each
tier and reports AUROC on that split.
"""

import argparse
from typing import Dict, List, Sequence, Tuple
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import logging
import json
import time

logger = logging.getLogger("ood_scoring")
logger.setLevel(logging.INFO)

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent


def load_insurance_queries(path: str | Path) -> list[Dict]:
    """Domain labels only: ID does not imply answerable from retrieved evidence."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    seen_ids, seen_queries, families = set(), set(), {}
    for row in rows:
        group, split = row.get("group"), row.get("split")
        query = " ".join(row.get("query", "").casefold().split())
        family = row.get("family")
        if group not in {"id", "near_ood", "far_ood"} or split not in {"train", "test"}:
            raise ValueError("Insurance queries require group=id/near_ood/far_ood and split=train/test")
        if not query or not row.get("id") or not family or row["id"] in seen_ids or query in seen_queries:
            raise ValueError("Insurance queries require unique IDs/text and a nonempty family")
        if family in families and families[family] != split:
            raise ValueError(f"Query family crosses train/test splits: {family}")
        families[family] = split
        seen_ids.add(row["id"])
        seen_queries.add(query)
    for split in ("train", "test"):
        if {row["group"] for row in rows if row["split"] == split} != {"id", "near_ood", "far_ood"}:
            raise ValueError(f"Each split requires ID, near-OOD and far-OOD queries: {split}")
    if sum(row["split"] == "train" and row["group"] == "id" for row in rows) < 2:
        raise ValueError("At least two training ID queries are required for leave-one-out kNN")
    return rows


def insurance_reference(collection, scope: Dict, embedding_model: str) -> tuple[list, Dict]:
    """Select a document allowlist from a pinned insurance index, not tier tags."""
    metadata = dict(collection.metadata or {})
    if scope.get("collection", collection.name) != collection.name:
        raise ValueError("Collection does not match the insurance scope")
    if scope.get("embedding_model", embedding_model) != embedding_model:
        raise ValueError("Embedding model does not match the insurance scope")
    if metadata.get("embedding_model") != embedding_model:
        raise ValueError("Embedding model does not match the insurance index")
    if metadata.get("hnsw:space") != "cosine":
        raise ValueError("Insurance reference requires the indexed cosine geometry")
    if not scope.get("corpus_fingerprint") or metadata.get("corpus_fingerprint") != scope["corpus_fingerprint"]:
        raise ValueError("Insurance corpus fingerprint mismatch; review the scope before refitting")
    allowed = set(scope.get("trusted_document_ids", []))
    if not allowed:
        raise ValueError("Insurance scope requires an explicit trusted document allowlist")
    data = collection.get(include=["embeddings", "metadatas"])
    vectors, selected_ids, documents, excluded = [], [], {}, {}
    for cid, vector, meta in zip(data["ids"], data["embeddings"], data["metadatas"]):
        doc_id = meta.get("doc_id")
        description = {"title": meta.get("title", ""), "source_type": meta.get("source_type", "")}
        if doc_id not in allowed:
            excluded[doc_id or "unknown"] = description
            continue
        vector = np.asarray(vector, dtype=np.float64)
        if vector.ndim != 1 or not np.isfinite(vector).all() or np.linalg.norm(vector) <= 0:
            raise ValueError(f"Invalid reference embedding: {cid}")
        vectors.append(vector)
        selected_ids.append(cid)
        documents[doc_id] = description
    if allowed - documents.keys():
        raise ValueError(f"Allowlisted documents absent from collection: {sorted(allowed - documents.keys())}")
    if len(vectors) < 2 or len({len(vector) for vector in vectors}) != 1:
        raise ValueError("Insurance reference requires at least two embeddings with consistent dimensions")
    import hashlib
    provenance = {
        "domain": "home_insurance", "embedding_model": embedding_model,
        "collection": collection.name, "collection_metadata": metadata,
        "corpus_fingerprint": metadata["corpus_fingerprint"],
        "reference_chunk_count": len(vectors), "embedding_dimension": len(vectors[0]),
        "trusted_documents": documents, "excluded_documents": excluded,
        "selected_chunk_ids_sha256": hashlib.sha256(json.dumps(sorted(selected_ids)).encode()).hexdigest(),
        "label_semantics": "Domain membership only; ID does not establish answer support",
    }
    return vectors, provenance


def evaluate_insurance_queries(detector, rows: list[Dict], embeddings) -> Dict:
    """Report untouched test-query scores and the existing band's actual action."""
    if len(rows) != len(embeddings):
        raise ValueError("Evaluation queries and embeddings must have matching lengths")
    predictions = [{**row, **detector.score_query(vector)} for row, vector in zip(rows, embeddings)]
    groups = {group: [row for row in predictions if row["group"] == group]
              for group in ("id", "near_ood", "far_ood")}
    summary = {}
    for group, cases in groups.items():
        summary[group] = {
            "count": len(cases),
            "reject_rate": float(np.mean([case["is_ood"] for case in cases])) if cases else None,
            "near_band_rate": float(np.mean([case["is_near_ood"] for case in cases])) if cases else None,
        }
    comparisons = {}
    for group in ("near_ood", "far_ood"):
        cases = groups["id"] + groups[group]
        labels = [case["group"] != "id" for case in cases]
        comparisons[f"id_vs_{group}"] = {
            f"auroc_{score}": auroc(labels, [case[score] for case in cases])
            for score in ("energy", "knn_distance", "ood_probability")
        }
    return {"groups": summary, "comparisons": comparisons, "predictions": predictions,
            "decision_policy": "Existing energy AND kNN thresholds; probabilities do not select the band",
            "interval_note": "ood_confidence_interval describes the energy percentile, not logistic probability uncertainty"}


def fit_insurance_ood(collection, scope: Dict, query_rows: list[Dict], model,
                      embedding_model: str = "all-MiniLM-L6-v2", **kwargs):
    """Keep the scoring architecture; fit query references on train only."""
    vectors, provenance = insurance_reference(collection, scope, embedding_model)
    encoded = np.asarray(model.encode([row["query"] for row in query_rows], show_progress_bar=False), dtype=np.float64)
    if encoded.shape != (len(query_rows), provenance["embedding_dimension"]) or not np.isfinite(encoded).all():
        raise ValueError("Query embeddings must match the indexed embedding dimension and be finite")
    if np.any(np.linalg.norm(encoded, axis=1) <= 0):
        raise ValueError("Query embeddings must be nonzero")
    train_id = [vector for row, vector in zip(query_rows, encoded) if row["split"] == "train" and row["group"] == "id"]
    train_ood = [vector for row, vector in zip(query_rows, encoded) if row["split"] == "train" and row["group"] != "id"]
    if len(train_id) < 2 or not train_ood:
        raise ValueError("Insurance fitting requires at least two ID and one OOD training queries")
    detector = EmbeddingOOD(**kwargs).fit(vectors)
    detector.fit_query_reference(train_id, train_ood)
    detector.artifact_metadata_ = provenance
    test = [(row, vector) for row, vector in zip(query_rows, encoded) if row["split"] == "test"]
    report = evaluate_insurance_queries(detector, [row for row, _ in test], [vector for _, vector in test])
    return detector, {"provenance": provenance, "test": report,
                      "training_query_counts": {group: sum(row["split"] == "train" and row["group"] == group for row in query_rows)
                                                for group in ("id", "near_ood", "far_ood")},
                      "chunk_probability_source": "reference_tail (no labelled OOD chunk calibration)"}

def _wilson_interval(successes: float, n: int, z: float = 1.96):
    """95% binomial interval, used for the empirical OOD percentile."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def auroc(labels, scores) -> float:
    """AUROC for a higher-is-more-OOD score. ``labels`` are 1=OOD, 0=ID.

    Mann-Whitney form, so ties count as 0.5.  Returns nan if a class is missing.
    """
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = np.sum(pos[:, None] > neg[None, :])
    equal = np.sum(pos[:, None] == neg[None, :])
    return float((greater + 0.5 * equal) / (len(pos) * len(neg)))


def holdout_split(rows: Sequence[tuple], eval_fraction: float = 0.10,
                  seed: int = 0) -> Tuple[list, list]:
    """Stratified train/eval split. Each row is ``(embedding, label, group, ...)``.

    ``eval_fraction`` is the share held out per group (default 10%).  Set it
    to 0 to fit on everything.  Groups (id / near_ood / far_ood) are split
    separately so the eval set is not all one tier.
    """
    if eval_fraction < 0 or eval_fraction >= 1:
        raise ValueError("eval_fraction must be in [0, 1)")
    if eval_fraction == 0 or not rows:
        return list(rows), []

    rng = np.random.default_rng(seed)
    by_group: Dict[str, list] = {}
    for row in rows:
        by_group.setdefault(row[2], []).append(row)

    train, eval_rows = [], []
    for items in by_group.values():
        order = rng.permutation(len(items))
        n_eval = int(round(len(items) * eval_fraction))
        n_eval = min(max(n_eval, 0), len(items) - 1)
        eval_sel = set(order[:n_eval].tolist())
        for i, item in enumerate(items):
            (eval_rows if i in eval_sel else train).append(item)
    return train, eval_rows


def labeled_rows_from_collection(collection, id_tier: str = "id_core_text",
                                 ood_tier_prefixes=("near_ood", "far_ood")):
    """Pack collection embeddings as ``(embedding, label, group, tier)`` rows."""
    data = collection.get(include=["embeddings", "metadatas"])
    rows = []
    for emb, meta in zip(data["embeddings"], data["metadatas"]):
        tier = meta.get("tier", "")
        if tier == id_tier:
            group, label = "id", 0
        elif tier.startswith(ood_tier_prefixes):
            group = "near_ood" if tier.startswith("near_ood") else "far_ood"
            label = 1
        else:
            continue
        rows.append((np.asarray(emb, dtype=np.float64), label, group, tier))
    return rows


class EmbeddingOOD:
    """Regularised Gaussian reference distribution for one embedding space.

    ``shrinkage`` makes covariance inversion stable when embedding dimensions
    approach (or exceed) the number of reference chunks.  ``threshold`` is an
    empirical reference-energy percentile, e.g. .95 flags the most unusual 5%.
    """

    def __init__(self, shrinkage: float = 0.60, threshold: float = 0.95,
                 near_threshold: float = 0.80, n_clusters: int = 1,
                 max_iter: int = 100, tol: float = 1e-4, knn_k: int = 5):
        if not 0 <= shrinkage <= 1 or not 0 < threshold < 1:
            raise ValueError("shrinkage must be in [0, 1] and threshold in (0, 1)")
        if not 0 < near_threshold < threshold:
            raise ValueError("near_threshold must be in (0, threshold)")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.shrinkage = shrinkage
        self.threshold = threshold
        self.near_threshold = near_threshold
        self.knn_k = knn_k
    
    def query_knn_distance(
        self,embeddings: Sequence[Sequence[float]],k: int = None,exclude_self: bool = False,
    ) -> np.ndarray:

        if not hasattr(self, "query_reference_embeddings_"):
            raise RuntimeError(
                "Query reference embeddings are not fitted."
            )

        k = k or self.knn_k
        X = self._normalize(embeddings)
        ref = self.query_reference_embeddings_
        dist = 1.0 - (X @ ref.T)

        if exclude_self:
            take = min(k + 1, len(ref))

            nearest = np.partition(dist,kth=take - 1,axis=1,)[:, :take]
            nearest = np.sort(nearest, axis=1)
            # First neighbour should be the point itself
            nearest = nearest[:, 1:k + 1]

        else:
            take = min(k, len(ref))
            nearest = np.partition(dist,kth=take - 1,axis=1,)[:, :take]

        return np.mean(nearest, axis=1)

    def fit_query_reference(
        self,
        id_queries: Sequence[Sequence[float]],
        ood_queries: Sequence[Sequence[float]] | None = None,
        steps: int = 2000,
        learning_rate: float = 0.05,
    ):
        """
        Fit query-specific ID thresholds and, when OOD queries are supplied,
        a separate query-level energy -> P(OOD) calibrator.

        id_queries / ood_queries must already be embeddings produced by
        the same embedding model used by the RAG system.
        """

        if id_queries is None or len(id_queries) == 0:
            raise ValueError("fit_query_reference requires at least one ID query")

        X_id = np.asarray(id_queries, dtype=np.float64)

        if X_id.ndim != 2:
            raise ValueError(
                "id_queries must have shape (n_queries, embedding_dimension)"
            )

        # energy() and knn_distance_min() will use the same normalized
        # embedding geometry as the main detector.
        id_energy = self.energy(X_id)
        self.query_reference_embeddings_ = self._normalize(X_id)
        id_knn = self.query_knn_distance(X_id,exclude_self=True)

        # ---------------------------------------------------------
        # Query-specific decision thresholds
        # ---------------------------------------------------------
        self.query_energy_threshold_ = float(
            np.quantile(id_energy, self.threshold)
        )
        self.query_knn_threshold_ = float(
            np.quantile(id_knn, self.threshold)
        )
        self.query_near_energy_threshold_ = float(
            np.quantile(id_energy, self.near_threshold)
        )
        self.query_near_knn_threshold_ = float(
            np.quantile(id_knn, self.near_threshold)
        )

        # Keep these for diagnostics
        self.query_reference_energy_ = id_energy
        self.query_reference_knn_ = id_knn

        # No labelled OOD queries -> thresholds are still usable,
        # but we cannot claim a calibrated P(OOD).
        if ood_queries is None or len(ood_queries) == 0:
            return self

        X_ood = np.asarray(ood_queries, dtype=np.float64)

        if X_ood.ndim != 2:
            raise ValueError(
                "ood_queries must have shape (n_queries, embedding_dimension)"
            )

        ood_energy = self.energy(X_ood)

        # ---------------------------------------------------------
        # Query-specific probability calibration
        # ---------------------------------------------------------
        energy = np.concatenate([id_energy, ood_energy])
        ood_knn = self.query_knn_distance(X_ood)

        # =========================================================
        # CHANGED: build 2D calibration features:
        # [energy, kNN distance]
        # =========================================================
        id_features = np.column_stack([
            id_energy,
            id_knn,
        ])

        ood_features = np.column_stack([
            ood_energy,
            ood_knn,
        ])

        calibration_features = np.vstack([
            id_features,
            ood_features,
        ])

        labels = np.concatenate([
            np.zeros(
                len(id_energy),
                dtype=np.int64,
            ),
            np.ones(
                len(ood_energy),
                dtype=np.int64,
            ),
        ])

        # =========================================================
        # NEW: standardize energy and kNN before logistic regression.
        #
        # Important because energy and kNN are on very different
        # numerical scales.
        # =========================================================
        self.query_calibration_scaler_ = StandardScaler()
        calibration_features_scaled = (
            self.query_calibration_scaler_.fit_transform(
                calibration_features
            )
        )
        # =========================================================
        # NEW: Logistic Regression calibrator
        # Learns:P(OOD | energy, kNN)
        # instead of the old:P(OOD | energy)
        # =========================================================
        self.query_calibrator_ = LogisticRegression(
            class_weight="balanced",
            C=1.0,
            random_state=42,
            max_iter=2000,
        )
        self.query_calibrator_.fit(
            calibration_features_scaled,
            labels,
        )
        # =========================================================
        # NEW: retain coefficients for diagnostics / logging
        # =========================================================
        self.query_calibration_energy_weight_ = float(
            self.query_calibrator_.coef_[0][0]
        )
        self.query_calibration_knn_weight_ = float(
            self.query_calibrator_.coef_[0][1]
        )
        self.query_calibration_bias_ = float(
            self.query_calibrator_.intercept_[0]
        )
        return self

    def fit(self, embeddings: Sequence[Sequence[float]]):
        X = np.asarray(embeddings, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(
                "fit requires embeddings shaped "
                "(n_samples, embedding_dimension)"
            )

        N, D = X.shape
        K = self.n_clusters

        if N < K:
            raise ValueError(
                f"fit needs at least {K} embeddings; got {N}"
            )

        # IMPORTANT:
        # Gaussian, Mahalanobis, energy and kNN now operate in
        # a consistent cosine-like embedding geometry.
        X = self._normalize(X)

        rng = np.random.default_rng(42)

        self.means_ = X[
            rng.choice(N, K, replace=False)
        ].copy()

        self.weights_ = np.ones(K, dtype=np.float64) / K
        self.precisions_ = np.array(
            [np.eye(D, dtype=np.float64) for _ in range(K)]
        )
        self.log_dets_ = np.zeros(K, dtype=np.float64)

        log_likelihood = -np.inf

        for _ in range(self.max_iter):

            # ---------------- E-step ----------------
            log_resp = np.zeros((N, K), dtype=np.float64)

            for k in range(K):
                diff = X - self.means_[k]
                mahal_sq = np.sum((diff @ self.precisions_[k]) * diff,axis=1,)
                log_pdf = -0.5 * (D * np.log(2.0 * np.pi) + self.log_dets_[k]+ mahal_sq)
                log_resp[:, k] = (np.log(self.weights_[k] + 1e-12) + log_pdf)
            max_log_resp = np.max(
                log_resp,
                axis=1,
                keepdims=True,
            )

            log_sum_exp = (
                max_log_resp
                + np.log(
                    np.sum(
                        np.exp(log_resp - max_log_resp),
                        axis=1,
                        keepdims=True,
                    )
                )
            )

            resp = np.exp(log_resp - log_sum_exp)
            current_log_likelihood = float(
                np.sum(log_sum_exp)
            )
            if (
                np.isfinite(log_likelihood)
                and abs(
                    current_log_likelihood - log_likelihood
                ) < self.tol
            ):
                break

            log_likelihood = current_log_likelihood

            # ---------------- M-step ----------------
            N_k = resp.sum(axis=0) + 1e-10
            self.weights_ = N_k / N

            for k in range(K):
                self.means_[k] = (np.sum(resp[:, [k]] * X, axis=0)/ N_k[k])
                diff = X - self.means_[k]
                cov = (
                    (diff.T * resp[:, k]) @ diff
                ) / N_k[k]

                scale = float(np.trace(cov) / D)

                # Avoid a degenerate covariance floor.
                scale = max(scale, 1e-8)

                cov = (
                    (1.0 - self.shrinkage) * cov
                    + self.shrinkage
                    * scale
                    * np.eye(D)
                )

                values, vectors = np.linalg.eigh(cov)

                eigen_floor = max(
                    np.finfo(float).eps,
                    scale * 1e-6,
                )

                values = np.maximum(
                    values,
                    eigen_floor,
                )

                self.precisions_[k] = (
                    vectors / values
                ) @ vectors.T

                self.log_dets_[k] = float(
                    np.log(values).sum()
                )

        # ---------------------------------------------------------
        # Reference distributions
        # ---------------------------------------------------------

        self.reference_embeddings_ = X
        self.reference_energy_ = self.energy(X)

        self.energy_threshold_ = float(
            np.quantile(
                self.reference_energy_,
                self.threshold,
            )
        )

        self.near_energy_threshold_ = float(
            np.quantile(
                self.reference_energy_,
                self.near_threshold,
            )
        )

        self.reference_knn_ = self.knn_distance_min(X,exclude_self=True,)

        self.knn_threshold_ = float(
            np.quantile(
                self.reference_knn_,
                self.threshold,
            )
        )

        self.near_knn_threshold_ = float(
            np.quantile(
                self.reference_knn_,
                self.near_threshold,
            )
        )

        logger.info(json.dumps({
            "event": "ood_reference_fit_complete",
            "reference_count": N,
            "embedding_dimension": D,
            "n_clusters": K,
            "energy_threshold": self.energy_threshold_,
            "knn_threshold": self.knn_threshold_,
        }))

        return self

    @classmethod
    def from_collection(cls, collection, tier: str = "id_core_text", **kwargs):
        """Fit from a Chroma collection without recomputing embeddings."""
        rows = labeled_rows_from_collection(collection, id_tier=tier, ood_tier_prefixes=())
        embeddings = [r[0] for r in rows]
        if not embeddings:
            raise ValueError(f"no embeddings found for tier={tier!r}")
        return cls(**kwargs).fit(embeddings)

    @classmethod
    def fit_from_collection(cls, collection,id_queries: Sequence[Sequence[float]] = None, ood_queries:Sequence[Sequence[float]] = None, id_tier: str = "id_core_text",
                            ood_tier_prefixes=("near_ood", "far_ood"),
                            eval_fraction: float = 0.10, eval_seed: int = 0, **kwargs):
        """Fit on a train split; hold out ``eval_fraction`` of each tier for AUROC."""
        rows = labeled_rows_from_collection(collection, id_tier, ood_tier_prefixes)
        train, eval_rows = holdout_split(rows, eval_fraction=eval_fraction, seed=eval_seed)
        id_embeddings = [r[0] for r in train if r[1] == 0]
        ood_embeddings = [r[0] for r in train if r[1] == 1]
        if not id_embeddings:
            raise ValueError(f"no embeddings found for tier={id_tier!r}")
        detector = cls(**kwargs).fit(id_embeddings)
        if ood_embeddings:
            detector.fit_probability_calibrator(ood_embeddings)
        detector.eval_report_ = detector.evaluate(eval_rows) if eval_rows else {}
        detector.fit_query_reference(id_queries,ood_queries)
        return detector

    def evaluate(self, eval_rows: Sequence[tuple]) -> Dict:
        """AUROC on a hold-out set of ``(embedding, label, group, ...)`` rows."""
        if not eval_rows:
            return {}
        embeddings = np.stack([r[0] for r in eval_rows])
        labels = np.array([r[1] for r in eval_rows])
        groups = [r[2] for r in eval_rows]
        energy = self.energy(embeddings)
        ood_energies = [e for e, l in zip(energy, labels) if l == 1] 
        tau = np.quantile(ood_energies, 0.05)
        false_positive_rate = np.mean([e for e, l in zip(energy, labels) if l == 0] > tau)
        
        knn = self.knn_distance_min(embeddings)
        report = {
            "n": int(len(labels)),
            "n_id": int((labels == 0).sum()),
            "n_ood": int((labels == 1).sum()),
            "auroc_energy": auroc(labels, energy),
            "auroc_knn": auroc(labels, knn),
            "false_positive_rate": false_positive_rate,
        }
        id_mask = np.array([g == "id" for g in groups])
        for name in ("near_ood", "far_ood"):
            mask = id_mask | np.array([g == name for g in groups])
            if id_mask.any() and mask.sum() > id_mask.sum():
                report[f"auroc_energy_{name}"] = auroc(labels[mask], energy[mask])
                report[f"auroc_knn_{name}"] = auroc(labels[mask], knn[mask])
        return report

    def fit_probability_calibrator(self, ood_embeddings: Sequence[Sequence[float]],
                                   steps: int = 2000, learning_rate: float = 0.05):
        """A small logistic calibrator: mixture energy -> P(OOD)."""
        id_energy = self.reference_energy_
        ood_energy = self.energy(ood_embeddings)
        energy = np.concatenate([id_energy, ood_energy])
        labels = np.concatenate([np.zeros(len(id_energy)), np.ones(len(ood_energy))])
        self.calibration_mean_ = float(energy.mean())
        self.calibration_scale_ = float(energy.std() + 1e-9)
        x = (energy - self.calibration_mean_) / self.calibration_scale_
        weight = 0.0
        bias = float(np.log((labels.mean() + 1e-9) / (1 - labels.mean() + 1e-9)))
        for _ in range(steps):
            residual = _sigmoid(weight * x + bias) - labels
            weight -= learning_rate * (np.mean(residual * x) + 1e-4 * weight)
            bias -= learning_rate * np.mean(residual)
        self.calibration_weight_, self.calibration_bias_ = float(weight), float(bias)
        return self

    def mahalanobis(self, embeddings: Sequence[Sequence[float]]) -> np.ndarray:
        """Computes the minimum Mahalanobis distance to any cluster center for each point."""
        #X = np.asarray(embeddings, dtype=np.float64)
        X = self._normalize(embeddings)
        N = X.shape[0]
        K = self.n_clusters
        
        # Store distance from each point to each cluster
        all_distances = np.zeros((N, K))
        
        for k in range(K):
            diff = X - self.means_[k]
            # d_M^2 = (x - μ)^T Σ^-1 (x - μ)
            squared_dist = np.sum((diff @ self.precisions_[k]) * diff, axis=1)
            all_distances[:, k] = np.sqrt(np.maximum(squared_dist, 0.0))
            
        # Return distance to the closest component center
        return np.min(all_distances, axis=1)
    
    def _normalize(self, embeddings: Sequence[Sequence[float]]) -> np.ndarray:
        """Helper to ensure uniform L2-normalization across all metrics."""
        X = np.asarray(embeddings, dtype=np.float64)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / (norms + 1e-9)

    def energy(self, embeddings: Sequence[Sequence[float]]) -> np.ndarray:
        """Calculates mixture score (e.g., negative log-likelihood/free energy) for each point."""
        X = self._normalize(embeddings)
        #X = np.asarray(embeddings, dtype=np.float64)
        N, D = X.shape
        K = self.n_clusters
        log_resp = np.zeros((N, K))
        
        for k in range(K):
            diff = X - self.means_[k]
            mahalanobis = np.sum((diff @ self.precisions_[k]) * diff, axis=1)
            log_pdf = -0.5 * ( self.log_dets_[k] + mahalanobis) #D * np.log(2 * np.pi) +
            log_resp[:, k] = np.log(self.weights_[k] + 1e-12) + log_pdf

        # Log-Sum-Exp over all components to get overall mixture likelihood
        max_log_resp = np.max(log_resp, axis=1, keepdims=True)
        log_likelihoods = (max_log_resp + np.log(np.sum(np.exp(log_resp - max_log_resp), axis=1, keepdims=True))).reshape(-1)
        
        return -log_likelihoods

    def knn_distance(self, embeddings: Sequence[Sequence[float]], k: int = None,
                     exclude_self: bool = False) -> np.ndarray:
        """Mean cosine distance to the k nearest ID reference embeddings.

        Near-OOD abstracts often sit inside a shrunk Gaussian but still away
        from actual ID neighbors.  kNN catches that local gap.
        """
        k = k or self.knn_k
        X = np.asarray(embeddings, dtype=np.float64)
        ref = self.reference_embeddings_
        take = min(k + int(exclude_self), len(ref))
        x_n = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        r_n = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-9)
        dist = 1.0 - (x_n @ r_n.T)
        knn = np.sort(np.partition(dist, kth=take - 1, axis=1)[:, :take], axis=1)
        if exclude_self:
            knn = knn[:, 1:k + 1]
        return np.atleast_1d(knn.mean(axis=1))
    
    def knn_distance_min(self, embeddings: Sequence[Sequence[float]], k: int = None,
                 exclude_self: bool = False) -> np.ndarray:
        k = k or self.knn_k
        X = self._normalize(np.asarray(embeddings, dtype=np.float64))
        ref = self.reference_embeddings_
        take = min(k + int(exclude_self), len(ref))
        
        # L2 normalize
        x_n = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        r_n = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-9)
        
        dist = 1.0 - (x_n @ r_n.T)
        knn = np.sort(np.partition(dist, kth=take - 1, axis=1)[:, :take], axis=1)
        
        if exclude_self:
            knn = knn[:, 1:k + 1]
            
        # FIX: Take the MINIMUM distance to the nearest neighbor (1-NN), NOT the mean!
        return np.atleast_1d(knn[:, 0])

    def _ood_band(self, energy: float, knn: float, query: bool = False) -> str:
        """id / near_ood / ood; both energy and kNN must exceed band thresholds."""
        if query:
            energy_threshold = self.query_energy_threshold_
            knn_threshold = self.query_knn_threshold_
            near_energy_threshold = self.query_near_energy_threshold_
            near_knn_threshold = self.query_near_knn_threshold_
        else:
            energy_threshold = self.energy_threshold_
            knn_threshold = self.knn_threshold_
            near_energy_threshold = self.near_energy_threshold_
            near_knn_threshold = getattr(self, "near_knn_threshold_", self.knn_threshold_)
        if energy >= energy_threshold and knn >= knn_threshold:
            return "ood"
        if energy >= near_energy_threshold and knn >= near_knn_threshold:
            return "near_ood"
        return "id"

    def score_embeddings(
        self,
        embeddings: Sequence[Sequence[float]],
        query: bool = False,
    ) -> List[Dict]:

        energies = self.energy(embeddings)
        distances = self.mahalanobis(embeddings)
        if query:
            knn_distances = self.query_knn_distance(embeddings)
        else:
            knn_distances = self.knn_distance_min(embeddings)

        reference_energy = np.asarray(
            self.query_reference_energy_
            if query and hasattr(self, "query_reference_energy_")
            else self.reference_energy_
        )

        if len(reference_energy) == 0:
            raise RuntimeError("OOD reference energy distribution is empty.")

        if query:
            calibration_available = (
                hasattr(self, "query_calibration_scaler_")
                and hasattr(self, "query_calibrator_")
            )
        else:
            calibration_available = all(
                hasattr(self, attr)
                for attr in (
                    "calibration_mean_",
                    "calibration_scale_",
                    "calibration_weight_",
                    "calibration_bias_",
                )
            )

        results = []

        for energy, distance, knn_d in zip(
            energies, distances, knn_distances
        ):
            energy = float(energy)
            distance = float(distance)
            knn_d = float(knn_d)

            percentile = float(
                np.mean(reference_energy <= energy)
            )

            low, high = _wilson_interval(
                percentile * len(reference_energy),
                len(reference_energy),
            )

            if query and calibration_available:
                features = np.array(
                    [[energy, knn_d]],
                    dtype=np.float64,
                )

                features = (
                    self.query_calibration_scaler_
                    .transform(features)
                )

                probability = float(
                    self.query_calibrator_
                    .predict_proba(features)[0, 1]
                )

                probability_source = (
                    "query_energy_knn_logistic"
                )

            elif not query and calibration_available:
                scale = max(
                    float(self.calibration_scale_),
                    1e-8,
                )

                z_energy = (
                    energy - float(self.calibration_mean_)
                ) / scale

                logit = (
                    float(self.calibration_weight_) * z_energy
                    + float(self.calibration_bias_)
                )

                probability = float(_sigmoid(logit))
                probability_source = "chunk_label_calibrated"

            else:
                probability = percentile
                probability_source = "reference_tail"

            band = self._ood_band(
                energy,
                knn_d,
                query=query,
            )
            results.append({
            "mahalanobis": distance,
            "energy": energy,
            "knn_distance": knn_d,
            "ood_percentile": percentile,
            "ood_confidence_interval": (
                float(low),
                float(high),
            ),
            "ood_probability": probability,
            "probability_source": probability_source,
            "ood_band": band,
            "is_near_ood": band == "near_ood",
            "is_ood": band == "ood",
            })
        return results

    def score_query(self, embedding: Sequence[float]) -> Dict:
        """Score a query embedding against the ID document reference.

        Same space as chunks (symmetric MiniLM).  An off-topic question can
        still retrieve ID snippets; those chunks look in-distribution while
        the query does not.
        """
        start = time.perf_counter()
        try:
            result = self.score_embeddings([embedding], query=True)[0]
            latency_ms = (time.perf_counter() - start) * 1000
            logger.info(json.dumps({
                "event": "ood_query_scored",
                "ood_band": result["ood_band"],
                "ood_probability": round(
                    result["ood_probability"], 4
                ),
                "energy": round(result["energy"], 4),
                "knn_distance": round(
                    result["knn_distance"], 4
                ),
                "probability_source": result[
                    "probability_source"
                ],
                "latency_ms": round(latency_ms, 2),
            }))
            return result
        except Exception:
            logger.exception(json.dumps({
                "event": "ood_query_scoring_failed",
            }))
            raise

    def score_chunks(self, chunks: List[Dict], collection) -> List[Dict]:
        """Annotate retrieval output in place with OOD scores and return it."""
        if not chunks:
            return chunks
        data = collection.get(ids=[c["id"] for c in chunks], include=["embeddings"])
        by_id = dict(zip(data["ids"], data["embeddings"]))
        missing = [c["id"] for c in chunks if c["id"] not in by_id]
        if missing:
            logger.error(json.dumps({
                "event": "ood_chunk_embeddings_missing",
                "missing_count": len(missing),
                "sample_ids": missing[:3],
            }))
            raise ValueError(f"retrieved IDs absent from collection: {missing[:3]}")
        for chunk, score in zip(chunks, self.score_embeddings([by_id[c["id"]] for c in chunks])):
            chunk["ood"] = score
        return chunks

    def score_retrieval(self, query_embedding: Sequence[float],
                        chunks: List[Dict], collection) -> List[Dict]:
        """Score the query and the retrieved chunks.  Request OOD follows the query.

        Wrong-question / plausible-ID-snippet: chunks may be ``is_ood=False``
        while ``request_is_ood`` is True.
        """
        self.score_chunks(chunks, collection)
        query_ood = self.score_query(query_embedding)
        for chunk in chunks:
            chunk["query_ood"] = query_ood
            chunk["request_is_ood"] = bool(query_ood["is_ood"])
            chunk["request_is_near_ood"] = bool(query_ood["is_near_ood"])
        return chunks

    def save(self, path: str):
        np.savez_compressed(
            path,
            artifact_metadata_json=json.dumps(getattr(self, "artifact_metadata_", {})),
            means=self.means_,
            precisions=self.precisions_,
            weights=self.weights_,
            log_dets=self.log_dets_,
            reference_energy=self.reference_energy_,
            reference_embeddings=self.reference_embeddings_,
            shrinkage=self.shrinkage,
            threshold=self.threshold,
            near_threshold=self.near_threshold,
            energy_threshold=self.energy_threshold_,
            near_energy_threshold=self.near_energy_threshold_,
            knn_k=self.knn_k,
            knn_threshold=self.knn_threshold_,
            near_knn_threshold=self.near_knn_threshold_,
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            tol=self.tol,
            calibration_mean=getattr(self, "calibration_mean_", np.nan),
            calibration_scale=getattr(self, "calibration_scale_", np.nan),
            calibration_weight=getattr(self, "calibration_weight_", np.nan),
            calibration_bias=getattr(self, "calibration_bias_", np.nan),
            query_calibration_mean=getattr(self, "query_calibration_mean_", np.nan),
            query_calibration_scale=getattr(self, "query_calibration_scale_", np.nan),
            query_calibration_weight=getattr(self, "query_calibration_weight_", np.nan),
            query_calibration_bias=getattr(self, "query_calibration_bias_", np.nan),

            query_energy_threshold=getattr(self, "query_energy_threshold_", np.nan),
            query_knn_threshold=getattr(self, "query_knn_threshold_", np.nan),
            query_near_energy_threshold=getattr(self, "query_near_energy_threshold_", np.nan),
            query_near_knn_threshold=getattr(self, "query_near_knn_threshold_", np.nan),
            query_reference_energy=getattr(self, "query_reference_energy_", np.array([])),
            query_reference_knn=getattr(self, "query_reference_knn_", np.array([])),
            query_scaler_mean=getattr(getattr(self, "query_calibration_scaler_", None),"mean_",np.array([]),),
            query_scaler_scale=getattr(getattr(self, "query_calibration_scaler_", None),"scale_",np.array([]),),
            query_lr_coef=getattr(getattr(self, "query_calibrator_", None),"coef_",np.array([]),),
            query_lr_intercept=getattr(getattr(self, "query_calibrator_", None),"intercept_",np.array([]),),
            query_lr_classes=getattr(getattr(self, "query_calibrator_", None),"classes_",np.array([]),),
            query_reference_embeddings=getattr(self,"query_reference_embeddings_",np.array([]),),

        )

    @classmethod
    def load(cls, path: str):
        try:
            with np.load(path) as archive:
                data = {key: archive[key] for key in archive.files}
            n_clusters = int(data["n_clusters"]) if "n_clusters" in data else 1
            max_iter = int(data["max_iter"]) if "max_iter" in data else 100
            tol = float(data["tol"]) if "tol" in data else 1e-4

            detector = cls(
                shrinkage=float(data["shrinkage"]),
                threshold=float(data["threshold"]),
                near_threshold=float(data["near_threshold"]) if "near_threshold" in data else 0.80,
                n_clusters=n_clusters,
                max_iter=max_iter,
                tol=tol,
                knn_k=int(data["knn_k"]) if "knn_k" in data else 5,
            )
            detector.artifact_metadata_ = json.loads(str(data["artifact_metadata_json"])) if "artifact_metadata_json" in data else {}
            
            detector.means_ = data["means"] if "means" in data else data["mean"]
            detector.precisions_ = data["precisions"] if "precisions" in data else data["precision"]
            detector.weights_ = data["weights"] if "weights" in data else np.ones(n_clusters) / n_clusters
            detector.log_dets_ = data["log_dets"] if "log_dets" in data else data["log_det"]
            detector.reference_energy_ = data["reference_energy"]
            detector.energy_threshold_ = float(data["energy_threshold"])
            detector.near_energy_threshold_ = float(data["near_energy_threshold"]) if "near_energy_threshold" in data else detector.energy_threshold_
            
            if "query_energy_threshold" in data:
                detector.query_energy_threshold_ = float(data["query_energy_threshold"])
                detector.query_knn_threshold_ = float(data["query_knn_threshold"])
                detector.query_near_energy_threshold_ = float(data["query_near_energy_threshold"])
                detector.query_near_knn_threshold_ = float(data["query_near_knn_threshold"])
                
            if "reference_embeddings" in data:
                detector.reference_embeddings_ = data["reference_embeddings"]
                detector.knn_threshold_ = float(data["knn_threshold"]) if "knn_threshold" in data else 1.0
                detector.near_knn_threshold_ = float(data["near_knn_threshold"]) if "near_knn_threshold" in data else detector.knn_threshold_
            else:
                detector.reference_embeddings_ = np.zeros((1, detector.means_.shape[-1]))
                detector.knn_threshold_ = 1.0
                detector.near_knn_threshold_ = 1.0
            if "calibration_weight" in data and np.isfinite(data["calibration_weight"]):
                detector.calibration_mean_ = float(data["calibration_mean"])
                detector.calibration_scale_ = float(data["calibration_scale"])
                detector.calibration_weight_ = float(data["calibration_weight"])
                detector.calibration_bias_ = float(data["calibration_bias"])
            if ("query_calibration_weight" in data and np.isfinite(data["query_calibration_weight"])):
                detector.query_calibration_mean_ = float(data["query_calibration_mean"])
                detector.query_calibration_scale_ = float(data["query_calibration_scale"])
                detector.query_calibration_weight_ = float(data["query_calibration_weight"])
                detector.query_calibration_bias_ = float(data["query_calibration_bias"])
            
            if "query_reference_energy" in data:
                detector.query_reference_energy_ = data["query_reference_energy"]
            if "query_reference_knn" in data:
                detector.query_reference_knn_ = data["query_reference_knn"]
            if "query_scaler_mean" in data and len(data["query_scaler_mean"]) > 0:
                scaler = StandardScaler()
                scaler.mean_ = data["query_scaler_mean"]
                scaler.scale_ = data["query_scaler_scale"]
                # Required by sklearn transform()
                scaler.var_ = scaler.scale_ ** 2
                scaler.n_features_in_ = len(scaler.mean_)
                detector.query_calibration_scaler_ = scaler
            if "query_lr_coef" in data and data["query_lr_coef"].size > 0:
                calibrator = LogisticRegression()
                calibrator.coef_ = data["query_lr_coef"]
                calibrator.intercept_ = data["query_lr_intercept"]
                calibrator.classes_ = data["query_lr_classes"]
                calibrator.n_features_in_ = calibrator.coef_.shape[1]
                detector.query_calibrator_ = calibrator
            if ("query_reference_embeddings" in data and data["query_reference_embeddings"].size > 0):
                detector.query_reference_embeddings_ = (data["query_reference_embeddings"])
            
            logger.info(json.dumps({
                "event": "ood_model_loaded",
                "path": str(path),
                "n_clusters": detector.n_clusters,
                "knn_k": detector.knn_k,
                "query_calibrator": hasattr(detector, "query_calibrator_"),
            }))

            return detector
        except Exception:
            logger.exception(json.dumps({"event":"ood_model_load_failerd","path":str(path)}))
            raise


def main():
    parser = argparse.ArgumentParser(description="Fit an ID embedding reference for OOD scoring.")
    parser.add_argument("--domain", choices=("research", "home_insurance"), default="research")
    parser.add_argument("--index", default=None)
    parser.add_argument("--collection", default=None,
                        help="Chroma collection name; must match the retrieval index")
    parser.add_argument("--output", default=None)
    parser.add_argument("--insurance-scope", type=Path, default=BASE_DIR / "data/home_insurance_scope.json")
    parser.add_argument("--insurance-queries", type=Path, default=BASE_DIR / "data/home_insurance_queries.jsonl")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--inspect-only", action="store_true", help="Record insurance index provenance without loading a model or fitting")
    parser.add_argument("--shrinkage", type=float, default=0.10)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--eval-fraction", type=float, default=0.10,
                        help="Fraction of each tier held out for AUROC (0 disables)")
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--query_ref", default="../query_reference.json")
    args = parser.parse_args()

    if args.domain == "home_insurance":
        import hashlib
        import chromadb
        import json
        args.index = args.index or str(BASE_DIR / "insurance_chroma_index_v3")
        args.collection = args.collection or "insurance_chunks"
        output = Path(args.output or BASE_DIR / "data/home_insurance_ood.npz")
        collection = chromadb.PersistentClient(path=args.index).get_collection(args.collection)
        scope = json.loads(args.insurance_scope.read_text(encoding="utf-8-sig"))
        if args.inspect_only:
            _, provenance = insurance_reference(collection, scope, args.embedding_model)
            provenance["index_dir"] = str(Path(args.index).resolve())
            output.parent.mkdir(parents=True, exist_ok=True)
            snapshot = output.with_suffix(".snapshot.json")
            snapshot.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
            print(f"Saved insurance index snapshot: {snapshot}")
            return
        rows = load_insurance_queries(args.insurance_queries)
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(args.embedding_model)
        detector, report = fit_insurance_ood(collection, scope, rows, model,
                                             embedding_model=args.embedding_model,
                                             shrinkage=args.shrinkage, threshold=args.threshold)
        detector.artifact_metadata_.update(
            index_dir=str(Path(args.index).resolve()),
            queries_sha256=hashlib.sha256(args.insurance_queries.read_bytes()).hexdigest(),
            scope_sha256=hashlib.sha256(args.insurance_scope.read_bytes()).hexdigest(),
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        detector.save(output)
        output.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        output.with_suffix(".snapshot.json").write_text(json.dumps(detector.artifact_metadata_, indent=2), encoding="utf-8")
        print(json.dumps({"saved_model": str(output), "test_groups": report["test"]["groups"],
                          "test_comparisons": report["test"]["comparisons"]}, indent=2))
        return
    if args.inspect_only:
        parser.error("--inspect-only requires --domain home_insurance")
    args.index = args.index or str(BASE_DIR / "chroma_index")
    args.collection = args.collection or "text_chunks"
    args.output = args.output or "./ood_reference.npz"

    import json
    with open(args.query_ref, "r") as f:
        query_reference = json.load(f)
    raw_queries = [q["query"] for q in query_reference]
    with open("../ood_query_reference.json", "r") as f:
        ood_query_reference = json.load(f)
    raw_near_ood_queries = [
        q["query"]
        for q in ood_query_reference["near_ood"]
    ]

    raw_far_ood_queries = [
        q["query"]
        for q in ood_query_reference["far_ood"]
    ]

    
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.embedding_model)
    id_queries = model.encode(raw_queries, show_progress_bar=False).tolist()
    near_ood_queries = model.encode(raw_near_ood_queries, show_progress_bar=False).tolist()
    far_ood_queries = model.encode(raw_far_ood_queries, show_progress_bar=False).tolist()
    ood_queries = near_ood_queries + far_ood_queries
    import chromadb
    client = chromadb.PersistentClient(path=args.index)
    try:
        collection = client.get_collection(args.collection)
    except chromadb.errors.NotFoundError as error:
        available = [c.name for c in client.list_collections()]
        raise SystemExit(
            f"Collection {args.collection!r} was not found in {args.index!r}. "
            f"Available collections: {available or 'none'}. "
            "Pass the same --index and --collection used by hybrid_retrieval.py."
        ) from error
    detector = EmbeddingOOD.fit_from_collection(collection, id_queries=id_queries, ood_queries=ood_queries,shrinkage=args.shrinkage,
                                                 threshold=args.threshold,
                                                 eval_fraction=args.eval_fraction,
                                                 eval_seed=args.eval_seed)
    detector.save(args.output)
    print(f"Saved {args.output}: {len(detector.reference_energy_)} ID chunks (train), "
          f"energy threshold={detector.energy_threshold_:.3f}, "
          f"P(OOD)={'calibrated' if hasattr(detector, 'calibration_weight_') else 'reference-tail'}")
    report = getattr(detector, "eval_report_", {}) or {}
    if report:
        near = report.get("auroc_energy_near_ood")
        far = report.get("auroc_energy_far_ood")
        extra = ""
        if near is not None:
            extra += f"  near={near:.3f}"
        if far is not None:
            extra += f"  far={far:.3f}"
        print(f"Eval hold-out {args.eval_fraction:.0%} (n={report['n']}, "
              f"id={report['n_id']}, ood={report['n_ood']}): "
              f"AUROC energy={report['auroc_energy']:.3f}  knn={report['auroc_knn']:.3f}{extra} fpr@tpr0.95={report['false_positive_rate']:.3f}")


if __name__ == "__main__":
    main()
