"""Embedding-space OOD detection for RAG queries and retrieved chunks.

Fit this once on trusted ``id_core_text`` chunks, save the reference, then
call ``score_retrieval`` after retrieval.  Score the query as well as the
chunks: an off-topic question can retrieve plausible ID snippets.

``fit_from_collection`` holds out ``eval_fraction`` (default 10%) of each
tier and reports AUROC on that split.
"""

import argparse
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

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
    
    def fit_query_reference(self, id_queries: Sequence[Sequence[float]]):
        """Fit query-specific energy and kNN thresholds using in-domain queries."""
        X_q = np.asarray(id_queries, dtype=np.float64)
        query_energies = self.energy(X_q)
        query_knn = self.knn_distance_min(X_q)
        
        self.query_energy_threshold_ = float(np.quantile(query_energies, self.threshold))
        self.query_knn_threshold_ = float(np.quantile(query_knn, self.threshold))
        self.query_near_energy_threshold_ = float(np.quantile(query_energies, self.near_threshold))
        self.query_near_knn_threshold_ = float(np.quantile(query_knn, self.near_threshold))

    def fit(self, embeddings: Sequence[Sequence[float]]):
        X = np.asarray(embeddings, dtype=np.float64)
        N, D = X.shape
        K = self.n_clusters
        
        if X.ndim != 2 or N < K:
            raise ValueError(f"fit needs at least {K} embeddings shaped (n_chunks, dimension)")

        # 1. Initialize parameters (k-means++ style or random point selection)
        np.random.seed(42)  # For reproducibility
        self.means_ = X[np.random.choice(N, K, replace=False)].copy()
        self.weights_ = np.ones(K) / K
        self.precisions_ = np.array([np.eye(D) for _ in range(K)])
        self.log_dets_ = np.zeros(K)

        log_likelihood = -np.inf

        # 2. EM Loop
        for iteration in range(self.max_iter):
            # --- E-STEP: Compute Responsibilities ---
            log_resp = np.zeros((N, K))
            for k in range(K):
                diff = X - self.means_[k]
                # Mahalanobis distance squared: (x - μ)^T Σ^-1 (x - μ)
                mahalanobis = np.sum((diff @ self.precisions_[k]) * diff, axis=1)
                # Log Gaussian density per component
                log_pdf = -0.5 * (D * np.log(2 * np.pi) + self.log_dets_[k] + mahalanobis)
                log_resp[:, k] = np.log(self.weights_[k] + 1e-12) + log_pdf
            
            # Log-sum-exp trick for numerical stability
            max_log_resp = np.max(log_resp, axis=1, keepdims=True)
            log_sum_exp = max_log_resp + np.log(np.sum(np.exp(log_resp - max_log_resp), axis=1, keepdims=True))
            resp = np.exp(log_resp - log_sum_exp)

            current_log_likelihood = np.sum(log_sum_exp)
            if abs(current_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = current_log_likelihood

            # --- M-STEP: Update Component Parameters ---
            N_k = resp.sum(axis=0) + 1e-10  # Soft counts
            self.weights_ = N_k / N

            for k in range(K):
                # Weighted Mean
                self.means_[k] = np.sum(resp[:, [k]] * X, axis=0) / N_k[k]
                
                # Weighted Covariance + Your Shrinkage
                diff = X - self.means_[k]
                cov = (diff.T * resp[:, k]) @ diff / N_k[k]
                
                scale = np.trace(cov) / D
                cov = ((1 - self.shrinkage) * cov + self.shrinkage * scale * np.eye(D))

                # Inverse and Log Determinant via eigh (retained from your code)
                values, vectors = np.linalg.eigh(cov)
                values = np.maximum(values, np.finfo(float).eps * max(scale, 1.0))
                
                self.precisions_[k] = (vectors / values) @ vectors.T
                self.log_dets_[k] = float(np.log(values).sum())

        # Reference energy calculation across the fitted mixture
        self.reference_embeddings_ = X
        self.reference_energy_ = self.energy(X)
        self.energy_threshold_ = float(np.quantile(self.reference_energy_, self.threshold))
        self.near_energy_threshold_ = float(np.quantile(self.reference_energy_, self.near_threshold))
        self.reference_knn_ = self.knn_distance_min(X, exclude_self=True)
        self.knn_threshold_ = float(np.quantile(self.reference_knn_, self.threshold))
        self.near_knn_threshold_ = float(np.quantile(self.reference_knn_, self.near_threshold))
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
    def fit_from_collection(cls, collection,id_queries: Sequence[Sequence[float]] = None, id_tier: str = "id_core_text",
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
        detector.fit_query_reference(id_queries)
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
        X = np.asarray(embeddings, dtype=np.float64)
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

    def energy(self, embeddings: Sequence[Sequence[float]]) -> np.ndarray:
        """Calculates mixture score (e.g., negative log-likelihood/free energy) for each point."""
        X = np.asarray(embeddings, dtype=np.float64)
        N, D = X.shape
        K = self.n_clusters
        log_resp = np.zeros((N, K))
        
        for k in range(K):
            diff = X - self.means_[k]
            mahalanobis = np.sum((diff @ self.precisions_[k]) * diff, axis=1)
            log_pdf = -0.5 * (D * np.log(2 * np.pi) + self.log_dets_[k] + mahalanobis)
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
        X = np.asarray(embeddings, dtype=np.float64)
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
        """id / near_ood / ood from energy or kNN, whichever is more extreme."""
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

    def score_embeddings(self, embeddings: Sequence[Sequence[float]], query: bool = False) -> List[Dict]:
        energies = self.energy(embeddings)
        distances = self.mahalanobis(embeddings)
        knn = self.knn_distance_min(embeddings)
        n = len(self.reference_energy_)
        results = []
        for energy, distance, knn_d in zip(energies, distances, knn):
            # This is an empirical tail percentile, not a learned posterior.
            percentile = float(np.mean(self.reference_energy_ <= energy))
            low, high = _wilson_interval(percentile * n, n)
            if hasattr(self, "calibration_weight_"):
                probability = float(_sigmoid(self.calibration_weight_ *
                    ((energy - self.calibration_mean_) / self.calibration_scale_)
                    + self.calibration_bias_))
                probability_source = "label_calibrated"
            else:
                probability, probability_source = percentile, "reference_tail"
            band = self._ood_band(float(energy), float(knn_d), query=query)
            results.append({
                "mahalanobis": float(distance),
                "energy": float(energy),
                "knn_distance": float(knn_d),
                "ood_percentile": percentile,
                "ood_confidence_interval": (float(low), float(high)),
                "ood_probability": probability,
                "probability_source": probability_source,
                "ood_band": band,
                "is_near_ood": band == "near_ood",
                # Query/doc flag uses the fitted energy quantile, not the Wilson bound.
                "is_ood": band == "ood",
            })
        return results

    def score_query(self, embedding: Sequence[float]) -> Dict:
        """Score a query embedding against the ID document reference.

        Same space as chunks (symmetric MiniLM).  An off-topic question can
        still retrieve ID snippets; those chunks look in-distribution while
        the query does not.
        """
        #return self.score_embeddings([embedding])[0]
        return self.score_embeddings([embedding], query=True)[0]


    def score_chunks(self, chunks: List[Dict], collection) -> List[Dict]:
        """Annotate retrieval output in place with OOD scores and return it."""
        if not chunks:
            return chunks
        data = collection.get(ids=[c["id"] for c in chunks], include=["embeddings"])
        by_id = dict(zip(data["ids"], data["embeddings"]))
        missing = [c["id"] for c in chunks if c["id"] not in by_id]
        if missing:
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
            query_energy_threshold=getattr(self, "query_energy_threshold_", np.nan),
            query_knn_threshold=getattr(self, "query_knn_threshold_", np.nan),
            query_near_energy_threshold=getattr(self, "query_near_energy_threshold_", np.nan),
            query_near_knn_threshold=getattr(self, "query_near_knn_threshold_", np.nan),
        )

    @classmethod
    def load(cls, path: str):
        data = np.load(path)
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
        
        detector.means_ = data["means"] if "means" in data else data["mean"]
        detector.precisions_ = data["precisions"] if "precisions" in data else data["precision"]
        detector.weights_ = data["weights"] if "weights" in data else np.ones(n_clusters) / n_clusters
        detector.log_dets_ = data["log_dets"] if "log_dets" in data else data["log_det"]
        detector.reference_energy_ = data["reference_energy"]
        detector.energy_threshold_ = float(data["energy_threshold"])
        detector.near_energy_threshold_ = float(data["near_energy_threshold"]) if "near_energy_threshold" in data else detector.energy_threshold_
        detector.query_energy_threshold_ = float(data["query_energy_threshold"])
        detector.query_knn_threshold_ = float(data["query_knn_threshold"])
        detector.query_near_energy_threshold_ = float(data["query_near_energy_threshold"])
        detector.query_near_knn_threshold_ = float(data["query_near_knn_threshold"])
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
        
        return detector


def main():
    parser = argparse.ArgumentParser(description="Fit an ID embedding reference for OOD scoring.")
    parser.add_argument("--index", default="./chroma_index")
    parser.add_argument("--collection", default="text_chunks",
                        help="Chroma collection name; must match the retrieval index")
    parser.add_argument("--output", default="./ood_reference.npz")
    parser.add_argument("--shrinkage", type=float, default=0.10)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--eval-fraction", type=float, default=0.10,
                        help="Fraction of each tier held out for AUROC (0 disables)")
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--query_ref", default="./query_reference.json")
    args = parser.parse_args()

    import json
    with open(args.query_ref, "r") as f:
        query_reference = json.load(f)
    raw_queries = [q["query"] for q in query_reference]
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    id_queries = model.encode(raw_queries, show_progress_bar=False).tolist()

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
    detector = EmbeddingOOD.fit_from_collection(collection, id_queries=id_queries, shrinkage=args.shrinkage,
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
