"""Embedding-space OOD detection for retrieved RAG chunks.

Fit this once on trusted ``id_core_text`` chunks, save the reference, then
call ``score_chunks`` after retrieval.  A high Mahalanobis distance (or its
equivalent Gaussian energy) means that a retrieved chunk is unlike the
trusted corpus.
"""

import argparse
from typing import Dict, List, Sequence

import numpy as np


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


class EmbeddingOOD:
    """Regularised Gaussian reference distribution for one embedding space.

    ``shrinkage`` makes covariance inversion stable when embedding dimensions
    approach (or exceed) the number of reference chunks.  ``threshold`` is an
    empirical reference-energy percentile, e.g. .95 flags the most unusual 5%.
    """

    def __init__(self, shrinkage: float = 0.10, threshold: float = 0.95, n_clusters: int = 1, max_iter: int = 100, tol: float = 1e-4):
        if not 0 <= shrinkage <= 1 or not 0 < threshold < 1:
            raise ValueError("shrinkage must be in [0, 1] and threshold in (0, 1)")
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.shrinkage = shrinkage
        self.threshold = threshold

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
                self.means_[k] = np.sum(resp[:, k:] * X, axis=0) / N_k[k]
                
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
        self.reference_energy_ = self.energy(X)
        self.energy_threshold_ = float(np.quantile(self.reference_energy_, self.threshold))
        return self

    @classmethod
    def from_collection(cls, collection, tier: str = "id_core_text", **kwargs):
        """Fit from a Chroma collection without recomputing embeddings."""
        data = collection.get(include=["embeddings", "metadatas"])
        embeddings = [e for e, meta in zip(data["embeddings"], data["metadatas"])
                      if meta.get("tier") == tier]
        if not embeddings:
            raise ValueError(f"no embeddings found for tier={tier!r}")
        return cls(**kwargs).fit(embeddings)

    @classmethod
    def fit_from_collection(cls, collection, id_tier: str = "id_core_text",
                            ood_tiers=("near_ood_text", "far_ood_text"), **kwargs):
        """Fit the ID mixture and calibrate energy to P(OOD) using labelled tiers."""
        data = collection.get(include=["embeddings", "metadatas"])
        pairs = list(zip(data["embeddings"], data["metadatas"]))
        id_embeddings = [e for e, m in pairs if m.get("tier") == id_tier]
        ood_embeddings = [e for e, m in pairs if m.get("tier") in ood_tiers]
        detector = cls(**kwargs).fit(id_embeddings)
        if ood_embeddings:
            detector.fit_probability_calibrator(ood_embeddings)
        return detector

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
        log_likelihoods = (max_log_resp + np.log(np.sum(np.exp(log_resp - max_log_resp), axis=1, keepdims=True))).squeeze()
        
        return -log_likelihoods

    def score_embeddings(self, embeddings: Sequence[Sequence[float]]) -> List[Dict]:
        energies = self.energy(embeddings)
        distances = self.mahalanobis(embeddings)
        n = len(self.reference_energy_)
        results = []
        for energy, distance in zip(energies, distances):
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
            results.append({
                "mahalanobis": float(distance),
                "energy": float(energy),
                "ood_percentile": percentile,
                "ood_confidence_interval": (float(low), float(high)),
                "ood_probability": probability,
                "probability_source": probability_source,
                # Conservative: flag only when even the lower 95% bound is beyond threshold.
                "is_ood": bool(low >= self.threshold),
            })
        return results

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

    def save(self, path: str):
        np.savez_compressed(
            path,
            means=self.means_,
            precisions=self.precisions_,
            weights=self.weights_,
            log_dets=self.log_dets_,
            reference_energy=self.reference_energy_,
            shrinkage=self.shrinkage,
            threshold=self.threshold,
            energy_threshold=self.energy_threshold_,
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            tol=self.tol,
            calibration_mean=getattr(self, "calibration_mean_", np.nan),
            calibration_scale=getattr(self, "calibration_scale_", np.nan),
            calibration_weight=getattr(self, "calibration_weight_", np.nan),
            calibration_bias=getattr(self, "calibration_bias_", np.nan),
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
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=tol,
        )
        
        detector.means_ = data["means"] if "means" in data else data["mean"]
        detector.precisions_ = data["precisions"] if "precisions" in data else data["precision"]
        detector.weights_ = data["weights"] if "weights" in data else np.ones(n_clusters) / n_clusters
        detector.log_dets_ = data["log_dets"] if "log_dets" in data else data["log_det"]
        detector.reference_energy_ = data["reference_energy"]
        detector.energy_threshold_ = float(data["energy_threshold"])
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
    args = parser.parse_args()

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
    detector = EmbeddingOOD.fit_from_collection(collection, shrinkage=args.shrinkage,
                                                 threshold=args.threshold)
    detector.save(args.output)
    print(f"Saved {args.output}: {len(detector.reference_energy_)} ID chunks, "
          f"energy threshold={detector.energy_threshold_:.3f}, "
          f"P(OOD)={'calibrated' if hasattr(detector, 'calibration_weight_') else 'reference-tail'}")


if __name__ == "__main__":
    main()
