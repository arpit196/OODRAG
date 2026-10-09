"""Lightweight post-retrieval grounding gate; no generation dependencies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import warnings

import joblib
import numpy as np
import sklearn

STAGE1_FEATURE_NAMES = (
    "embedding_norm", "ood_probability", "energy", "mahalanobis", "knn_distance",
)
FEATURE_NAMES = STAGE1_FEATURE_NAMES + (
    "cosine_1", "cosine_2", "cosine_3",
    "bm25_normalized_1", "bm25_normalized_2", "bm25_normalized_3",
    "cosine_max", "cosine_mean", "cosine_min", "chunk_count",
)


def extract_grounding_features(
    query: str,
    top_3_chunks: Sequence[Mapping[str, Any]],
    stage1_features: Mapping[str, float],
) -> np.ndarray:
    """Return a fixed-order float64 vector matching ``FEATURE_NAMES``.

    Chunks must carry actual ``cosine_similarity`` values (not reranker
    scores or distances from an unspecified metric). BM25 uses the bounded
    transform max(score, 0)/(1 + max(score, 0)), preserving score magnitude
    across queries. Absent ranks are zero-padded; aggregates use present ranks
    only. Missing Stage 1 values are zero. Nonfinite inputs are rejected.
    The query identifies the already-embedded request; no model is run here.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a nonempty string")
    chunks = list(top_3_chunks[:3])
    cosine = np.zeros(3, dtype=np.float64)
    bm25 = np.zeros(3, dtype=np.float64)
    for rank, chunk in enumerate(chunks):
        similarity = float(chunk["cosine_similarity"])
        raw_score = float(chunk.get("bm25_score") or 0.0)
        if not np.isfinite(similarity) or not -1.000001 <= similarity <= 1.000001:
            raise ValueError("cosine_similarity must be finite and in [-1, 1]")
        if not np.isfinite(raw_score):
            raise ValueError("bm25_score must be finite")
        cosine[rank] = np.clip(similarity, -1.0, 1.0)
        positive_score = max(raw_score, 0.0)
        bm25[rank] = positive_score / (1.0 + positive_score)
    present = cosine[:len(chunks)]
    aggregates = [present.max(), present.mean(), present.min()] if chunks else [0.0] * 3
    vector = np.asarray(
        [float(stage1_features.get(name) or 0.0) for name in STAGE1_FEATURE_NAMES]
        + cosine.tolist() + bm25.tolist() + aggregates + [float(len(chunks))],
        dtype=np.float64,
    )
    if not np.isfinite(vector).all():
        raise ValueError("grounding features must be finite")
    return vector


@dataclass(frozen=True)
class GroundingDecision:
    grounded: bool
    probability: float
    threshold: float
    features: tuple[float, ...]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "feature_names": list(FEATURE_NAMES)}


class GroundingEvaluator:
    """Load a trusted local sklearn artifact once and infer per request."""

    def __init__(self, classifier: Any, threshold: float = 0.5) -> None:
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("grounding threshold must be in [0, 1]")
        if set(classifier.classes_) != {0, 1}:
            raise ValueError("grounding classifier must have labels 0 and 1 (grounded)")
        if classifier.n_features_in_ != len(FEATURE_NAMES):
            raise ValueError("grounding classifier feature count mismatch")
        self.classifier = classifier
        self.threshold = threshold

    @classmethod
    def load(cls, path: str | Path, threshold: float = 0.5) -> GroundingEvaluator:
        # Older sklearn releases report this as UserWarning, newer ones as
        # InconsistentVersionWarning. Handle both before serving requests.
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", UserWarning)
            artifact = joblib.load(path)
        mismatches = [str(item.message) for item in captured
                      if "Trying to unpickle estimator" in str(item.message)]
        trained_version = artifact.get("environment", {}).get("scikit-learn")
        if mismatches or (trained_version and trained_version != sklearn.__version__):
            detail = mismatches[0] if mismatches else f"Model sklearn={trained_version}; runtime={sklearn.__version__}."
            raise ValueError(
                f"Grounding model is incompatible with this scikit-learn environment. {detail} "
                "Refit in the active environment: python train_grounding.py "
                "--training-rows data/grounding.training.jsonl --output data/grounding.joblib"
            )
        for item in captured:
            warnings.warn(item.message, item.category, stacklevel=2)
        if tuple(artifact["feature_names"]) != FEATURE_NAMES:
            raise ValueError("grounding classifier feature schema mismatch; retrain model")
        return cls(artifact["classifier"], threshold)

    def evaluate(
        self, query: str, chunks: Sequence[Mapping[str, Any]],
        stage1_features: Mapping[str, float],
    ) -> GroundingDecision:
        features = extract_grounding_features(query, chunks, stage1_features)
        if not chunks:
            return GroundingDecision(False, 0.0, self.threshold, tuple(features), "no_chunks")
        positive_index = list(self.classifier.classes_).index(1)
        probability = float(self.classifier.predict_proba(features.reshape(1, -1))[0, positive_index])
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("classifier returned an invalid grounding probability")
        grounded = probability >= self.threshold
        return GroundingDecision(
            grounded, probability, self.threshold, tuple(features),
            "grounded" if grounded else "below_threshold",
        )
