"""Regression/diagnostic test suite for EmbeddingOOD query classification.

Usage:
    python test_ood_regression.py \
        --detector ./ood_reference.npz \
        --ood-module './ood_scoring(3).py'

Optional CI-style gates:
    python test_ood_regression.py \
        --detector ./ood_reference.npz \
        --ood-module './ood_scoring(3).py' \
        --assert-minimums

The suite deliberately separates:
  * ID: legitimate corpus-domain queries, including hard/paraphrased ID.
  * near_ood: deceptively similar wording, but outside the corpus knowledge domain.
  * far_ood: clearly unrelated topics.

Near-OOD "caught" means predicted near_ood OR ood. Far-OOD "caught" means
predicted ood. ID is correct only when predicted id.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer


# -----------------------------------------------------------------------------
# Held-out regression cases.
# Keep these OUT of query calibration/training data.
# -----------------------------------------------------------------------------
TEST_CASES: Dict[str, List[str]] = {
    "id": [
        "What is domain generalization?",
        "Why do models struggle when deployment data differs from training data?",
        "How can a model remain reliable in an unseen environment?",
        "How can learned features stay stable across changing domains?",
        "What is the difference between domain adaptation and domain generalization?",
        "How can a classifier detect that an input is outside its known distribution?",
        "Why can neural networks be overconfident on unfamiliar inputs?",
        "How does adversarial training affect robustness under distribution shift?",
        "Can data augmentation improve robustness to image corruptions?",
        "How does feature alignment help unsupervised domain adaptation?",
        "How can models avoid relying on spurious style information?",
        "What makes representations transferable to unseen domains?",
        "How can medical imaging models generalize across hospitals?",
        "Can uncertainty estimation help with out-of-distribution detection?",
        "How do nearest-neighbour methods help identify OOD samples?",
        "What happens when source and target distributions are different?",
        "How can a vision model become robust to blur, noise, and weather corruptions?",
        "How does test-time adaptation respond to a changing target domain?",
    ],
    "near_ood": [
        "How can a recommendation model remain stable when user preferences suddenly change?",
        "How should a language model handle distribution shift caused by new slang?",
        "How can speech recognition adapt to unseen accents without labelled data?",
        "How can fraud detection models respond to behavioural drift in transactions?",
        "How can a time-series forecasting model generalize to a regime it has never observed?",
        "How should a RAG system detect queries outside its indexed knowledge?",
        "Can embedding distance identify unsupported customer questions in a RAG system?",
        "How can a recommender distinguish preference drift from anomalous behaviour?",
        "How do graph neural networks generalize when graph topology changes?",
        "How can reinforcement-learning agents remain robust when environment dynamics change?",
        "How should an NLP classifier identify text from an unseen topic?",
        "Can uncertainty estimation identify novel intents in a chatbot?",
        "How can network intrusion detection remain robust as attack patterns evolve?",
        "How can tabular credit-risk models handle population drift over time?",
        "How can a forecasting system distinguish structural change from temporary noise?",
        "Can nearest-neighbour distance detect unfamiliar prompts in an LLM application?",
        "How should a search ranking model adapt when query intent changes over time?",
        "How can an ad-click prediction model remain calibrated after user behaviour shifts?",
    ],
    "far_ood": [
        "What is the capital of France?",
        "How does photosynthesis work in plants?",
        "Why do black holes have event horizons?",
        "How do I make sourdough bread?",
        "What caused the fall of the Roman Empire?",
        "How does CRISPR gene editing work?",
        "What is the difference between TCP and UDP?",
        "How do I center a div using CSS?",
        "What are the rules of chess castling?",
        "Why does inflation affect interest rates?",
        "How does a combustion engine work?",
        "What is the chemical formula for table salt?",
        "How do solar panels generate electricity?",
        "What are the main causes of earthquakes?",
        "How does DNS resolve a website name?",
        "Who wrote Pride and Prejudice?",
        "How can I calculate compound interest?",
        "What is the difference between a comet and an asteroid?",
    ],
}


@dataclass
class Result:
    group: str
    query: str
    predicted_band: str
    caught_correctly: bool
    strict_correct: bool
    p_ood: float
    energy: float
    knn_distance: float
    mahalanobis: float
    probability_source: str


def load_ood_module(path: str):
    """Load EmbeddingOOD from an arbitrary .py path, including filenames with ()."""
    module_path = Path(path).resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"OOD module not found: {module_path}")

    spec = importlib.util.spec_from_file_location("ood_scoring_under_test", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def is_caught(group: str, band: str) -> bool:
    """Operational detection success used for group-level recall."""
    if group == "id":
        return band == "id"
    if group == "near_ood":
        return band in {"near_ood", "ood"}
    if group == "far_ood":
        return band == "ood"
    raise ValueError(f"Unknown group: {group}")


def is_strict(group: str, band: str) -> bool:
    expected = {"id": "id", "near_ood": "near_ood", "far_ood": "ood"}[group]
    return band == expected


def evaluate(detector, model: SentenceTransformer, show_detector_output: bool = False) -> List[Result]:
    rows: List[Result] = []

    for group, queries in TEST_CASES.items():
        embeddings = model.encode(queries, show_progress_bar=False)

        for query, embedding in zip(queries, embeddings):
            if show_detector_output:
                score = detector.score_query(embedding)
            else:
                # Current _ood_band() prints debugging output; suppress it so the
                # regression report remains readable.
                with contextlib.redirect_stdout(io.StringIO()):
                    score = detector.score_query(embedding)

            band = score["ood_band"]
            rows.append(
                Result(
                    group=group,
                    query=query,
                    predicted_band=band,
                    caught_correctly=is_caught(group, band),
                    strict_correct=is_strict(group, band),
                    p_ood=float(score["ood_probability"]),
                    energy=float(score["energy"]),
                    knn_distance=float(score["knn_distance"]),
                    mahalanobis=float(score["mahalanobis"]),
                    probability_source=str(score["probability_source"]),
                )
            )

    return rows


def print_per_query(rows: List[Result]) -> None:
    print("\nPER-QUERY RESULTS")
    print("=" * 118)
    print(
        f"{'GROUP':<10} {'OK':<3} {'BAND':<9} {'P(OOD)':>8} "
        f"{'ENERGY':>12} {'kNN':>9}  QUERY"
    )
    print("-" * 118)

    for r in rows:
        ok = "Y" if r.caught_correctly else "N"
        print(
            f"{r.group:<10} {ok:<3} {r.predicted_band:<9} {r.p_ood:>8.3f} "
            f"{r.energy:>12.3f} {r.knn_distance:>9.4f}  {r.query}"
        )


def summarize(rows: List[Result]) -> Dict[str, dict]:
    summary: Dict[str, dict] = {}

    for group in TEST_CASES:
        group_rows = [r for r in rows if r.group == group]
        caught = np.array([r.caught_correctly for r in group_rows], dtype=float)
        strict = np.array([r.strict_correct for r in group_rows], dtype=float)
        probs = np.array([r.p_ood for r in group_rows], dtype=float)
        energy = np.array([r.energy for r in group_rows], dtype=float)
        knn = np.array([r.knn_distance for r in group_rows], dtype=float)

        summary[group] = {
            "n": len(group_rows),
            "caught": int(caught.sum()),
            "detection_rate": float(caught.mean()),
            "strict_accuracy": float(strict.mean()),
            "mean_p_ood": float(probs.mean()),
            "median_p_ood": float(np.median(probs)),
            "mean_energy": float(energy.mean()),
            "mean_knn": float(knn.mean()),
        }

    return summary


def print_summary(summary: Dict[str, dict]) -> None:
    print("\nGROUP SUMMARY")
    print("=" * 96)
    print(
        f"{'GROUP':<10} {'N':>4} {'CAUGHT':>9} {'RATE':>9} {'STRICT':>9} "
        f"{'MEAN P':>9} {'MED P':>9} {'MEAN kNN':>10}"
    )
    print("-" * 96)

    for group in ("id", "near_ood", "far_ood"):
        s = summary[group]
        print(
            f"{group:<10} {s['n']:>4} {s['caught']:>9} "
            f"{s['detection_rate']:>8.1%} {s['strict_accuracy']:>8.1%} "
            f"{s['mean_p_ood']:>9.3f} {s['median_p_ood']:>9.3f} "
            f"{s['mean_knn']:>10.4f}"
        )

    id_rate = summary["id"]["detection_rate"]
    near_rate = summary["near_ood"]["detection_rate"]
    far_rate = summary["far_ood"]["detection_rate"]

    print("\nInterpretation:")
    print(f"  ID acceptance rate       : {id_rate:.1%}")
    print(f"  Near-OOD detection recall: {near_rate:.1%}")
    print(f"  Far-OOD detection recall : {far_rate:.1%}")
    print(f"  ID false-positive rate   : {1.0 - id_rate:.1%}")


def write_csv(rows: List[Result], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "query",
                "predicted_band",
                "caught_correctly",
                "strict_correct",
                "p_ood",
                "energy",
                "knn_distance",
                "mahalanobis",
                "probability_source",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r.__dict__)

    print(f"\nSaved detailed results to: {out}")


def enforce_minimums(summary: Dict[str, dict], min_id: float, min_near: float, min_far: float) -> None:
    failures = []

    if summary["id"]["detection_rate"] < min_id:
        failures.append(
            f"ID acceptance {summary['id']['detection_rate']:.1%} < required {min_id:.1%}"
        )
    if summary["near_ood"]["detection_rate"] < min_near:
        failures.append(
            f"Near-OOD recall {summary['near_ood']['detection_rate']:.1%} < required {min_near:.1%}"
        )
    if summary["far_ood"]["detection_rate"] < min_far:
        failures.append(
            f"Far-OOD recall {summary['far_ood']['detection_rate']:.1%} < required {min_far:.1%}"
        )

    if failures:
        raise AssertionError("OOD regression failed:\n  - " + "\n  - ".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser(description="Regression suite for query OOD detection")
    parser.add_argument("--detector", required=True, help="Path to saved ood_reference.npz")
    parser.add_argument(
        "--ood-module",
        default="./ood_scoring.py",
        help="Path to the Python file containing EmbeddingOOD",
    )
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Must match the embedding model used when fitting the detector",
    )
    parser.add_argument("--csv", default="ood_regression_results.csv")
    parser.add_argument("--show-detector-output", action="store_true")
    parser.add_argument("--assert-minimums", action="store_true")
    parser.add_argument("--min-id", type=float, default=0.90)
    parser.add_argument("--min-near", type=float, default=0.70)
    parser.add_argument("--min-far", type=float, default=0.90)
    args = parser.parse_args()

    module = load_ood_module(args.ood_module)
    detector = module.EmbeddingOOD.load(args.detector)
    model = SentenceTransformer(args.embedding_model)

    rows = evaluate(detector, model, args.show_detector_output)
    print_per_query(rows)
    summary = summarize(rows)
    print_summary(summary)
    write_csv(rows, args.csv)

    if args.assert_minimums:
        enforce_minimums(summary, args.min_id, args.min_near, args.min_far)
        print("\nRegression gates: PASS")


if __name__ == "__main__":
    main()
