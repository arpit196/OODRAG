"""Offline tests: no model downloads, vector database, or LLM calls."""

import importlib
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np

from src.grounding import FEATURE_NAMES, GroundingEvaluator, extract_grounding_features
from train_grounding import TrainingQuery, collect_training_rows, fit_classifier, load_training_queries


def load_agent_module():
    # Stub only heavyweight import dependencies absent in a minimal test runtime.
    # Retrieval, embeddings and generation are explicitly injected below.
    optional = {}
    for name in ("sentence_transformers", "langchain_openai", "langchain_core.messages"):
        try:
            importlib.import_module(name)
        except ImportError:
            optional[name] = MagicMock()
    with patch.dict("sys.modules", optional):
        return importlib.import_module("src.agent")


class GroundingTests(unittest.TestCase):
    def test_features_preserve_magnitude_and_pad_missing_ranks(self):
        vector = extract_grounding_features("question", [
            {"cosine_similarity": -.5, "bm25_score": 3},
            {"cosine_similarity": .8, "bm25_score": -2},
        ], {"embedding_norm": 2})
        self.assertEqual(vector.shape, (len(FEATURE_NAMES),))
        np.testing.assert_allclose(vector[5:11], [-.5, .8, 0, .75, 0, 0])
        np.testing.assert_allclose(vector[11:], [.8, .15, -.5, 2])

    def test_invalid_features_rejected(self):
        with self.assertRaises(ValueError):
            extract_grounding_features("question", [{"cosine_similarity": np.nan}], {})
        with self.assertRaises(KeyError):
            extract_grounding_features("question", [{"dense_distance": .1}], {})

    def test_train_save_load_and_online_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            # Deterministic fixtures test serialization, not production training.
            rows = []
            for group, label, similarity, score in (("id", 1, .9, 8.), ("near_ood", 0, .05, .1), ("far_ood", 0, -.1, 0.)):
                for index in range(10):
                    vector = extract_grounding_features("question", [
                        {"cosine_similarity": similarity, "bm25_score": score} for _ in range(3)
                    ], {"embedding_norm": 1, "ood_probability": .2, "energy": 3,
                        "mahalanobis": 3, "knn_distance": .2})
                    rows.append({"id": f"{group}{index}", "group": group, "label": label,
                                 "stage1_would_reject": group == "far_ood", "features": vector.tolist()})
            report = fit_classifier(rows, path)
            self.assertFalse(set(report["train_ids"]) & set(report["validation_ids"]))
            evaluator = GroundingEvaluator.load(path)
            good = evaluator.evaluate("question", [
                {"cosine_similarity": .9, "bm25_score": 8} for _ in range(3)
            ], {"embedding_norm": 1, "ood_probability": .2, "energy": 3,
                "mahalanobis": 3, "knn_distance": .2})
            bad = evaluator.evaluate("question", [
                {"cosine_similarity": .05, "bm25_score": .1} for _ in range(3)
            ], {"embedding_norm": 1, "ood_probability": .2, "energy": 3,
                "mahalanobis": 3, "knn_distance": .2})
            self.assertTrue(good.grounded)
            self.assertFalse(bad.grounded)
            self.assertFalse(evaluator.evaluate("question", [], {}).grounded)
            artifact = joblib.load(path)
            artifact["feature_names"] = FEATURE_NAMES[::-1]
            joblib.dump(artifact, path)
            with self.assertRaises(ValueError):
                GroundingEvaluator.load(path)

    def test_class_order_and_threshold_boundary(self):
        classifier = MagicMock(classes_=np.array([1, 0]), n_features_in_=len(FEATURE_NAMES))
        classifier.predict_proba.return_value = np.array([[.5, .5]])
        evaluator = GroundingEvaluator(classifier, .5)
        self.assertTrue(evaluator.evaluate("question", [{"cosine_similarity": .8}], {}).grounded)
        with self.assertRaises(ValueError):
            GroundingEvaluator(classifier, float("nan"))

    def test_incompatible_saved_model_fails_before_online_inference(self):
        classifier = MagicMock(classes_=np.array([0, 1]), n_features_in_=len(FEATURE_NAMES))
        artifact = {"classifier": classifier, "feature_names": FEATURE_NAMES,
                    "environment": {"scikit-learn": "0.0.0"}}
        with patch("src.grounding.joblib.load", return_value=artifact):
            with self.assertRaisesRegex(ValueError, "--training-rows"):
                GroundingEvaluator.load("unused.joblib")
        artifact.pop("environment")
        def old_version_load(path):
            warnings.warn("Trying to unpickle estimator LogisticRegression from version 1.9.1", UserWarning)
            return artifact
        with patch("src.grounding.joblib.load", side_effect=old_version_load):
            with self.assertRaisesRegex(ValueError, "incompatible"):
                GroundingEvaluator.load("unused.joblib")
        classifier.predict_proba.assert_not_called()


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_agent_module()

    def setUp(self):
        self.agent = self.module.ThreeStageRAGPipeline.__new__(self.module.ThreeStageRAGPipeline)
        self.agent.router = MagicMock()
        self.agent.router.route.return_value = {"decision": "domain"}
        self.agent.embed_model = MagicMock()
        self.agent.embed_model.encode.return_value = np.array([[1., 0.]])
        self.agent.detector = MagicMock()
        self.agent.detector.score_query.return_value = {
            "is_ood": False, "ood_probability": .2, "energy": 2.,
            "mahalanobis": 2., "knn_distance": .1,
        }
        self.agent.text_collection = MagicMock()
        self.agent.text_collection.get.return_value = {"ids": ["a", "b", "c"],
                                                       "embeddings": [[1., 0.]] * 3}
        bm25 = MagicMock()
        bm25.get_scores.return_value = [8., 8., 8.]
        self.agent.bm25_state = (bm25, ["a", "b", "c"], [], [])
        self.agent.cross_encoder = MagicMock()
        self.agent.grounding_evaluator = GroundingEvaluator(
            MagicMock(classes_=np.array([0, 1]), n_features_in_=len(FEATURE_NAMES)))
        self.agent.grounding_evaluator.classifier.predict_proba.return_value = np.array([[.1, .9]])
        self.agent.generator = MagicMock(name="generator")
        self.agent.generator.name = "test"
        self.agent.generator.model = "test"
        self.agent.generator.last_usage = {}
        self.agent.generator.generate.return_value = "Supported answer [1]"
        self.agent._fallback_generator = MagicMock()
        self.agent.index_dir = "missing_index"
        self.agent.embed_model_name = "test"
        self.agent.reranker_model_name = "test"
        self.agent.log_path = None
        self.chunks = [{"id": cid, "text": "Evidence", "ood": {"ood_band": "id"}}
                       for cid in ["a", "b", "c"]]
        self.retrieval_patch = patch.object(self.module, "hybrid_retrieve", return_value=self.chunks)
        self.retrieve = self.retrieval_patch.start()
        self.addCleanup(self.retrieval_patch.stop)

    def test_far_ood_never_retrieves_or_generates(self):
        self.agent.detector.score_query.return_value["is_ood"] = True
        result = self.agent.answer("question")
        self.assertEqual(result["execution_path"], ["stage1"])
        self.retrieve.assert_not_called()
        self.agent.generator.generate.assert_not_called()
        self.agent.grounding_evaluator.classifier.predict_proba.assert_not_called()

    def test_offline_ood_examples_retrieve_without_generating_or_self_labeling(self):
        self.agent.detector.score_query.return_value["is_ood"] = True
        rows = collect_training_rows(self.agent, [TrainingQuery("NOOD001", "question", "near_ood", 0)])
        self.retrieve.assert_called_once()
        self.assertEqual(rows[0]["label"], 0)
        self.assertTrue(rows[0]["stage1_would_reject"])
        self.assertEqual(rows[0]["features"][5:8], [1., 1., 1.])
        self.agent.generator.generate.assert_not_called()
        self.agent._fallback_generator.generate.assert_not_called()
        self.agent.grounding_evaluator.classifier.predict_proba.assert_not_called()
        json.dumps(rows)

    def test_offline_features_match_online_features(self):
        offline = self.agent.collect_grounding_example("question")
        online = self.agent.answer("question")
        np.testing.assert_allclose(offline["features"], online["grounding_metrics"]["features"])

    def test_loader_uses_supplied_labels_and_utf16_paper_titles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queries = root / "queries.json"
            docs = root / "papers.txt"
            queries.write_text(json.dumps({"near_ood": [{"id": "N1", "query": "A related question"}],
                                           "far_ood": [{"id": "F1", "query": "An unrelated question"}]}))
            docs.write_text("=== FILE: paper.txt ===\nPaper Title\n\nAn actual abstract.", encoding="utf-16")
            cases = load_training_queries(queries, docs)
            self.assertEqual([c.label for c in cases], [0, 0, 1])
            self.assertIn("Paper Title", cases[-1].query)
            self.assertEqual(cases[-1].source_document, "paper.txt")

    def test_ungrounded_never_generates(self):
        self.agent.grounding_evaluator.classifier.predict_proba.return_value = np.array([[.9, .1]])
        result = self.agent.answer("question")
        self.assertEqual(result["answer"], "No relevant context found.")
        self.assertEqual(result["execution_path"], ["stage1", "stage2"])
        self.agent.generator.generate.assert_not_called()
        self.agent._fallback_generator.generate.assert_not_called()
        json.dumps(result)

    def test_empty_retrieval_never_generates(self):
        self.retrieve.return_value = []
        result = self.agent.answer("question")
        self.assertEqual(result["grounding_metrics"]["reason"], "no_chunks")
        self.agent.generator.generate.assert_not_called()

    def test_grounded_runs_existing_generation_with_verified_chunks(self):
        # The fourth result must never reach generation in the three-stage path.
        self.chunks.append({"id": "d", "text": "Other", "ood": {"ood_band": "id"}})
        result = self.agent.run("question")
        self.assertEqual(result.execution_path, ("stage1", "stage2", "stage3"))
        self.assertEqual(result.response_text, "Supported answer [1]")
        self.assertEqual(len(self.agent.generator.generate.call_args.kwargs["chunks"]), 3)
        self.assertIn('<document id="1"', self.agent.generator.generate.call_args.args[0])
        self.assertEqual(result.debug_metadata["grounding_metrics"]["features"][5:8], (1., 1., 1.))

    def test_disabled_gate_preserves_existing_answer_path(self):
        self.agent.grounding_evaluator = None
        result = self.agent.answer("question")
        self.assertEqual(result["answer"], "Supported answer [1]")
        self.assertFalse(result["grounding_metrics"]["enabled"])
        self.agent.text_collection.get.assert_not_called()

    def test_chitchat_keeps_existing_bypass(self):
        self.agent.router.route.return_value = {"decision": "chitchat"}
        result = self.agent.answer("hello")
        self.assertEqual(result["action"], "chitchat")
        self.retrieve.assert_not_called()
        self.agent.grounding_evaluator.classifier.predict_proba.assert_not_called()


if __name__ == "__main__":
    unittest.main()
