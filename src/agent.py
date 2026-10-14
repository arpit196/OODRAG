"""
agent.py
---------
The orchestrator tying together everything built so far:

    query
      -> QueryRouter            : chitchat, or does this need the corpus at all?
      -> hybrid_retrieve()       : dense + BM25 + RRF + rerank   (hybrid_retrieval.py)
      -> EmbeddingOOD.score_retrieval()  : query-level + chunk-level OOD  (ood_scoring.py)
      -> decide_action()         : generate | hedge | abstain | chitchat
      -> GroundingEvaluator      : optional top-3 relevance probability gate
      -> guardrails.*            : chunk filtering, prompt building, citations
      -> Generator.generate()    : pluggable backend  (generator.py)

Two design decisions worth being explicit about:

1. Chit-chat detection happens BEFORE retrieval, not after. "Hi, how are
   you?" and "what's the weather today" both score as OOD against a
   domain-specific corpus, but they need opposite handling: chit-chat should
   never mention the corpus at all, while a genuine off-topic question
   should hit the "abstain" path and say so explicitly. Only the router can
   tell these apart — OOD distance alone can't, since retrieval always
   returns its nearest neighbors regardless of relevance.

2. The action decision is driven primarily by the QUERY's own OOD status,
   not the retrieved chunks' bands. ood_scoring.py's docstring calls out
   exactly why: an off-topic query can still retrieve plausible-looking
   ID-band chunks (nearest neighbors of a bad query are still something).
   Trusting chunk bands alone would let a wrong-question case slip through
   as "generate". Chunk composition is used as a secondary signal only when
   the query itself scores as "id".

Usage:
    python agent.py --index ./chroma_index --ood-reference ./ood_reference.npz \\
        --generator extractive --query "how does domain generalization work?"
"""

import argparse
import json
import time
import traceback
import os
from dotenv import load_dotenv
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
import math
import sys
import logging
from dataclasses import dataclass
# Add project root directory to sys.path so 'src' can be imported cleanly
logger = logging.getLogger("rag_agent")
logger.setLevel(logging.INFO)

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import src.guardrails
from src.timing import measure, skipped_metrics

from src.hybrid_retrieval import build_bm25_index, hybrid_retrieve
from src.ood_scoring import EmbeddingOOD
from src.grounding import GroundingEvaluator, extract_grounding_features
from src.query_router import QueryRouter
#import src.guardrails
from src.guardrails import (
    select_chunks_for_generation,
    build_generation_prompt,
    ground_citations,
    build_repro_bundle,
)
from src.generator import Generator, load_generator, ExtractiveGenerator, OpenAICompatibleGenerator
#import sys
import numpy as np

load_dotenv()

LOG_FILE = "ood_abstains.jsonl"
MODEL_PRICING_PATH = BASE_DIR / "src" / "model_pricing.json"

def load_model_pricing(
    filepath: str | Path = MODEL_PRICING_PATH
) -> Dict[str, Dict[str, float]]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

MODEL_PRICING = load_model_pricing()

def calculate_llm_cost(
    model_name: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> Dict[str, float]:

    pricing = MODEL_PRICING.get(model_name)
    if not pricing:
        return {
            "input_cost_usd": 0.0,
            "output_cost_usd": 0.0,
            "total_cost_usd": 0.0,
        }

    input_cost = (
        input_tokens / 1000000
    ) * pricing["input_per_1m_tokens_usd"]

    output_cost = (
        output_tokens / 1000000
    ) * pricing["output_per_1m_tokens_usd"]

    return {
        "input_cost_usd": round(input_cost, 8),
        "output_cost_usd": round(output_cost, 8),
        "total_cost_usd": round(input_cost + output_cost, 8),
    }

def fetch_google_search_chunks(query: str, max_results: int = 3) -> List[Dict]:
    """Fetches web search snippets and formats them as RAG chunk dictionaries."""
    try:
        # Example using duckduckgo_search (no API key required)
        from ddgs import DDGS
        results = []
        with DDGS() as ddgs:
            search_results = list(ddgs.text(query, max_results=max_results))
            for i, r in enumerate(search_results):
                results.append({
                    "id": f"web_{i}",
                    "text": r.get("body", ""),
                    "title": r.get("title", "Web Result"),
                    "url": r.get("href", ""),
                    "ood": {"ood_band": "id", "ood_probability": 0.05},
                    "source_tier": "web"
                })
        return results
    except Exception as e:
        print(f"[Warning] Web search failed: {e}")
        return []


def log_abstain_query(query: str, embedding: object, filepath: str = LOG_FILE) -> None:
    """Logs abstained queries alongside their vector embeddings and timestamps."""
    logger.info(
        json.dumps({
            "event": "rag_abstain",
            "query": query,
        })
    )
    # Convert numpy arrays to Python list for JSON serialization
    if isinstance(embedding, np.ndarray):
        embedding_list = embedding.tolist()
    elif hasattr(embedding, "tolist"):
        embedding_list = embedding.tolist()
    else:
        embedding_list = list(embedding)

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "embedding": embedding_list
    }

    # Append entry as a single line JSON
    #with open(filepath, "a", encoding="utf-8") as f:
    #    f.write(json.dumps(log_entry) + "\n")

def decide_action(chunks: List[Dict]) -> str:
    """generate | hedge | abstain, from query-level OOD first, chunk
    composition second. See module docstring point 2 for why query-level
    takes priority."""
    if not chunks:
        return "abstain"

    query_ood = chunks[0].get("query_ood") or {}
    low, high = query_ood.get("ood_confidence_interval", (0.0, 1.0))
    ci_width = high - low
    max_ci_width = 0.37
    
    #if ci_width >= max_ci_width and (low <= 0.79 <= high):
    #    return "clarify"
    if query_ood.get("is_ood"):
        return "abstain"
    if query_ood.get("is_near_ood"):
        return "hedge"

    # Query itself looked "id" — fall back to what was actually retrieved.
    bands = [c["ood"]["ood_band"] for c in chunks if c.get("ood")]
    if not bands:
        return "hedge"  # scoring didn't run for some reason — don't over-trust it
    if bands.count("id") == 0:
        return "abstain"  # nothing trustworthy came back despite an on-topic query
    if bands.count("id") < len(bands):
        return "hedge"
    return "generate"

def prepare_model(model_path: str, model_name: str) -> str:
    source = Path(model_path)
    target = Path("/tmp/models") / model_name

    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)

    return str(target)

class RAGAgent:
    def __init__(self, index_dir: str, ood_reference_path: str,
                 embed_model_name: str = "all-MiniLM-L6-v2",
                 reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 generator: Optional[Generator] = None,
                 log_path: Optional[str] = None,
                 grounding_model_path: Optional[str] = None,
                 grounding_threshold: float = 0.5): #"./agent_log.jsonl"
        import chromadb
        from sentence_transformers import SentenceTransformer, CrossEncoder

        self.index_dir = index_dir
        
        self.log_path = Path(log_path) if log_path else None

        self.client = chromadb.PersistentClient(path=index_dir)
        self.text_collection = self.client.get_collection("text_chunks")
        embed_model_name = os.getenv(
            "RAG_EMBED_MODEL",
            embed_model_name
        )
        reranker_model_name = os.getenv(
            "RAG_CROSS_MODEL",
            reranker_model_name
        )

        self.embed_model_name = embed_model_name
        self.reranker_model_name = reranker_model_name
        logger.info(
            "Loading models: embedder=%s reranker=%s",
            embed_model_name,
            reranker_model_name,
        )
        self.embed_model = SentenceTransformer(embed_model_name)
        self.cross_encoder = CrossEncoder(reranker_model_name)
        self.bm25_state = build_bm25_index(self.text_collection)

        self.router = QueryRouter(self.text_collection, self.embed_model)
        self.detector = EmbeddingOOD.load(ood_reference_path)
        self.grounding_evaluator = (
            GroundingEvaluator.load(grounding_model_path, grounding_threshold)
            if grounding_model_path else None
        )

        self.generator = generator or load_generator("extractive")
        #print(f"generator: {self.generator}")
        self._fallback_generator = ExtractiveGenerator()  # used if self.generator errors

    def answer(self, query: str, history: Optional[List[Dict]] = None,
              top_k_final: int = 5) -> Dict:
        t0 = time.time()
        route = self.router.route(query, ood_detector=self.detector)
        query_metrics = {}
        chunk_metrics = []
        retrieval_metrics = {}
        grounding_metrics = {"enabled": self.grounding_evaluator is not None}
        execution_path = ["stage1"]
        grounding_rejected = False

        if route["decision"] == "chitchat":
            chunks: List[Dict] = []
            action = "chitchat"
        else:
            
            query_embedding = self.embed_model.encode([query])[0]

            #Extract Query-level P(OOD) & Confidence Interval
            query_ood = self.detector.score_query(query_embedding)
            query_ci = query_ood.get("ood_confidence_interval")
            query_metrics = {
                "p_ood": query_ood.get("ood_probability"),
                "ood_probability": query_ood.get("ood_probability"),
                "p_ood_ci_95": list(query_ci) if query_ci is not None else None,  # (low, high)
                "probability_source": query_ood.get("probability_source"),
                "ood_band": query_ood.get("ood_band"),
                "energy": query_ood.get("energy"),
                "mahalanobis": query_ood.get("mahalanobis"),
                "knn_distance": query_ood.get("knn_distance"),
            }
            if query_ood["is_ood"]:
               action = "abstain"
               chunks = []
            else:
                execution_path.append("stage2")
                chunks = hybrid_retrieve(query, self.text_collection, self.embed_model,
                                        self.bm25_state, self.cross_encoder, top_k_final=top_k_final)
                self.detector.score_retrieval(query_embedding, chunks, self.text_collection)
                for chunk in chunks:
                    ood_info = chunk.get("ood", {})
                    chunk_ci = ood_info.get("ood_confidence_interval")
                    chunk_metrics.append({
                        "id": chunk.get("id"),
                        "title": chunk.get("title", ""),
                        "p_ood": ood_info.get("ood_probability"),
                        "ood_probability": ood_info.get("ood_probability"),
                        "p_ood_ci_95": list(chunk_ci) if chunk_ci is not None else None,  # (low, high)
                        "ood_band": ood_info.get("ood_band"),
                        "probability_source": ood_info.get("probability_source"),
                        "energy": ood_info.get("energy"),
                        "mahalanobis": ood_info.get("mahalanobis"),
                        "knn_distance": ood_info.get("knn_distance"),
                    })

                trace = chunks[0].get("retrieval_trace") if chunks else None
                if trace is not None:
                    retrieval_metrics = {
                        "stage_latency_ms": dict(trace.stage_latency_ms),
                        "agreement": dict(trace.agreement),
                    }

            action = decide_action(chunks)
            if not query_ood["is_ood"] and self.grounding_evaluator is not None:
                # Evaluate exactly the evidence that existing generation filtering
                # can accept, in retrieval rank order, limited to three chunks.
                chunks, features = self._grounding_inputs(query, query_embedding, chunks, query_ood)
                decision = self.grounding_evaluator.evaluate(query, chunks, features)
                grounding_metrics.update(decision.as_dict())
                if not decision.grounded:
                    grounding_rejected = True
                    action = "abstain"
            if action in ("abstain","hedge"):
                log_abstain_query(query=query, embedding=query_embedding)
                '''web_chunks = fetch_google_search_chunks(query, max_results=3)
                if web_chunks:
                    # Append web chunks to local context
                    chunks.extend(web_chunks)
                    web_search_used = True'''
                    #action = "hedge_with_web"
                
        if action!="abstain":
            chunks = select_chunks_for_generation(chunks, drop_ood=True, drop_near_ood=False)
            prompt = build_generation_prompt(query, chunks, action=action, history=history)
        else:
            prompt=""

        if hasattr(self.generator, "reset_usage"):
            self.generator.reset_usage()

        if grounding_rejected:
            answer_text, generator_used, generation_error = "No relevant context found.", "none", None
        else:
            if action != "abstain":
                execution_path.append("stage3")
            answer_text, generator_used, generation_error = self._generate_with_fallback(prompt, chunks, action)
        token_usage = getattr(self.generator, "last_usage", {})
        
        generator_model = getattr(
            self.generator,
            "model",
            getattr(self.generator, "name", "unknown"),
        )
        cost_metrics = calculate_llm_cost(
            model_name=generator_model,
            input_tokens=token_usage.get("input_tokens", 0),
            output_tokens=token_usage.get("output_tokens", 0),
        )
        logger.info(json.dumps({
            "event": "generation_metrics",
            "model": generator_model,
            "token_usage": token_usage,
            "cost_usd": cost_metrics["total_cost_usd"],
        }))
        citations = ground_citations(answer_text, chunks) if action in ("generate", "hedge") else []

        repro = build_repro_bundle(
            self.index_dir, self.embed_model_name, self.reranker_model_name,
            generation_model_name=generator_used,
            extra={
                "route": route["decision"],
                "action": action,
                "generation_error": generation_error,
                "latency_seconds": round(time.time() - t0, 3),
                "query_ood_metrics": query_metrics,
                "retrieval": retrieval_metrics,
                "grounding": grounding_metrics,
                "execution_path": execution_path,
                "cost_metrics": cost_metrics["total_cost_usd"],
            },
        )
        contexts_text = [chunk["text"] for chunk in chunks if isinstance(chunk, dict) and "text" in chunk]

        result = {
            "query": query,
            "execution_path": execution_path,
            "grounding_metrics": grounding_metrics,
            "contexts": contexts_text,
            "action": action,
            "answer": answer_text,
            "citations": citations,
            "route": route,
            "query_metrics": query_metrics,
            "chunk_metrics": chunk_metrics,
            "cost_metrics": cost_metrics["total_cost_usd"],
            "repro": repro,
        }
        self._log(result)
        return result

    def collect_grounding_example(self, query: str, top_k_final: int = 5) -> Dict[str, Any]:
        """Offline retrieval-only path for training; never calls generation.

        Deliberately retrieve even when Stage 1 would reject: the supplied
        OOD labels must have actual retrieval features, not synthetic scores.
        The online Stage 1 short circuit is unchanged. The current grounding
        classifier is not consulted, so it cannot label its own training set.
        """
        query_embedding = np.asarray(self.embed_model.encode([query])[0])
        query_ood = self.detector.score_query(query_embedding)
        retrieved = hybrid_retrieve(
            query, self.text_collection, self.embed_model, self.bm25_state,
            self.cross_encoder, top_k_final=top_k_final,
        )
        self.detector.score_retrieval(query_embedding, retrieved, self.text_collection)
        chunks, stage1_features = self._grounding_inputs(query, query_embedding, retrieved, query_ood)
        return {
            "query": query,
            "stage1_would_reject": bool(query_ood["is_ood"]),
            "stage1_features": stage1_features,
            "features": extract_grounding_features(query, chunks, stage1_features).tolist(),
            "retrieved_ids": [chunk["id"] for chunk in retrieved],
            "chunks": [{key: chunk.get(key) for key in
                        ("id", "title", "text", "tier", "cosine_similarity", "bm25_score")}
                       for chunk in chunks],
        }

    def _grounding_inputs(
        self, query: str, query_embedding: np.ndarray, chunks: List[Dict], query_ood: Dict,
    ) -> tuple[List[Dict], Dict[str, float]]:
        """Shared evidence selection and features for offline and online use."""
        selected = select_chunks_for_generation(chunks, drop_ood=True, drop_near_ood=False)[:3]
        features = {
            "embedding_norm": float(np.linalg.norm(query_embedding)),
            **{name: float(query_ood.get(name) or 0.0) for name in
               ("ood_probability", "energy", "mahalanobis", "knn_distance")},
        }
        self._annotate_grounding_scores(query, query_embedding, selected)
        return selected, features

    def _annotate_grounding_scores(
        self, query: str, query_embedding: np.ndarray, chunks: List[Dict],
    ) -> None:
        """Compute cosine from stored vectors, including BM25-only hits.

        Dense-only hits also need real BM25 scores rather than treating their
        absence from the sparse candidate list as a zero score.
        """
        if not chunks:
            return
        stored = self.text_collection.get(ids=[c["id"] for c in chunks], include=["embeddings"])
        by_id = dict(zip(stored["ids"], stored["embeddings"]))
        bm25, ids, _, _ = self.bm25_state
        sparse_scores = dict(zip(ids, bm25.get_scores(query.lower().split())))
        for chunk in chunks:
            vector = np.asarray(by_id[chunk["id"]], dtype=np.float64)
            denominator = float(np.linalg.norm(query_embedding) * np.linalg.norm(vector))
            chunk["cosine_similarity"] = float(query_embedding @ vector / denominator) if denominator else 0.0
            chunk["bm25_score"] = float(sparse_scores[chunk["id"]])

    def _generate_with_fallback(self, prompt: str, chunks: List[Dict], action: str):
        """Try the configured generator; on failure (e.g. API outage, missing
        key), degrade to the extractive generator rather than crashing the
        whole request. This is exactly the "no defined behavior for things
        going wrong" gap discussed earlier — here it's handled explicitly."""
        try:
            if action=="abstain":
                text = """> ⚠️ I cannot answer this question based on the provided documents.""" #Please rephrase your query or ask a question related to our internal knowledge base.
            else:
                text = self.generator.generate(prompt, chunks=chunks, action=action)
            return text, self.generator.name, None
        except Exception as e:
            error_summary = f"{type(e).__name__}: {e}"
            fallback_text = self._fallback_generator.generate(prompt, chunks=chunks, action=action)
            return fallback_text, f"{self._fallback_generator.name} (fallback after {error_summary})", error_summary

    def _log(self, result: Dict):
        if not self.log_path:
            return
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({**result, "logged_at": datetime.now(timezone.utc).isoformat()}) + "\n")
        except Exception:
            pass  # logging must never break the actual response

def load_insurance_retriever(index_dir, scope, collection_name):
    """Shared index checks for guarded and standard insurance agents."""
    import chromadb
    from sentence_transformers import SentenceTransformer
    from insurance_retrieval_fixed import InsuranceHybridRetriever

    if collection_name != scope["collection"]:
        raise ValueError("Insurance collection must match the scope file")
    collection = chromadb.PersistentClient(path=index_dir).get_collection(collection_name)
    metadata = collection.metadata or {}
    if (metadata.get("embedding_model") != scope["embedding_model"]
            or metadata.get("corpus_fingerprint") != scope["corpus_fingerprint"]):
        raise ValueError("Insurance index does not match the scoped embedding model/corpus")
    return InsuranceHybridRetriever(collection, SentenceTransformer(scope["embedding_model"]))


class InsuranceRAGAgent(RAGAgent):
    """Opt-in insurance path; replaces the research relevance gate with Stage 2.

    Stage 1 uses the existing fitted embedding detector. No research query router
    or research-trained grounding classifier is used on insurance requests.
    """

    def __init__(self, index_dir: str, ood_reference_path: str,
                 insurance_scope_path: str = "data/home_insurance_scope.json",
                 collection_name: str = "insurance_chunks",
                 assessment_model: str = "gpt-4o-mini",
                 generator: Optional[Generator] = None, log_path: Optional[str] = None,
                 retriever=None, detector=None, assessor=None):
        from src.evidence_gate import InsuranceEvidencePipeline, LangChainEvidenceAssessor

        scope = json.loads(Path(insurance_scope_path).read_text(encoding="utf-8-sig"))
        if collection_name != scope["collection"]:
            raise ValueError("Insurance collection must match the scope file")
        self.index_dir, self.log_path = index_dir, Path(log_path) if log_path else None
        self.embed_model_name = scope["embedding_model"]
        self.reranker_model_name = "none (insurance hybrid RRF)"
        if retriever is None:
            retriever = load_insurance_retriever(index_dir, scope, collection_name)
        self.embed_model = retriever.model
        self.detector = detector if detector is not None else EmbeddingOOD.load(ood_reference_path)
        provenance = getattr(self.detector, "artifact_metadata_", {})
        if detector is None and any(provenance.get(key) != scope.get(key) for key in
                                    ("domain", "collection", "embedding_model", "corpus_fingerprint")):
            raise ValueError("OOD artifact does not match the insurance scope; refit Stage 1")
        self.generator = generator or load_generator("openai", model=assessment_model)
        self.evidence_pipeline = InsuranceEvidencePipeline(
            retriever, assessor if assessor is not None else LangChainEvidenceAssessor(assessment_model),
            scope["trusted_document_ids"],
        )

    def answer(self, query: str, history: Optional[List[Dict]] = None,
               top_k_final: int = 5, customer_context: Optional[Dict] = None,
               include_evidence: bool = False) -> Dict:
        from src.guardrails import INSURANCE_PROMPT_TEMPLATES, insurance_gate_response

        t0 = time.perf_counter()
        timings = {"scope": "request_only_excludes_initialization_and_output_logging",
                   "stages": skipped_metrics(("stage1", "stage2", "stage3")),
                   "stage1_steps": skipped_metrics(("query_embedding", "ood_scoring")),
                   "stage2_steps": {}, "stage3_steps": skipped_metrics(("prompt_build", "generation_and_verification"))}
        with measure(timings["stages"], "stage1"):
            with measure(timings["stage1_steps"], "query_embedding"):
                query_embedding = self.embed_model.encode([query])[0]
            with measure(timings["stage1_steps"], "ood_scoring"):
                query_ood = self.detector.score_query(query_embedding)
        execution_path = ["stage1"]
        chunks, assessment, usage = [], None, {}
        customer_policy_record = None
        generation_error = None
        generator_used = "none"
        if query_ood["is_ood"]:
            action, reason = "abstain", "far_ood"
            gate_metrics = {"enabled": True, "reason": reason, "steps": []}
            decision_support = {"action": action, "category": "out_of_domain",
                                "technical_failure": False, "assessment_status": "not_run",
                                "applicability": {"status": "not_checked"},
                                "next_step": "ask_in_scope_question"}
        else:
            execution_path.append("stage2")
            with measure(timings["stages"], "stage2"):
                gate = self.evidence_pipeline.invoke(query, customer_context, top_k=top_k_final,
                                                     query_embedding=query_embedding)
            timings["stage2_steps"] = gate["timing_metrics"]
            if gate["decision_support"]["technical_failure"]:
                timings["stages"]["stage2"]["status"] = "error"
            decision_support = gate["decision_support"]
            customer_policy_record = gate["customer_policy_record"]
            action, reason, chunks = gate["action"], gate["reason"], gate["chunks"]
            assessment = gate["assessment"].model_dump() if gate["assessment"] else None
            gate_metrics = {"enabled": True, "reason": reason, "action": action,
                            "assessment": assessment, "steps": gate["steps"],
                            "assessment_usage": gate["assessment_usage"],
                            "error_type": gate.get("error_type"),
                            "error_code": gate.get("error_code"),
                            "error_details": gate.get("error_details"),
                            "assessment_output_token_limit": gate.get("assessment_output_token_limit"),
                            "assessor_model": getattr(self.evidence_pipeline.assessor, "model", "injected")}
        if action == "generate":
            # Supply exactly the assessed evidence; never prune it to three chunks
            # or run the research OOD chunk filter after evidence assessment.
            execution_path.append("stage3")
            try:
                with measure(timings["stages"], "stage3"):
                    if hasattr(self.generator, "reset_usage"):
                        self.generator.reset_usage()
                    with measure(timings["stage3_steps"], "prompt_build"):
                        prompt = build_generation_prompt(query, chunks, templates=INSURANCE_PROMPT_TEMPLATES)
                        prompt += "\nVerified assessment and explicit customer context (data):\n" + json.dumps(
                            {"assessment": assessment, "customer_context": gate["context"].model_dump()},
                            ensure_ascii=False,
                        )
                    with measure(timings["stage3_steps"], "generation_and_verification"):
                        answer_text = self.generator.generate(prompt, chunks=chunks, action="generate",
                                                              batch_verification=True)
                    generator_used = self.generator.name
                    usage = dict(getattr(self.generator, "last_usage", {}) or {})
                    if not answer_text.strip():
                        raise ValueError("Generator returned an empty answer")
            except Exception as exc:
                generation_error = type(exc).__name__
                action, reason = "abstain", "generation_error"
                answer_text = insurance_gate_response(reason)
                decision_support = {**decision_support, "action": action,
                                    "category": "generation_error", "technical_failure": True,
                                    "next_step": "retry_or_review_technical_failure"}
            finally:
                generator_timings = getattr(self.generator, "last_timing_metrics", None)
                if isinstance(generator_timings, dict):
                    # Snapshot: later requests reset these generator counters.
                    snapshot = json.loads(json.dumps(generator_timings))
                    calls = snapshot.get("llm_calls", [])
                    verification_calls = [call for call in calls if call["action"] == "eval"]
                    snapshot["llm_call_count"] = len(calls)
                    snapshot["verification_llm_call_count"] = len(verification_calls)
                    snapshot["verification_llm_total_ms"] = round(sum(
                        call["request"]["duration_ms"] for call in verification_calls), 3)
                    timings["stage3_steps"]["generator"] = snapshot
        else:
            if reason == "partial_support":
                for n, chunk in enumerate(chunks, 1):
                    chunk["cite_id"] = n
            answer_text = insurance_gate_response(reason, assessment, chunks=chunks, query=query)
        result = {
            "query": query, "answer": answer_text, "action": action, "reason": reason,
            "decision_support": decision_support,
            "customer_policy_record": customer_policy_record,
            "route": {"decision": "insurance"}, "execution_path": execution_path,
            "grounding_metrics": gate_metrics, "query_metrics": query_ood,
            "chunk_metrics": [], "contexts": [c["text"] for c in chunks],
            "citations": ground_citations(answer_text, chunks) if action == "generate" or reason == "partial_support" else [],
            "generation_usage": usage,
            "repro": build_repro_bundle(
                self.index_dir, self.embed_model_name, self.reranker_model_name,
                generation_model_name=generator_used,
                extra={"grounding": gate_metrics, "generation_error": generation_error,
                       "latency_seconds": round(time.perf_counter() - t0, 3)}),
        }
        timings["request_total_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        measured = {name: entry["duration_ms"] for name, entry in timings["stages"].items()
                    if entry["duration_ms"] is not None}
        timings["slowest_stage"] = max(measured, key=measured.get)
        timings["other_request_overhead_ms"] = round(max(
            0., timings["request_total_ms"] - sum(measured.values())), 3)
        result["timing_metrics"] = timings
        result["repro"]["timing_metrics"] = timings
        result["repro"]["latency_seconds"] = round(timings["request_total_ms"] / 1000, 3)
        if include_evidence:
            result["evidence_chunks"] = chunks
        logger.info(json.dumps({"event": "insurance_stage_timings", **timings}))
        self._log(result)
        return result


class StandardInsuranceRAGAgent(RAGAgent):
    """Insurance adaptation of standard RAG: retrieve and generate, no OOD gate.

    Uses the same index, allowlist, grounded prompt, generator and citation checks
    as the guarded path. It does not force answers or suppress textual abstention.
    """

    def __init__(self, index_dir: str, insurance_scope_path="data/home_insurance_scope.json",
                 collection_name="insurance_chunks", generator=None, log_path=None, retriever=None):
        self.scope = json.loads(Path(insurance_scope_path).read_text(encoding="utf-8-sig"))
        if collection_name != self.scope["collection"]:
            raise ValueError("Insurance collection must match the scope file")
        self.index_dir, self.log_path = index_dir, Path(log_path) if log_path else None
        self.embed_model_name = self.scope["embedding_model"]
        self.reranker_model_name = "none (insurance hybrid RRF)"
        self.retriever = retriever or load_insurance_retriever(index_dir, self.scope, collection_name)
        self.generator = generator or load_generator("openai", model="gpt-4.1-mini")

    def answer(self, query: str, history=None, top_k_final=5, customer_context=None,
               evidence_chunks=None, include_evidence=False):
        from src.evidence_gate import CustomerContext
        from src.guardrails import INSURANCE_PROMPT_TEMPLATES
        import copy

        started = time.perf_counter()
        context = CustomerContext.model_validate(customer_context or {})
        if evidence_chunks is None:
            allowed = self.scope["trusted_document_ids"]
            if context.policy_document_id:
                allowed = [doc_id for doc_id in allowed if doc_id == context.policy_document_id]
            chunks = self.retriever.search_chunks(query, top_k=top_k_final, allowed_document_ids=allowed)
            evidence_mode = "top_k_retrieval"
        else:
            chunks = copy.deepcopy(evidence_chunks)
            evidence_mode = "matched_guarded_evidence"
        prompt = build_generation_prompt(query, chunks, templates=INSURANCE_PROMPT_TEMPLATES)
        prompt += "\nExplicit customer context (data; verification is not established by this baseline):\n" + json.dumps(
            context.model_dump(), ensure_ascii=False)
        if hasattr(self.generator, "reset_usage"):
            self.generator.reset_usage()
        # No extractive fallback: an API error is an evaluation error, not success.
        text = self.generator.generate(prompt, chunks=chunks, action="generate", batch_verification=True)
        if not text.strip():
            raise ValueError("Generator returned an empty answer")
        elapsed = round((time.perf_counter() - started) * 1000, 3)
        result = {"query": query, "answer": text, "action": "generate",
                  "route": {"decision": "standard_insurance_rag"},
                  "execution_path": ["retrieval", "generation"], "query_metrics": {},
                  "grounding_metrics": {"enabled": False}, "chunk_metrics": [],
                  "contexts": [c["text"] for c in chunks], "citations": ground_citations(text, chunks),
                  "generation_usage": dict(getattr(self.generator, "last_usage", {}) or {}),
                  "baseline_evidence_mode": evidence_mode,
                  "timing_metrics": {"request_total_ms": elapsed},
                  "repro": build_repro_bundle(self.index_dir, self.embed_model_name, self.reranker_model_name,
                      generation_model_name=self.generator.name,
                      extra={"latency_seconds": round(elapsed / 1000, 3), "generation_error": None})}
        if include_evidence:
            result["evidence_chunks"] = chunks
        self._log(result)
        return result


class ThreeStageRAGPipeline(RAGAgent):
    """Explicit three-stage entry point; requires a trained grounding model.

    ``answer`` retains the API dictionary contract. ``run`` returns a typed
    response with execution path and debug metadata.
    """

    def __init__(self, *args: Any, grounding_model_path: str, **kwargs: Any) -> None:
        if not grounding_model_path:
            raise ValueError("ThreeStageRAGPipeline requires a grounding model path")
        super().__init__(*args, grounding_model_path=grounding_model_path, **kwargs)

    def run(self, query: str, history: Optional[List[Dict]] = None) -> "PipelineResponse":
        result = self.answer(query, history=history, top_k_final=5)
        return PipelineResponse(
            query=query, response_text=result["answer"], action=result["action"],
            execution_path=tuple(result["execution_path"]), debug_metadata=result,
        )


@dataclass(frozen=True)
class PipelineResponse:
    query: str
    response_text: str
    action: str
    execution_path: tuple[str, ...]
    debug_metadata: Dict[str, Any]


class StandardRAGAgent(RAGAgent):
    """A standard, vanilla RAG model without any OOD detection, 
    routing, fallback web-search, or action decisions (always 'generate')."""
    
    def answer(self, query: str, history: Optional[List[Dict]] = None, top_k_final: int = 5) -> Dict:
        t0 = time.time()
        
        # 1. Standard Retrieval (Dense + BM25 + RRF + Rerank)
        chunks = hybrid_retrieve(
            query, 
            self.text_collection, 
            self.embed_model,
            self.bm25_state, 
            self.cross_encoder, 
            top_k_final=top_k_final
        )

        # 2. Force action to 'generate' (No OOD checks, no abstain, no hedge)
        action = "generate"

        #print("In standard answer")
        # 3. Build Prompt without chunk dropping/filtering
        prompt = build_generation_prompt(query, chunks, action=action, history=history)
        #print(prompt)
        # 4. Generate Answer
        answer_text, generator_used, generation_error = self._generate_with_fallback(prompt, chunks, action)

        # 5. Extract Citations
        citations = ground_citations(answer_text, chunks)

        # 6. Assemble Standard Result Bundle
        contexts_text = [chunk["text"] for chunk in chunks if isinstance(chunk, dict) and "text" in chunk]

        repro = build_repro_bundle(
            self.index_dir, 
            self.embed_model_name, 
            self.reranker_model_name,
            generation_model_name=generator_used,
            extra={
                "route": "standard_rag",
                "action": action,
                "generation_error": generation_error,
                "latency_seconds": round(time.time() - t0, 3),
            },
        )

        result = {
            "query": query,
            "contexts": contexts_text,
            "action": action,
            "answer": answer_text,
            "citations": citations,
            "route": {"decision": "generate"},
            "query_metrics": {},
            "chunk_metrics": [],
            "repro": repro,
        }
        self._log(result)
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-turn RAG Chat Agent")
    parser.add_argument("--index", type=str, default="./chroma_index", help="Path to index directory")
    parser.add_argument("--ood-reference", type=str, default="./ood_reference.npz", help="Path to OOD reference file")
    parser.add_argument("--generator", type=str, default="extractive", help="Generator backend: extractive | openai")
    parser.add_argument("--grounding-model", default=None, help="Trusted joblib grounding classifier")
    parser.add_argument("--grounding-threshold", type=float, default=0.5)
    parser.add_argument("--domain", choices=("research", "home_insurance"), default="research")
    parser.add_argument("--pipeline", choices=("guarded", "standard"), default="guarded")
    parser.add_argument("--insurance-scope", default="data/home_insurance_scope.json")
    parser.add_argument("--collection", default="insurance_chunks")
    parser.add_argument("--assessment-model", default="gpt-4.1-mini")
    parser.add_argument("--customer-context", help="JSON file with explicit customer context")
    parser.add_argument("--log-path", help="Append response records to this JSONL file")
    parser.add_argument("--query", type=str, default=None, help="Optional single query mode. Omit to launch interactive chat.")
    return parser.parse_args()


def display_turn_result(result: Dict[str, Any]) -> None:
    """Helper to display routing, citations, and the answer cleanly."""
    print(f"\n[route: {result['route']['decision']} | action: {result['action']} | model: {result['repro']['generation_model']}]")
    print(f"\nAgent: {result['answer']}")
    
    if result.get("citations"):
        print("\nCitations:")
        for c in result["citations"]:
            title = c.get('title', '')[:100]
            print(f"  [{c['n']}] ({c.get('tier')}, ood_band={c.get('ood_band')}) {title}")
    print("\n" + "-" * 60)
    p_ood = 1
    se_list = []; p_ood_list = []
    if result.get("chunk_metrics") and result.get("action") == "generate":
        print("Retrieval confidence:")
        for c in result["chunk_metrics"]:
            title = c.get('title', '')[:100]
            p_ood = c.get('p_ood')
            low, high = c.get('p_ood_ci_95')
                # Avoid division by zero in log-space/delta propagation
            p_safe = max(p_ood, 1e-6) 
            p_ood_list.append(p_safe)
        
            # Approximate standard error from 95% CI (1.96 z-score)
            se = (high - low) / (2 * 1.96)
            se_list.append(se)
            #print(f"  [{c['n']}] ({c.get('tier')}, ood_band={c.get('ood_band')}) {title}")
        rel_variance_sum = sum((se / p)**2 for se, p in zip(se_list, p_ood_list))
        p_ood = max(p_ood_list)
        overall_se = p_ood * math.sqrt(rel_variance_sum)
        # 3. Construct 95% CI
        ci_low = max(0.0, p_ood - 1.96 * overall_se)
        ci_high = min(1.0, p_ood + 1.96 * overall_se)
        print(f"Overall Retrieval Confidence: {max(p_ood_list)*100:.4f}% (±{1.96*overall_se:.4f} CI: {ci_low:.4f}-{ci_high:.4f})")
def run_single_turn(agent: Any, query: str) -> None:
    """Executes a single query and prints output."""
    result = agent.answer(query)
    display_turn_result(result)


def run_interactive_chat(agent: Any) -> None:
    """Executes an interactive multi-turn REPL loop while tracking history."""
    conversation_history: List[Dict[str, str]] = []
    
    print("\n" + "=" * 60)
    print("Multi-Turn RAG Chatbot Initialized.")
    print("Type 'exit', 'quit', or 'q' to end the session.")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("User: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ("exit", "quit", "q"):
                print("Ending session. Goodbye!")
                break

            # Pass history into the agent call if your RAGAgent.answer supports it
            result = agent.answer(user_input, history=conversation_history)
            display_turn_result(result)

            # Record turn state for the next pass
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": result["answer"]})

        except (KeyboardInterrupt, EOFError):
            print("\nSession interrupted. Exiting.")
            sys.exit(0)


def main() -> None:
    args = parse_args()

    if args.domain == "home_insurance":
        if args.generator != "openai":
            raise ValueError("Insurance answers require --generator openai; extractive output can omit question parts")
        common = dict(index_dir=args.index, insurance_scope_path=args.insurance_scope,
                      collection_name=args.collection,
                      generator=load_generator("openai", model=args.assessment_model), log_path=args.log_path)
        agent = (StandardInsuranceRAGAgent(**common) if args.pipeline == "standard" else
                 InsuranceRAGAgent(**common, ood_reference_path=args.ood_reference,
                                   assessment_model=args.assessment_model))
        context = json.loads(Path(args.customer_context).read_text(encoding="utf-8-sig")) if args.customer_context else {}
        if args.query:
            result = agent.answer(args.query, customer_context=context)
            print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        else:
            while True:
                query = input("User: ").strip()
                if query.lower() in ("exit", "quit", "q"):
                    break
                if query:
                    print(json.dumps(agent.answer(query, customer_context=context),
                                     indent=2, ensure_ascii=False, default=str))
        return

    # Initialize components
    gen = load_generator(args.generator)
    agent_class = StandardRAGAgent if args.pipeline == "standard" else RAGAgent
    agent = agent_class(
        index_dir=args.index, 
        ood_reference_path=args.ood_reference, 
        generator=gen,
        grounding_model_path=args.grounding_model,
        grounding_threshold=args.grounding_threshold,
        log_path=args.log_path,
    )

    # Route between single-turn CLI mode and multi-turn interactive chat mode
    if args.query:
        run_single_turn(agent, args.query)
    else:
        run_interactive_chat(agent)


if __name__ == "__main__":
    main()
