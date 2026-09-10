"""
agent.py
---------
The orchestrator tying together everything built so far:

    query
      -> QueryRouter            : chitchat, or does this need the corpus at all?
      -> hybrid_retrieve()       : dense + BM25 + RRF + rerank   (hybrid_retrieval.py)
      -> EmbeddingOOD.score_retrieval()  : query-level + chunk-level OOD  (ood_scoring.py)
      -> decide_action()         : generate | hedge | abstain | chitchat
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import math

from hybrid_retrieval import build_bm25_index, hybrid_retrieve
from ood_scoring import EmbeddingOOD
from query_router import QueryRouter
import guardrails
from generator import Generator, load_generator, ExtractiveGenerator, OpenAICompatibleGenerator
import sys
from typing import Dict, Any, List


def decide_action(chunks: List[Dict]) -> str:
    """generate | hedge | abstain, from query-level OOD first, chunk
    composition second. See module docstring point 2 for why query-level
    takes priority."""
    if not chunks:
        return "abstain"

    query_ood = chunks[0].get("query_ood") or {}
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


class RAGAgent:
    def __init__(self, index_dir: str, ood_reference_path: str,
                 embed_model_name: str = "all-MiniLM-L6-v2",
                 reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 generator: Optional[Generator] = None,
                 log_path: Optional[str] = "./agent_log.jsonl"):
        import chromadb
        from sentence_transformers import SentenceTransformer, CrossEncoder

        self.index_dir = index_dir
        self.embed_model_name = embed_model_name
        self.reranker_model_name = reranker_model_name
        self.log_path = Path(log_path) if log_path else None

        self.client = chromadb.PersistentClient(path=index_dir)
        self.text_collection = self.client.get_collection("text_chunks")
        self.embed_model = SentenceTransformer(embed_model_name)
        self.cross_encoder = CrossEncoder(reranker_model_name)
        self.bm25_state = build_bm25_index(self.text_collection)

        self.router = QueryRouter(self.text_collection, self.embed_model)
        self.detector = EmbeddingOOD.load(ood_reference_path)

        self.generator = generator or load_generator("extractive")
        print(f"generator: {self.generator}")
        self._fallback_generator = ExtractiveGenerator()  # used if self.generator errors

    def answer(self, query: str, history: Optional[List[Dict]] = None,
              top_k_final: int = 5) -> Dict:
        t0 = time.time()
        route = self.router.route(query, ood_detector=self.detector)
        query_metrics = {}
        chunk_metrics = []
        retrieval_metrics = {}

        if route["decision"] == "chitchat":
            chunks: List[Dict] = []
            action = "chitchat"
        else:
            chunks = hybrid_retrieve(query, self.text_collection, self.embed_model,
                                     self.bm25_state, self.cross_encoder, top_k_final=top_k_final)
            query_embedding = self.embed_model.encode([query])[0]
            self.detector.score_retrieval(query_embedding, chunks, self.text_collection)

            #Extract Query-level P(OOD) & Confidence Interval
            query_ood = self.detector.score_query(query_embedding)
            query_metrics = {
                "p_ood": query_ood.get("ood_probability"),
                "ood_probability": query_ood.get("ood_probability"),
                "p_ood_ci_95": query_ood.get("ood_confidence_interval"),  # (low, high)
                "probability_source": query_ood.get("probability_source"),
                "ood_band": query_ood.get("ood_band"),
                "energy": query_ood.get("energy"),
                "mahalanobis": query_ood.get("mahalanobis"),
                "knn_distance": query_ood.get("knn_distance"),
            }
            for chunk in chunks:
                ood_info = chunk.get("ood", {})
                chunk_metrics.append({
                    "id": chunk.get("id"),
                    "title": chunk.get("title", ""),
                    "p_ood": ood_info.get("ood_probability"),
                    "ood_probability": ood_info.get("ood_probability"),
                    "p_ood_ci_95": ood_info.get("ood_confidence_interval"),  # (low, high)
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
            chunks = guardrails.select_chunks_for_generation(chunks, drop_ood=True, drop_near_ood=False)

        prompt = guardrails.build_generation_prompt(query, chunks, action=action, history=history)

        answer_text, generator_used, generation_error = self._generate_with_fallback(prompt, chunks, action)

        citations = guardrails.ground_citations(answer_text, chunks) if action in ("generate", "hedge") else []

        repro = guardrails.build_repro_bundle(
            self.index_dir, self.embed_model_name, self.reranker_model_name,
            generation_model_name=generator_used,
            extra={
                "route": route["decision"],
                "action": action,
                "generation_error": generation_error,
                "latency_seconds": round(time.time() - t0, 3),
                "query_ood_metrics": query_metrics,
                "retrieval": retrieval_metrics,
            },
        )

        result = {
            "query": query,
            "action": action,
            "answer": answer_text,
            "citations": citations,
            "route": route,
            "query_metrics": query_metrics,
            "chunk_metrics": chunk_metrics,
            "repro": repro,
        }
        self._log(result)
        return result

    def _generate_with_fallback(self, prompt: str, chunks: List[Dict], action: str):
        """Try the configured generator; on failure (e.g. API outage, missing
        key), degrade to the extractive generator rather than crashing the
        whole request. This is exactly the "no defined behavior for things
        going wrong" gap discussed earlier — here it's handled explicitly."""
        try:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-turn RAG Chat Agent")
    parser.add_argument("--index", type=str, default="./chroma_index", help="Path to index directory")
    parser.add_argument("--ood-reference", type=str, default="./ood_reference.npz", help="Path to OOD reference file")
    parser.add_argument("--generator", type=str, default="extractive", help="Generator backend: extractive | openai")
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
        print(f"Overall Retrieval Confidence: {max(p_ood_list)*100:.4f %} (±{1.96*overall_se:.4f} CI: {ci_low:.4f}-{ci_high:.4f})")
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

    # Initialize components
    gen = load_generator(args.generator)
    agent = RAGAgent(
        index_dir=args.index, 
        ood_reference_path=args.ood_reference, 
        generator=gen
    )

    # Route between single-turn CLI mode and multi-turn interactive chat mode
    if args.query:
        run_single_turn(agent, args.query)
    else:
        run_interactive_chat(agent)


if __name__ == "__main__":
    main()
