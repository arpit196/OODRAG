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

from hybrid_retrieval import build_bm25_index, hybrid_retrieve
from ood_scoring import EmbeddingOOD
from query_router import QueryRouter
import guardrails
from generator import Generator, load_generator, ExtractiveGenerator, OpenAICompatibleGenerator


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

        if route["decision"] == "chitchat":
            chunks: List[Dict] = []
            action = "chitchat"
        else:
            chunks = hybrid_retrieve(query, self.text_collection, self.embed_model,
                                     self.bm25_state, self.cross_encoder, top_k_final=top_k_final)
            query_embedding = self.embed_model.encode([query])[0]
            self.detector.score_retrieval(query_embedding, chunks, self.text_collection)
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
            },
        )

        result = {
            "query": query,
            "action": action,
            "answer": answer_text,
            "citations": citations,
            "route": route,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str, default="./chroma_index")
    parser.add_argument("--ood-reference", type=str, default="./ood_reference.npz")
    parser.add_argument("--generator", type=str, default="extractive", help="extractive | openai")
    parser.add_argument("--query", type=str, required=True)
    args = parser.parse_args()

    gen = load_generator(args.generator)
    agent = RAGAgent(index_dir=args.index, ood_reference_path=args.ood_reference, generator=gen)

    result = agent.answer(args.query)

    print(f"\nroute      : {result['route']['decision']}")
    print(f"action     : {result['action']}")
    print(f"generator  : {result['repro']['generation_model']}")
    print(f"\nanswer:\n{result['answer']}")
    if result["citations"]:
        print("\ncitations:")
        for c in result["citations"]:
            print(f"  [{c['n']}] ({c['tier']}, ood_band={c['ood_band']}) {c['title'][:100]}")


if __name__ == "__main__":
    main()
