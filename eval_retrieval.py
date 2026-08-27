"""
eval_retrieval.py
------------------
Rigorous(ish) evaluation of hybrid_retrieval.py, built around two
complementary eval sets since you don't have human relevance judgments:

  A. SELF-RETRIEVAL RECALL (does the pipeline find a document given a query
     clearly derived from it?)
     For a sample of id_core documents, use the title as a synthetic query
     and check whether the document's own chunk appears in the top-k
     results. This is a floor, not a ceiling — a system that fails this
     is definitely broken; a system that passes it isn't necessarily good
     at harder, less on-the-nose queries. Cheap, and needs no manual
     labeling.

  B. TIER PRECISION on hand-written in-domain queries
     For a small set of realistic queries about domain generalization /
     robustness, measure what fraction of top-k results are id_core vs
     near_ood vs far_ood. This is the metric that actually matters for your
     project's thesis: a good retriever should pull almost entirely from
     id_core for a clearly in-domain query, with far_ood essentially never
     appearing.

Both are run as ABLATIONS across four configurations — dense-only,
BM25-only, hybrid (RRF fused, no rerank), hybrid+reranked — so you can see
whether each stage is actually earning its place, not just assume it is.

Usage:
    python eval_retrieval.py --index ./chroma_index
"""

import argparse
import random
import time
from typing import List, Dict

from hybrid_retrieval import (
    dense_search, bm25_search, fuse_rrf, rerank, build_bm25_index,
)

# Hand-written in-domain queries for the tier-precision eval (part B).
# Edit these to match your actual corpus topic if you change DEFAULT_TIERS
# in corpus_selection.py.
IN_DOMAIN_QUERIES = [
    "domain generalization under distribution shift",
    "how do neural networks handle out-of-distribution inputs",
    "robustness of vision models to covariate shift",
    "unsupervised domain adaptation techniques",
    "Hi, how are you doing?"
]


# ---------------------------------------------------------------------------
# Config-to-function mapping for the ablation
# ---------------------------------------------------------------------------

def run_config(config_name: str, query: str, text_collection, embed_model,
               bm25_state, cross_encoder, top_k: int = 10) -> List[Dict]:
    if config_name == "dense_only":
        return dense_search(query, text_collection, embed_model, top_k=top_k)
    if config_name == "bm25_only":
        return bm25_search(query, *bm25_state, top_k=top_k)
    if config_name == "hybrid_fused":
        dense_hits = dense_search(query, text_collection, embed_model, top_k=20)
        bm25_hits = bm25_search(query, *bm25_state, top_k=20)
        return fuse_rrf([dense_hits, bm25_hits], top_k=top_k)
    if config_name == "hybrid_reranked":
        dense_hits = dense_search(query, text_collection, embed_model, top_k=20)
        bm25_hits = bm25_search(query, *bm25_state, top_k=20)
        fused = fuse_rrf([dense_hits, bm25_hits], top_k=20)
        return rerank(query, fused, cross_encoder, top_k=top_k)
    raise ValueError(config_name)


# ---------------------------------------------------------------------------
# Eval A: self-retrieval recall / MRR
# ---------------------------------------------------------------------------

def eval_self_retrieval(config_name, text_collection, embed_model, bm25_state,
                         cross_encoder, n_samples: int = 30, top_k: int = 10, seed: int = 0):
    all_docs = text_collection.get(include=["documents", "metadatas"])
    id_core_indices = [i for i, m in enumerate(all_docs["metadatas"]) if m["tier"] == "id_core_text"]
    random.Random(seed).shuffle(id_core_indices)
    sample = id_core_indices[:n_samples]

    hits, reciprocal_ranks = 0, []
    for i in sample:
        target_id = all_docs["ids"][i]
        query = all_docs["metadatas"][i]["title"]  # use the title as a synthetic query
        results = run_config(config_name, query, text_collection, embed_model, bm25_state, cross_encoder, top_k=top_k)
        result_ids = [r["id"] for r in results]
        if target_id in result_ids:
            hits += 1
            reciprocal_ranks.append(1.0 / (result_ids.index(target_id) + 1))
        else:
            reciprocal_ranks.append(0.0)

    recall_at_k = hits / len(sample)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return {"recall_at_k": recall_at_k, "mrr": mrr, "n": len(sample)}


# ---------------------------------------------------------------------------
# Eval B: tier precision on hand-written in-domain queries
# ---------------------------------------------------------------------------

def eval_tier_precision(config_name, text_collection, embed_model, bm25_state,
                         cross_encoder, queries: List[str], top_k: int = 10):
    id_frac, far_present_count = [], 0
    for q in queries:
        results = run_config(config_name, q, text_collection, embed_model, bm25_state, cross_encoder, top_k=top_k)
        tiers = [r["tier"] for r in results]
        id_frac.append(tiers.count("id_core_text") / len(tiers))
        if any(t.startswith("far_ood") for t in tiers):
            far_present_count += 1

    return {
        "avg_id_core_fraction": sum(id_frac) / len(id_frac),
        "queries_with_far_ood_leakage": far_present_count,
        "n_queries": len(queries),
    }


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str, default="./chroma_index")
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    import chromadb
    from sentence_transformers import SentenceTransformer, CrossEncoder

    client = chromadb.PersistentClient(path=args.index)
    text_collection = client.get_collection("text_chunks")

    print("Loading models...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    bm25_state = build_bm25_index(text_collection)

    configs = ["dense_only", "bm25_only", "hybrid_fused", "hybrid_reranked"]

    print(f"\n{'Config':18s} {'Recall@k':>10s} {'MRR':>8s} {'AvgID%':>8s} {'FarLeak':>9s} {'Latency(s)':>11s}")
    print("-" * 68)
    for cfg in configs:
        t0 = time.time()
        self_ret = eval_self_retrieval(cfg, text_collection, embed_model, bm25_state, cross_encoder,
                                        n_samples=args.n_samples, top_k=args.top_k)
        tier_prec = eval_tier_precision(cfg, text_collection, embed_model, bm25_state, cross_encoder,
                                         IN_DOMAIN_QUERIES, top_k=args.top_k)
        elapsed = time.time() - t0
        per_query = elapsed / (args.n_samples + len(IN_DOMAIN_QUERIES))

        print(f"{cfg:18s} {self_ret['recall_at_k']:>10.2%} {self_ret['mrr']:>8.3f} "
              f"{tier_prec['avg_id_core_fraction']:>8.2%} "
              f"{tier_prec['queries_with_far_ood_leakage']:>5d}/{tier_prec['n_queries']:<3d} "
              f"{per_query:>10.3f}s")

    print("\nHow to read this:")
    print("  Recall@k / MRR   - can the pipeline find a document given an obvious query for it?")
    print("                     (self-retrieval floor — should be high, e.g. >0.8, for all configs)")
    print("  AvgID%           - of top-k results for realistic in-domain queries, what fraction")
    print("                     came from id_core? Higher is better; this is the metric that")
    print("                     matters most for whether OOD detection has a foundation to build on.")
    print("  FarLeak          - how many of the in-domain queries pulled ANY far_ood chunk into")
    print("                     top-k. Should be 0 or close to it for a well-separated corpus.")
    print("  Latency(s)       - per-query wall time for that config, at the same top_k.")
    print("\nIf hybrid_reranked doesn't beat dense_only on AvgID% and FarLeak, the reranker isn't")
    print("earning its latency cost for this corpus — worth knowing before you build on top of it.")


if __name__ == "__main__":
    main()
