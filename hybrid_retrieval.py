"""
hybrid_retrieval.py
--------------------
Hybrid retrieval (dense + BM25, fused with reciprocal rank fusion) plus a
cross-encoder reranker, over the text_chunks Chroma collection from
build_index.py.

Deliberately structured as four independent stages, each a plain function
taking/returning a list of dicts, so any one stage can be swapped later
without touching the others:

    dense_search()  -->  bm25_search()  -->  fuse_rrf()  -->  rerank()

Every candidate dict keeps its `tier` (id_core / near_ood / far_ood) and its
raw `dense_distance` all the way through every stage — even after fusion and
reranking discard/reorder things. That's deliberate: the OOD scorer you'll
build next needs the actual embedding-space distance, not just a final rank
position, and you'll want to be able to check things like "hit rate by tier"
without re-querying anything.

Usage:
    pip install rank_bm25 sentence-transformers chromadb
    python hybrid_retrieval.py --index ./chroma_index --query "your query here"
"""

import argparse
from typing import List, Dict


# ---------------------------------------------------------------------------
# Stage 1: dense retrieval (existing Chroma text_chunks collection)
# ---------------------------------------------------------------------------

def dense_search(query: str, text_collection, embed_model, top_k: int = 20) -> List[Dict]:
    q_emb = embed_model.encode([query]).tolist()
    res = text_collection.query(query_embeddings=q_emb, n_results=top_k)

    candidates = []
    for doc, meta, dist, cid in zip(res["documents"][0], res["metadatas"][0],
                                     res["distances"][0], res["ids"][0]):
        candidates.append({
            "id": cid, "text": doc, "tier": meta["tier"], "title": meta.get("title", ""),
            "dense_distance": dist,   # raw cosine distance — kept for the OOD scorer later
            "bm25_score": None,
            "fused_rank": None,
            "rerank_score": None,
        })
    return candidates


# ---------------------------------------------------------------------------
# Stage 2: sparse retrieval (BM25 over the same corpus, kept in-memory —
# swap for Elasticsearch/Whoosh later without touching anything else)
# ---------------------------------------------------------------------------

def build_bm25_index(text_collection):
    """Pulls every document out of the Chroma collection once, tokenizes,
    and builds an in-memory BM25 index. Cheap to rebuild at startup for a
    corpus this size (a few hundred chunks) — for a much bigger corpus this
    is the first thing you'd swap for a real search engine."""
    from rank_bm25 import BM25Okapi

    all_docs = text_collection.get(include=["documents", "metadatas"])
    ids = all_docs["ids"]
    texts = all_docs["documents"]
    metas = all_docs["metadatas"]

    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    return bm25, ids, texts, metas


def bm25_search(query: str, bm25, ids, texts, metas, top_k: int = 20) -> List[Dict]:
    scores = bm25.get_scores(query.lower().split())
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    candidates = []
    for i in ranked_idx:
        candidates.append({
            "id": ids[i], "text": texts[i], "tier": metas[i]["tier"], "title": metas[i].get("title", ""),
            "dense_distance": None,
            "bm25_score": float(scores[i]),
            "fused_rank": None,
            "rerank_score": None,
        })
    return candidates


# ---------------------------------------------------------------------------
# Stage 3: fuse dense + BM25 rankings with Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def fuse_rrf(ranked_lists: List[List[Dict]], k: int = 60, top_k: int = 10) -> List[Dict]:
    """RRF: score(doc) = sum over lists of 1 / (k + rank_in_that_list).
    Rank-based, not score-based — this is what lets us fuse dense cosine
    distances and BM25 scores without normalizing them onto a shared scale,
    and it's the same reason it'll extend cleanly to a third ranked list
    (e.g. image retrieval) later without any rework."""
    fused_scores: Dict[str, float] = {}
    merged: Dict[str, Dict] = {}

    for ranked_list in ranked_lists:
        for rank, cand in enumerate(ranked_list):
            fused_scores[cand["id"]] = fused_scores.get(cand["id"], 0.0) + 1.0 / (k + rank + 1)
            # merge fields (dense list carries dense_distance, bm25 list carries bm25_score)
            if cand["id"] not in merged:
                merged[cand["id"]] = dict(cand)
            else:
                for field in ("dense_distance", "bm25_score"):
                    if cand.get(field) is not None:
                        merged[cand["id"]][field] = cand[field]

    fused_order = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)[:top_k]
    results = []
    for rank, cid in enumerate(fused_order):
        c = merged[cid]
        c["fused_rank"] = rank
        c["fused_score"] = fused_scores[cid]
        results.append(c)
    return results


# ---------------------------------------------------------------------------
# Stage 4: cross-encoder reranking
# ---------------------------------------------------------------------------

def rerank(query: str, candidates: List[Dict], cross_encoder, top_k: int = 5) -> List[Dict]:
    """Small cross-encoder (ms-marco-MiniLM-L-6-v2) for the prototype —
    swap for cross-encoder/bge-reranker-base later for better quality at
    higher latency/compute cost; the call signature below doesn't change."""
    pairs = [(query, c["text"]) for c in candidates]
    scores = cross_encoder.predict(pairs)

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def hybrid_retrieve(query: str, text_collection, embed_model, bm25_state,
                     cross_encoder, top_k_stage1: int = 20, top_k_fused: int = 10,
                     top_k_final: int = 5, ood_detector=None) -> List[Dict]:
    # Dense and BM25 have no dependency on each other — running them
    # sequentially wastes latency for nothing. This is one of the cheap
    # production wins most people miss: budget latency for "the LLM call"
    # and forget the retrieval stages in front of it add up too.
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=2) as pool:
        dense_future = pool.submit(dense_search, query, text_collection, embed_model, top_k_stage1)
        bm25_future = pool.submit(bm25_search, query, *bm25_state, top_k_stage1)
        dense_hits = dense_future.result()
        bm25_hits = bm25_future.result()

    fused = fuse_rrf([dense_hits, bm25_hits], top_k=top_k_fused)
    final = rerank(query, fused, cross_encoder, top_k=top_k_final)
    if ood_detector is not None:
        ood_detector.score_chunks(final, text_collection)
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str, default="./chroma_index")
    parser.add_argument("--query", type=str, default="domain generalization under distribution shift")
    parser.add_argument("--ood-reference", type=str, default=None,
                        help=".npz made by ood_scoring.py; annotates final chunks")
    args = parser.parse_args()

    import chromadb
    from sentence_transformers import SentenceTransformer, CrossEncoder

    client = chromadb.PersistentClient(path=args.index)
    text_collection = client.get_collection("text_chunks")

    print("Loading models...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    print("Building BM25 index...")
    bm25_state = build_bm25_index(text_collection)

    print(f"\nQuery: \"{args.query}\"\n")
    detector = None
    if args.ood_reference:
        from ood_scoring import EmbeddingOOD
        detector = EmbeddingOOD.load(args.ood_reference)
    results = hybrid_retrieve(args.query, text_collection, embed_model, bm25_state, cross_encoder,
                              ood_detector=detector)

    for r in results:
        dd = f"{r['dense_distance']:.3f}" if r["dense_distance"] is not None else "  -  "
        bm = f"{r['bm25_score']:.2f}" if r["bm25_score"] is not None else "  -  "
        if "ood" in r:
            score = r["ood"]
            low, high = score["ood_confidence_interval"]
            ood = (f"  P(OOD)={score['ood_probability']:.0%} "
                   f"CI95%=[{low:.0%}, {high:.0%}] OOD={score['is_ood']}")
        else:
            ood = ""
        print(f"  [{r['tier']:15s}] rerank={r['rerank_score']:.3f}  dense_dist={dd}  bm25={bm}  {r['title'][:60]}{ood}")

    print("\nEvery field above (dense_distance, bm25_score, fused_rank, rerank_score,")
    print("tier) is preserved on each candidate dict — this is what the OOD scorer")
    print("will read from next, without needing to re-run retrieval.")


if __name__ == "__main__":
    main()
