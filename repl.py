"""
repl.py
--------
Loads the index + models ONCE, then lets you type queries in a loop and see
results instantly — for fast manual iteration on prompts/queries. Use
eval_retrieval.py instead when you want actual numbers across many queries;
use this when you just want to poke at a specific query and eyeball what
comes back.

Usage:
    python repl.py --index ./chroma_index

Commands once running:
    <any text>          run it through hybrid_retrieve (fused + reranked)
    :compare <query>     run the same query through all 4 configs side by side
    :k <number>          change top_k for subsequent queries (default 5)
    :quit / :q            exit
"""

import argparse

from hybrid_retrieval import build_bm25_index, hybrid_retrieve
from eval_retrieval import run_config


def print_results(results, label=""):
    if label:
        print(f"  -- {label} --")
    for r in results:
        dd = f"{r['dense_distance']:.3f}" if r.get("dense_distance") is not None else "  -  "
        bm = f"{r['bm25_score']:.2f}" if r.get("bm25_score") is not None else "  -  "
        rr = f"{r['rerank_score']:.3f}" if r.get("rerank_score") is not None else "  -  "
        print(f"    [{r['tier']:15s}] rerank={rr}  dense={dd}  bm25={bm}  {r['title'][:55]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str, default="./chroma_index")
    args = parser.parse_args()

    print("Loading index + models (one-time cost)...")
    import chromadb
    from sentence_transformers import SentenceTransformer, CrossEncoder

    client = chromadb.PersistentClient(path=args.index)
    text_collection = client.get_collection("text_chunks")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    bm25_state = build_bm25_index(text_collection)
    print("Ready. Type a query, or :compare <query>, or :quit.\n")

    top_k = 5
    while True:
        try:
            line = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line in (":quit", ":q"):
            break
        if line.startswith(":k "):
            top_k = int(line.split(" ", 1)[1])
            print(f"  top_k set to {top_k}")
            continue
        if line.startswith(":compare "):
            query = line[len(":compare "):]
            for cfg in ["dense_only", "bm25_only", "hybrid_fused", "hybrid_reranked"]:
                results = run_config(cfg, query, text_collection, embed_model, bm25_state, cross_encoder, top_k=top_k)
                print_results(results, label=cfg)
            print()
            continue

        # default: full hybrid + rerank pipeline
        results = hybrid_retrieve(line, text_collection, embed_model, bm25_state, cross_encoder, top_k_final=top_k)
        print_results(results)
        print()


if __name__ == "__main__":
    main()
