"""
query_router.py
-----------------
Runs BEFORE hybrid_retrieve(). Decides whether a query needs retrieval at
all, using the same idea as OOD distance-scoring — just with a second,
small reference cluster instead of one.

Two reference centroids in embedding space:
  - CHITCHAT_EXAMPLES : greetings, small talk, thanks, etc.
  - the corpus centroid (id_core_text) : what "on-topic" looks like

A query is routed to whichever centroid it's closer to. This matters
because retrieval always returns its top-k nearest neighbors regardless of
whether any of them are actually relevant — there's no "no results" option
in nearest-neighbor search. So "high distance from the corpus" alone can't
tell you whether a query is (a) chit-chat that doesn't need the corpus at
all, or (b) a real question the corpus genuinely fails to cover. Those need
opposite responses: (a) gets a plain conversational reply, no retrieval,
no hedging; (b) is the actual case your low-confidence/abstention logic
should handle.

Usage:
    from query_router import QueryRouter
    router = QueryRouter(text_collection, embed_model)
    route = router.route("Hi, how are you?")   # -> "chitchat"
    route = router.route("how does domain generalization work?")  # -> "domain"
"""

from typing import List
import numpy as np


# Small, deliberately generic set — extend this with real examples you see
# users send once you have any usage data. Doesn't need to be large; it just
# needs to define "what does casual conversation look like" as a cluster.
CHITCHAT_EXAMPLES = [
    "Hi, how are you?",
    "Hello!",
    "Good morning",
    "Thanks a lot",
    "Thank you, that's helpful",
    "What can you do?",
    "Who are you?",
    "Can you help me?",
    "Bye, see you later",
    "That's interesting",
    "Ok got it",
    "Nice to meet you",
    "How does this work?",
    "What is this tool for?",
]


class QueryRouter:
    def __init__(self, text_collection, embed_model, chitchat_examples: List[str] = None):
        self.embed_model = embed_model
        self.chitchat_examples = chitchat_examples or CHITCHAT_EXAMPLES

        # Reference centroid 1: chit-chat cluster (mean embedding of examples)
        chitchat_embs = self.embed_model.encode(self.chitchat_examples)
        self.chitchat_centroid = np.mean(chitchat_embs, axis=0)

        # Reference centroid 2: the corpus itself (id_core only — this is
        # "what on-topic looks like", not the whole index including far_ood)
        all_docs = text_collection.get(include=["embeddings", "metadatas"])
        id_core_embs = [e for e, m in zip(all_docs["embeddings"], all_docs["metadatas"])
                         if m["tier"] == "id_core_text"]
        self.corpus_centroid = np.mean(np.array(id_core_embs), axis=0)

    @staticmethod
    def _cosine_dist(a, b):
        a, b = np.asarray(a), np.asarray(b)
        return 1.0 - (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

    def route(self, query: str) -> dict:
        q_emb = self.embed_model.encode([query])[0]

        dist_chitchat = self._cosine_dist(q_emb, self.chitchat_centroid)
        dist_corpus = self._cosine_dist(q_emb, self.corpus_centroid)

        # Whichever centroid is closer wins. Ties/near-ties lean toward
        # "domain" — better to run retrieval unnecessarily than to skip it
        # for a real question.
        decision = "chitchat" if dist_chitchat < dist_corpus else "domain"

        return {
            "query": query,
            "decision": decision,
            "dist_to_chitchat": float(dist_chitchat),
            "dist_to_corpus": float(dist_corpus),
        }


if __name__ == "__main__":
    import chromadb
    from sentence_transformers import SentenceTransformer

    client = chromadb.PersistentClient(path="./chroma_index")
    text_collection = client.get_collection("text_chunks")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    router = QueryRouter(text_collection, embed_model)

    test_queries = [
        "Hi, how are you?",
        "Thanks, that's helpful!",
        "how does domain generalization handle distribution shift?",
        "what's a good approach to unsupervised domain adaptation?",
        "what's the weather like today",  # genuinely OOD, not chitchat
    ]
    for q in test_queries:
        r = router.route(q)
        print(f"[{r['decision']:8s}] chitchat_dist={r['dist_to_chitchat']:.3f}  "
              f"corpus_dist={r['dist_to_corpus']:.3f}  {q}")
