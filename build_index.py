"""
build_index.py
---------------
Step-by-step embedding + indexing:
  1. Load the manifest from corpus_selection.py
  2. Embed text chunks with a sentence embedder (all-MiniLM-L6-v2)
  3. Embed images with CLIP (clip-ViT-B-32) — same sentence-transformers
     library handles both, so a query embedded with the CLIP text tower can
     be compared directly against image embeddings (aligned space)
  4. Store each modality in its own Chroma collection (kept separate on
     purpose — text and image embeddings live in different vector spaces
     even when the underlying model family overlaps; results get fused
     later, at retrieval time, not mixed into one index now)
  5. Run one sanity query against each collection so you can SEE whether
     ID-tier documents rank above near/far-OOD before building anything
     downstream — this is your first real signal on whether the tiers
     you built are actually going to support OOD detection.

Usage:
    pip install chromadb sentence-transformers pillow
    python build_index.py --corpus ./corpus --index ./chroma_index
"""

import argparse
import json
from pathlib import Path


def load_manifest(corpus_dir: Path):
    manifest = json.loads((corpus_dir / "manifest.json").read_text())
    text_records = [r for r in manifest if r["modality"] == "text"]
    image_records = [r for r in manifest if r["modality"] == "image"]
    return text_records, image_records


def build_text_index(text_records, client, batch_size: int = 64):
    from sentence_transformers import SentenceTransformer

    print(f"Embedding {len(text_records)} text chunks (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    collection = client.get_or_create_collection("text_chunks")

    for i in range(0, len(text_records), batch_size):
        batch = text_records[i:i + batch_size]
        texts, kept = [], []
        for r in batch:
            try:
                texts.append(Path(r["path"]).read_text(encoding="utf-8"))
                kept.append(r)
            except Exception:
                continue  # skip unreadable files rather than crash the whole run
        if not texts:
            continue
        embeddings = model.encode(texts, show_progress_bar=False).tolist()
        collection.add(
            ids=[f"{r['tier']}__{r['doc_id']}__{i + j}" for j, r in enumerate(kept)],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"tier": r["tier"], "title": r["title"], "source": r["source"]} for r in kept],
        )
        print(f"  indexed {min(i + batch_size, len(text_records))}/{len(text_records)}")
    return collection


def build_image_index(image_records, client, batch_size: int = 32):
    from sentence_transformers import SentenceTransformer
    from PIL import Image as PILImage

    print(f"\nEmbedding {len(image_records)} images (CLIP ViT-B/32)...")
    model = SentenceTransformer("clip-ViT-B-32")
    collection = client.get_or_create_collection("images")

    for i in range(0, len(image_records), batch_size):
        batch = image_records[i:i + batch_size]
        images, kept = [], []
        for r in batch:
            try:
                images.append(PILImage.open(r["path"]).convert("RGB"))
                kept.append(r)
            except Exception:
                continue  # skip corrupt/unreadable images
        if not images:
            continue
        embeddings = model.encode(images, show_progress_bar=False).tolist()
        collection.add(
            ids=[f"{r['tier']}__{r['doc_id']}__{i + j}" for j, r in enumerate(kept)],
            embeddings=embeddings,
            # Chroma requires a `documents` field even for images — store the
            # file path as a placeholder; the actual content lives in metadata.
            documents=[r["path"] for r in kept],
            metadatas=[{"tier": r["tier"], "title": r["title"], "source": r["source"], "path": r["path"]}
                       for r in kept],
        )
        print(f"  indexed {min(i + batch_size, len(image_records))}/{len(image_records)}")
    return collection


def sanity_query(text_collection, image_collection, query: str = "domain generalization under distribution shift"):
    from sentence_transformers import SentenceTransformer

    print(f"\n--- Sanity query: \"{query}\" ---")

    print("\n[text_chunks collection]")
    text_model = SentenceTransformer("all-MiniLM-L6-v2")
    res = text_collection.query(query_embeddings=text_model.encode([query]).tolist(), n_results=5)
    for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
        print(f"  [{meta['tier']:20s}] dist={dist:.3f}  {meta['title'][:70]}")

    print("\n[images collection] (query embedded via CLIP text tower)")
    clip_model = SentenceTransformer("clip-ViT-B-32")
    res = image_collection.query(query_embeddings=clip_model.encode([query]).tolist(), n_results=5)
    for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
        print(f"  [{meta['tier']:20s}] dist={dist:.3f}  {meta['title'][:70]}")

    print("\nWhat to look for: id_core results should cluster at the lowest")
    print("distances, near_ood should sit further out but still appear, and")
    print("far_ood should barely show up (or rank last) for an in-domain")
    print("query like this one. If far_ood results rank ABOVE id_core, the")
    print("tiers aren't separable enough yet — go back to the")
    print("`*_separability_check.png` plots from corpus_selection.py before")
    print("building the OOD scorer on top of this index.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, default="./corpus")
    parser.add_argument("--index", type=str, default="./chroma_index")
    args = parser.parse_args()

    import chromadb

    corpus_dir = Path(args.corpus)
    text_records, image_records = load_manifest(corpus_dir)
    print(f"Loaded manifest: {len(text_records)} text records, {len(image_records)} image records")

    client = chromadb.PersistentClient(path=args.index)

    text_collection = build_text_index(text_records, client)
    image_collection = build_image_index(image_records, client)

    sanity_query(text_collection, image_collection)

    print(f"\nIndex saved to {args.index}")
    print(f"  text_chunks : {text_collection.count()} items")
    print(f"  images      : {image_collection.count()} items")


if __name__ == "__main__":
    main()
