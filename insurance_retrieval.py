"""Small BM25 + Chroma retriever for real_insurance_corpus.

python insurance_retrieval.py build
python insurance_retrieval.py search --query "Can gradual damage exclusions apply to a water leak?"
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterator

from src.hybrid_retrieval import fuse_rrf

ROOT = Path(__file__).resolve().parent
COLLECTION_NAME = "insurance_chunks"
SOURCE_TYPES = {"policies": "policy", "policy": "policy", "fca": "fca",
                "legislation": "legislation", "fos_guidance": "fos", "fos_decisions": "fos", "fos": "fos"}


def tokenize(text: str) -> list[str]:
    """Keep numbers and rule references useful for ICOBS/PRIN/DRN searches."""
    return re.findall(r"\w+(?:[.'-]\w+)*", text.casefold())


def resolve_corpus_path(root: Path, relative_path: str) -> Path:
    # Scraper manifests produced on Windows must also work from WSL/Linux.
    path = (root / relative_path.replace("\\", "/")).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Document path leaves corpus directory: {relative_path}")
    return path


POLICY_HEADINGS = {
    "definitions",
    "buildings",
    "contents",
    "escape of water",
    "escape of water or oil",
    "storm",
    "storm or flood",
    "flood",
    "fire",
    "theft",
    "theft or attempted theft",
    "accidental damage",
    "subsidence",
    "heave",
    "landslip",
    "alternative accommodation",
    "trace and access",
    "personal possessions",
    "general exclusions",
    "general exceptions",
    "policy exclusions",
    "claims conditions",
    "policy conditions",
    "making a claim",
    "how to claim",
    "what is covered",
    "what is not covered",
}


def normalize_heading(line: str) -> str:
    """Normalize heading text for matching."""
    text = line.strip()

    # Remove leading numbering:
    # 14. Fatal injury
    # 14.1 What is covered
    # 7) Escape of water
    text = re.sub(
        r"^\s*\d+(?:\.\d+)*[.)]?\s+",
        "",
        text,
    )

    text = re.sub(r"\s+", " ", text)

    return text.casefold().strip(" :-–—")


def is_numbered_heading(line: str) -> bool:
    """
    Detect headings such as:
        14. Fatal injury
        7. Escape of water
        14.1 What is covered

    Avoid matching monetary values, dates, or normal sentences.
    """
    text = line.strip()

    if not text or len(text) > 100:
        return False

    return bool(
        re.match(
            r"^\d+(?:\.\d+)*[.)]?\s+"
            r"[A-Za-z][A-Za-z0-9 ,/&'’()-]*$",
            text,
        )
    )


def is_policy_heading(line: str) -> bool:
    """Detect likely semantic headings in insurance policy text."""
    text = line.strip()

    if not text:
        return False

    if len(text) > 100:
        return False

    # Strongest structural signal.
    if is_numbered_heading(text):
        return True

    normalized = normalize_heading(text)

    # Known insurance heading.
    if normalized in POLICY_HEADINGS:
        return True

    # Slight variations:
    # Escape of water - What is covered
    # Theft: What is not covered
    for heading in POLICY_HEADINGS:
        if normalized.startswith(heading + " -"):
            return True

        if normalized.startswith(heading + ":"):
            return True

    # Generic heading heuristic.
    # Short title-case / uppercase lines are often headings.
    words = text.split()

    if len(words) <= 8:
        if text.isupper():
            return True

        if text.istitle() and not text.endswith((".", ",", ";")):
            return True

    return False


def sliding_word_chunks(
    text: str,
    words: int = 220,
    overlap: int = 40,
) -> Iterator[str]:
    """
    Fallback splitter for sections that are too large.
    """
    tokens = text.split()

    if not tokens:
        return

    if len(tokens) <= words:
        yield text.strip()
        return

    step = words - overlap

    for start in range(0, len(tokens), step):
        chunk = tokens[start:start + words]

        if not chunk:
            break

        yield " ".join(chunk)

        if start + words >= len(tokens):
            break


def split_policy_page_into_sections(
    page_text: str,
) -> list[tuple[str, str]]:
    """
    Split one PDF page into semantic sections based on headings.

    Returns:
        [
            ("14. Fatal injury", "..."),
            ("15. Alternative accommodation", "..."),
        ]
    """
    lines = page_text.splitlines()

    sections = []

    current_heading = "unclassified"
    current_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        if is_policy_heading(stripped):

            if current_lines:
                content = "\n".join(current_lines).strip()

                if content:
                    sections.append(
                        (current_heading, content)
                    )

            current_heading = stripped
            current_lines = []

        else:
            current_lines.append(line)

    if current_lines:
        content = "\n".join(current_lines).strip()

        if content:
            sections.append(
                (current_heading, content)
            )

    return sections

def merge_small_sections(
    sections: list[tuple[str, str]],
    min_words: int = 60,
) -> list[tuple[str, str]]:
    if not sections:
        return []

    merged = []
    i = 0

    while i < len(sections):
        heading, text = sections[i]

        if len(text.split()) < min_words and i + 1 < len(sections):
            next_heading, next_text = sections[i + 1]

            combined_text = (
                f"{heading}\n{text}\n\n"
                f"{next_heading}\n{next_text}"
            ).strip()

            merged.append((heading, combined_text))
            i += 2
        else:
            merged.append((heading, text))
            i += 1

    return merged
    
def iter_chunks(
    text: str,
    words: int = 220,
    overlap: int = 40,
) -> Iterator[tuple[int, str, str]]:
    """
    Structure-aware policy chunker.

    Returns:
        page_number,
        section_name,
        chunk_text

    Strategy:
        PDF page
            -> detect headings
            -> keep small semantic sections intact
            -> sliding-window split only if section is too large
    """
    if not 0 <= overlap < words:
        raise ValueError(
            "Require 0 <= overlap < words"
        )

    pages = re.split(
        r"(?m)^--- PAGE (\d+) ---\s*$",
        text,
    )

    page_sections = (
        [(0, pages[0])]
        +
        [
            (int(pages[i]), pages[i + 1])
            for i in range(1, len(pages), 2)
        ]
    )

    for page, page_text in page_sections:

        sections = split_policy_page_into_sections(
            page_text
        )
        sections = merge_small_sections(sections, min_words=60)
        for section_name, section_text in sections:

            for chunk in sliding_word_chunks(
                section_text,
                words=words,
                overlap=overlap,
            ):

                # Add section identity to the embedded text.
                content = (
                    f"Section: {section_name}\n\n"
                    f"{chunk}"
                )

                yield (
                    page,
                    section_name,
                    content,
                )

def iter_generic_chunks(
    text: str,
    words: int = 220,
    overlap: int = 40,
) -> Iterator[tuple[int, str, str]]:

    pages = re.split(
        r"(?m)^--- PAGE (\d+) ---\s*$",
        text,
    )

    page_sections = (
        [(0, pages[0])]
        +
        [
            (int(pages[i]), pages[i + 1])
            for i in range(1, len(pages), 2)
        ]
    )

    for page, content in page_sections:

        for chunk in sliding_word_chunks(
            content,
            words,
            overlap,
        ):
            yield (
                page,
                "unclassified",
                chunk,
            )


def normalize_metadata(record: dict[str, Any], corpus: Path) -> dict[str, Any]:
    """Normalize scraper source types; keep dates/references only when known."""
    source_type = SOURCE_TYPES[record["source_type"]]
    metadata = {key: value for key, value in record.items()
                if isinstance(value, (str, int, float, bool)) and value != ""}
    metadata.update(source_type=source_type, source_subtype=record["source_type"],
                    extracted_path=str(resolve_corpus_path(corpus, record["extracted_path"])),
                    raw_path=str(resolve_corpus_path(corpus, record["raw_path"])))
    if source_type != "policy":
        metadata.setdefault("authority", record.get("provider", ""))
    title = record.get("title", "")
    if source_type == "fca":
        rule = re.search(r"\b(?:ICOBS|PRIN)\s+\d+[A-Z]?(?:\.\d+)*", title, re.I)
        if rule:
            metadata.setdefault("rule", rule.group().upper())
    if source_type == "fos":
        decision = re.search(r"DRN[- ]?\d+", title + " " + record.get("source_url", ""), re.I)
        if decision:
            metadata.setdefault("decision_id", decision.group().upper().replace(" ", "-"))
        normalized_title = " ".join(tokenize(title.replace("-", " ")))
        for issue in ("escape of water", "underinsurance", "gradual damage", "storm damage", "subsidence", "flood"):
            if issue in normalized_title:
                metadata.setdefault("issue", issue.replace(" ", "_"))
                break
    return metadata


def load_chunks(corpus: Path, words: int = 140, overlap: int = 30) -> list[dict[str, Any]]:
    """Ingest only documents listed in the scraper manifest, never benchmark/."""
    records = {}
    for line in (corpus / "metadata/documents.jsonl").read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("source_type") in SOURCE_TYPES:
                records[record["doc_id"]] = record  # Scraper reruns may append duplicates.
    chunks = []
    for record in records.values():
        metadata = normalize_metadata(record, corpus)
        path = Path(metadata["extracted_path"])
        text = path.read_text(encoding="utf-8-sig")
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if metadata["source_type"] == "policy":
            iterator = iter_chunks(
                text,
                words,
                overlap,
            )
        else:
            iterator = iter_generic_chunks(
                text,
                words,
                overlap,
            )
        for index, (
            page,
            section,
            content,
        ) in enumerate(
            iterator
        ):
            chunks.append({
                "id": (
                    f"{record['doc_id']}__"
                    f"{text_hash[:12]}__"
                    f"{index}"
                ),
                "text": content,
                "metadata": {
                    **metadata,
                    "page": page,
                    "section": section,
                    "chunk_index": index,
                    "extracted_sha256": text_hash,
                },
            })
    return chunks


def searchable_text(chunk: dict[str, Any]) -> str:
    metadata = chunk["metadata"]
    heading = " ".join(str(metadata.get(key, "")) for key in ("title", "provider", "section", "rule", "issue"))
    return f"{heading}\n{chunk['text']}"


def build_index(corpus: Path, index: Path, model: Any, model_name: str,
                words: int = 220, overlap: int = 40, batch_size: int = 64) -> dict[str, Any]:
    import chromadb

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    chunks = load_chunks(corpus, words, overlap)
    if not chunks:
        raise ValueError("The manifest has no nonempty extracted documents")
    fingerprint = hashlib.sha256(json.dumps(chunks, sort_keys=True).encode("utf-8")).hexdigest()
    metadata = {"hnsw:space": "cosine", "embedding_model": model_name,
                "corpus_fingerprint": fingerprint, "chunk_words": words, "chunk_overlap": overlap}
    client = chromadb.PersistentClient(path=str(index))
    collection = client.get_or_create_collection(COLLECTION_NAME, metadata=metadata)
    if any((collection.metadata or {}).get(key) != value for key, value in metadata.items()):
        raise ValueError("Corpus, embedding model or chunk settings changed; build into a fresh --index directory")
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        embeddings = model.encode([searchable_text(chunk) for chunk in batch],
                                  normalize_embeddings=True, show_progress_bar=False).tolist()
        collection.upsert(ids=[chunk["id"] for chunk in batch], embeddings=embeddings,
                          documents=[chunk["text"] for chunk in batch],
                          metadatas=[chunk["metadata"] for chunk in batch])
    return {"index": str(index), "collection": COLLECTION_NAME, "chunks": collection.count(),
            "documents": len({chunk["metadata"]["doc_id"] for chunk in chunks}),
            "source_counts": dict(Counter({chunk["metadata"]["doc_id"]: chunk["metadata"]["source_type"]
                                           for chunk in chunks}.values()))}


@dataclass(frozen=True)
class FileHit:
    doc_id: str
    title: str
    source_type: str
    provider: str
    source_url: str
    raw_path: str
    extracted_path: str
    score: float
    chunks: list[dict[str, Any]]


class InsuranceHybridRetriever:
    def __init__(self, collection: Any, model: Any) -> None:
        from rank_bm25 import BM25Okapi

        self.collection, self.model = collection, model
        data = collection.get(include=["documents", "metadatas"])
        self.chunks = [{"id": cid, "text": text, "metadata": metadata}
                       for cid, text, metadata in zip(data["ids"], data["documents"], data["metadatas"])]
        self.bm25 = BM25Okapi([tokenize(searchable_text(chunk)) for chunk in self.chunks]) if self.chunks else None

    def search(self, query: str, top_k: int = 5, source_type: str | None = None,
               provider: str | None = None, candidates: int = 40) -> list[FileHit]:
        """Apply identical metadata filters to dense and sparse search, then RRF.

        Files rank by their best matching chunk, so long booklets cannot win
        merely by having more chunks. Scores express rank, not confidence.
        """
        if not query.strip() or top_k < 1 or candidates < 1:
            raise ValueError("Provide a nonempty query and positive top_k/candidates")
        if source_type is not None and source_type not in set(SOURCE_TYPES.values()):
            raise ValueError("source_type must be policy, fca, fos or legislation")
        filters = {key: value for key, value in (("source_type", source_type), ("provider", provider)) if value is not None}
        selected = [(i, chunk) for i, chunk in enumerate(self.chunks)
                    if all(chunk["metadata"].get(key) == value for key, value in filters.items())]
        if not selected:
            return []
        where = {"$and": [{key: value} for key, value in filters.items()]} if len(filters) > 1 else filters
        embedding = self.model.encode([query], normalize_embeddings=True).tolist()
        dense = self.collection.query(query_embeddings=embedding, n_results=min(candidates, len(selected)),
                                      include=["documents", "metadatas", "distances"], **({"where": where} if where else {}))
        dense_hits = [{"id": cid, "text": text, "metadata": meta,
                       "dense_distance": float(distance), "bm25_score": None}
                      for cid, text, meta, distance in zip(dense["ids"][0], dense["documents"][0],
                                                          dense["metadatas"][0], dense["distances"][0])]
        scores = self.bm25.get_scores(tokenize(query))
        sparse_hits = [{**chunk, "bm25_score": float(scores[i]), "dense_distance": None}
                       for i, chunk in sorted(selected, key=lambda pair: scores[pair[0]], reverse=True)[:candidates]
                       if scores[i] > 0]
        fused = fuse_rrf([dense_hits, sparse_hits], top_k=len(dense_hits) + len(sparse_hits))
        grouped: dict[str, FileHit] = {}
        for chunk in fused:
            meta = chunk["metadata"]
            doc_id = meta["doc_id"]
            if doc_id not in grouped:
                grouped[doc_id] = FileHit(doc_id, meta.get("title", ""), meta["source_type"],
                                          meta.get("provider", ""), meta.get("source_url", ""),
                                          meta["raw_path"], meta["extracted_path"], chunk["fused_score"], [])
            if len(grouped[doc_id].chunks) < 3:
                grouped[doc_id].chunks.append(chunk)
        return list(grouped.values())[:top_k]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "search"))
    parser.add_argument("--corpus", type=Path, default=ROOT / "real_insurance_corpus")
    parser.add_argument("--index", type=Path, default=ROOT / "insurance_chroma_index")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--source-type", choices=("policy", "fca", "fos", "legislation"))
    parser.add_argument("--provider")
    parser.add_argument("--overlap", type=int, default=40)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--chunk-words",
        type=int,
        default=220,
    )
    args = parser.parse_args()
    if args.command == "search" and not args.query:
        parser.error("search requires --query")
    from sentence_transformers import SentenceTransformer
    import chromadb
    model = SentenceTransformer(args.embedding_model)
    if args.command == "build":
        print(json.dumps(build_index(args.corpus, args.index, model, args.embedding_model,
                                     args.chunk_words, args.overlap), indent=2))
        return
    collection = chromadb.PersistentClient(path=str(args.index)).get_collection(COLLECTION_NAME)
    if (collection.metadata or {}).get("embedding_model") != args.embedding_model:
        parser.error("Query embedding model must match the indexed embedding model")
    hits = InsuranceHybridRetriever(collection, model).search(args.query, args.top_k, args.source_type, args.provider)
    if args.json:
        print(json.dumps([asdict(hit) for hit in hits], indent=2, ensure_ascii=False))
    else:
        for rank, hit in enumerate(hits, 1):
            print(f"\n{rank}. [{hit.source_type}] {hit.title} | RRF={hit.score:.4f}")
            print(f"File: {hit.raw_path}\nText: {hit.extracted_path}\nURL: {hit.source_url}")
            best = hit.chunks[0]
            print(f"Page: {best['metadata']['page']}\nExcerpt: {best['text'][:600]}")
        if not hits:
            print("No documents matched the metadata filters.")


if __name__ == "__main__":
    main()
