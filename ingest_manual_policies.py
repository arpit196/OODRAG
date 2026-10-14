#!/usr/bin/env python3
"""
Append manually downloaded home-insurance policy PDFs to an EXISTING corpus.

This utility does NOT scrape the web and does NOT delete/rebuild documents.jsonl.

Expected input:
manual_policies/
    example.pdf
    example.json

The sidecar may contain:
    provider
    product
    document_type
    source_type
    effective_from
    effective_to
    document_code
    version_label
    source_url
    metadata_notes

Install:
    pip install pypdf

Example:
    python ingest_manual_policies.py \
        --corpus ./real_insurance_corpus_v4 \
        --manual-policy-dir ./manual_policies
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
from pathlib import Path
from typing import Optional

from pypdf import PdfReader


def utc_stamp() -> str:
    import datetime as dt
    return dt.datetime.now(dt.timezone.utc).isoformat()


def safe_name(value: str, max_len: int = 120) -> str:
    value = re.sub(r"https?://", "", value)
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = value.strip("_")
    return (value[:max_len] or "document").lower()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def pdf_to_text(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append(f"\n--- PAGE {i} ---\n{text}")
    return "\n".join(pages)


def load_existing_hashes(meta_path: Path) -> set[str]:
    hashes: set[str] = set()
    if not meta_path.exists():
        return hashes

    with meta_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"Invalid JSON in existing manifest {meta_path} line {lineno}: {e}"
                ) from e
            digest = row.get("sha256")
            if digest:
                hashes.add(str(digest))
    return hashes


def load_sidecar(pdf_path: Path) -> dict:
    sidecar = pdf_path.with_suffix(".json")
    if not sidecar.exists():
        raise FileNotFoundError(
            f"Missing metadata sidecar for {pdf_path.name}: expected {sidecar.name}"
        )

    data = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{sidecar.name} must contain a single JSON object")

    provider = str(data.get("provider") or "").strip()
    product = str(data.get("product") or "").strip()

    if not provider:
        raise ValueError(f"{sidecar.name}: 'provider' is required")
    if not product:
        raise ValueError(f"{sidecar.name}: 'product' is required")

    return data


def make_record(
    *,
    pdf_path: Path,
    digest: str,
    sidecar: dict,
    raw_rel: Path,
    extracted_rel: Path,
) -> dict:
    provider = str(sidecar["provider"]).strip()
    product = str(sidecar["product"]).strip()

    source_type = str(sidecar.get("source_type") or "policy").strip().lower()
    # Existing corpus uses "policies"; preserve compatibility.
    corpus_source_type = "policies" if source_type in {"policy", "policies"} else source_type

    document_type = sidecar.get("document_type") or "policy_wording"

    notes = "Manually downloaded public policy PDF"
    extra_note = sidecar.get("metadata_notes")
    if extra_note:
        notes += f"; {extra_note}"

    return {
        "doc_id": f"{safe_name(provider, 30)}_{digest[:12]}",
        "source_type": corpus_source_type,
        "provider": provider,
        "title": product,
        "source_url": str(sidecar.get("source_url") or ""),
        "retrieved_at_utc": utc_stamp(),
        "content_type": "application/pdf",
        "raw_path": str(raw_rel),
        "extracted_path": str(extracted_rel),
        "sha256": digest,
        "notes": notes,
        "product": product,
        "document_type": str(document_type) if document_type is not None else None,
        "effective_from": (
            str(sidecar["effective_from"])
            if sidecar.get("effective_from") is not None
            else None
        ),
        "effective_to": (
            str(sidecar["effective_to"])
            if sidecar.get("effective_to") is not None
            else None
        ),
        "document_code": (
            str(sidecar["document_code"])
            if sidecar.get("document_code") is not None
            else None
        ),
        "version_label": (
            str(sidecar["version_label"])
            if sidecar.get("version_label") is not None
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Existing corpus root containing metadata/documents.jsonl",
    )
    parser.add_argument(
        "--manual-policy-dir",
        type=Path,
        required=True,
        help="Directory containing PDF + same-name JSON sidecar pairs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report what would be added without writing anything",
    )
    args = parser.parse_args()

    corpus = args.corpus
    manual_dir = args.manual_policy_dir

    if not corpus.exists():
        raise SystemExit(f"Corpus directory does not exist: {corpus}")
    if not manual_dir.exists() or not manual_dir.is_dir():
        raise SystemExit(f"Manual policy directory does not exist: {manual_dir}")

    meta_path = corpus / "metadata" / "documents.jsonl"
    failures_path = corpus / "metadata" / "manual_policy_failures.jsonl"

    existing_hashes = load_existing_hashes(meta_path)
    seen_this_run: set[str] = set()

    pdfs = sorted(manual_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {manual_dir}")

    added = 0
    skipped = 0
    failed = 0

    for pdf_path in pdfs:
        try:
            data = pdf_path.read_bytes()

            if not data.lstrip().startswith(b"%PDF-"):
                raise ValueError("file does not begin with PDF magic bytes")

            digest = sha256_bytes(data)

            if digest in existing_hashes:
                print(f"[skip-existing] {pdf_path.name}")
                skipped += 1
                continue

            if digest in seen_this_run:
                print(f"[skip-duplicate] {pdf_path.name}")
                skipped += 1
                continue

            sidecar = load_sidecar(pdf_path)
            provider = str(sidecar["provider"]).strip()
            product = str(sidecar["product"]).strip()

            base = safe_name(pdf_path.stem)
            raw_rel = Path("raw") / "policies" / f"{base}_{digest[:8]}.pdf"
            extracted_rel = Path("extracted") / "policies" / f"{base}_{digest[:8]}.txt"

            extracted_text = pdf_to_text(data)
            if not extracted_text.strip():
                raise ValueError("PDF extraction produced no text")

            record = make_record(
                pdf_path=pdf_path,
                digest=digest,
                sidecar=sidecar,
                raw_rel=raw_rel,
                extracted_rel=extracted_rel,
            )

            if args.dry_run:
                print(
                    f"[would-add] {provider} | {product} | "
                    f"{pdf_path.name} | {digest[:12]}"
                )
            else:
                raw_path = corpus / raw_rel
                extracted_path = corpus / extracted_rel
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                extracted_path.parent.mkdir(parents=True, exist_ok=True)

                # Preserve the exact downloaded PDF bytes.
                shutil.copyfile(pdf_path, raw_path)
                extracted_path.write_text(
                    extracted_text,
                    encoding="utf-8",
                    errors="ignore",
                )
                append_jsonl(meta_path, record)

                print(f"[added] {provider} | {product}")

            seen_this_run.add(digest)
            added += 1

        except Exception as e:
            failed += 1
            entry = {
                "path": str(pdf_path),
                "error": repr(e),
                "stage": "manual_policy_append",
                "timestamp_utc": utc_stamp(),
            }
            if args.dry_run:
                print(f"[error] {pdf_path.name}: {e}")
            else:
                append_jsonl(failures_path, entry)
                print(f"[error] {pdf_path.name}: {e}")

    print("\nDone.")
    print(f"  added:   {added}")
    print(f"  skipped: {skipped}")
    print(f"  failed:  {failed}")
    print(f"  manifest: {meta_path.resolve()}")

    if failed and not args.dry_run:
        print(f"  failures: {failures_path.resolve()}")


if __name__ == "__main__":
    main()
