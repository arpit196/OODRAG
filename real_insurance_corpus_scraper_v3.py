#!/usr/bin/env python3
"""
Build a REAL-SOURCE UK home-insurance corpus for an OOD-aware RAG system.

Core corpus sources:
  1) Public insurer policy wording / policy booklet PDFs
  2) FCA Handbook pages / PDFs
  3) UK legislation pages
  4) Financial Ombudsman Service (FOS) home-insurance guidance
  5) Published, anonymised FOS decisions / case studies

Synthetic content is OPTIONAL and limited to benchmark queries / OOD prompts.
The script does NOT fabricate policy wording, FCA rules, claim forms, or legal text.

IMPORTANT:
- Respect each site's robots.txt, terms, copyright, and rate limits.
- Use downloaded material for research/prototyping unless you have rights for wider redistribution.
- FOS decisions are anonymised and public, but are case-specific and are not definitive law.
- Policy wordings change over time. Preserve retrieval timestamps and source URLs.
- For real legal/compliance decisions, always re-check the current authoritative source.

Install:
    pip install requests beautifulsoup4 pypdf lxml

Example:
    python real_insurance_corpus_scraper.py \
        --output ./real_insurance_corpus \
        --fos-pages 40 \
        --max-fos-decisions 100 \
        --delay 1.2

Directory layout:
real_insurance_corpus/
    raw/
        policies/
        fca/
        legislation/
        fos_guidance/
        fos_decisions/
    extracted/
        policies/
        fca/
        legislation/
        fos_guidance/
        fos_decisions/
    metadata/
        documents.jsonl
        failures.jsonl
    benchmark/
        ood_queries.jsonl     # optional synthetic benchmark only
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader


USER_AGENT = (
    "InsuranceRAGResearchBot/1.0 "
    "(research prototype; respectful crawler; contact: replace-with-your-email)"
)

HOME_INSURANCE_TERMS = (
    "home insurance",
    "buildings insurance",
    "contents insurance",
    "escape of water",
    "storm damage",
    "flood",
    "subsidence",
    "accidental damage",
    "underinsurance",
    "theft",
    "fire damage",
    "wear and tear",
    "gradual damage",
)

POLICY_LINK_TERMS = (
    "policy wording",
    "sample policy wording",
    "policy booklet",
    "policy book",
    "policy document",
    "home insurance policy",
    "home insurance documents",
    "document of insurance",
    "insurance policy wording",
)

SKIP_POLICY_TERMS = (
    "insurance product information document",
    "ipid",
    "summary of cover",
    "privacy",
    "terms of use",
)

# Curated CURRENT/PUBLIC source pages. The code discovers actual document links at runtime.
INSURER_POLICY_PAGES = [
    {
        "provider": "AXA UK",
        "url": "https://www.axa.co.uk/home-insurance/policy-details/",
    },
    {
        "provider": "Admiral",
        "url": "https://www.admiral.com/existing-customers/policy-documents.php",
    },
    {
        "provider": "Aviva",
        "url": "https://www.aviva.co.uk/insurance/home-products/home-insurance/",
    },
    {
        "provider": "Aviva Private Clients",
        "url": "https://www.aviva.co.uk/aviva-private-clients/",
    },
]

# Known public policy documents that are useful fallbacks when discovery pages change.
DIRECT_POLICY_URLS = [
    {
        "provider": "AXA UK",
        "title": "AXA Home Insurance sample policy wording",
        "url": "https://www.axa.co.uk/globalassets/pdfs/home/policy-docs/june-2025/axa-direct-home-policy-wording-acpd0400p-d.pdf/",
    },
    {
        "provider": "AXA UK",
        "title": "AXA Plus Home Insurance sample policy wording",
        "url": "https://www.axa.co.uk/globalassets/pdfs/home/policy-docs/june-2025/axa-direct-home-plus-wording-acpd0401p-d.pdf/",
    },
    {
        "provider": "AXA UK",
        "title": "AXA Premier Home Insurance sample policy wording",
        "url": "https://www.axa.co.uk/globalassets/pdfs/home/policy-docs/june-2025/axa-direct-home-premier-policy-wording-acpd0402p-d.pdf/",
    },
    {
        "provider": "Direct Line",
        "title": "Direct Line Home Insurance policy document",
        "url": "https://www.directline.com/assets/pdf/dl-home-insurance-policy-document.pdf",
    },
    {
        "provider": "LV=",
        "title": "LV= Home Insurance policy document",
        "url": (
            "https://www.lv.com/promotion/-/insurance/media/gi/home-insurance/"
            "pdfs/policy-documents/0031893-2020-home-doi.pdf?la=en&u=20210224143552"
        ),
    },
]

FCA_SOURCES = [
    {
        "authority": "Financial Conduct Authority",
        "title": "ICOBS 8 - Claims handling",
        "url": "https://handbook.fca.org.uk/handbook/icobs8",
    },
    {
        "authority": "Financial Conduct Authority",
        "title": "ICOBS 8 PDF - Claims handling",
        "url": "https://www.handbook.fca.org.uk/handbook/ICOBS/8.pdf",
    },
    {
        "authority": "Financial Conduct Authority",
        "title": "PRIN 2A - Consumer Duty",
        "url": "https://handbook.fca.org.uk/handbook/PRIN/2A/",
    },
]

LEGISLATION_SOURCES = [
    {
        "authority": "UK Legislation",
        "title": "Insurance Act 2015",
        "url": "https://www.legislation.gov.uk/ukpga/2015/4/contents",
    },
    {
        "authority": "UK Legislation",
        "title": "Consumer Insurance (Disclosure and Representations) Act 2012",
        "url": "https://www.legislation.gov.uk/ukpga/2012/6/contents",
    },
]

FOS_GUIDANCE_SEEDS = [
    "https://www.financial-ombudsman.org.uk/consumers/complaints-can-help/insurance/home-insurance",
    "https://www.financial-ombudsman.org.uk/businesses/resolving-complaint/our-insight/escape-water-claims-unoccupied-properties",
]

FOS_DECISION_SEARCH = (
    "https://www.financial-ombudsman.org.uk/businesses/"
    "resolving-complaint/ombudsman-decisions/search"
)

# Useful real FOS case-study pages. The crawler also discovers additional related pages.
FOS_CASE_STUDIES = [
    "https://www.financial-ombudsman.org.uk/businesses/resolving-complaint/case-studies/insurer-voided-home-insurance-policy-refused-claim-fire-damage",
    "https://www.financial-ombudsman.org.uk/businesses/resolving-complaint/case-studies/consumer-complains-impact-underinsurance",
    "https://www.financial-ombudsman.org.uk/businesses/resolving-complaint/case-studies/slow-leak-damage-not-covered-risks-policy",
    "https://www.financial-ombudsman.org.uk/businesses/resolving-complaint/case-studies/ceiling-collapse-caused-insured-event-insurer-still-declines-claim",
]


@dataclass
class DocMeta:
    doc_id: str
    source_type: str
    provider: str
    title: str
    source_url: str
    retrieved_at_utc: str
    content_type: str
    raw_path: str
    extracted_path: str
    sha256: str
    notes: str = ""


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


class RespectfulSession:
    def __init__(self, delay: float = 1.0, timeout: int = 30):
        self.delay = max(delay, 0.0)
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/pdf,*/*;q=0.8",
        })
        self._robots: dict[str, RobotFileParser] = {}
        self._last_request = 0.0

    def _sleep(self):
        elapsed = time.time() - self._last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)

    def allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self._robots:
            rp = RobotFileParser()
            rp.set_url(urljoin(base, "/robots.txt"))
            try:
                self._sleep()
                r = self.session.get(rp.url, timeout=self.timeout)
                self._last_request = time.time()
                if r.ok:
                    rp.parse(r.text.splitlines())
                else:
                    # If robots cannot be fetched, be conservative but allow explicit public docs.
                    rp.parse([])
            except requests.RequestException:
                rp.parse([])
            self._robots[base] = rp
        try:
            return self._robots[base].can_fetch(USER_AGENT, url)
        except Exception:
            return True

    def get(self, url: str, *, allow_redirects: bool = True) -> requests.Response:
        if not self.allowed(url):
            raise PermissionError(f"robots.txt disallows crawling: {url}")
        self._sleep()
        r = self.session.get(
            url,
            timeout=self.timeout,
            allow_redirects=allow_redirects,
        )
        self._last_request = time.time()
        r.raise_for_status()
        return r


def html_to_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer"]):
        tag.decompose()

    title = ""
    if soup.title:
        title = soup.title.get_text(" ", strip=True)

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(attrs={"role": "main"})
        or soup.body
        or soup
    )
    text = main.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return title, text


def pdf_to_text(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text(
                extraction_mode="layout"
            ) or ""
        except Exception:
            text = page.extract_text() or ""

        pages.append(f"\n--- PAGE {i} ---\n{text}")
    return "\n".join(pages)


def looks_like_pdf(url: str, content_type: str = "") -> bool:
    """URL/header hint only. Do not trust this alone when parsing."""
    clean = urlparse(url).path.lower().rstrip("/")
    return clean.endswith(".pdf") or "application/pdf" in content_type.lower()


def response_is_real_pdf(response: requests.Response) -> bool:
    """Validate the actual response body before passing it to pypdf."""
    body = response.content.lstrip()
    return body.startswith(b"%PDF-")


def response_looks_html(response: requests.Response) -> bool:
    content_type = response.headers.get("Content-Type", "").lower()
    body = response.content.lstrip()[:100].lower()
    return (
        "text/html" in content_type
        or body.startswith(b"<!doctype html")
        or body.startswith(b"<html")
    )


def save_document(
    root: Path,
    response: requests.Response,
    *,
    source_type: str,
    provider: str,
    title_hint: str = "",
    notes: str = "",
) -> Optional[DocMeta]:
    data = response.content
    final_url = response.url
    content_type = response.headers.get("Content-Type", "")

    base = safe_name(title_hint or Path(urlparse(final_url).path).stem or final_url)
    digest = sha256_bytes(data)
    doc_id = f"{safe_name(provider, 30)}_{digest[:12]}"

    if response_is_real_pdf(response):
        raw_rel = Path("raw") / source_type / f"{base}_{digest[:8]}.pdf"
        extracted_rel = Path("extracted") / source_type / f"{base}_{digest[:8]}.txt"
        text = pdf_to_text(data)
        actual_type = "application/pdf"
        title = title_hint or base
    else:
        raw_rel = Path("raw") / source_type / f"{base}_{digest[:8]}.html"
        extracted_rel = Path("extracted") / source_type / f"{base}_{digest[:8]}.txt"
        html = response.text
        page_title, text = html_to_text(html)
        actual_type = content_type or "text/html"
        title = title_hint or page_title or base

    raw_path = root / raw_rel
    extracted_path = root / extracted_rel
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    extracted_path.parent.mkdir(parents=True, exist_ok=True)

    raw_path.write_bytes(data)
    extracted_path.write_text(text, encoding="utf-8", errors="ignore")

    return DocMeta(
        doc_id=doc_id,
        source_type=source_type,
        provider=provider,
        title=title,
        source_url=final_url,
        retrieved_at_utc=utc_stamp(),
        content_type=actual_type,
        raw_path=str(raw_rel),
        extracted_path=str(extracted_rel),
        sha256=digest,
        notes=notes,
    )


def anchor_links(html: str, base_url: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html, "lxml")
    out = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = urldefrag(urljoin(base_url, a["href"]))[0]
        label = " ".join(a.stripped_strings)
        if not href.startswith(("http://", "https://")):
            continue
        if href in seen:
            continue
        seen.add(href)
        out.append((label, href))
    return out


def policy_candidate(label: str, url: str) -> bool:
    """
    Return True when a discovered link looks like a full home-policy wording.

    IPIDs and summary-of-cover documents are deliberately excluded because they are
    not substitutes for the full contractual wording.
    """
    haystack = f"{label} {url}".lower()

    if any(x in haystack for x in SKIP_POLICY_TERMS):
        return False

    if not looks_like_pdf(url):
        return False

    if any(x in haystack for x in POLICY_LINK_TERMS):
        return True

    # Useful fallback for filenames such as home_wording_default_v8.pdf.
    if "home" in haystack and any(x in haystack for x in ("policy", "wording", "booklet")):
        return True

    return False


def scrape_policy_sources(root: Path, client: RespectfulSession, failures: Path):
    """
    Discover real policy PDFs from official insurer pages and curated fallbacks.

    A document is counted as a policy only if the downloaded response actually begins
    with the PDF magic bytes '%PDF-'. This prevents HTML error/redirect pages from
    being passed to pypdf or counted as policy documents.
    """
    seen_urls: set[str] = set()
    emitted_hashes: set[str] = set()

    def maybe_emit(meta: Optional[DocMeta]) -> Optional[DocMeta]:
        if meta is None or meta.sha256 in emitted_hashes:
            return None
        emitted_hashes.add(meta.sha256)
        return meta

    def download_policy(url: str, provider: str, title: str, notes: str):
        canonical = urldefrag(url)[0]
        if canonical in seen_urls:
            return None
        seen_urls.add(canonical)

        try:
            r = client.get(url)
        except Exception as e:
            append_jsonl(
                failures,
                {"url": url, "error": repr(e), "stage": "policy_download"},
            )
            return None

        if not response_is_real_pdf(r):
            append_jsonl(
                failures,
                {
                    "url": url,
                    "final_url": r.url,
                    "content_type": r.headers.get("Content-Type", ""),
                    "body_prefix": r.content[:40].decode("utf-8", errors="replace"),
                    "error": "Expected policy PDF but response body is not a PDF",
                    "stage": "policy_not_pdf",
                },
            )
            return None

        meta = save_document(
            root,
            r,
            source_type="policies",
            provider=provider,
            title_hint=title or "Home insurance policy wording",
            notes=notes,
        )
        return maybe_emit(meta)

    # Discover policy links from official insurer landing pages.
    for source in INSURER_POLICY_PAGES:
        try:
            page = client.get(source["url"])
        except Exception as e:
            append_jsonl(
                failures,
                {"url": source["url"], "error": repr(e), "stage": "policy_landing"},
            )
            continue

        for label, href in anchor_links(page.text, page.url):
            if not policy_candidate(label, href):
                continue

            meta = download_policy(
                href,
                source["provider"],
                label or "Home insurance policy wording",
                f"Discovered from {source['url']}",
            )
            if meta:
                yield meta

    # Curated official public fallbacks.
    for source in DIRECT_POLICY_URLS:
        meta = download_policy(
            source["url"],
            source["provider"],
            source.get("title") or f"{source['provider']} home insurance policy wording",
            "Curated official public policy-document URL",
        )
        if meta:
            yield meta


def scrape_static_sources(
    root: Path,
    client: RespectfulSession,
    sources: Iterable[dict],
    source_type: str,
    failures: Path,
):
    for src in sources:
        try:
            r = client.get(src["url"])

            # If a source was explicitly intended to be a PDF, do not feed an HTML
            # error/redirect page into pypdf or duplicate an HTML handbook page.
            if looks_like_pdf(src["url"]) and not response_is_real_pdf(r):
                append_jsonl(
                    failures,
                    {
                        "url": src["url"],
                        "final_url": r.url,
                        "content_type": r.headers.get("Content-Type", ""),
                        "body_prefix": r.content[:40].decode("utf-8", errors="replace"),
                        "error": "Expected PDF but response body is not a PDF",
                        "stage": f"{source_type}_not_pdf",
                    },
                )
                continue

            meta = save_document(
                root,
                r,
                source_type=source_type,
                provider=src.get("authority", src.get("provider", "unknown")),
                title_hint=src["title"],
                notes="Authoritative/public source",
            )
            if meta:
                yield meta
        except Exception as e:
            append_jsonl(
                failures,
                {"url": src["url"], "error": repr(e), "stage": source_type},
            )


def relevant_fos_guidance(label: str, url: str) -> bool:
    p = urlparse(url)
    if "financial-ombudsman.org.uk" not in p.netloc:
        return False

    low = f"{label} {url}".lower()
    allowed_paths = (
        "/insurance/home-insurance",
        "/resolving-complaint/case-studies/",
        "/resolving-complaint/our-insight/",
    )
    if not any(path in p.path for path in allowed_paths):
        return False

    # Keep pages that look home-insurance/claims relevant.
    return any(term in low for term in (
        "home", "insurance", "claim", "water", "storm", "flood",
        "subsidence", "damage", "underinsurance", "policy", "fire"
    ))


def crawl_fos_guidance(
    root: Path,
    client: RespectfulSession,
    failures: Path,
    max_pages: int = 40,
):
    queue = list(dict.fromkeys(FOS_GUIDANCE_SEEDS + FOS_CASE_STUDIES))
    seen = set()
    count = 0

    while queue and count < max_pages:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        try:
            r = client.get(url)
        except Exception as e:
            append_jsonl(failures, {"url": url, "error": repr(e), "stage": "fos_guidance"})
            continue

        title, text = html_to_text(r.text)
        # Avoid indexing irrelevant generic pages.
        if not any(term in text.lower() for term in HOME_INSURANCE_TERMS):
            continue

        meta = save_document(
            root, r,
            source_type="fos_guidance",
            provider="Financial Ombudsman Service",
            title_hint=title,
            notes="Public FOS guidance/case-study page; case-specific where applicable",
        )
        if meta:
            yield meta
            count += 1

        for label, href in anchor_links(r.text, r.url):
            if href not in seen and relevant_fos_guidance(label, href):
                queue.append(href)


DECISION_REF_RE = re.compile(r"\bDRN-\d+\b", re.I)


def is_home_insurance_decision(text: str) -> bool:
    low = text.lower()
    # Require insurance + at least one strong home/property/claim indicator.
    return (
        "insurance" in low
        and any(x in low for x in (
            "home insurance",
            "buildings insurance",
            "contents insurance",
            "property insurance",
            "escape of water",
            "storm damage",
            "subsidence",
            "flood damage",
            "accidental damage",
            "underinsured",
            "gradual damage",
        ))
    )


def decision_links_from_search(html: str, base_url: str) -> list[tuple[str, str]]:
    links = []
    for label, href in anchor_links(html, base_url):
        if DECISION_REF_RE.search(label) or "/ombudsman-decisions/decision/" in href:
            links.append((label, href))
    return links


def fetch_fos_decision(
    client: RespectfulSession,
    href: str,
) -> requests.Response:
    """
    FOS search results have historically linked to published decision documents.
    Try the discovered link first, then sensible HTML/PDF fallbacks.
    """
    attempts = [href]

    if href.lower().endswith(".pdf"):
        attempts.append(href[:-4])
    else:
        attempts.append(href.rstrip("/") + ".pdf")

    last_exc = None
    for candidate in dict.fromkeys(attempts):
        try:
            return client.get(candidate)
        except Exception as e:
            last_exc = e
    raise last_exc or RuntimeError(f"Unable to fetch FOS decision {href}")


def scrape_fos_decisions(
    root: Path,
    client: RespectfulSession,
    failures: Path,
    pages_to_scan: int,
    max_decisions: int,
):
    """
    Scans public FOS decision search result pages and keeps only decisions whose
    result snippet/document appears related to home/property insurance.

    The site currently paginates with ?Start=<offset> in batches of 10.
    This avoids inventing undocumented filter parameters and filters locally.
    """
    kept = 0
    seen_refs = set()

    for page_idx in range(pages_to_scan):
        if kept >= max_decisions:
            break

        start = page_idx * 10
        search_url = FOS_DECISION_SEARCH if start == 0 else f"{FOS_DECISION_SEARCH}?Start={start}"

        try:
            search = client.get(search_url)
        except Exception as e:
            append_jsonl(failures, {"url": search_url, "error": repr(e), "stage": "fos_search"})
            continue

        for label, href in decision_links_from_search(search.text, search.url):
            if kept >= max_decisions:
                break

            ref_match = DECISION_REF_RE.search(label + " " + href)
            ref = ref_match.group(0).upper() if ref_match else href
            if ref in seen_refs:
                continue
            seen_refs.add(ref)

            # Search-result snippets often contain enough information to avoid
            # downloading clearly irrelevant banking/investment decisions.
            if not is_home_insurance_decision(label):
                continue

            try:
                r = fetch_fos_decision(client, href)
                if looks_like_pdf(r.url, r.headers.get("Content-Type", "")):
                    extracted = pdf_to_text(r.content)
                    title = ref
                else:
                    title, extracted = html_to_text(r.text)

                if not is_home_insurance_decision(extracted + " " + label):
                    continue

                meta = save_document(
                    root, r,
                    source_type="fos_decisions",
                    provider="Financial Ombudsman Service",
                    title_hint=f"{ref} home-insurance decision",
                    notes=(
                        "Published anonymised FOS final decision. "
                        "Case-specific; not definitive law."
                    ),
                )
                if meta:
                    yield meta
                    kept += 1
            except Exception as e:
                append_jsonl(failures, {"url": href, "error": repr(e), "stage": "fos_decision"})


def generate_synthetic_benchmark(root: Path):
    """
    ONLY benchmark/OOD queries are synthetic. They are never placed in the
    retrievable real-source corpus.
    """
    out = root / "benchmark" / "ood_queries.jsonl"
    if out.exists():
        out.unlink()

    examples = [
        # In-domain / answerable against typical home-policy corpus
        {
            "query": "What policy terms commonly apply to escape-of-water claims?",
            "expected_class": "IN_DOMAIN",
        },
        {
            "query": "What issues can arise when a home is unoccupied during an escape-of-water claim?",
            "expected_class": "IN_DOMAIN",
        },

        # Near OOD: insurance language but different product
        {
            "query": "Will travel insurance cover a cancelled international flight?",
            "expected_class": "NEAR_OOD",
        },
        {
            "query": "Does pet insurance cover surgery for a dog?",
            "expected_class": "NEAR_OOD",
        },
        {
            "query": "Will motor insurance pay for damage to another driver's car?",
            "expected_class": "NEAR_OOD",
        },

        # Far OOD
        {
            "query": "What dose of ibuprofen should an adult take?",
            "expected_class": "FAR_OOD",
        },
        {
            "query": "Write a Python transformer training loop.",
            "expected_class": "FAR_OOD",
        },
        {
            "query": "Recommend a hotel in Barcelona.",
            "expected_class": "FAR_OOD",
        },
    ]
    for i, item in enumerate(examples, start=1):
        append_jsonl(out, {"id": f"Q-{i:03d}", **item})


def write_readme(root: Path):
    text = """# Real-source UK Home Insurance RAG Corpus

This corpus is built primarily from real public sources:

- insurer home-insurance policy wording / policy booklets;
- FCA Handbook material;
- UK legislation;
- Financial Ombudsman Service (FOS) home-insurance guidance and case studies;
- published anonymised FOS decisions.

The script does **not** fabricate policy wordings, FCA rules, legislation, or Ombudsman decisions.

Only `benchmark/` contains synthetic evaluation/OOD questions.

## Important caveats

1. Policy documents change. Preserve retrieval date and source URL.
2. FCA/legislation sources are authoritative only in their current official form.
3. FOS decisions are case-specific and not definitive statements of law.
4. Public documents may be copyrighted. Use responsibly and check terms before redistribution.
5. Do not scrape or ingest private customer claims.
6. For production/regulated use, add legal/compliance review and source-version controls.

## Recommended RAG metadata

Keep at least:
- provider / authority
- source URL
- retrieval date
- source type
- document hash
- policy/product name if discoverable
- effective/version dates if extracted
- decision reference for FOS cases

## Suggested architecture

query
  -> domain classifier
  -> policy/regulatory retriever
  -> FOS guidance/decision retriever
  -> reranker
  -> evidence sufficiency / OOD gate
  -> answer | abstain | escalate

Do not ingest `benchmark/` into your retriever.

## MVP source-balance target

Before evaluating retrieval seriously, aim for at least:
- 5-8 distinct full home-insurance policy wordings;
- FCA / legislation material;
- 30+ FOS guidance/decision documents.

Multiple policy wordings matter because different insurers/products phrase similar
cover, exclusions and conditions differently.
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("./real_insurance_corpus"))
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between requests per session")
    parser.add_argument("--fos-pages", type=int, default=30, help="FOS decision search pages to scan")
    parser.add_argument("--max-fos-decisions", type=int, default=75)
    parser.add_argument("--fos-guidance-pages", type=int, default=30)
    parser.add_argument(
        "--min-policy-docs",
        type=int,
        default=5,
        help="Warn if fewer than this many distinct real policy wordings are downloaded",
    )
    parser.add_argument("--skip-policies", action="store_true")
    parser.add_argument("--skip-fca", action="store_true")
    parser.add_argument("--skip-legislation", action="store_true")
    parser.add_argument("--skip-fos-guidance", action="store_true")
    parser.add_argument("--skip-fos-decisions", action="store_true")
    parser.add_argument("--no-synthetic-benchmark", action="store_true")
    args = parser.parse_args()

    root = args.output
    root.mkdir(parents=True, exist_ok=True)

    meta_path = root / "metadata" / "documents.jsonl"
    failures = root / "metadata" / "failures.jsonl"
    for p in (meta_path, failures):
        if p.exists():
            p.unlink()

    client = RespectfulSession(delay=args.delay)

    def record(stream):
        n = 0
        for meta in stream:
            append_jsonl(meta_path, asdict(meta))
            print(f"[saved] {meta.source_type:15s} {meta.provider:28s} {meta.title[:70]}")
            n += 1
        return n

    counts = {}

    if not args.skip_policies:
        counts["policies"] = record(scrape_policy_sources(root, client, failures))
        if counts["policies"] < args.min_policy_docs:
            print(
                f"\nWARNING: only {counts['policies']} distinct policy wordings were downloaded; "
                f"target is at least {args.min_policy_docs}. "
                "Inspect metadata/failures.jsonl before building the retrieval index."
            )

    if not args.skip_fca:
        counts["fca"] = record(
            scrape_static_sources(root, client, FCA_SOURCES, "fca", failures)
        )

    if not args.skip_legislation:
        counts["legislation"] = record(
            scrape_static_sources(root, client, LEGISLATION_SOURCES, "legislation", failures)
        )

    if not args.skip_fos_guidance:
        counts["fos_guidance"] = record(
            crawl_fos_guidance(
                root, client, failures,
                max_pages=args.fos_guidance_pages,
            )
        )

    if not args.skip_fos_decisions:
        counts["fos_decisions"] = record(
            scrape_fos_decisions(
                root, client, failures,
                pages_to_scan=args.fos_pages,
                max_decisions=args.max_fos_decisions,
            )
        )

    if not args.no_synthetic_benchmark:
        generate_synthetic_benchmark(root)

    write_readme(root)

    print("\nDone.")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print(f"\nCorpus:   {root.resolve()}")
    print(f"Metadata: {meta_path.resolve()}")
    print(f"Failures: {failures.resolve()}")
    print("\nNOTE: FOS decision search can change its HTML/link format.")
    print("If decisions stop downloading, inspect metadata/failures.jsonl and update the parser.")


if __name__ == "__main__":
    main()
