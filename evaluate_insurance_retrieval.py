#!/usr/bin/env python3
"""Run insurance_retrieval_fixed.py search repeatedly and score source-document retrieval.

Only evaluates whether returned *source documents* match a human-labelled gold set.
It does not evaluate answer faithfulness, passage relevance, or abstention.
No external Python dependencies.

Usage:
 python evaluate_insurance_retrieval.py --queries insurance_retrieval_benchmark.csv \\
   --index ./insurance_chroma_index_v3 --retrieval-script insurance_retrieval_fixed.py \\
   --top-k 10 --ks 1,3,5,10

If your search CLI prints a different result format, inspect outputs/raw/*.txt and
adapt parse_ranked_blocks() / classify_document() below. Never treat a parse failure
as a negative retrieval result.
"""
import argparse
import csv
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# Normalised document IDs must match CSV gold_docs, not broad provider names.
DOCUMENTS = {
    'lv_home_insurance_policy_document': ['lv_home_insurance_policy_document'],
    'aviva_home_insurance': ['aviva_home_insurance'],
    'axa_direct_home_2025': ['axa_direct_home_2025'],
    'axa_home_plus_2025': ['axa_home_plus_2025'],
    'axa_direct_home_premier_2023': ['axa_direct_home_premier_2023'],
    'axa_direct_home_premier_2025': ['axa_direct_home_premier_2025'],
}

# Only accept the actual search CLI headers. Numbered policy clauses like
# "3. Escape of water" or "4. Storm or flood" are NOT result headers.
# Example: 1. [policy] LV= Home Insurance policy document | RRF=0.0285
RANK_HEADER = re.compile(
    r'^\s*(\d+)\.\s+\[([a-z][a-z0-9_-]*)\]\s+.+?\s*\|\s*RRF\s*=\s*[-+]?\d',
    re.I,
)

METADATA = re.compile(r'\b(?:source|source_file|file|text|filename|file_name|file_path|path|doc_id|document|title|provider|product|version_label|document_code)\s*[:=]\s*(.*)', re.I)


def parse_ranked_blocks(output):
    """Return [(rank, searchable metadata/header string)] from human-readable CLI output."""
    blocks, rank, lines = [], None, []
    for line in output.splitlines():
        m = RANK_HEADER.match(line)
        if m:
            new_rank = int(m.group(1))
            if rank is not None:
                blocks.append((rank, '\n'.join(lines)))
            rank, lines = new_rank, [line]
        elif rank is not None:
            # Limit to identifying metadata; do NOT infer source from passage text.
            if METADATA.search(line):
                lines.append(line)
    if rank is not None:
        blocks.append((rank, '\n'.join(lines)))
    # Reject overlapping or non-sequential ranking, which commonly means numbered
    # source text got parsed as a result header.
    ranks = [r for r, _ in blocks]
    if not ranks or ranks != list(range(1, len(ranks)+1)):
        raise ValueError(f'Unrecognised or ambiguous rank format: ranks={ranks[:20]}')
    return blocks


def classify_document(block):
    """Conservative identity mapping. Unknown identity is never assigned at random."""
    s = block.casefold().replace('\\\\', '/').replace('\\', '/')
    # Exact filename stems, e.g. axa_direct_home_premier_2025_b2dc1c42.txt
    for doc, aliases in DOCUMENTS.items():
        if any(re.search(r'(?<![a-z0-9_])' + re.escape(a) + r'(?:_[a-f0-9]{8,})?(?=\.[a-z0-9]+|[^a-z0-9_]|$)', s) for a in aliases):
            return doc
    # Only use descriptive product labels if they unambiguously select a source.
    if re.search(r'\blv\s*=?\s*home insurance\b', s):
        return 'lv_home_insurance_policy_document'
    if 'aviva signature home' in s:
        return 'aviva_home_insurance'
    # AXA siblings must not be inferred from provider AXA alone.
    is_axa = 'axa' in s
    if is_axa:
        product = ('premier' if 'premier' in s else 'plus' if 'plus' in s else
                   'home' if re.search(r'axa\s+home(?:\s+insurance)?', s) else None)
        if product == 'premier':
            has23 = bool(re.search(r'\b2023\b', s)); has25 = bool(re.search(r'\b2025\b', s))
            if has23 != has25:
                return 'axa_direct_home_premier_2023' if has23 else 'axa_direct_home_premier_2025'
        elif product == 'plus':
            return 'axa_home_plus_2025'
        elif product == 'home':
            return 'axa_direct_home_2025'
    return None


def score_one(gold, retrieved, ks):
    # Multiple gold docs supported. Each relevant doc is credited at most once.
    seen = set(); positions = {}
    for i, doc in enumerate(retrieved, start=1):
        if doc in gold and doc not in seen:
            positions[doc] = i; seen.add(doc)
    reciprocal_rank = 1 / min(positions.values()) if positions else 0.0
    row = {'rr': reciprocal_rank}
    for k in ks:
        relevant = sum(pos <= k for pos in positions.values())
        row[f'hit@{k}'] = int(relevant > 0)
        row[f'recall@{k}'] = relevant / len(gold)
        row[f'precision@{k}'] = relevant / k
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queries', type=Path, default=Path('insurance_retrieval_benchmark.csv'))
    p.add_argument('--index', default='./insurance_chroma_index_v3')
    p.add_argument('--retrieval-script', type=Path, default=Path('insurance_retrieval_fixed.py'))
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--top-k', type=int, default=10)
    p.add_argument('--ks', default='1,3,5,10')
    p.add_argument('--out', type=Path, default=Path('benchmark_results'))
    p.add_argument('--timeout', type=int, default=180)
    p.add_argument('--limit', type=int, default=0, help='Run first N supported queries for debugging')
    args = p.parse_args()
    ks = sorted(set(int(x) for x in args.ks.split(',')))
    if not ks or min(ks) < 1 or max(ks) > args.top_k:
        p.error('All ks must be >=1 and <= --top-k')
    with args.queries.open(encoding='utf-8-sig', newline='') as f:
        queries = list(csv.DictReader(f))
    supported = [q for q in queries if q['category'] == 'supported']
    if args.limit: supported = supported[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    raw_dir = args.out / 'raw'; raw_dir.mkdir(exist_ok=True)
    results = []; failed = []; unknown_docs = 0
    for number, q in enumerate(supported, 1):
        cmd = [args.python, str(args.retrieval_script), 'search', '--index', args.index,
               '--query', q['query'], '--top-k', str(args.top_k)]
        print(f'[{number}/{len(supported)}] {q["id"]}: {q["query"]}', flush=True)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
            raw_file = raw_dir / f'{q["id"]}.txt'
            raw_file.write_text(proc.stdout + ('\nSTDERR:\n' + proc.stderr if proc.stderr else ''), encoding='utf-8')
            if proc.returncode:
                raise RuntimeError(f'CLI return code {proc.returncode}; inspect {raw_file}')
            blocks = parse_ranked_blocks(proc.stdout)
            if len(blocks) < args.top_k:
                # Valid if source filters or corpus size reduce available hits, but warn.
                print(f'  Warning: parsed only {len(blocks)} results (requested {args.top_k})')
            retrieved = [classify_document(text) for _, text in blocks[:args.top_k]]
            unknown_docs += sum(d is None for d in retrieved)
            gold = set(x.strip() for x in q['gold_docs'].split('|') if x.strip())
            if not gold or any(g not in DOCUMENTS for g in gold):
                raise ValueError(f'Gold labels missing/unknown: {gold}')
            metrics = score_one(gold, retrieved, ks)
            results.append({'id': q['id'], 'query': q['query'], 'category': q['category'],
                            'gold_docs': '|'.join(sorted(gold)),
                            'retrieved_docs': '|'.join(x or 'UNKNOWN' for x in retrieved),
                            **metrics})
            print(f'  first relevant RR={metrics["rr"]:.3f}; hits={metrics[f"hit@{ks[-1]}"]}')
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            failed.append({'id': q['id'], 'error': str(exc)})
            print(f'  ERROR: {exc}', file=sys.stderr)
    fields = ['id','query','category','gold_docs','retrieved_docs','rr'] + [f'{name}@{k}' for k in ks for name in ('hit','recall','precision')]
    with (args.out/'per_query.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(results)
    metric_cols = ['rr'] + [f'{name}@{k}' for k in ks for name in ('hit','recall','precision')]
    summary = {'requested_supported_queries': len(supported), 'evaluated': len(results),
               'failures': failed, 'unknown_source_results': unknown_docs,
               'metrics': {('mrr' if c == 'rr' else c): round(sum(r[c] for r in results)/len(results), 4)
                           for c in metric_cols} if results else {}}
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('\n=== Aggregate document-level retrieval metrics ===')
    print(f'Evaluated: {len(results)}/{len(supported)} | failed: {len(failed)} | unknown source hits: {unknown_docs}')
    for name, value in summary['metrics'].items():
        print(f'{name:16s} {value:.4f}')
    print('Files:', args.out/'summary.json', args.out/'per_query.csv', args.out/'raw/')
    if failed:
        print('WARNING: some queries failed; aggregate metrics exclude them.', file=sys.stderr)
        sys.exit(2)

if __name__ == '__main__':
    main()
