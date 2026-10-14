"""Structured evaluation of the existing hybrid retriever and optional RAG agents.

No retrieval algorithms are changed. --build-cases extends the existing CSV.
Retrieval-only runs make no LLM calls; optional agent runs do. Blank means unknown
or not applicable, never a synthetic zero. Read BENCHMARK_EVALUATION.md for scope.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import sqlite3
import statistics
import time

ROOT = Path(__file__).resolve().parent


def source_span(source, quote):
    """Locate a literal passage allowing only extraction whitespace differences."""
    match = re.search(r'\s+'.join(re.escape(t) for t in quote.split()), source)
    if not quote.strip() or match is None:
        raise ValueError(f'Passage absent from supplied source: {quote[:100]}')
    return match


def reviewed_annotations(cases, by_doc, corpus):
    annotations = json.loads((ROOT / 'data/insurance_evidence_annotations.json').read_text(encoding='utf-8'))
    for case in cases:
        if case['id'] not in annotations:
            continue
        case['gold_evidence'] = []
        for selector in annotations[case['id']]:
            doc_id = case['gold_document_ids'][selector.get('document_index', 0)]
            source = (corpus / by_doc[doc_id]['extracted_path']).read_text(encoding='utf-8-sig')
            page = re.search(r'--- PAGE ' + str(selector['page']) + r' ---\s*(.*?)(?=--- PAGE|\Z)', source, re.S)
            if page is None:
                raise ValueError(f'Source page missing: {case["id"]}')
            start = source_span(page[1], selector['start'])
            end = source_span(page[1][start.start():], selector['end'])
            quote = page[1][start.start():start.start() + end.end()]
            case['gold_evidence'].append({'doc_id': doc_id, 'quote': quote, 'source_page': selector['page']})
        case['gold_origin'] = 'reviewed literal supplied document passages'
        case['annotation_note'] = 'Core answer passages; selected coverage, not an exhaustive legal interpretation'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def indexed_chunks(index, collection):
    """Read existing indexed text/provenance without embedding or rebuilding."""
    with sqlite3.connect((Path(index).resolve() / 'chroma.sqlite3').as_uri() + '?mode=ro', uri=True) as db:
        rows = db.execute('''SELECT e.embedding_id,m.key,m.string_value,m.int_value,m.float_value,m.bool_value
            FROM embeddings e JOIN segments s ON s.id=e.segment_id JOIN collections c ON c.id=s.collection
            JOIN embedding_metadata m ON m.id=e.id WHERE c.name=? ORDER BY e.id''', (collection,)).fetchall()
    chunks = {}
    for cid, key, *values in rows:
        c = chunks.setdefault(cid, {'id': cid, 'text': '', 'metadata': {}})
        value = next((v for v in values if v is not None), None)
        if key == 'chroma:document':
            c['text'] = value
        else:
            c['metadata'][key] = value
    if not chunks:
        raise ValueError('Collection contains no indexed chunks')
    return list(chunks.values())


def build_cases(legacy, corpus, chunks):
    documents = [json.loads(line) for line in (corpus / 'metadata/documents.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    by_doc = {d['doc_id']: d for d in documents}
    aliases = ['lv_home_insurance_policy_document', 'aviva_home_insurance', 'axa_direct_home_2025',
               'axa_home_plus_2025', 'axa_direct_home_premier_2023', 'axa_direct_home_premier_2025']
    mapping = {}
    for alias in aliases:
        matches = [d['doc_id'] for d in documents if Path(d['extracted_path']).stem.startswith(alias + '_')]
        if len(matches) != 1:
            raise ValueError(f'Ambiguous legacy document identity: {alias}')
        mapping[alias] = matches[0]
    with legacy.open(encoding='utf-8-sig', newline='') as f:
        legacy_cases = list(csv.DictReader(f))
    cases = []
    for old in legacy_cases:
        gold = [mapping[x] for x in old['gold_docs'].split('|') if x]
        category = old['category']
        expected = ['generate'] if gold else (['clarify', 'abstain'] if category == 'unsupported_in_domain' else ['abstain'])
        cases.append({'id': old['id'], 'query': old['query'], 'category': category,
                      'expected_actions': expected, 'gold_document_ids': gold,
                      'gold_evidence': None, 'gold_origin': 'existing insurance_retrieval_benchmark.csv',
                      'annotation_note': old['evidence_hint'] + '; hints are not passage annotations'})
    # References below are literal passages in the supplied corpus, checked below.
    additions = [
        ('G001', 'fca', 'What does FCA ICOBS 8.1.1 require about handling claims promptly and fairly?',
         'financial_conduct_authority_a7473f38887e', 'handle claims promptly and fairly;'),
        ('G002', 'fca', 'Under ICOBS 8.1.1, when must an insurer settle a claim after settlement terms are agreed?',
         'financial_conduct_authority_a7473f38887e', 'settle claims promptly once settlement terms are agreed.'),
        ('G003', 'fca', 'What does the Consumer Duty say about unfairly exploiting behavioural biases or vulnerability?',
         'financial_conduct_authority_0e750f77c2ce', 'ensuring that no aspect of its business involves unfairly exploiting behavioural biases displayed or characteristics of vulnerability held by retail customers'),
        ('G004', 'fos', 'What does the FOS escape-of-water guidance say about determining whether a property was unoccupied?',
         'financial_ombudsman_service_f28f6a85f908', 'As an insurer, you need to consider whether the property was unoccupied or not, in line with their policy wording.'),
        ('G005', 'fos', 'In FOS guidance on unoccupied properties, why should an insurer establish the date of the leak?',
         'financial_ombudsman_service_f28f6a85f908', 'A policy might exclude cover once a property has been unoccupied for 30 days, but a leak was discovered on day 31. If the evidence suggests the leak started on day 29, then we’re unlikely to support an insurer applying the exclusion.'),
        ('G006', 'fos_case', 'In the FOS Barry underinsurance case, what proportion of the total claim did FOS consider fair?',
         'financial_ombudsman_service_4d8518cf2a71', 'paying out 85% of the total claim value.'),
        ('G007', 'fos_case', 'Why did FOS not uphold Andrea’s complaint about a burglary claim and understated contents?',
         'financial_ombudsman_service_2da959bb8a34', "It didn't look like Andrea had taken enough care to complete the forms with up-to-date information. We also agreed that if the insurer had known the true cost of the items, they wouldn't have given the insurance."),
        ('E001', 'supported', 'According to this LV policy wording, which document specifies my excess?',
         mapping['lv_home_insurance_policy_document'], 'If your personal details shows you have to pay an excess, this is the amount you must pay as the first part of any claim.'),
        ('E002', 'supported', 'Are motor vehicles included in the definition of contents in this LV home wording?',
         mapping['lv_home_insurance_policy_document'],
         'Contents are not:\n• motor vehicles (including motorbikes, quad bikes and motorised scooters),\ncaravans, trailers, watercraft, aircraft and all their accessories;'),
    ]
    for cid, category, query, doc_id, anchor in additions:
        cases.append({'id': cid, 'query': query, 'category': category, 'expected_actions': ['generate'],
                      'gold_document_ids': [doc_id], 'gold_evidence': [{'doc_id': doc_id, 'quote': anchor}],
                      'gold_origin': 'literal supplied document passage',
                      'annotation_note': 'Selected anchor coverage, not exhaustive answer evidence'})
    cases.extend([
        {'id': 'C001', 'query': 'What is my exact LV escape-of-water excess? My personal details document is unavailable.',
         'category': 'missing_customer_context', 'expected_actions': ['abstain', 'clarify'],
         'gold_document_ids': [mapping['lv_home_insurance_policy_document']], 'gold_evidence': None,
         'annotation_note': 'LV wording is contextual; private amount is absent, not annotated as public evidence'},
        {'id': 'C002', 'query': 'Is accidental damage selected on my LV customer policy?',
         'category': 'missing_customer_context', 'expected_actions': ['clarify'],
         'gold_document_ids': [mapping['lv_home_insurance_policy_document']], 'gold_evidence': None,
         'customer_context': {'request_type': 'customer_claim', 'policy_document_id': mapping['lv_home_insurance_policy_document'],
                              'policy_document_verified': False, 'facts': {}},
         'annotation_note': 'No customer schedule supplied; requires verification before cover confirmation'},
        {'id': 'V001', 'query': 'Compare the unoccupied definitions in the AXA Premier 2023 file and 2025 file.',
         'category': 'version_comparison', 'expected_actions': ['generate'],
         'gold_document_ids': [mapping['axa_direct_home_premier_2023'], mapping['axa_direct_home_premier_2025']],
         'gold_evidence': None, 'annotation_note': 'Local file-version labels, not policy effective dates'},
    ])
    reviewed_annotations(cases, by_doc, corpus)
    for case in cases:
        gold = case['gold_document_ids']
        case['gold_documents'] = [{k: by_doc[d].get(k) for k in
                                  ('doc_id', 'provider', 'product', 'version_label', 'sha256', 'extracted_path')} for d in gold]
        case.setdefault('customer_context', {'request_type': 'general', 'facts': {},
                         **({'policy_document_id': gold[0]} if len(gold) == 1 else {})})
        for evidence in case['gold_evidence'] or []:
            doc = by_doc[evidence['doc_id']]
            source = (corpus / doc['extracted_path']).read_text(encoding='utf-8-sig')
            norm = lambda s: ' '.join(s.split())
            page_text = source
            if evidence.get('source_page') is not None:
                page_text = re.search(r'--- PAGE ' + str(evidence['source_page']) + r' ---\s*(.*?)(?=--- PAGE|\Z)', source, re.S)[1]
            match = source_span(page_text, evidence['quote'])
            evidence['quote'] = match[0]
            evidence.setdefault('source_page', int(re.findall(r'--- PAGE (\d+) ---', source[:match.start()])[-1])
                                if '--- PAGE' in source[:match.start()] else None)
            evidence['evidence_id'] = evidence['doc_id'] + '::span_' + hashlib.sha256(evidence['quote'].encode()).hexdigest()[:12]
            evidence['source_path'] = (corpus / doc['extracted_path']).relative_to(ROOT).as_posix()
            evidence['source_text_sha256'] = digest(corpus / doc['extracted_path'])
            evidence['matching_chunk_ids'] = [c['id'] for c in chunks if c['metadata']['doc_id'] == evidence['doc_id']
                                             and norm(evidence['quote']) in norm(c['text'])]
            evidence['index_status'] = 'indexed' if evidence['matching_chunk_ids'] else 'source_passage_missing_from_index'
    return cases


def validate_cases(cases, chunks):
    by_id = {c['id']: c for c in chunks}
    docs = {c['metadata']['doc_id'] for c in chunks}
    ids = set()
    for case in cases:
        if case['id'] in ids or not case['query'].strip():
            raise ValueError('Case IDs must be unique and queries nonempty')
        ids.add(case['id'])
        if not set(case['gold_document_ids']) <= docs:
            raise ValueError(f'Gold document missing from index: {case["id"]}')
        if not set(case['expected_actions']) <= {'generate', 'abstain', 'clarify'}:
            raise ValueError('Unknown expected action')
        for e in case['gold_evidence'] or []:
            source = ROOT / e['source_path'].replace('\\', '/')
            if not source.resolve().is_relative_to(ROOT) or digest(source) != e['source_text_sha256']:
                raise ValueError('Gold source changed or outside project')
            if ' '.join(e['quote'].split()) not in ' '.join(source.read_text(encoding='utf-8-sig').split()):
                raise ValueError('Gold quote absent from source')
            if e.get('source_page') is not None:
                page = re.search(r'--- PAGE ' + str(e['source_page']) + r' ---\s*(.*?)(?=--- PAGE|\Z)',
                                 source.read_text(encoding='utf-8-sig'), re.S)
                if page is None or ' '.join(e['quote'].split()) not in ' '.join(page[1].split()):
                    raise ValueError('Gold quote absent from annotated page')
            if not e['matching_chunk_ids'] and e.get('index_status') != 'source_passage_missing_from_index':
                raise ValueError('Gold evidence lacks indexed alternatives')
            for cid in e['matching_chunk_ids']:
                c = by_id[cid]
                if c['metadata']['doc_id'] != e['doc_id'] or ' '.join(e['quote'].split()) not in ' '.join(c['text'].split()):
                    raise ValueError('Gold passage/index provenance mismatch')


def export_gold_evidence(cases, chunks, output_dir):
    """Export source gold separately from observed retrieval results."""
    by_id = {c['id']: c for c in chunks}
    passages, alternatives = [], []
    for case in cases:
        for e in case['gold_evidence'] or []:
            row = {'query_id': case['id'], 'query': case['query'], 'evidence_id': e['evidence_id'],
                   'document_id': e['doc_id'], 'source_page': e.get('source_page'),
                   'source_path': e['source_path'], 'source_text_sha256': e['source_text_sha256'],
                   'source_quote': e['quote'], 'index_status': e.get('index_status'),
                   'matching_chunk_ids': e['matching_chunk_ids']}
            passages.append(row)
            for cid in e['matching_chunk_ids']:
                chunk = by_id[cid]
                alternatives.append({**row, 'chunk_id': cid, 'indexed_page': chunk['metadata'].get('page'),
                                     'indexed_text': chunk['text']})
    write_csv(output_dir / 'gold_evidence.csv', passages, list(passages[0]) if passages else ['query_id'])
    write_csv(output_dir / 'gold_evidence_chunks.csv', alternatives,
              list(alternatives[0]) if alternatives else ['query_id'])


def retrieval_metrics(case, hits, ks):
    """Chunk-ranked document recall; evidence anchors deduplicated across overlap."""
    gold = set(case['gold_document_ids'])
    evidence = case['gold_evidence']
    metrics = {}
    ranks = [i for i, h in enumerate(hits, 1) if h['metadata'].get('doc_id') in gold]
    metrics['mrr'] = 1 / ranks[0] if ranks else (0.0 if gold else None)
    evidence_hits = lambda h: {e['evidence_id'] for e in evidence or [] if h['id'] in e['matching_chunk_ids']}
    evidence_ranks = [i for i, h in enumerate(hits, 1) if evidence_hits(h)]
    metrics['evidence_mrr'] = (1 / evidence_ranks[0] if evidence_ranks else 0.0) if evidence else None
    for k in ks:
        top = hits[:k]
        retrieved_docs = {h['metadata'].get('doc_id') for h in top}
        metrics[f'hit@{k}'] = int(bool(gold & retrieved_docs)) if gold else None
        metrics[f'document_recall@{k}'] = len(gold & retrieved_docs) / len(gold) if gold else None
        metrics[f'document_precision@{k}'] = sum(h['metadata'].get('doc_id') in gold for h in top) / k if gold else None
        found = set().union(*(evidence_hits(h) for h in top)) if top else set()
        metrics[f'evidence_hit@{k}'] = int(bool(found)) if evidence else None
        metrics[f'evidence_recall@{k}'] = len(found) / len(evidence) if evidence else None
        metrics[f'evidence_precision@{k}'] = sum(bool(evidence_hits(h)) for h in top) / k if evidence else None
    return metrics


def metadata_mismatches(case, meta):
    """Unknown metadata is unknown; FCA/FOS providers are not policy insurers."""
    if meta.get('source_type') != 'policy':
        return {k: None for k in ('wrong_insurer', 'wrong_product', 'wrong_version')}
    expected = case.get('gold_documents', [])
    def mismatch(key, among):
        allowed = {d.get(key) for d in among if d.get(key)}
        return int(meta[key] not in allowed) if allowed and meta.get(key) else None
    insurer = mismatch('provider', expected)
    same_insurer = [d for d in expected if d.get('provider') == meta.get('provider')]
    product = mismatch('product', same_insurer)
    same_product = [d for d in same_insurer if d.get('product') == meta.get('product')]
    version = mismatch('version_label', same_product)
    return {'wrong_insurer': insurer, 'wrong_product': product, 'wrong_version': version}


def usage_and_cost(result, prices):
    usages = []
    grounding = result.get('grounding_metrics', {})
    if grounding.get('assessment_usage'):
        usages.append((grounding.get('assessor_model'), grounding['assessment_usage']))
    if result.get('generation_usage'):
        model = result.get('repro', {}).get('generation_model', '').removeprefix('openai:')
        usages.append((model, result['generation_usage']))
    counts = {'input_tokens': None, 'output_tokens': None, 'cached_input_tokens': None}
    costs = []
    for model, u in usages:
        inp, out = u.get('input_tokens'), u.get('output_tokens')
        cached = u.get('cached_tokens', u.get('input_token_details', {}).get('cache_read', 0))
        if inp is None or out is None:
            continue
        for key, value in [('input_tokens', inp), ('output_tokens', out), ('cached_input_tokens', cached)]:
            counts[key] = (counts[key] or 0) + value
        rate = (prices or {}).get('models', {}).get(model)
        if rate is not None and (not cached or 'cached_input_per_million' in rate):
            costs.append(((inp - cached) * rate['input_per_million'] + out * rate['output_per_million']
                          + cached * rate.get('cached_input_per_million', 0)) / 1e6)
    # Never present a partial cost as total. Failure usage may be incomplete.
    complete = len(costs) == len(usages) and bool(usages) and not result.get('decision_support', {}).get('technical_failure')
    return {**counts, 'estimated_cost': sum(costs) if complete else None,
            'total_tokens': counts['input_tokens'] + counts['output_tokens'] if counts['input_tokens'] is not None else None,
            'cost_currency': (prices or {}).get('currency') if complete else None,
            'usage_status': 'reported' if usages else 'not_reported'}


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * p
    lo = int(position)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (position - lo)


def aggregate(rows, metrics):
    output = []
    groups = {'all': rows}
    for key in ('variant', 'category', 'expected_actions'):
        for value in sorted({str(r[key]) for r in rows}):
            groups[f'{key}={value}'] = [r for r in rows if str(r[key]) == value]
    for group, selected in groups.items():
        for metric in metrics:
            values = [r[metric] for r in selected if r.get(metric) is not None]
            output.append({'group': group, 'metric': metric, 'value': statistics.mean(values) if values else None,
                           'measured_count': len(values), 'attempt_count': len(selected)})
        for scope in ('request_latency_ms', 'retrieval_latency_ms'):
            values = [r[scope] for r in selected if r.get(scope) is not None]
            for name, p in [('p50', .5), ('p95', .95), ('p99', .99)]:
                output.append({'group': group, 'metric': scope + '_' + name, 'value': percentile(values, p),
                               'measured_count': len(values), 'attempt_count': len(selected)})
        output.append({'group': group, 'metric': 'error_rate', 'value': sum(bool(r['error']) for r in selected) / len(selected),
                       'measured_count': len(selected), 'attempt_count': len(selected)})
        costs = [r['estimated_cost'] for r in selected if r.get('estimated_cost') is not None]
        output.append({'group': group, 'metric': 'total_estimated_cost',
                       'value': sum(costs) if len(costs) == len(selected) else None,
                       'measured_count': len(costs), 'attempt_count': len(selected)})
    return output


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row[key], ensure_ascii=False) if isinstance(row.get(key), (list, dict))
                             else row.get(key) for key in fields})


class RetrievalObserver:
    """Capture actual calls; delegation preserves the existing retrieval behavior."""
    def __init__(self, retriever):
        self.retriever, self.calls = retriever, []

    def __getattr__(self, name):
        return getattr(self.retriever, name)

    def search_chunks(self, query, **kwargs):
        started = time.perf_counter()
        hits = self.retriever.search_chunks(query, **kwargs)
        self.calls.append({'query': query, 'latency_ms': (time.perf_counter() - started) * 1000,
                           'hits': copy.deepcopy(hits), 'filters': {k: v for k, v in kwargs.items() if k != 'query_embedding'}})
        return hits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, default=ROOT / 'data/insurance_retrieval_cases_v2.jsonl')
    parser.add_argument('--build-cases', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--index', type=Path, default=ROOT / 'insurance_chroma_index_v3')
    parser.add_argument('--collection', default='insurance_chunks')
    parser.add_argument('--legacy', type=Path, default=ROOT / 'insurance_retrieval_benchmark.csv')
    parser.add_argument('--corpus', type=Path, default=ROOT / 'real_insurance_corpus_v2')
    parser.add_argument('--scope', type=Path, default=ROOT / 'data/home_insurance_scope.json')
    parser.add_argument('--pipeline', choices=['retrieval', 'guarded', 'standard', 'both'], default='retrieval')
    parser.add_argument('--model', default='gpt-4o-mini')
    parser.add_argument('--model-cache', type=Path, default=ROOT / '.benchmark-model-cache')
    parser.add_argument('--allow-model-download', action='store_true')
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--ks', default='1,3,5,10')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--query-id', action='append')
    parser.add_argument('--prices', type=Path, help='User-supplied dated price table; no assumed rates')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'benchmark_results/insurance_structured')
    args = parser.parse_args()
    ks = sorted(set(int(k) for k in args.ks.split(',')))
    if not ks or min(ks) < 1 or max(ks) > args.top_k or args.repeats < 1 or (args.limit is not None and args.limit < 1):
        parser.error('Require positive K/repeats/limit and max(K) <= top-k')
    chunks = indexed_chunks(args.index, args.collection)
    if args.build_cases:
        cases = build_cases(args.legacy, args.corpus, chunks)
        args.cases.write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in cases), encoding='utf-8')
    cases = [json.loads(line) for line in args.cases.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    validate_cases(cases, chunks)
    if args.query_id:
        if set(args.query_id) - {c['id'] for c in cases}:
            parser.error('Unknown query ID')
        cases = [c for c in cases if c['id'] in args.query_id]
    if args.limit:
        cases = cases[:args.limit]
    if args.validate_only:
        print(f'Validated {len(cases)} queries; {sum(bool(c["gold_evidence"]) for c in cases)} have real passage annotations.')
        return 0
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Choose an empty output directory to preserve previous results')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'cases_snapshot.jsonl').write_bytes(args.cases.read_bytes())
    prices = json.loads(args.prices.read_text()) if args.prices else None
    if prices and (not prices.get('as_of') or not prices.get('currency')):
        parser.error('Price table requires as_of and currency')
    variants = ['guarded', 'standard'] if args.pipeline == 'both' else [args.pipeline]
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(),
                'script_sha256': digest(__file__), 'cases_sha256': digest(args.cases),
                'corpus_manifest_sha256': digest(args.corpus / 'metadata/documents.jsonl'),
                'index_text_metadata_sha256': hashlib.sha256(json.dumps(chunks, sort_keys=True).encode()).hexdigest(),
                'index': str(args.index), 'collection': args.collection, 'query_count': len(cases),
                'variants': variants, 'top_k': args.top_k, 'ks': ks, 'repeats': args.repeats,
                'selected_query_ids': [c['id'] for c in cases],
                'retrieval_stage': 'initial_chunk_ranking; no metadata filters in retrieval-only mode',
                'precision_denominator': 'K; missing slots count as nonrelevant',
                'mrr_scope': 'first relevant document in returned top-k chunk ranking',
                'evidence_scope': 'annotated anchors only, not exhaustive answer evidence', 'prices': prices,
                'scope_sha256': digest(args.scope),
                'precision_unit': 'chunk slots whose document/evidence is relevant; recall deduplicates documents/anchors'}
    init_started = time.perf_counter()
    agents, observer, startup_error = {}, None, None
    try:
        import chromadb
        from importlib.metadata import version
        from sentence_transformers import SentenceTransformer
        from insurance_retrieval_fixed import InsuranceHybridRetriever
        collection = chromadb.PersistentClient(path=str(args.index)).get_collection(args.collection)
        metadata = collection.metadata or {}
        scope = json.loads(args.scope.read_text(encoding='utf-8-sig'))
        if metadata.get('embedding_model') != scope['embedding_model'] or metadata.get('corpus_fingerprint') != scope['corpus_fingerprint']:
            raise ValueError('Index does not match existing model/corpus scope')
        model = SentenceTransformer(metadata['embedding_model'], cache_folder=str(args.model_cache),
                                    local_files_only=not args.allow_model_download)
        observer = RetrievalObserver(InsuranceHybridRetriever(collection, model))
        if args.pipeline != 'retrieval':
            from dotenv import load_dotenv
            load_dotenv(ROOT / '.env')
            from src.agent import InsuranceRAGAgent, StandardInsuranceRAGAgent
            from src.generator import load_generator
            for variant in variants:
                common = {'index_dir': str(args.index), 'insurance_scope_path': str(args.scope),
                          'retriever': observer, 'generator': load_generator('openai', model=args.model)}
                agents[variant] = (InsuranceRAGAgent(**common, ood_reference_path=str(ROOT / 'data/home_insurance_ood.npz'),
                                  assessment_model=args.model) if variant == 'guarded' else StandardInsuranceRAGAgent(**common))
        manifest['collection_metadata'] = metadata
        manifest['package_versions'] = {name: version(name) for name in ('chromadb', 'sentence-transformers', 'numpy', 'rank-bm25')}
        manifest['embedding_cache_snapshots'] = [p.name for p in args.model_cache.glob('models--*/snapshots/*') if p.is_dir()]
    except Exception as exc:
        startup_error = type(exc).__name__ + ': ' + str(exc)
    manifest['initialization_latency_ms'] = (time.perf_counter() - init_started) * 1000
    manifest['startup_error'] = startup_error
    metrics = list(retrieval_metrics(cases[0], [], ks))
    per_query, retrieved = [], []
    with (args.output_dir / 'raw_results.jsonl').open('w', encoding='utf-8') as raw:
        for repeat in range(1, args.repeats + 1):
            for case in cases:
                for variant in variants:
                    row = {'query_id': case['id'], 'query': case['query'], 'variant': variant, 'repeat': repeat,
                           'category': case['category'], 'expected_actions': case['expected_actions'],
                           'gold_document_ids': case['gold_document_ids'],
                           'gold_evidence_ids': [e['evidence_id'] for e in case['gold_evidence'] or []],
                           'gold_chunk_ids': sorted({cid for e in case['gold_evidence'] or [] for cid in e['matching_chunk_ids']}),
                           'evidence_annotated': bool(case['gold_evidence']), 'error': startup_error,
                           'request_latency_ms': None, 'retrieval_latency_ms': None, 'estimated_cost': None,
                           'actual_action': None, 'action_matches': None, 'retrieval_executed': False}
                    result, hits = {}, []
                    if not startup_error:
                        observer.calls.clear()
                        started = time.perf_counter()
                        try:
                            if variant == 'retrieval':
                                hits = observer.search_chunks(case['query'], top_k=args.top_k)
                            else:
                                result = agents[variant].answer(case['query'], customer_context=copy.deepcopy(case['customer_context']),
                                                               top_k_final=args.top_k, include_evidence=True)
                                row['actual_action'] = result['action']
                                if result.get('decision_support', {}).get('technical_failure'):
                                    row['error'] = result.get('reason', 'technical_failure')
                                if variant == 'guarded':
                                    row['action_matches'] = int(not row['error'] and result['action'] in case['expected_actions'])
                        except Exception as exc:
                            row['error'] = type(exc).__name__ + ': ' + str(exc)
                        row['request_latency_ms'] = (time.perf_counter() - started) * 1000
                        calls = [call for call in observer.calls if call['query'] == case['query']]
                        if calls:
                            hits = calls[0]['hits']
                            row.update(retrieval_executed=True, retrieval_latency_ms=calls[0]['latency_ms'],
                                       retrieval_filters=calls[0]['filters'])
                    row.update(usage_and_cost(result, prices))
                    if row['retrieval_executed']:
                        row.update(retrieval_metrics(case, hits, ks))
                    else:
                        row.update({metric: None for metric in metrics})
                    row['retrieved_count'] = len(hits)
                    for rank, hit in enumerate(hits, 1):
                        meta = hit['metadata']
                        matched = [e['evidence_id'] for e in case['gold_evidence'] or [] if hit['id'] in e['matching_chunk_ids']]
                        retrieved.append({'query_id': case['id'], 'variant': variant, 'repeat': repeat, 'rank': rank,
                            'chunk_id': hit['id'], 'document_id': meta.get('doc_id'), 'insurer': meta.get('provider') if meta.get('source_type') == 'policy' else None,
                            'authority': meta.get('authority', meta.get('provider')) if meta.get('source_type') != 'policy' else None,
                            'product': meta.get('product'), 'version': meta.get('version_label'), 'page': meta.get('page'),
                            'source_type': meta.get('source_type'), 'source_url': meta.get('source_url'),
                            'dense_distance': hit.get('dense_distance'), 'cosine_similarity': hit.get('cosine_similarity'),
                            'bm25_score': hit.get('bm25_score'), 'rrf_score': hit.get('fused_score'), 'excerpt': hit['text'][:800],
                            'document_relevant': int(meta.get('doc_id') in case['gold_document_ids']),
                            'evidence_relevant': int(bool(matched)) if case['gold_evidence'] else None,
                            'matched_evidence_ids': matched, **metadata_mismatches(case, meta)})
                    own = [h for h in retrieved if h['query_id'] == case['id'] and h['variant'] == variant and h['repeat'] == repeat]
                    for flag in ('wrong_insurer', 'wrong_product', 'wrong_version'):
                        observed = [h[flag] for h in own if h[flag] is not None]
                        row[flag + '_rate'] = statistics.mean(observed) if observed else None
                    raw.write(json.dumps({'case': case, 'row': row, 'initial_hits': hits, 'result': result}, ensure_ascii=False, default=str) + '\n')
                    raw.flush()
                    per_query.append(row)
                    print(f'{case["id"]} {variant}: {row["error"] or "completed"}; hits={len(hits)}', flush=True)
    query_fields = list(dict.fromkeys(k for r in per_query for k in r))
    query_fields += [k for k in ('input_tokens', 'output_tokens', 'cached_input_tokens', 'cost_currency') if k not in query_fields]
    hit_fields = ['query_id', 'variant', 'repeat', 'rank', 'chunk_id', 'document_id', 'insurer', 'authority', 'product', 'version', 'page',
                  'source_type', 'source_url', 'dense_distance', 'cosine_similarity', 'bm25_score', 'rrf_score', 'excerpt',
                  'document_relevant', 'evidence_relevant', 'matched_evidence_ids', 'wrong_insurer', 'wrong_product', 'wrong_version']
    write_csv(args.output_dir / 'per_query_results.csv', per_query, query_fields)
    write_csv(args.output_dir / 'retrieved_results.csv', retrieved, hit_fields)
    export_gold_evidence(cases, chunks, args.output_dir)
    write_csv(args.output_dir / 'aggregate_metrics.csv', aggregate(per_query, metrics + ['wrong_insurer_rate', 'wrong_product_rate', 'wrong_version_rate', 'action_matches']),
              ['group', 'metric', 'value', 'measured_count', 'attempt_count'])
    (args.output_dir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    failures = [r for r in per_query if r['error'] or r.get(f'hit@{max(ks)}') == 0 or r.get('action_matches') == 0
                or (r.get(f'evidence_recall@{max(ks)}') is not None and r[f'evidence_recall@{max(ks)}'] < 1)]
    report = ['# Failure analysis', '', f'Attempts: {len(per_query)}; errors: {sum(bool(r["error"]) for r in per_query)}.',
              f'Retrieval executed: {sum(r["retrieval_executed"] for r in per_query)}; passage-annotated queries: {sum(bool(c["gold_evidence"]) for c in cases)}.',
              '', 'Metrics use initial chunk ranks, not final expanded evidence. Blank evidence scores mean unannotated.',
              'OOD retrieval returns neighbours; it does not constitute permission to answer. Expected actions are gold targets, not measured decisions in retrieval-only mode.',
              'FOS sources are indexed but excluded from the current guarded trusted-document allowlist; agent runs may reject answerable FOS questions.',
              'Wrong-version flags compare local file labels only. Blank metadata is unknown. Costs are blank without reported usage and a dated price table.', '', '## Misses/errors']
    report += [f'- {r["query_id"]} ({r["variant"]}): {r["error"] or "gold document/evidence miss or wrong action"}; evidence recall@{max(ks)}={r.get(f"evidence_recall@{max(ks)}")}.' for r in failures]
    report += ['', '## Source passages missing from individual indexed chunks']
    report += [f'- {c["id"]}, {e["doc_id"]}, source page {e.get("source_page")}: {e["evidence_id"]}. Preserved source gold; no indexed alternative contains this literal passage.'
               for c in cases for e in c['gold_evidence'] or [] if not e['matching_chunk_ids']]
    report += ['', '## Early-rank passage misses']
    report += [f'- {r["query_id"]}: document hit@1={r.get("hit@1")}; evidence recall@1={r.get("evidence_recall@1")}; evidence MRR={r.get("evidence_mrr")}.'
               for r in per_query if r.get('evidence_recall@1') is not None and r['evidence_recall@1'] < 1]
    report += ['', '## Metadata mismatch observations']
    report += [f'- {flag}: {sum(h[flag] == 1 for h in retrieved)} mismatched hits / {sum(h[flag] is not None for h in retrieved)} comparable hits.'
               for flag in ('wrong_insurer', 'wrong_product', 'wrong_version')]
    (args.output_dir / 'failure_analysis.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    from report_insurance_results import report as report_results
    report_results(args.output_dir / 'raw_results.jsonl')
    print(f'Outputs: {args.output_dir}')
    return int(any(r['error'] for r in per_query))


if __name__ == '__main__':
    raise SystemExit(main())
