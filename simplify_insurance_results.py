"""Export a readable view of recorded results, without running the RAG pipeline."""
import argparse
import json
from pathlib import Path


def simplify(record):
    result = record['result']
    support = result.get('decision_support') or {}
    assessment = (result.get('grounding_metrics') or {}).get('assessment') or {}
    # A failed assessment is diagnostic output, not trusted supporting evidence.
    parts = [] if support.get('technical_failure') else assessment.get('parts', [])
    chunks = {c['id']: c for c in result.get('evidence_chunks', [])}
    references = {}
    for citation in result.get('citations', []):
        references.setdefault(citation['id'], {'roles': [], 'quotes': []})['roles'].append('answer_citation')
    for part in parts:
        for ref in part.get('evidence', []):
            entry = references.setdefault(ref['chunk_id'], {'roles': [], 'quotes': []})
            if 'assessment_reference' not in entry['roles']:
                entry['roles'].append('assessment_reference')
            if ref.get('quote') and ref['quote'] not in entry['quotes']:
                entry['quotes'].append(ref['quote'])
    evidence = []
    for cid, ref in references.items():
        chunk = chunks.get(cid, {})
        meta = chunk.get('metadata') or {}
        page = meta.get('page')
        evidence.append({'chunk_id': cid, 'role': list(dict.fromkeys(ref['roles'])),
                         'document': {'id': meta.get('doc_id'), 'title': meta.get('title')},
                         # HTML sources use page=0: it is not a real PDF page.
                         'page': page if isinstance(page, int) and page > 0 else None,
                         'selected_quotes': ref['quotes'], 'text': chunk.get('text')})
    return {'query_id': record.get('case', {}).get('id'),
            'variant': record.get('row', {}).get('variant'),
            'repeat': record.get('row', {}).get('repeat'),
            'query': result.get('query', record.get('case', {}).get('query')),
            'answer': result.get('answer'),
            'reasoning': {'action': result.get('action'), 'reason': result.get('reason'),
                          'technical_failure': bool(support.get('technical_failure')),
                          'missing_evidence': support.get('missing_evidence', []),
                          'missing_customer_facts': support.get('missing_customer_facts', []),
                          'conflicts': support.get('conflicts', []),
                          'error': (result.get('grounding_metrics') or {}).get('error_details'),
                          'question_parts': [{k: p.get(k) for k in
                              ('question_part', 'status', 'missing_evidence',
                               'missing_customer_facts', 'conflicts')} for p in parts]},
            'supporting_chunks': evidence}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.input.with_name('simple_results.json')
    if output.resolve() == args.input.resolve():
        parser.error('Output must differ from the original results file')
    records = [json.loads(line) for line in args.input.read_text(encoding='utf-8').splitlines() if line.strip()]
    output.write_text(json.dumps([simplify(r) for r in records], ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote {len(records)} results to {output}')


if __name__ == '__main__':
    main()
