"""Offline reporting of recorded benchmark results; no model calls or new scores."""
import argparse
import csv
import json
from pathlib import Path

from evaluate_insurance_benchmark import aggregate, write_csv
from simplify_insurance_results import simplify

REVIEW_FIELDS = ['actually_answered', 'fully_answered', 'factually_correct',
                 'citations_supported', 'answer_relevant', 'unsupported_claim', 'notes']


def report(input_path, output_dir=None):
    input_path = Path(input_path)
    output = Path(output_dir) if output_dir else input_path.parent
    output.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in input_path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    review_path = output / 'manual_review.csv'
    key = lambda r: (str(r['query_id']), str(r['variant']), str(r['repeat']))
    previous = {}
    if review_path.exists():
        with review_path.open(encoding='utf-8', newline='') as f:
            previous = {key(r): r for r in csv.DictReader(f)}
    rows, reviews = [], []
    for record in records:
        row = dict(record['row'])
        result = record.get('result') or {}
        support = result.get('decision_support') or {}
        action = result.get('action')
        agent_run = row['variant'] != 'retrieval'
        row['error'] = row.get('error') or (result.get('reason', 'technical_failure') if support.get('technical_failure') else None)
        row.update(answer=result.get('answer'), reason=result.get('reason'))
        for name in ('abstain', 'clarify', 'generate'):
            row[name + '_rate'] = int(action == name) if agent_run else None
        row['technical_abstention_rate'] = int(action == 'abstain' and bool(row['error'])) if agent_run else None
        row['partial_support_rate'] = int(result.get('reason') == 'partial_support') if agent_run else None
        row['response_present_rate'] = int(bool(result.get('answer', '').strip())) if agent_run else None
        row['citation_count'] = len(result.get('citations') or []) if agent_run else None
        for stage, value in (result.get('timing_metrics', {}).get('stages') or {}).items():
            row[stage + '_latency_ms'] = value.get('duration_ms')
        review = {**{k: row[k] for k in ('query_id', 'variant', 'repeat', 'query')},
                  'answer': result.get('answer'), **{f: previous.get(key(row), {}).get(f, '') for f in REVIEW_FIELDS}}
        for field in REVIEW_FIELDS[:-1]:
            value = review[field]
            if value not in ('', '0', '1'):
                raise ValueError(f'{row["query_id"]}: {field} must be blank, 0 or 1')
            row['manual_' + field] = int(value) if value else None
        reviews.append(review)
        rows.append(row)
    metrics = sorted({k for r in rows for k in r if '@' in k or k in ('mrr', 'evidence_mrr', 'action_matches', 'citation_count')
                      or k.endswith('_rate') or k.startswith('manual_') or k.startswith(('stage1_latency', 'stage2_latency', 'stage3_latency'))})
    totals = aggregate(rows, metrics) if rows else []
    # Keep known subtotals separate from complete totals, including failed requests.
    for group in sorted({r['group'] for r in totals}):
        if group == 'all':
            selected = rows
        else:
            field, value = group.split('=', 1)
            selected = [r for r in rows if str(r[field]) == value]
        for field in ('input_tokens', 'output_tokens', 'cached_input_tokens', 'total_tokens', 'estimated_cost'):
            values = [r[field] for r in selected if r.get(field) is not None]
            for suffix, value in [('known_subtotal', sum(values) if values else None),
                                  ('complete_total', sum(values) if len(values) == len(selected) and not any(r['error'] for r in selected) else None)]:
                totals.append(dict(group=group, metric=field + '_' + suffix, value=value,
                                   measured_count=len(values), attempt_count=len(selected)))
    write_csv(output / 'report_per_query.csv', rows, list(dict.fromkeys(k for r in rows for k in r)))
    write_csv(output / 'report_metrics.csv', totals, ['group', 'metric', 'value', 'measured_count', 'attempt_count'])
    write_csv(review_path, reviews, ['query_id', 'variant', 'repeat', 'query', 'answer'] + REVIEW_FIELDS)
    (output / 'simple_results.json').write_text(json.dumps([simplify(r) for r in records], ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'report_summary.json').write_text(json.dumps(totals, indent=2), encoding='utf-8')
    (output / 'REPORT.md').write_text('''# Reading the report

Retrieval metrics retain the benchmark's initial chunk ranking. Document recall deduplicates documents;
evidence recall deduplicates annotated passages. Blank/unannotated/OOD values are excluded, not zero.
Every aggregate includes its measured denominator. Compare variants separately; the all group pools attempts.
MRR is reciprocal rank of the first relevant result within the retrieved list.

Abstain/clarify/generate are routing decisions, not correctness judgements. Error rate includes technical failures.
Response present does not mean the question was answered. Fill manual_review.csv with 0/1 (blank = unknown),
then rerun this script; existing ratings are preserved. Actually answered includes useful partial answers;
fully answered requires all requested parts. Judge factual correctness and citation support against the sources.
An appropriate refusal can be correct without answering. Unsupported claim flags unsafe assertions.

Latency is per request and excludes initialization; P50 is median. P95/P99 on small samples are unstable.
Stage averages exclude skipped stages. Tokens/cost are reported only from recorded measurements;
known subtotals may omit failed/unreported calls. Complete totals remain blank when coverage is incomplete.
Cost currency/rates come from the original run. Cached input is a subset of input, not extra tokens.

Before deployment: evaluate a held-out set, review answer correctness/completeness, citation entailment,
unnecessary abstentions, unsafe answers on unsupported queries, insurer/version leakage, and PDF table extraction.
Load-test concurrent requests, timeouts, cold starts and memory; test access control, data retention,
prompt injection and sensitive customer data handling. This benchmark alone does not establish production readiness.
''', encoding='utf-8')
    return rows, totals


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    rows, _ = report(args.input, args.output_dir)
    print(f'Reported {len(rows)} attempts in {args.output_dir or args.input.parent}')
