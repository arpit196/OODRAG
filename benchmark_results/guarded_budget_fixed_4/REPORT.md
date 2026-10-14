# Reading the report

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
