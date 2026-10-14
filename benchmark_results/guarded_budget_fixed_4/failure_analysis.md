# Failure analysis

Attempts: 54; errors: 0.
Retrieval executed: 44; passage-annotated queries: 44.

Metrics use initial chunk ranks, not final expanded evidence. Blank evidence scores mean unannotated.
OOD retrieval returns neighbours; it does not constitute permission to answer. Expected actions are gold targets, not measured decisions in retrieval-only mode.
FOS sources are indexed but excluded from the current guarded trusted-document allowlist; agent runs may reject answerable FOS questions.
Wrong-version flags compare local file labels only. Blank metadata is unknown. Costs are blank without reported usage and a dated price table.

## Misses/errors
- Q002 (guarded): gold document/evidence miss or wrong action; evidence recall@10=1.0.
- Q003 (guarded): gold document/evidence miss or wrong action; evidence recall@10=0.0.
- Q005 (guarded): gold document/evidence miss or wrong action; evidence recall@10=0.0.
- Q007 (guarded): gold document/evidence miss or wrong action; evidence recall@10=0.0.
- Q010 (guarded): gold document/evidence miss or wrong action; evidence recall@10=1.0.
- Q014 (guarded): gold document/evidence miss or wrong action; evidence recall@10=1.0.
- Q016 (guarded): gold document/evidence miss or wrong action; evidence recall@10=0.0.
- Q017 (guarded): gold document/evidence miss or wrong action; evidence recall@10=0.0.
- Q018 (guarded): gold document/evidence miss or wrong action; evidence recall@10=0.5.
- Q019 (guarded): gold document/evidence miss or wrong action; evidence recall@10=0.5.
- Q020 (guarded): gold document/evidence miss or wrong action; evidence recall@10=1.0.
- Q022 (guarded): gold document/evidence miss or wrong action; evidence recall@10=1.0.
- Q024 (guarded): gold document/evidence miss or wrong action; evidence recall@10=1.0.
- Q026 (guarded): gold document/evidence miss or wrong action; evidence recall@10=0.0.
- Q028 (guarded): gold document/evidence miss or wrong action; evidence recall@10=0.0.
- Q031 (guarded): gold document/evidence miss or wrong action; evidence recall@10=1.0.
- Q033 (guarded): gold document/evidence miss or wrong action; evidence recall@10=0.5.
- G002 (guarded): gold document/evidence miss or wrong action; evidence recall@10=1.0.
- G004 (guarded): gold document/evidence miss or wrong action; evidence recall@10=None.
- G005 (guarded): gold document/evidence miss or wrong action; evidence recall@10=None.
- G006 (guarded): gold document/evidence miss or wrong action; evidence recall@10=None.
- G007 (guarded): gold document/evidence miss or wrong action; evidence recall@10=None.
- V001 (guarded): gold document/evidence miss or wrong action; evidence recall@10=0.5.

## Source passages missing from individual indexed chunks
- Q003, lv_03904e7e51ee, source page 4: lv_03904e7e51ee::span_48c630aeb819. Preserved source gold; no indexed alternative contains this literal passage.
- Q028, axa_insurance_uk_plc_405909992a8f, source page 5: axa_insurance_uk_plc_405909992a8f::span_fb44211fb057. Preserved source gold; no indexed alternative contains this literal passage.
- Q028, axa_insurance_uk_plc_405909992a8f, source page 5: axa_insurance_uk_plc_405909992a8f::span_e2678c849723. Preserved source gold; no indexed alternative contains this literal passage.
- Q028, axa_insurance_uk_plc_405909992a8f, source page 5: axa_insurance_uk_plc_405909992a8f::span_e2a0cab40275. Preserved source gold; no indexed alternative contains this literal passage.

## Early-rank passage misses
- Q001: document hit@1=1; evidence recall@1=0.0; evidence MRR=0.5.
- Q002: document hit@1=1; evidence recall@1=0.0; evidence MRR=0.25.
- Q003: document hit@1=1; evidence recall@1=0.0; evidence MRR=0.0.
- Q004: document hit@1=1; evidence recall@1=0.0; evidence MRR=0.125.
- Q005: document hit@1=1; evidence recall@1=0.0; evidence MRR=0.0.
- Q007: document hit@1=1; evidence recall@1=0.0; evidence MRR=0.0.
- Q009: document hit@1=1; evidence recall@1=0.0; evidence MRR=0.3333333333333333.
- Q011: document hit@1=1; evidence recall@1=0.0; evidence MRR=0.3333333333333333.
- Q014: document hit@1=1; evidence recall@1=0.0; evidence MRR=0.5.
- Q015: document hit@1=1; evidence recall@1=0.0; evidence MRR=0.5.
- Q016: document hit@1=1; evidence recall@1=0.0; evidence MRR=0.0.
- Q017: document hit@1=1; evidence recall@1=0.0; evidence MRR=0.0.
- Q018: document hit@1=1; evidence recall@1=0.0; evidence MRR=0.3333333333333333.
- Q019: document hit@1=1; evidence recall@1=0.0; evidence MRR=0.25.
- Q020: document hit@1=1; evidence recall@1=0.0; evidence MRR=0.16666666666666666.
- Q021: document hit@1=1; evidence recall@1=0.0; evidence MRR=0.2.
- Q023: document hit@1=1; evidence recall@1=0.5; evidence MRR=1.0.
- Q024: document hit@1=1; evidence recall@1=0.0; evidence MRR=0.2.
- Q025: document hit@1=1; evidence recall@1=0.0; evidence MRR=0.5.
- Q026: document hit@1=1; evidence recall@1=0.0; evidence MRR=0.0.
- Q027: document hit@1=1; evidence recall@1=0.5; evidence MRR=1.0.
- Q028: document hit@1=1; evidence recall@1=0.0; evidence MRR=0.0.
- Q029: document hit@1=1; evidence recall@1=0.0; evidence MRR=0.5.
- Q030: document hit@1=1; evidence recall@1=0.5; evidence MRR=1.0.
- Q031: document hit@1=1; evidence recall@1=0.0; evidence MRR=0.2.
- Q032: document hit@1=1; evidence recall@1=0.0; evidence MRR=0.5.
- Q033: document hit@1=1; evidence recall@1=0.5; evidence MRR=1.0.
- Q034: document hit@1=1; evidence recall@1=0.5; evidence MRR=1.0.
- E001: document hit@1=1; evidence recall@1=0.0; evidence MRR=0.1111111111111111.
- V001: document hit@1=1; evidence recall@1=0.0; evidence MRR=0.3333333333333333.

## Metadata mismatch observations
- wrong_insurer: 0 mismatched hits / 380 comparable hits.
- wrong_product: 0 mismatched hits / 250 comparable hits.
- wrong_version: 0 mismatched hits / 180 comparable hits.
