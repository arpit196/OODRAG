"""Run corpus-grounded boundary cases against guarded and standard insurance RAG.

--list is offline. Other modes make paid API calls. Action agreement is not a
hallucination score: review baseline answer text against actual supplied evidence.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time
import uuid

ROOT = Path(__file__).resolve().parent


def load_cases(path: Path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    ids = set()
    for row in rows:
        if not row.get("id") or row["id"] in ids or not row.get("query", "").strip():
            raise ValueError("Cases require unique IDs and nonempty queries")
        ids.add(row["id"])
        if row.get("domain_label") not in {"id", "near_ood", "mixed"}:
            raise ValueError("Invalid domain label")
        if not row.get("expected_actions") or not set(row["expected_actions"]) <= {"abstain", "clarify", "generate"}:
            raise ValueError("Invalid expected actions")
        if not isinstance(row.get("customer_context"), dict) or not row.get("source_refs"):
            raise ValueError("Cases require explicit context and source anchors")
    return rows


def validate_source_anchors(cases):
    """Check that evidence anchors really occur in the documents used to author cases."""
    cache = {}
    for case in cases:
        for ref in case["source_refs"]:
            path = (ROOT / ref["path"]).resolve()
            if not path.is_relative_to(ROOT.resolve()):
                raise ValueError("Source path leaves the project")
            if path not in cache:
                cache[path] = " ".join(path.read_text(encoding="utf-8-sig").casefold().split())
            if " ".join(ref["anchor"].casefold().split()) not in cache[path]:
                raise ValueError(f"Source anchor missing for {case['id']}: {ref['anchor']}")


def result_is_error(result):
    return (bool(result.get("evaluation_error")) or bool(result.get("repro", {}).get("generation_error"))
            or bool(result.get("decision_support", {}).get("technical_failure"))
            or result.get("reason") in {"stage2_error", "generation_error"})


def source_anchor_checks(case, result):
    """Expected source anchors reveal retrieval misses; not a semantic answer grade."""
    normalize = lambda text: " ".join(text.casefold().split())
    chunks = result.get("evidence_chunks", [])
    return [{"doc_id": ref["doc_id"], "anchor": ref["anchor"],
             "retrieved": any(c["metadata"].get("doc_id") == ref["doc_id"]
                              and normalize(ref["anchor"]) in normalize(c["text"]) for c in chunks),
             "matching_chunks": [{"id": c["id"], "page": c["metadata"].get("page")}
                 for c in chunks if c["metadata"].get("doc_id") == ref["doc_id"]
                 and normalize(ref["anchor"]) in normalize(c["text"])]} for ref in case["source_refs"]]


def evaluate_one(agent, case, *, stage2_only=False, evidence_chunks=None):
    started = time.perf_counter()
    try:
        if stage2_only:
            from src.guardrails import insurance_gate_response
            gate = agent.evidence_pipeline.invoke(case["query"], case["customer_context"])
            report = gate["assessment"].model_dump() if gate["assessment"] else None
            result = {"query": case["query"], "action": gate["action"], "reason": gate["reason"],
                      "answer": ("[Diagnostic: Stage 2 permits generation; no answer generated]"
                                 if gate["action"] == "generate" else insurance_gate_response(gate["reason"], report)),
                      "execution_path": ["stage2_diagnostic"], "query_metrics": {}, "citations": [],
                      "contexts": [c["text"] for c in gate["chunks"]], "evidence_chunks": gate["chunks"],
                      "grounding_metrics": {"enabled": True, "assessment": report,
                                            "assessment_usage": gate["assessment_usage"],
                                            "error_type": gate.get("error_type"),
                                            "error_code": gate.get("error_code"),
                                            "error_details": gate.get("error_details"),
                                            "assessment_output_token_limit": gate.get("assessment_output_token_limit"),
                                            "steps": gate["steps"]},
                      "decision_support": gate.get("decision_support", {}),
                      "customer_policy_record": gate.get("customer_policy_record"),
                      "timing_metrics": {"stage2_steps": gate["timing_metrics"]}}
        else:
            kwargs = {"customer_context": case["customer_context"], "include_evidence": True}
            if evidence_chunks is not None:
                kwargs["evidence_chunks"] = evidence_chunks
            result = agent.answer(case["query"], **kwargs)
    except Exception as exc:
        result = {"query": case["query"], "action": "error", "answer": "",
                  "evaluation_error": type(exc).__name__, "contexts": [], "citations": []}
    result["evaluation_wall_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return result


def make_record(case, variant, repeat, result):
    error = result_is_error(result)
    return {"case_id": case["id"], "domain_label": case["domain_label"], "family": case["family"],
            "variant": variant, "repeat": repeat, "query": case["query"],
            "customer_context": case["customer_context"], "expected_actions": case["expected_actions"],
            "expected_reason": case["expected_reason"], "source_refs": case["source_refs"],
            "failure_risk": case["failure_risk"], "review_rule": case["review_rule"],
            "status": "error" if error else "completed",
            "guarded_action_matches": (not error and result["action"] in case["expected_actions"])
                                      if variant.startswith("guarded") else None,
            "source_anchor_checks": source_anchor_checks(case, result),
            "review_verdict": "unreviewed", "review_notes": "", "result": result}


def summarize(records):
    summary = {"note": "Action agreement checks control flow only. Hallucination requires human review.",
               "human_reviewed_count": 0, "variants": {}}
    for variant in sorted({r["variant"] for r in records}):
        rows = [r for r in records if r["variant"] == variant]
        complete = [r for r in rows if r["status"] == "completed"]
        entry = {"runs": len(rows), "completed": len(complete), "errors": len(rows) - len(complete),
                 "median_request_ms": round(statistics.median(r["result"]["evaluation_wall_ms"] for r in complete), 3)
                                      if complete else None,
                 "stage2_exercised": sum("stage2" in r["result"].get("execution_path", [])
                                         or "stage2_diagnostic" in r["result"].get("execution_path", []) for r in complete)}
        if variant.startswith("guarded"):
            entry["action_agreement_rate"] = sum(r["guarded_action_matches"] for r in complete) / len(complete) if complete else None
            entry["stage1_rejections"] = sum(r["result"].get("reason") == "far_ood" for r in complete)
            entry["by_domain_label"] = {}
            for label in sorted({r["domain_label"] for r in rows}):
                selected = [r for r in complete if r["domain_label"] == label]
                entry["by_domain_label"][label] = {"completed": len(selected),
                    "action_matches": sum(r["guarded_action_matches"] for r in selected)}
        summary["variants"][variant] = entry
    return summary


def write_review_csv(path, records):
    fields = ["case_id", "variant", "repeat", "domain_label", "query", "expected_actions", "actual_action",
              "reason", "status", "guarded_action_matches", "wall_ms", "answer", "citations",
              "source_anchor_checks", "retrieval_diagnostics", "question_parts", "customer_schedule_evidence",
              "failure_risk", "review_rule", "review_verdict", "review_notes"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            result = row["result"]
            writer.writerow({**{key: row[key] for key in ("case_id", "variant", "repeat", "domain_label", "query",
                                "status", "guarded_action_matches", "failure_risk", "review_rule", "review_verdict", "review_notes")},
                             "expected_actions": "/".join(row["expected_actions"]), "actual_action": result["action"],
                             "reason": result.get("reason", ""), "wall_ms": result["evaluation_wall_ms"],
                             "answer": result["answer"], "citations": json.dumps(result.get("citations", []), ensure_ascii=False),
                             "source_anchor_checks": json.dumps(row["source_anchor_checks"], ensure_ascii=False),
                             **{key: json.dumps(result.get("decision_support", {}).get(key), ensure_ascii=False)
                                for key in ("retrieval_diagnostics", "question_parts", "customer_schedule_evidence")}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "data/insurance_boundary_cases.jsonl")
    parser.add_argument("--mode", choices=("both", "guarded", "standard"), default="both")
    parser.add_argument("--index", default=str(ROOT / "insurance_chroma_index_v3"))
    parser.add_argument("--collection", default="insurance_chunks")
    parser.add_argument("--ood-reference", default=str(ROOT / "data/home_insurance_ood.npz"))
    parser.add_argument("--insurance-scope", default=str(ROOT / "data/home_insurance_scope.json"))
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--baseline-evidence", choices=("top_k", "matched"), default="top_k",
                        help="matched supplies guarded evidence when available; otherwise ordinary retrieval")
    parser.add_argument("--stage2-only", action="store_true", help="Diagnostic: bypass Stage 1; no guarded answer generation")
    parser.add_argument("--case-id", action="append", help="Select one or more case IDs")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--list", action="store_true", help="Validate/list cases without loading models or calling APIs")
    args = parser.parse_args()
    cases = load_cases(args.cases)
    if args.case_id:
        unknown = set(args.case_id) - {c["id"] for c in cases}
        if unknown:
            parser.error(f"Unknown case IDs: {sorted(unknown)}")
        cases = [case for case in cases if case["id"] in args.case_id]
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        cases = cases[:args.limit]
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    validate_source_anchors(cases)
    if args.list:
        for case in cases:
            print(f"{case['id']} [{case['domain_label']}] -> {'/'.join(case['expected_actions'])}: {case['query']}")
        print(f"Validated {len(cases)} cases and their local source anchors. No API calls made.")
        return 0
    if args.baseline_evidence == "matched" and args.mode != "both":
        parser.error("--baseline-evidence matched requires --mode both")
    if args.stage2_only and args.mode == "standard":
        parser.error("--stage2-only requires a guarded run")

    # Heavy imports only for actual runs. Both variants share one index/embedder.
    from src.agent import InsuranceRAGAgent, StandardInsuranceRAGAgent, load_insurance_retriever
    from src.generator import load_generator
    scope = json.loads(Path(args.insurance_scope).read_text(encoding="utf-8-sig"))
    output = args.output_dir or ROOT / "benchmark_results" / (
        "insurance_boundary_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6])
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory is not empty; choose a new directory to preserve previous results")
    output.mkdir(parents=True, exist_ok=True)
    initialization_started = time.perf_counter()
    retriever = load_insurance_retriever(args.index, scope, args.collection)
    common = dict(index_dir=args.index, insurance_scope_path=args.insurance_scope,
                  collection_name=args.collection, retriever=retriever)
    guarded = (InsuranceRAGAgent(**common, ood_reference_path=args.ood_reference,
                assessment_model=args.model, generator=load_generator("openai", model=args.model))
               if args.mode != "standard" else None)
    standard = (StandardInsuranceRAGAgent(**common, generator=load_generator("openai", model=args.model))
                if args.mode != "guarded" else None)
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "model": args.model,
                "index": args.index, "collection": args.collection, "corpus_fingerprint": scope["corpus_fingerprint"],
                "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
                "case_ids": [c["id"] for c in cases], "mode": args.mode, "repeats": args.repeats,
                "baseline_evidence": args.baseline_evidence, "stage2_only": args.stage2_only,
                "initialization_ms": round((time.perf_counter() - initialization_started) * 1000, 3),
                "comparison_note": "Top-k baseline vs full gated pipeline; matched mode controls evidence only when available. Same model and grounded generation instructions; no forced hallucination."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    records = []
    with (output / "results.jsonl").open("w", encoding="utf-8") as handle:
        for repeat in range(1, args.repeats + 1):
            for case in cases:
                guarded_result = None
                if guarded:
                    guarded_result = evaluate_one(guarded, case, stage2_only=args.stage2_only)
                    row = make_record(case, "guarded_stage2_only" if args.stage2_only else "guarded", repeat, guarded_result)
                    records.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                    handle.flush()
                    print(f"{case['id']} guarded: {guarded_result['action']} [{row['status']}] {guarded_result['evaluation_wall_ms']:.0f} ms")
                if standard:
                    supplied = (guarded_result.get("evidence_chunks") if args.baseline_evidence == "matched" and guarded_result else None)
                    result = evaluate_one(standard, case, evidence_chunks=supplied if supplied else None)
                    if args.baseline_evidence == "matched" and not supplied:
                        result["baseline_evidence_mode"] = "top_k_fallback_no_guarded_evidence"
                    row = make_record(case, "standard", repeat, result)
                    records.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                    handle.flush()
                    print(f"{case['id']} standard: [{row['status']}] {result['evaluation_wall_ms']:.0f} ms; review answer text")
    write_review_csv(output / "review.csv", records)
    (output / "summary.json").write_text(json.dumps(summarize(records), indent=2), encoding="utf-8")
    print(f"Results: {output}\nReview hallucination/abstention manually in review.csv against results.jsonl evidence.")
    return 1 if any(row["status"] == "error" for row in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
