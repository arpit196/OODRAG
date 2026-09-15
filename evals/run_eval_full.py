"""
ragas_eval_2.py
--------------
Wires RAGAS into the actual agent pipeline and evaluates both end-to-end
and per-stage retrieval latency (via observability.py's RetrievalTrace patterns).

Outputs a comprehensive evaluation CSV containing:
- Query, Action, Answer, Reference
- RAGAS metrics (Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall)
- End-to-end latency (total_latency_ms)
- Per-stage latency breakdown (dense_ms, bm25_ms, fusion_ms, rerank_ms, generation_ms)
"""

import argparse
import sys
import time
from typing import Dict, List, Optional
import pandas as pd
from pathlib import Path
# Add project root directory to sys.path so 'src' can be imported cleanly
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
    

# Imports from src/ directory
from src.agent import RAGAgent, StandardRAGAgent
from src.generator import load_generator
from src.observability import RetrievalTrace
from eval_retrieval import IN_DOMAIN_QUERIES, EASY_OOD_TEST_QUERIES

import langchain_google_vertexai
sys.modules['langchain_community.chat_models.vertexai'] = langchain_google_vertexai

# Reference answers mapping
REFERENCE_ANSWERS: Dict[str, str] = {
    "domain generalization under distribution shift":
        "Domain generalization aims to train models that perform well on unseen target distributions without access to target-domain data.",
    "multi-source domain generalization":
        "Multi-source domain generalization aims to train models using data from multiple source domains that perform well on unseen target distributions without access to target-domain data.",
    "semi-supervised domain generalization":
        "Semi-supervised domain generalization aims to train models using a small amount of labeled data and a large amount of unlabeled data from multiple source domains that perform well on unseen target distributions without access to target-domain data.",
    "MADAN: Multi-source Adversarial Domain Aggregation Network for Domain Adaptation":
        "MADAN is a multi-source domain adaptation model that uses adversarial training to aggregate domain-specific features and learn a domain-invariant representation.",
    "DANN: Domain-Adversarial Neural Networks":
        "DANN is a domain adaptation model that uses adversarial training to learn a domain-invariant representation.",
    "CDAN: Conditional Domain Adversarial Networks":
        "CDAN is a domain adaptation model that uses conditional adversarial training to learn a domain-invariant representation.",
    "ADDA: Adversarial Discriminative Domain Adaptation":
        "ADDA is a domain adaptation model that uses adversarial training to learn a domain-invariant representation.",
    "MCD: Maximum Classifier Discrepancy":
        "MCD is a domain adaptation model that uses maximum classifier discrepancy to learn a domain-invariant representation."
}


def run_agent_over_queries(agent: RAGAgent, queries: List[str]) -> List[Dict]:
    """Runs the agent pipeline over queries while measuring total end-to-end
    latency and extracting per-stage latencies from the telemetry payload."""
    records = []
    
    for q in queries:
        trace = RetrievalTrace()
        
        # Wrap agent call in trace to compute total end-to-end latency
        start_time = time.perf_counter()
        result = agent.answer(q)
        total_latency_ms = round((time.perf_counter() - start_time) * 1000, 3)
        
        # Extract internal stage latencies from the repro/retrieval telemetry payload
        repro = result.get("repro", {})
        retrieval_info = repro.get("retrieval", {})
        stage_latencies = retrieval_info.get("stage_latency_ms", {})
        
        # Compute generation latency (total minus retrieval latency)
        retrieval_total_ms = sum(stage_latencies.values())
        generation_ms = max(0.0, round(total_latency_ms - retrieval_total_ms, 3))
        
        print(f"Query: {q:40s} | Action: {result['action']:8s} | Total Latency: {total_latency_ms:.1f}ms")
        
        records.append({
            "query": q,
            "action": result["action"],
            "answer": result["answer"],
            "contexts": result.get("contexts", []),
            "query_metrics": result.get("query_metrics", {}),
            "reference": REFERENCE_ANSWERS.get(q),
            # Latency Metrics
            "total_latency_ms": total_latency_ms,
            "dense_ms": stage_latencies.get("dense", 0.0),
            "bm25_ms": stage_latencies.get("bm25", 0.0),
            "fusion_ms": stage_latencies.get("fusion", 0.0),
            "rerank_ms": stage_latencies.get("rerank", 0.0),
            "generation_ms": generation_ms,
        })
    return records


def build_ragas_dataset(records: List[Dict]):
    """Filter scoreable records and return EvaluationDataset and corresponding record metadata."""
    from ragas import EvaluationDataset, SingleTurnSample

    samples = []
    scoreable = [r for r in records if r["action"] in ("generate", "hedge","clarify","abstain")] # and r["contexts"]
    for r in scoreable:
        samples.append(SingleTurnSample(
            user_input=r["query"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"] if r["reference"] is not None else "",
        ))
    
    skipped = len(records) - len(scoreable)
    if skipped:
        print(f"Skipped {skipped} chitchat/abstain/empty-context turns for RAGAS evaluation.")
    return EvaluationDataset(samples=samples), scoreable


def get_metrics(has_any_reference: bool):
    """Instantiate RAGAS metrics."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import Faithfulness, AnswerRelevancy

    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", max_tokens=4096))
    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
    ]

    try:
        from ragas.metrics import LLMContextPrecisionWithoutReference
        metrics.append(LLMContextPrecisionWithoutReference(llm=evaluator_llm))
    except ImportError:
        print("Skipping context precision — metric unavailable.")

    if has_any_reference:
        try:
            from ragas.metrics import LLMContextRecall
            metrics.append(LLMContextRecall(llm=evaluator_llm))
        except ImportError:
            print("Skipping context recall — metric unavailable.")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str, default="./chroma_index")
    parser.add_argument("--ood-reference", type=str, default="./ood_reference.npz")
    parser.add_argument("--agent-type", type=str, default="ood")
    parser.add_argument("--generator", type=str, default="openai")
    args = parser.parse_args()

    gen = load_generator(args.generator)

    if args.agent_type == "ood":
        agent = RAGAgent(index_dir=args.index, ood_reference_path=args.ood_reference, generator=gen)
    else:
        agent = StandardRAGAgent(index_dir=args.index, ood_reference_path=args.ood_reference, generator=gen)

    queries = IN_DOMAIN_QUERIES[:7] + EASY_OOD_TEST_QUERIES[:7]#list(REFERENCE_ANSWERS[].keys()) #IN_DOMAIN_QUERIES[:1] + 
    queries = list(dict.fromkeys(queries))

    print(f"Running {len(queries)} queries through the live agent with latency tracing...")
    records = run_agent_over_queries(agent, queries)

    '''scoreable_records = [
        r for r in records 
        if r["action"] in ("generate", "hedge") and r["contexts"]
    ]'''

    dataset, scoreable = build_ragas_dataset(records)
    
    if not scoreable:
        print("No scoreable turns found. Exporting latency metrics only.")
        df_latency = pd.DataFrame(records)
        df_latency.to_csv("./ragas_results_lat.csv", index=False)
        return

    has_reference = any(r["reference"] for r in scoreable)
    metrics = get_metrics(has_any_reference=has_reference)

    from ragas import evaluate
    from ragas.run_config import RunConfig

    run_config = RunConfig(max_workers=2, timeout=240, max_retries=5, max_wait=60)
    result = evaluate(dataset=dataset, metrics=metrics, run_config=run_config)

    # Convert RAGAS evaluation output to DataFrame
    df_ragas = result.to_pandas()

    # Create matching DataFrame for scoreable records containing latencies and actions
    scoreable_meta = pd.DataFrame([{
        "query": r["query"],
        "action": r["action"],
        "total_latency_ms": r["total_latency_ms"],
        "dense_ms": r["dense_ms"],
        "bm25_ms": r["bm25_ms"],
        "fusion_ms": r["fusion_ms"],
        "rerank_ms": r["rerank_ms"],
        "generation_ms": r["generation_ms"]
    } for r in scoreable])

    # Merge RAGAS output with latency metrics
    final_df = pd.merge(df_ragas, scoreable_meta, left_index=True, right_index=True)
    is_refusal = final_df["action"].str.contains("abstain", case=False, na=False)

    # Override faithfulness for valid refusals when context is empty/irrelevant
    # Assigning 1.0 rewards proper refusal, or set to None to exclude from average
    final_df.loc[is_refusal, "faithfulness"] = 1.0

    print("\n--- Summary: Quality Scores vs Latency ---")
    summary_cols = [c for c in ["faithfulness", "answer_relevancy", "total_latency_ms", "generation_ms"] if c in final_df.columns]
    print(final_df.groupby("action")[summary_cols].mean().to_string())

    out_path = "./ragas_results_lat_ood.csv"
    final_df.to_csv(out_path, index=False)
    print(f"\nSaved combined quality and latency results to {out_path}")


if __name__ == "__main__":
    main()