"""
ragas_eval.py
--------------
Wires RAGAS into the actual agent pipeline (not a synthetic test harness) —
every question is run through the real RAGAgent, and RAGAS scores the real
retrieved contexts and real generated answer.

RAGAS's API has changed shape more than once (HuggingFace Dataset -> its own
EvaluationDataset/SingleTurnSample objects; metrics went from pre-built
objects to classes you instantiate with an LLM). This targets the current
object-based API. If imports fail on your installed version, check
docs.ragas.io/en/stable/howtos/migrations/ — the WIRING below (what goes into
each sample, why) doesn't change even if a class name does.

What each metric actually tells you, and what it needs:

  Faithfulness       : does the answer avoid claims unsupported by the
                       retrieved context? Needs: query, answer, contexts.
                       This is the rigorous version of your own
                       estimate_grounding_overlap() heuristic in
                       guardrails.py — worth comparing the two directly.
  AnswerRelevancy     : does the answer actually address the question (not
                       evasive/off-topic)? Needs: query, answer, an
                       embedding model (it embeds a few reverse-engineered
                       questions from the answer and compares to the real
                       query).
  ContextPrecision    : are the genuinely relevant chunks ranked near the
                       top of what was retrieved? Needs: query, contexts,
                       and — for the reference-based variant — a
                       ground-truth reference answer.
  ContextRecall       : did retrieval surface everything needed to answer
                       fully? This is the ONE metric that needs a real
                       reference answer — it is not fully ground-truth-free
                       despite RAGAS's reputation for being so.

Usage:
    pip install ragas langchain-openai
    export OPENAI_API_KEY=...
    python ragas_eval.py --index ./chroma_index --ood-reference ./ood_reference.npz
"""

import argparse
from typing import Dict, List, Optional
from pathlib import Path
import sys
# Add project root directory to sys.path so 'src' can be imported cleanly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agent import RAGAgent, StandardRAGAgent
from eval_retrieval import IN_DOMAIN_QUERIES
from src.generator import load_generator

import sys
import langchain_google_vertexai
sys.modules['langchain_community.chat_models.vertexai'] = langchain_google_vertexai


# Extend this with real reference answers where you have them (needed only
# for ContextRecall / reference-based ContextPrecision). Queries without a
# reference here still get scored on Faithfulness and AnswerRelevancy,
# which don't need one.
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
    """Runs the REAL pipeline (router -> retrieval -> OOD -> generation) for
    each query and keeps every field RAGAS needs, plus the agent's own
    action/confidence so we can cross-check the two afterward."""
    records = []
    for q in queries:
        result = agent.answer(q)
        print(f"Query: {q} | Action: {result['action']} | Contexts retrieved: {len(result.get('contexts', []))}")
        records.append({
            "query": q,
            "action": result["action"],
            "answer": result["answer"],
            "contexts": result.get("contexts", []),
            "query_metrics": result.get("query_metrics", {}),
            "reference": REFERENCE_ANSWERS.get(q),
        })
    return records


def build_ragas_dataset(records: List[Dict]):
    """Only 'generate'/'hedge' turns have real contexts to score — chitchat
    and abstain turns produce no retrieved evidence, so scoring their
    (non-existent) grounding would be meaningless, not just uninformative."""
    from ragas import EvaluationDataset, SingleTurnSample

    samples = []
    scoreable = [r for r in records if r["action"] in ("generate", "hedge") and r["contexts"]]
    for r in scoreable:
        samples.append(SingleTurnSample(
            user_input=r["query"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference = r["reference"] if r["reference"] is not None else "",  # None is fine for metrics that don't need it
        ))
    skipped = len(records) - len(scoreable)
    if skipped:
        print(f"Skipped {skipped} chitchat/abstain/empty-context turns (nothing to score).")
    return EvaluationDataset(samples=samples), scoreable


def get_metrics(has_any_reference: bool):
    """Instantiates metrics with an evaluator LLM/embeddings, per the current
    (v0.2+) RAGAS pattern. Swap ChatOpenAI/OpenAIEmbeddings for whatever
    provider you actually have a key for — RAGAS accepts any LangChain
    chat model / embeddings via these wrappers, not just OpenAI."""
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

    # Context precision/recall class names have shifted between RAGAS
    # versions (ContextPrecision vs LLMContextPrecisionWithoutReference,
    # etc.) — try the current names, fall back gracefully rather than
    # crashing the whole run over one metric.
    try:
        from ragas.metrics import LLMContextPrecisionWithoutReference
        metrics.append(LLMContextPrecisionWithoutReference(llm=evaluator_llm))
    except ImportError:
        print("Skipping context precision — class name not found in this RAGAS version; "
              "check docs.ragas.io for the current name and add it back in.")

    if has_any_reference:
        try:
            from ragas.metrics import LLMContextRecall
            metrics.append(LLMContextRecall(llm=evaluator_llm))
        except ImportError:
            print("Skipping context recall — class name not found in this RAGAS version.")
    else:
        print("No reference answers provided in REFERENCE_ANSWERS — skipping ContextRecall "
              "(it's the one metric that genuinely needs ground truth, unlike the others).")

    return metrics


def summarize_by_action(results_df, scoreable: List[Dict]):
    """The most useful cross-check available here: does RAGAS — an
    independent, externally-validated metric — actually agree that 'hedge'
    answers are less grounded than 'generate' answers? If your OOD-based
    action decision is doing its job, this should show a visible gap."""
    results_df = results_df.copy()
    results_df["action"] = [r["action"] for r in scoreable]

    print("\n--- RAGAS scores by agent action (cross-checking your OOD confidence framework) ---")
    metric_cols = [c for c in results_df.columns if c not in ("user_input", "response",
                                                                "retrieved_contexts", "reference", "action")]
    grouped = results_df.groupby("action")[metric_cols].mean()
    print(grouped.to_string())
    print("\nExpectation: 'hedge' rows should score visibly lower on faithfulness/context")
    print("precision than 'generate' rows. If they don't, either your action thresholds need")
    print("retuning, or these queries weren't hard enough to actually separate the two cases —")
    print("check that your query set includes real near-OOD-hard examples, not just easy ones.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str, default="./chroma_index")
    parser.add_argument("--ood-reference", type=str, default="./ood_reference.npz")
    parser.add_argument("--agent-type",type=str,default="ood")
    parser.add_argument("--generator", type=str, default="openai",
                        help="Use a real generator, not 'extractive' — RAGAS scores actual "
                             "generated text, and the extractive fallback isn't representative")
    args = parser.parse_args()

    gen = load_generator(args.generator)

    if args.agent_type == "ood":
        agent = RAGAgent(index_dir=args.index, ood_reference_path=args.ood_reference, generator=gen)
    else:
        agent = StandardRAGAgent(index_dir=args.index, ood_reference_path=args.ood_reference, generator=gen)

    queries = IN_DOMAIN_QUERIES + list(REFERENCE_ANSWERS.keys())
    queries = list(dict.fromkeys(queries))  # de-dupe, preserve order

    print(f"Running {len(queries)} queries through the live agent...")
    records = run_agent_over_queries(agent, queries)

    scoreable_records = [
        r for r in records 
        if r["action"] in ("generate", "hedge") and r["contexts"]
    ]

    # 2. Track Abstain Accuracy / Refusal Rate as a separate business metric
    total_queries = len(records)
    abstained_queries = [r for r in records if r["action"] == "abstain"]

    print(f"System Refusal Rate: {len(abstained_queries) / total_queries * 100:.1f}%")

    dataset, scoreable = build_ragas_dataset(scoreable_records)
    if not scoreable:
        print("Nothing to score — every query hit chitchat/abstain, or had empty contexts. "
              "Check that IN_DOMAIN_QUERIES actually triggers retrieval for your corpus.")
        return

    has_reference = any(r["reference"] for r in scoreable)
    metrics = get_metrics(has_any_reference=has_reference)

    from ragas import evaluate
    from ragas.run_config import RunConfig

    # Conservative concurrency/timeout for a first run — RAGAS calls an LLM
    # judge per metric per sample, which adds up fast and costs real money.
    run_config = RunConfig(max_workers=2, timeout=240,max_retries=5,max_wait=60)
    result = evaluate(dataset=dataset, metrics=metrics, run_config=run_config)

    df = result.to_pandas()
    print("\n--- Per-query RAGAS scores ---")
    print(df.to_string())

    summarize_by_action(df, scoreable)

    out_path = "./ragas_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()
