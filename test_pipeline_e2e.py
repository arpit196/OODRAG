"""Real-retrieval smoke test for all three RAG execution paths.

Run: python test_pipeline_e2e.py
Optional: --generator openai to exercise the configured LLM backend.
These known examples verify wiring, not generalization to unseen queries.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent


class CountingGenerator:
    """Delegate to the real generator while counting top-level calls."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.backend, name)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self.calls += 1
        return self.backend.generate(prompt, **kwargs)


def main() -> int:
    # Refusal messages and corpus text contain Unicode on Windows consoles.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", choices=("extractive", "openai"), default="extractive")
    parser.add_argument("--index", type=Path, default=ROOT / "chroma_index")
    parser.add_argument("--ood-reference", type=Path, default=ROOT / "data/ood_reference5.npz")
    parser.add_argument("--grounding-model", type=Path, default=ROOT / "data/grounding.joblib")
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--local-runtime", action="store_true",
                        help="Use dependencies and cached models in .grounding-runtime")
    args = parser.parse_args()

    if args.local_runtime:
        runtime = ROOT / ".grounding-runtime"
        if not runtime.is_dir():
            parser.error(".grounding-runtime is missing; install project dependencies instead")
        sys.path.insert(0, str(runtime))
        os.environ["HF_HOME"] = str(runtime / "huggingface")
        os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    os.environ.setdefault("OMP_NUM_THREADS", "2")

    # Import after selecting the optional local dependency runtime.
    from src.agent import ThreeStageRAGPipeline
    from src.generator import load_generator

    generator = CountingGenerator(load_generator(args.generator))
    pipeline = ThreeStageRAGPipeline(
        index_dir=str(args.index),
        ood_reference_path=str(args.ood_reference),
        grounding_model_path=str(args.grounding_model),
        grounding_threshold=args.threshold,
        generator=generator,
    )
    rows = {
        row["id"]: row
        for row in (json.loads(line) for line in
                    (ROOT / "data/grounding.training.jsonl").read_text(encoding="utf-8").splitlines())
    }
    cases = (
        ("Far-OOD", "FOOD001", ("stage1",), 0),
        ("Near-OOD", "NOOD001", ("stage1", "stage2"), 0),
        ("ID", "ID:2207.03061v1.txt", ("stage1", "stage2", "stage3"), 1),
    )
    failures = []
    for name, query_id, expected_path, expected_calls in cases:
        generator.calls = 0
        query = rows[query_id]["query"]
        response = pipeline.run(query)
        debug = response.debug_metadata
        grounding = debug["grounding_metrics"]
        problems = []
        if response.execution_path != expected_path:
            problems.append(f"expected path {expected_path}, got {response.execution_path}")
        if generator.calls != expected_calls:
            problems.append(f"expected {expected_calls} generation calls, got {generator.calls}")
        if debug["repro"]["generation_error"] is not None:
            problems.append(f"generation fell back: {debug['repro']['generation_error']}")
        if not response.response_text.strip():
            problems.append("empty response")
        if expected_calls and len(debug["contexts"]) > 3:
            problems.append("more than three chunks reached generation")
        if name == "Near-OOD" and response.response_text != "No relevant context found.":
            problems.append("expected Stage 2 grounding refusal")

        print(f"\n{'FAIL' if problems else 'PASS'}: {name} ({query_id})")
        print(f"Query: {query}")
        print(f"Path: {' -> '.join(response.execution_path)}")
        if "probability" in grounding:
            print(f"Grounding probability: {grounding['probability']:.4f}; threshold: {grounding['threshold']}")
        print(f"Generation calls: {generator.calls}")
        print(f"Answer: {response.response_text}")
        for problem in problems:
            print(f"  {problem}")
        failures.extend(problems)

    print(f"\n{'FAILED' if failures else 'All three end-to-end paths passed.'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
