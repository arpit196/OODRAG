"""Interactive, cache-aware multi-turn terminal client for the RAG agent."""

from __future__ import annotations

import argparse
from typing import Sequence

from agent import RAGAgent, display_turn_result
from caching import AnswerCache, AnswerRequest, CacheKeyFactory, CacheSettings, Conversation, InMemoryTTLCache
from generator import load_generator


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line interface without performing application I/O."""
    parser = argparse.ArgumentParser(description="OOD-aware RAG terminal chat")
    parser.add_argument("--index", default="./chroma_index")
    parser.add_argument("--ood-reference", default="./ood_reference.npz")
    parser.add_argument("--generator", default="extractive")
    return parser


def run_repl(agent: RAGAgent, cache: AnswerCache, settings: CacheSettings) -> None:
    """Run a bounded multi-turn conversation with deterministic cache identity."""
    conversation = Conversation(max_turns=settings.max_history_turns)
    print("Ready. Commands: :clear, :quit, :q")
    while True:
        try:
            query = input("user> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not query:
            continue
        if query in {":quit", ":q"}:
            return
        if query == ":clear":
            conversation = Conversation(max_turns=settings.max_history_turns)
            print("Conversation history cleared.")
            continue

        request = AnswerRequest(query=query, history=conversation.snapshot(), top_k=5)
        result = cache.answer(agent, request)
        display_turn_result(result.payload)
        print(f"[cache: {'hit' if result.cache_hit else 'miss'}]")
        conversation.append_exchange(query, str(result.payload["answer"]))


def main(argv: Sequence[str] | None = None) -> None:
    """Construct terminal dependencies and start the REPL."""
    args = build_argument_parser().parse_args(argv)
    settings = CacheSettings.from_env()
    agent = RAGAgent(
        index_dir=args.index,
        ood_reference_path=args.ood_reference,
        generator=load_generator(args.generator),
    )
    cache = AnswerCache(InMemoryTTLCache(settings), CacheKeyFactory(settings.namespace))
    run_repl(agent, cache, settings)


if __name__ == "__main__":
    main()
