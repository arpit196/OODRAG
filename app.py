"""FastAPI adapter for the OOD-aware RAG agent."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Callable, Mapping

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from agent import RAGAgent
from caching import AnswerCache, AnswerRequest, CacheKeyFactory, CacheSettings, ConversationTurn, InMemoryTTLCache
from generator import Generator, load_generator
from observability import JsonlTelemetryStore, monitoring_summary, request_event


@dataclass(frozen=True, slots=True)
class AppSettings:
    """Runtime configuration supplied through ``RAG_*`` environment variables."""

    index_dir: str = "./chroma_index"
    ood_reference_path: str = "./ood_reference2.npz"
    generator_kind: str = "openai"
    frontend_path: Path = Path("index.html")
    telemetry_path: Path = Path("telemetry.jsonl")
    cache: CacheSettings = field(default_factory=CacheSettings)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "AppSettings":
        """Create settings without making import-time environment decisions."""
        values = os.environ if env is None else env
        return cls(
            index_dir=values.get("RAG_INDEX_DIR", "./chroma_index"),
            ood_reference_path=values.get("RAG_OOD_REFERENCE_PATH", "./ood_reference2.npz"),
            generator_kind=values.get("RAG_GENERATOR", "openai"),
            frontend_path=Path(values.get("RAG_FRONTEND_PATH", "index.html")),
            telemetry_path=Path(values.get("RAG_TELEMETRY_PATH", "telemetry.jsonl")),
            cache=CacheSettings.from_env(values),
        )


class Turn(BaseModel):
    """Validated API representation of one preceding conversation message."""

    role: str = Field(min_length=1, max_length=32)
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    """Inputs that may affect a RAG response."""

    query: str = Field(min_length=1)
    history: list[Turn] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=20)


AgentFactory = Callable[[AppSettings], RAGAgent]


def build_agent(settings: AppSettings) -> RAGAgent:
    """Composition root for production dependencies."""
    generator: Generator = load_generator(settings.generator_kind)
    return RAGAgent(
        index_dir=settings.index_dir,
        ood_reference_path=settings.ood_reference_path,
        generator=generator,
    )


def create_app(settings: AppSettings | None = None, agent_factory: AgentFactory = build_agent) -> FastAPI:
    """Create an application with injectable configuration and dependencies."""
    resolved_settings = settings or AppSettings.from_env()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.agent = agent_factory(resolved_settings)
        application.state.answer_cache = AnswerCache(
            backend=InMemoryTTLCache(resolved_settings.cache),
            key_factory=CacheKeyFactory(resolved_settings.cache.namespace),
        )
        application.state.telemetry_store = JsonlTelemetryStore(resolved_settings.telemetry_path)
        yield

    service = FastAPI(title="OOD-Aware RAG API", lifespan=lifespan)
    service.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @service.post("/api/chat")
    async def chat_endpoint(payload: ChatRequest, request: Request) -> dict[str, object]:
        """Return a cache-aware answer without blocking the event loop."""
        agent: RAGAgent | None = getattr(request.app.state, "agent", None)
        cache: AnswerCache | None = getattr(request.app.state, "answer_cache", None)
        if agent is None or cache is None:
            raise HTTPException(status_code=503, detail="RAG service is still starting")
        query = payload.query.strip()
        if not query:
            raise HTTPException(status_code=422, detail="Query string cannot be blank")
        answer_request = AnswerRequest(
            query=query,
            history=tuple(ConversationTurn(turn.role, turn.content) for turn in payload.history),
            top_k=payload.top_k,
        )
        result = await run_in_threadpool(cache.answer, agent, answer_request)
        telemetry_store: JsonlTelemetryStore = request.app.state.telemetry_store
        telemetry_store.append(request_event(result.payload, cache_hit=result.cache_hit))
        return {**result.payload, "cached": result.cache_hit}

    @service.get("/", response_class=HTMLResponse)
    async def serve_frontend() -> HTMLResponse:
        """Serve the static local demonstration UI."""
        try:
            return HTMLResponse(resolved_settings.frontend_path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail="Frontend file is not configured") from error

    @service.get("/api/monitoring/summary")
    async def monitoring_endpoint(request: Request, limit: int = 500) -> dict[str, object]:
        """Return a bounded aggregate of live telemetry for the monitoring UI."""
        telemetry_store: JsonlTelemetryStore = request.app.state.telemetry_store
        return monitoring_summary(telemetry_store.read_recent(max(1, min(limit, 10_000))))

    @service.get("/monitoring", response_class=HTMLResponse)
    async def monitoring_dashboard() -> HTMLResponse:
        """Serve the lightweight operational dashboard."""
        dashboard_path = resolved_settings.frontend_path.with_name("monitoring.html")
        try:
            return HTMLResponse(dashboard_path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise HTTPException(status_code=404, detail="Monitoring dashboard is not configured") from error

    return service


app = create_app()
