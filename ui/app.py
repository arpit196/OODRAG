"""FastAPI adapter for the OOD-aware RAG agent."""

from __future__ import annotations

from pathlib import Path
#from mangum import Mangum
import os
import logging
import json
import time
import uuid
from dotenv import load_dotenv
#import shutil

logger = logging.getLogger("rag_api")
logger.setLevel(logging.INFO)

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Callable, Mapping
import sys
# Add project root directory to sys.path so 'src' can be imported cleanly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))



from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.agent import RAGAgent, StandardRAGAgent
from src.caching import AnswerCache, AnswerRequest, CacheKeyFactory, CacheSettings, ConversationTurn, InMemoryTTLCache
from src.generator import Generator, load_generator
from src.observability import JsonlTelemetryStore, monitoring_summary, request_event

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INDEX_DIR = str(BASE_DIR / "chroma_index")
DEFAULT_OOD_PATH = str(BASE_DIR / "data" / "ood_reference5.npz")
DEFAULT_FRONTEND_PATH = BASE_DIR / "ui" / "index.html"
#DEFAULT_TELEMETRY_PATH = BASE_DIR / "logs" / "telemetry2.jsonl"
DEFAULT_TELEMETRY_PATH = Path("/tmp/telemetry2.jsonl")

@dataclass(frozen=True, slots=True)
class AppSettings:
    """Runtime configuration supplied through ``RAG_*`` environment variables."""

    index_dir: str = DEFAULT_INDEX_DIR
    ood_reference_path: str = DEFAULT_OOD_PATH
    generator_kind: str = "openai"
    agent_type: str = "ood"
    frontend_path: Path = field(default_factory=lambda: BASE_DIR / "ui" / "index.html")
    telemetry_path: Path = Path("telemetry.jsonl")
    cache: CacheSettings = field(default_factory=CacheSettings)
    environment: str = "development"
    app_version: str = "dev"
    grounding_model_path: str | None = None
    grounding_threshold: float = 0.5
    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "AppSettings":
        """Create settings without making import-time environment decisions."""
        values = os.environ if env is None else env
        return cls(
            index_dir=values.get("RAG_INDEX_DIR", DEFAULT_INDEX_DIR),
            ood_reference_path=values.get("RAG_OOD_REFERENCE_PATH", DEFAULT_OOD_PATH),
            generator_kind=values.get("RAG_GENERATOR", "openai"),
            agent_type=values.get("RAG_AGENT_TYPE", "ood").lower(),
            frontend_path=Path(values.get("RAG_FRONTEND_PATH", str(DEFAULT_FRONTEND_PATH))),
            telemetry_path=Path(values.get("RAG_TELEMETRY_PATH", str(DEFAULT_TELEMETRY_PATH))),
            environment=os.environ.get("RAG_ENVIRONMENT", "development"),
            app_version=os.environ.get("RAG_APP_VERSION", "dev"),
            cache=CacheSettings.from_env(values),
            grounding_model_path=values.get("RAG_GROUNDING_MODEL_PATH") or None,
            grounding_threshold=float(values.get("RAG_GROUNDING_THRESHOLD", "0.5")),
        )


class Turn(BaseModel):
    """Validated API representation of one preceding conversation message."""

    role: str = Field(min_length=1, max_length=32)
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    """Inputs that may affect a RAG response."""

    query: str = Field(min_length=1,max_length=6000)
    history: list[Turn] = Field(default_factory=list,max_length=20)
    top_k: int = Field(default=5, ge=1, le=20)


AgentFactory = Callable[[AppSettings], RAGAgent]


def prepare_chroma_index(index_dir: str) -> str:
    if not os.getenv("AWS_LAMBDA_FUNCTION_NAME"):
        return index_dir

    source = Path(index_dir)
    target = Path("/tmp/chroma_index")

    if not target.exists():
        shutil.copytree(source, target)

    return str(target)

def build_agent(settings: AppSettings) -> RAGAgent:
    """Composition root for production dependencies."""
    generator: Generator = load_generator(settings.generator_kind)
    #index_dir = prepare_chroma_index(settings.index_dir)
    index_dir = settings.index_dir
    if settings.agent_type == "standard":
        return StandardRAGAgent(
            index_dir=index_dir,
            ood_reference_path=settings.ood_reference_path,
            generator=generator,
        )
    return RAGAgent(
        index_dir=index_dir,
        ood_reference_path=settings.ood_reference_path,
        generator=generator,
        grounding_model_path=settings.grounding_model_path,
        grounding_threshold=settings.grounding_threshold,
    )


def create_app(settings: AppSettings | None = None, agent_factory: AgentFactory = build_agent) -> FastAPI:
    """Create an application with injectable configuration and dependencies."""
    resolved_settings = settings or AppSettings.from_env()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        logger.info(json.dumps({
            "event": "service_starting",
            "environment": resolved_settings.environment,
            "version": resolved_settings.app_version,
        }))
        try:
            application.state.agent = agent_factory(resolved_settings)
            logger.info(json.dumps({
                "event": "agent_started",
                "agent_type": resolved_settings.agent_type,
                "generator": resolved_settings.generator_kind,
            }))
            application.state.answer_cache = AnswerCache(
                backend=InMemoryTTLCache(resolved_settings.cache),
                key_factory=CacheKeyFactory(resolved_settings.cache.namespace),
            )
            application.state.telemetry_store = JsonlTelemetryStore(resolved_settings.telemetry_path)
            logger.info(json.dumps({
                "event": "service_ready",
            }))
            yield
        except Exception:
            logger.exception("RAG service startup failed.")
            raise
        finally:
            logger.info("RAG service shutting down")

    service = FastAPI(title="OOD-Aware RAG API", lifespan=lifespan)
    service.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:9000",
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization"],
    )
    @service.get("/health")
    async def health():
        return {"status": "ok"}
    
    @service.get("/ready")
    async def ready(request: Request):
        agent = getattr(request.app.state, "agent", None)
        if agent is None:
            raise HTTPException(
                status_code=503,
                detail="Agent not ready",
            )

        return {
            "status": "ready",
            "agent_type": resolved_settings.agent_type,
        }

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
        try:
            result = await run_in_threadpool(cache.answer,agent,answer_request)
        except TimeoutError as error:
                logger.warning(
                    "RAG request timed out",
                    exc_info=error,
                )
                raise HTTPException(
                    status_code=504,
                    detail="Request timed out",
                ) from error
        except Exception as error:
                logger.exception(
                    "RAG inference failed",
                )
                raise HTTPException(
                    status_code=500,
                    detail="Unable to process request",
                ) from error
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
        #print(telemetry_store.read_recent(max(1, min(limit, 10_000))))
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
