"""Local insurance chat: uvicorn ui.insurance_app:app --host 127.0.0.1 --port 8000."""
import asyncio
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from urllib.parse import quote, urlparse

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)


def build_agent():
    import json
    import chromadb
    from dotenv import load_dotenv
    from sentence_transformers import SentenceTransformer
    from insurance_retrieval_fixed import InsuranceHybridRetriever
    from src.agent import InsuranceRAGAgent
    load_dotenv(ROOT / '.env')
    scope_path = ROOT / 'data/home_insurance_scope.json'
    scope = json.loads(scope_path.read_text(encoding='utf-8-sig'))
    index = str(ROOT / 'insurance_chroma_index_v3')
    collection = chromadb.PersistentClient(path=index).get_collection(scope['collection'])
    if any((collection.metadata or {}).get(k) != scope[k] for k in ('embedding_model', 'corpus_fingerprint')):
        raise ValueError('Index does not match insurance scope')
    model = SentenceTransformer(scope['embedding_model'], cache_folder=str(ROOT / '.benchmark-model-cache'),
                                local_files_only=os.getenv('INSURANCE_ALLOW_MODEL_DOWNLOAD') != '1')
    return InsuranceRAGAgent(index, str(ROOT / 'data/home_insurance_ood.npz'),
                            insurance_scope_path=str(scope_path),
                            retriever=InsuranceHybridRetriever(collection, model),
                            assessment_model=os.getenv('INSURANCE_MODEL', 'gpt-4o-mini'))


def document_path(meta, corpus):
    """Resolve indexed WSL/Windows paths only inside the configured corpus."""
    raw = str(meta.get('raw_path', '')).replace('\\', '/')
    if '/raw/' in raw:
        raw = 'raw/' + raw.split('/raw/', 1)[1]
    path = (corpus / raw).resolve()
    return path if path.is_relative_to(corpus.resolve()) and path.is_file() and path.suffix.lower() == '.pdf' else None


def presentation(result, documents):
    support = result.get('decision_support') or {}
    assessment = (result.get('grounding_metrics') or {}).get('assessment') or {}
    sources = []
    chunks = {c['id']: c for c in result.get('evidence_chunks', [])}
    if not support.get('technical_failure') and result.get('action') in ('generate', 'clarify'):
        for citation in result.get('citations') or []:
            chunk = chunks.get(citation['id'])
            if not chunk:
                continue
            meta = chunk.get('metadata') or {}
            doc_id = meta.get('doc_id')
            if doc_id not in documents:
                continue
            page = meta.get('page')
            page = page if isinstance(page, int) and page > 0 else None
            url = meta.get('source_url', '')
            sources.append({'chunk_id': chunk['id'], 'citation': citation.get('n'),
                            'document_id': doc_id, 'title': meta.get('title', doc_id),
                            'version': meta.get('version_label'), 'page': page, 'excerpt': chunk['text'],
                            'viewer_url': f'/api/documents/{quote(doc_id, safe="")}#page={page}' if documents[doc_id].get('path') and page else None,
                            'source_url': url if urlparse(url).scheme in ('https', 'http') else None})
    compact = {'answer': result.get('answer'), 'action': result.get('action'), 'reason': result.get('reason'),
               'technical_failure': bool(support.get('technical_failure'))}
    for field in ('missing_evidence', 'missing_customer_facts', 'conflicts'):
        compact[field] = support.get(field) or list(dict.fromkeys(x for p in assessment.get('parts', []) for x in p.get(field, [])))
    compact['parts'] = [{'question': p.get('question_part'), 'status': p.get('status')} for p in assessment.get('parts', [])]
    compact['request_latency_ms'] = (result.get('timing_metrics') or {}).get('request_total_ms')
    return {'response': compact, 'sources': sources}


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=10, ge=1, le=20)
    customer_context: dict = Field(default_factory=lambda: {'request_type': 'general', 'facts': {}})


def create_app(agent_factory=build_agent, corpus=ROOT / 'real_insurance_corpus_v2'):
    @asynccontextmanager
    async def lifespan(app):
        app.state.agent = await run_in_threadpool(agent_factory)
        app.state.lock = asyncio.Lock()  # Agent usage/verification state is mutable.
        app.state.documents = {}
        for chunk in app.state.agent.evidence_pipeline.retriever.chunks:
            meta = chunk.get('metadata') or {}
            if meta.get('doc_id'):
                app.state.documents[meta['doc_id']] = {'path': document_path(meta, Path(corpus))}
        yield

    app = FastAPI(title='Insurance evidence chat', lifespan=lifespan)

    @app.get('/')
    def home():
        return FileResponse(Path(__file__).with_name('insurance.html'))

    @app.get('/api/documents/{doc_id}')
    def document(doc_id: str):
        path = app.state.documents.get(doc_id, {}).get('path')
        if path is None:
            raise HTTPException(404, 'Document unavailable')
        return FileResponse(path, media_type='application/pdf', headers={'Content-Disposition': 'inline'})

    @app.post('/api/chat')
    async def chat(request: ChatRequest):
        if not request.query.strip():
            raise HTTPException(422, 'Query cannot be blank')
        async with app.state.lock:
            try:
                result = await run_in_threadpool(app.state.agent.answer, request.query.strip(),
                                                customer_context=request.customer_context,
                                                top_k_final=request.top_k, include_evidence=True)
            except Exception:
                logger.exception('Insurance chat request failed')
                raise HTTPException(503, 'Request failed; consult server logs')
        return presentation(result, app.state.documents)

    return app


app = create_app()
