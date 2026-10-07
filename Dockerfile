FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RAG_INDEX_DIR=/app/chroma_index \
    RAG_OOD_REFERENCE_PATH=/app/data/ood_reference5.npz \
    RAG_FRONTEND_PATH=/app/ui/index.html \
    RAG_AGENT_TYPE=ood \
    RAG_GENERATOR=openai \
    RAG_ENVIRONMENT=production \
    RAG_EMBED_MODEL=/app/models/embedder \
    RAG_CROSS_MODEL=/app/models/reranker \
    HF_HOME=/tmp/huggingface \
    XDG_CACHE_HOME=/tmp/.cache \
    TMPDIR=/tmp

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY src ./src
COPY ui ./ui
COPY chroma_index ./chroma_index
COPY data ./data

RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2').save('/app/models/embedder'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2').save('/app/models/reranker')"

CMD ["uvicorn", "ui.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]