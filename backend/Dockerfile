FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_XET=1 \
    RAG_CONFIG=/srv/config/settings.yaml \
    RAG__DATA_DIR=/data \
    RAG__MODELS__CACHE_DIR=/models

WORKDIR /srv/backend
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app
COPY config /srv/config
COPY scripts/download_models.py /srv/scripts/download_models.py

RUN useradd --create-home --uid 1000 rag && mkdir -p /data /models && chown rag /data /models
USER rag

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

# Models are fetched into the /models volume on first start (or provisioned with scripts/download_models.py).
CMD ["sh", "-c", "python /srv/scripts/download_models.py && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
