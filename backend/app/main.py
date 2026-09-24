from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from app.api import documents, jobs, kbs, query
from app.api.deps import get_container
from app.config import Settings
from app.container import Container, build_container
from app.ingestion.parsers.registry import UnsupportedFormat
from app.storage.repository import Conflict, NotFound

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app(container: Container | None = None, *, start_worker: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        c = container or build_container(Settings())
        app.state.container = c
        missing = getattr(c.answerer.llm, "missing_keys", list)()
        if missing:
            logging.getLogger(__name__).warning("LLM %s: missing env var(s) %s; answers will fail", c.settings.models.llm, missing)
        if start_worker:
            c.worker.start()
        yield
        if start_worker:
            c.worker.stop()

    app = FastAPI(title="RAG Generator", version="1.0", lifespan=lifespan)
    for router in (kbs.router, documents.router, jobs.router, query.router):
        app.include_router(router)

    @app.exception_handler(NotFound)
    async def _not_found(_: Request, e: NotFound) -> JSONResponse:
        return JSONResponse({"detail": str(e)}, status_code=404)

    @app.exception_handler(Conflict)
    async def _conflict(_: Request, e: Conflict) -> JSONResponse:
        return JSONResponse({"detail": str(e)}, status_code=409)

    @app.exception_handler(UnsupportedFormat)
    async def _unsupported(_: Request, e: UnsupportedFormat) -> JSONResponse:
        return JSONResponse({"detail": str(e)}, status_code=415)

    @app.get("/health", tags=["meta"])
    def health(c: Container = Depends(get_container)) -> dict:
        qdrant_ok = c.store.healthy()
        missing = getattr(c.answerer.llm, "missing_keys", list)()
        return {
            "status": "ok" if qdrant_ok else "degraded",
            "qdrant": qdrant_ok,
            "llm": {"model": c.settings.models.llm, "missing_keys": missing},
        }

    @app.get("/config/formats", tags=["meta"])
    def formats(c: Container = Depends(get_container)) -> dict:
        return {"extensions": c.parsers.extensions, "max_upload_mb": c.settings.max_upload_mb}

    return app


app = create_app()
