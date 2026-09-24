from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import iterate_in_threadpool

from app.api.deps import get_container
from app.api.schemas import QueryIn, QueryOut
from app.container import Container
from app.generation.answerer import FinalEvent, TokenEvent

log = logging.getLogger(__name__)
router = APIRouter(prefix="/kbs/{kb_id}", tags=["query"])


def failure_detail(e: Exception) -> str:
    first_line = str(e).strip().splitlines()[0][:200] if str(e).strip() else ""
    return f"answer generation failed: {type(e).__name__}" + (f": {first_line}" if first_line else "")


@router.post("/query", response_model=QueryOut)
def query(kb_id: str, body: QueryIn, c: Container = Depends(get_container)) -> QueryOut:
    cfg = c.settings.kb_config(c.repo.get_kb(kb_id).overrides)
    try:
        return QueryOut.of(c.answerer.answer(kb_id, body.question, cfg))
    except Exception as e:
        log.exception("query failed kb=%s", kb_id)
        raise HTTPException(502, failure_detail(e)) from e


@router.post("/query/stream")
async def query_stream(kb_id: str, body: QueryIn, c: Container = Depends(get_container)) -> EventSourceResponse:
    """SSE events: `token` {text}* -> `citations` {verified sources + checks} -> `done` {full QueryOut}.
    On failure a single `error` {detail} event ends the stream."""
    cfg = c.settings.kb_config(c.repo.get_kb(kb_id).overrides)

    async def events() -> AsyncIterator[dict[str, str]]:
        try:
            async for event in iterate_in_threadpool(c.answerer.stream(kb_id, body.question, cfg)):
                if isinstance(event, TokenEvent):
                    yield {"event": "token", "data": json.dumps({"text": event.text})}
                elif isinstance(event, FinalEvent):
                    out = QueryOut.of(event.answer)
                    yield {
                        "event": "citations",
                        "data": out.model_dump_json(include={"refused", "refusal_reason", "verified", "sources", "citations", "uncited_sentences"}),
                    }
                    yield {"event": "done", "data": out.model_dump_json()}
        except Exception as e:
            log.exception("streamed query failed kb=%s", kb_id)
            yield {"event": "error", "data": json.dumps({"detail": failure_detail(e)})}

    return EventSourceResponse(events())
