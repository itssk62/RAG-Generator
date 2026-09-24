from __future__ import annotations

import shutil

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import ValidationError

from app.api.deps import get_container
from app.api.schemas import KbCreate, KbOut
from app.container import Container

router = APIRouter(prefix="/kbs", tags=["knowledge bases"])


@router.post("", response_model=KbOut, status_code=status.HTTP_201_CREATED)
def create_kb(body: KbCreate, c: Container = Depends(get_container)) -> KbOut:
    try:
        c.settings.kb_config(body.overrides)
    except ValidationError as e:
        raise HTTPException(422, detail={"overrides": e.errors(include_url=False, include_context=False)}) from e
    return KbOut.of(c.repo.create_kb(body.name, body.description, body.overrides))


@router.get("", response_model=list[KbOut])
def list_kbs(c: Container = Depends(get_container)) -> list[KbOut]:
    return [KbOut.of(kb) for kb in c.repo.list_kbs()]


@router.get("/{kb_id}", response_model=KbOut)
def get_kb(kb_id: str, c: Container = Depends(get_container)) -> KbOut:
    return KbOut.of(c.repo.get_kb(kb_id))


@router.delete("/{kb_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_kb(kb_id: str, c: Container = Depends(get_container)) -> None:
    c.repo.get_kb(kb_id)
    c.store.drop_collection(kb_id)
    c.repo.delete_kb(kb_id)
    shutil.rmtree(c.uploads_dir / kb_id, ignore_errors=True)
