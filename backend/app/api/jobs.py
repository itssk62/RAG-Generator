from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_container
from app.api.schemas import JobOut
from app.container import Container

router = APIRouter(tags=["jobs"])


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str, c: Container = Depends(get_container)) -> JobOut:
    return JobOut.of(c.repo.get_job(job_id))


@router.get("/kbs/{kb_id}/jobs", response_model=list[JobOut])
def list_jobs(kb_id: str, c: Container = Depends(get_container)) -> list[JobOut]:
    c.repo.get_kb(kb_id)
    return [JobOut.of(j) for j in c.repo.list_jobs(kb_id)]
