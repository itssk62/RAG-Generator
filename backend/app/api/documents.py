from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.deps import get_container
from app.api.schemas import DocumentOut, JobOut, UploadOut
from app.container import Container
from app.storage.repository import new_id

router = APIRouter(prefix="/kbs/{kb_id}/documents", tags=["documents"])

_CHUNK = 1 << 20


@router.post("", response_model=list[UploadOut], status_code=status.HTTP_202_ACCEPTED)
def upload(kb_id: str, files: list[UploadFile] = File(...), c: Container = Depends(get_container)) -> list[UploadOut]:
    c.repo.get_kb(kb_id)
    for f in files:
        if not f.filename or not c.parsers.supports(f.filename):
            raise HTTPException(
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                f"unsupported file '{f.filename}'; supported: {', '.join(c.parsers.extensions)}",
            )
    limit = c.settings.max_upload_mb * _CHUNK
    target_dir = c.uploads_dir / kb_id
    target_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for f in files:
        doc_id = new_id()
        filename = Path(f.filename).name  # drop any client-supplied directories
        path = target_dir / f"{doc_id}{Path(filename).suffix.lower()}"
        size = _save(f, path, limit)
        doc, job = c.repo.create_document_with_job(kb_id, filename, str(path), size, doc_id)
        out.append(UploadOut(document=DocumentOut.of(doc), job=JobOut.of(job)))
    c.worker.notify()
    return out


def _save(f: UploadFile, path: Path, limit: int) -> int:
    size = 0
    with path.open("wb") as dst:
        while block := f.file.read(_CHUNK):
            size += len(block)
            if size > limit:
                dst.close()
                path.unlink(missing_ok=True)
                raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"'{f.filename}' exceeds {limit // _CHUNK} MB")
            dst.write(block)
    if size == 0:
        path.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"'{f.filename}' is empty")
    return size


@router.get("", response_model=list[DocumentOut])
def list_documents(kb_id: str, c: Container = Depends(get_container)) -> list[DocumentOut]:
    c.repo.get_kb(kb_id)
    return [DocumentOut.of(d) for d in c.repo.list_documents(kb_id)]


@router.get("/{doc_id}", response_model=DocumentOut)
def get_document(kb_id: str, doc_id: str, c: Container = Depends(get_container)) -> DocumentOut:
    return DocumentOut.of(c.repo.get_document(doc_id, kb_id))


@router.post("/{doc_id}/reindex", response_model=JobOut, status_code=status.HTTP_202_ACCEPTED)
def reindex(kb_id: str, doc_id: str, c: Container = Depends(get_container)) -> JobOut:
    c.repo.get_document(doc_id, kb_id)
    job = c.repo.enqueue_job(kb_id, doc_id)
    c.worker.notify()
    return JobOut.of(job)


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(kb_id: str, doc_id: str, c: Container = Depends(get_container)) -> None:
    doc = c.repo.get_document(doc_id, kb_id)
    c.store.delete_document(kb_id, doc_id)
    c.repo.delete_document(doc_id)
    Path(doc.path).unlink(missing_ok=True)
