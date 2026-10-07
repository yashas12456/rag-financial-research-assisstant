"""Document upload / management endpoints."""
from uuid import UUID

from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.orm import Session

from app.core.errors import DocumentNotFoundError
from app.db.session import get_db
from app.models.models import Document
from app.schemas.schemas import (
    DocumentListResponse,
    DocumentResponse,
    DocumentUploadResponse,
    IndexDocumentResponse,
)
from app.services.ingestion_service import delete_document, index_document, upload_document

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse, status_code=201)
async def upload(file: UploadFile = File(...), db: Session = Depends(get_db)):
    file_bytes = await file.read()
    document = upload_document(db, file.filename, file_bytes)
    return DocumentUploadResponse(
        document=DocumentResponse.model_validate(document),
        message="Document uploaded. Call POST /api/documents/{id}/index to process it.",
    )


@router.get("", response_model=DocumentListResponse)
def list_documents(db: Session = Depends(get_db)):
    docs = db.query(Document).order_by(Document.created_at.desc()).all()
    return DocumentListResponse(
        documents=[DocumentResponse.model_validate(d) for d in docs], total=len(docs)
    )


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(document_id: UUID, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise DocumentNotFoundError(f"Document {document_id} not found.")
    return DocumentResponse.model_validate(doc)


@router.post("/{document_id}/index", response_model=IndexDocumentResponse)
def index(document_id: UUID, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise DocumentNotFoundError(f"Document {document_id} not found.")
    import os
    from app.core.config import get_settings
    settings = get_settings()
    # Re-read the stored file from disk (uploaded in the /upload step).
    matches = [
        f for f in os.listdir(settings.UPLOAD_DIR) if f.endswith(os.path.basename(doc.filename))
    ]
    if not matches:
        raise DocumentNotFoundError("Original uploaded file could not be located on disk.")
    with open(os.path.join(settings.UPLOAD_DIR, matches[0]), "rb") as f:
        file_bytes = f.read()

    num_chunks = index_document(db, doc, file_bytes)
    return IndexDocumentResponse(document_id=doc.id, num_chunks=num_chunks, status=doc.status.value)


@router.delete("/{document_id}", status_code=204)
def delete(document_id: UUID, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise DocumentNotFoundError(f"Document {document_id} not found.")
    delete_document(db, doc)
