"""
Ingestion service: orchestrates upload -> parse -> chunk -> embed -> index.
"""
import os
import uuid

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.errors import DuplicateDocumentError, FileTooLargeError, InvalidFileError
from app.core.logging import get_logger
from app.embeddings.embedder import Embedder
from app.ingestion.chunking import chunk_document
from app.ingestion.pdf_parser import compute_content_hash, extract_pdf
from app.models.models import Document, DocumentChunk, DocumentPage, DocumentStatus
from app.retrieval.bm25_retriever import BM25Index
from app.retrieval.vector_store import VectorStore

logger = get_logger(__name__)
settings = get_settings()

# Process-wide singletons: a fresh BM25 index is cheap to rebuild and the
# vector store client is a thin network wrapper -- safe to share.
_bm25_index = BM25Index()
_vector_store: VectorStore | None = None
_embedder: Embedder | None = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def get_bm25_index() -> BM25Index:
    return _bm25_index


def validate_upload(filename: str, file_bytes: bytes) -> None:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in settings.ALLOWED_UPLOAD_EXTENSIONS:
        raise InvalidFileError(f"Only {settings.ALLOWED_UPLOAD_EXTENSIONS} files are supported.")
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise FileTooLargeError(f"File exceeds the {settings.MAX_UPLOAD_SIZE_MB}MB upload limit.")


def upload_document(db: Session, filename: str, file_bytes: bytes) -> Document:
    """Validate, deduplicate, persist to disk, and create the Document row (status=uploaded)."""
    validate_upload(filename, file_bytes)

    content_hash = compute_content_hash(file_bytes)
    existing = db.query(Document).filter(Document.content_hash == content_hash).first()
    if existing:
        raise DuplicateDocumentError(
            f"This exact file was already uploaded as '{existing.filename}' "
            f"(document_id={existing.id})."
        )

    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    safe_name = f"{uuid.uuid4()}_{os.path.basename(filename)}"
    dest_path = os.path.join(settings.UPLOAD_DIR, safe_name)
    with open(dest_path, "wb") as f:
        f.write(file_bytes)

    document = Document(
        filename=filename,
        content_hash=content_hash,
        file_size_bytes=len(file_bytes),
        status=DocumentStatus.UPLOADED,
    )
    db.add(document)
    db.flush()
    logger.info("Uploaded document %s (%s)", document.id, filename)
    return document


def index_document(db: Session, document: Document, file_bytes: bytes) -> int:
    """
    Parse -> chunk -> embed -> store in Qdrant -> persist chunk rows -> rebuild BM25.

    Returns the number of chunks created. Raises AppError subclasses on failure
    and marks the document as FAILED with the error message preserved.
    """
    document.status = DocumentStatus.PROCESSING
    db.flush()

    try:
        extracted = extract_pdf(document.filename, file_bytes)
        document.num_pages = extracted.num_pages
        for page in extracted.pages:
            db.add(
                DocumentPage(
                    document_id=document.id,
                    page_number=page.page_number,
                    raw_text=page.text,
                    is_empty=page.is_empty,
                )
            )
        db.flush()

        chunks = chunk_document(extracted, str(document.id))
        if not chunks:
            raise InvalidFileError(f"No chunkable text extracted from '{document.filename}'.")

        embedder = get_embedder()
        vectors = embedder.embed_texts([c.text for c in chunks])

        payloads = [
            {
                "document_id": c.document_id,
                "filename": c.filename,
                "page_number": c.page_number,
                "section": c.section,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]
        get_vector_store().upsert_chunks(
            chunk_ids=[c.chunk_id for c in chunks], vectors=vectors, payloads=payloads
        )

        for c in chunks:
            db.add(
                DocumentChunk(
                    id=uuid.UUID(c.chunk_id),
                    document_id=document.id,
                    chunk_index=c.chunk_index,
                    page_number=c.page_number,
                    section=c.section,
                    text=c.text,
                    char_count=len(c.text),
                    vector_id=c.chunk_id,
                )
            )
        db.flush()

        rebuild_bm25_index(db)

        document.status = DocumentStatus.INDEXED
        from datetime import datetime
        document.indexed_at = datetime.utcnow()
        db.flush()
        logger.info("Indexed document %s: %d chunks", document.id, len(chunks))
        return len(chunks)

    except Exception as exc:
        document.status = DocumentStatus.FAILED
        document.error_message = str(exc)
        db.flush()
        raise


def rebuild_bm25_index(db: Session) -> None:
    """Rebuild the in-memory BM25 index from all currently-indexed chunks."""
    rows = (
        db.query(DocumentChunk.id, DocumentChunk.text, DocumentChunk.document_id)
        .join(Document)
        .filter(Document.status == DocumentStatus.INDEXED)
        .all()
    )
    text_map = {str(r.id): r.text for r in rows}
    doc_map = {str(r.id): str(r.document_id) for r in rows}
    _bm25_index.build(text_map, doc_map)


def delete_document(db: Session, document: Document) -> None:
    get_vector_store().delete_document(str(document.id))
    db.delete(document)
    db.flush()
    rebuild_bm25_index(db)
