"""
Text chunking.

ML CONCEPT: Text chunking
WHAT: Splits each page's extracted text into overlapping, fixed-size
      windows ("chunks") that will each become a single embedding vector
      and a single BM25 document.
WHY:  Embedding models have a limited context window and produce a
      single vector per input -- embedding an entire 100-page annual
      report as one vector would blur together dozens of unrelated
      facts (revenue, risk factors, executive bios) into a near-useless
      average. Chunking lets each vector represent one coherent idea,
      so retrieval can pinpoint the exact paragraph that answers a
      question instead of the whole document.

TRADE-OFFS (see docs/ml_concepts.md for full discussion):
  - Smaller chunks -> more precise retrieval, but less surrounding
    context for the LLM to reason over, and more chunks to store/search.
  - Larger chunks -> more context per chunk, but noisier embeddings
    (multiple topics mixed together) and worse retrieval precision.
  - Overlap prevents a sentence that would answer the question from
    being split exactly at a chunk boundary and losing its meaning in
    both halves.
  - Poor chunking (e.g. splitting mid-table, mid-sentence with no
    overlap) is one of the most common causes of bad RAG answers even
    when the embedding model and LLM are both good -- garbage in,
    garbage out.
"""
import re
import uuid
from dataclasses import dataclass

from app.core.config import get_settings
from app.ingestion.pdf_parser import ExtractedDocument

settings = get_settings()

# A cheap heuristic to tag likely section headers (ALL CAPS short lines, or
# common financial-report headings). Not a full layout-analysis model --
# good enough to give chunks a helpful "section" metadata field.
_SECTION_HEADING_RE = re.compile(
    r"^(item\s+\d+[a-z]?\.?|part\s+[ivx]+|management'?s?\s+discussion|"
    r"risk\s+factors|financial\s+statements|notes\s+to.*statements|"
    r"[A-Z][A-Z\s&,.'-]{6,60})$"
)


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    section: str | None
    chunk_index: int
    text: str


def _clean_text(text: str) -> str:
    """Normalize whitespace and drop common PDF extraction artifacts."""
    text = text.replace("\x00", "")
    # Collapse runs of whitespace but keep paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _detect_section(text: str) -> str | None:
    for line in text.splitlines()[:3]:
        stripped = line.strip()
        if 3 <= len(stripped) <= 80 and _SECTION_HEADING_RE.match(stripped):
            return stripped
    return None


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    """
    Split `text` into overlapping fixed-size character windows.

    We chunk by characters (not tokens) to keep the implementation simple
    and dependency-free; CHUNK_SIZE is tuned conservatively so chunks stay
    well within any embedding model's token limit (see docs/ml_concepts.md).
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP
    if overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    text = _clean_text(text)
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    n = len(text)
    step = chunk_size - overlap

    while start < n:
        end = min(start + chunk_size, n)
        # Try not to cut a chunk mid-word: extend to the next whitespace if close by.
        if end < n:
            lookahead = text.find(" ", end, end + 40)
            if lookahead != -1:
                end = lookahead
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start += step

    return chunks


def chunk_document(document: ExtractedDocument, document_id: str) -> list[Chunk]:
    """
    Chunk every non-empty page of an extracted document, preserving
    document_id / filename / page_number / section / chunk_id metadata
    on every chunk (required for citations -- see docs/architecture.md).
    """
    all_chunks: list[Chunk] = []
    running_index = 0

    for page in document.pages:
        if page.is_empty:
            continue
        section = _detect_section(page.text)
        for piece in chunk_text(page.text):
            all_chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    filename=document.filename,
                    page_number=page.page_number,
                    section=section,
                    chunk_index=running_index,
                    text=piece,
                )
            )
            running_index += 1

    return all_chunks
