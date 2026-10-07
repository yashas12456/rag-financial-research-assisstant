#!/usr/bin/env python3
"""
Bulk-ingest all PDFs in a folder (default: data/sample_documents) through the
same upload -> index pipeline the API uses.

Usage:
    python scripts/ingest.py [--folder data/sample_documents]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.core.errors import AppError
from app.core.logging import configure_logging, get_logger
from app.db.session import Base, SessionLocal, engine
from app.services.ingestion_service import index_document, upload_document

configure_logging()
logger = get_logger("scripts.ingest")


def main():
    parser = argparse.ArgumentParser(description="Bulk-ingest PDFs into the RAG system")
    parser.add_argument("--folder", default="data/sample_documents")
    args = parser.parse_args()

    Base.metadata.create_all(bind=engine)

    pdf_files = [f for f in os.listdir(args.folder) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in {args.folder}. Add some annual report / earnings "
              f"PDFs there and re-run this script.")
        return

    db = SessionLocal()
    try:
        for filename in pdf_files:
            path = os.path.join(args.folder, filename)
            with open(path, "rb") as f:
                file_bytes = f.read()

            print(f"Uploading {filename}...")
            try:
                document = upload_document(db, filename, file_bytes)
                db.commit()
            except AppError as e:
                print(f"  Skipped ({e.__class__.__name__}): {e.message}")
                db.rollback()
                continue

            print(f"  Indexing {filename}...")
            try:
                num_chunks = index_document(db, document, file_bytes)
                db.commit()
                print(f"  Indexed: {num_chunks} chunks created.")
            except AppError as e:
                print(f"  Indexing failed ({e.__class__.__name__}): {e.message}")
                db.commit()  # persist the FAILED status set by index_document
    finally:
        db.close()


if __name__ == "__main__":
    main()
