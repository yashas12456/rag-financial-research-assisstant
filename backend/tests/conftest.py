"""Shared pytest fixtures."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")

from app.db.session import Base


@pytest.fixture()
def db_session():
    """An isolated in-memory SQLite session per test (fast, no external deps)."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def sample_pdf_bytes() -> bytes:
    """Builds a tiny real PDF in-memory using PyMuPDF so parser tests don't need a fixture file."""
    import fitz
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), "Item 1. Business\n\nOur revenue increased significantly in FY2025.")
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Risk Factors\n\nWe face competition from other companies.")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes
