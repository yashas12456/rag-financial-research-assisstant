"""
SQLAlchemy engine/session management.
"""
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings

settings = get_settings()

engine_kwargs = {}
if settings.DATABASE_URL.startswith("sqlite"):
    from sqlalchemy.pool import StaticPool
    engine_kwargs = {"connect_args": {"check_same_thread": False}, "poolclass": StaticPool}
else:
    engine_kwargs = {"pool_pre_ping": True}

engine = create_engine(settings.DATABASE_URL, future=True, **engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a request-scoped DB session."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
