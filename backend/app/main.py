"""FastAPI application entrypoint."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import documents, evaluation, health, query
from app.core.config import get_settings
from app.core.errors import AppError, app_error_handler, unhandled_exception_handler
from app.core.logging import configure_logging, get_logger
from app.db.session import Base, engine

settings = get_settings()
configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting %s (env=%s)", settings.APP_NAME, settings.ENV)
    # In production, use Alembic migrations instead of create_all.
    # create_all is used here for a friction-free local/demo setup.
    Base.metadata.create_all(bind=engine)
    yield
    logger.info("Shutting down %s", settings.APP_NAME)


app = FastAPI(
    title=settings.APP_NAME,
    description="Production-style domain-specific RAG assistant for financial documents.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.ENV != "production" else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(AppError, app_error_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

app.include_router(health.router, prefix=settings.API_V1_PREFIX)
app.include_router(documents.router, prefix=settings.API_V1_PREFIX)
app.include_router(query.router, prefix=settings.API_V1_PREFIX)
app.include_router(evaluation.router, prefix=settings.API_V1_PREFIX)


@app.get("/")
def root():
    return {"name": settings.APP_NAME, "docs": "/docs", "health": f"{settings.API_V1_PREFIX}/health"}
