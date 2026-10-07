"""
Custom application exceptions and FastAPI exception handlers.

WHY: Returning raw stack traces or generic 500s leaks internals and gives
     the frontend nothing useful to show the user. Each domain error below
     maps to a specific HTTP status and a safe, descriptive message.
"""
from fastapi import Request, status
from fastapi.responses import JSONResponse


class AppError(Exception):
    """Base class for all handled application errors."""
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(self, message: str, details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


class InvalidFileError(AppError):
    status_code = status.HTTP_400_BAD_REQUEST


class FileTooLargeError(AppError):
    status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE


class DuplicateDocumentError(AppError):
    status_code = status.HTTP_409_CONFLICT


class DocumentNotFoundError(AppError):
    status_code = status.HTTP_404_NOT_FOUND


class EmptyDocumentError(AppError):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY


class EmbeddingError(AppError):
    status_code = status.HTTP_502_BAD_GATEWAY


class VectorStoreError(AppError):
    status_code = status.HTTP_502_BAD_GATEWAY


class LLMError(AppError):
    status_code = status.HTTP_502_BAD_GATEWAY


class InvalidQueryError(AppError):
    status_code = status.HTTP_400_BAD_REQUEST


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.__class__.__name__, "message": exc.message, "details": exc.details},
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Never leak internal exception text/stack traces to the client.
    from app.core.logging import get_logger
    get_logger("app.errors").exception("Unhandled exception on %s", request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "InternalServerError", "message": "An unexpected error occurred."},
    )
