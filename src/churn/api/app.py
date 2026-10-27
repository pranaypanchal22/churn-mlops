"""FastAPI prediction service.

Run locally:
    uvicorn churn.api.app:app --port 8000

Request bodies are never logged. Logs carry request id, method, path, status, and latency.
"""

# No `from __future__ import annotations` here: FastAPI must see the request model classes
# (built at runtime from the schema) as real types, not strings.
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from churn.api.schemas import (
    BatchPrediction,
    BatchPredictionItem,
    ErrorResponse,
    Prediction,
    build_batch_model,
    build_customer_model,
)
from churn.api.service import ModelService
from churn.api.settings import ApiSettings
from churn.schema.config import load_schema

log = logging.getLogger("churn.api")


def _not_ready(service: ModelService) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content=ErrorResponse(detail=f"Model not ready: {service.not_ready_reason}").model_dump(),
    )


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    settings = settings or ApiSettings()
    logging.basicConfig(level=settings.log_level)
    schema = load_schema(settings.schema_path)
    CustomerFeatures = build_customer_model(schema)
    BatchRequest = build_batch_model(CustomerFeatures, settings.max_batch_size)
    service = ModelService()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service.load(settings.registry_dir, schema, settings.model_version)
        yield

    app = FastAPI(
        title="Churn Prediction API",
        version="0.1.0",
        description="Estimates churn probability for one telecom customer profile. "
        "Portfolio project; estimates are not guarantees.",
        lifespan=lifespan,
    )
    app.state.service = service

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["x-request-id"] = request_id
        log.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": round(elapsed_ms, 2),
            },
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError):
        # Pydantic's default error body echoes the submitted value; leave it out so
        # customer data isn't reflected back or captured by proxies and logs.
        errors = [
            {
                "field": ".".join(str(p) for p in e["loc"] if p != "body") or "body",
                "message": e["msg"],
            }
            for e in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(detail="Invalid request", errors=errors).model_dump(),
        )

    @app.get("/health", tags=["ops"])
    def health() -> dict[str, str]:
        """Process is up. Does not check the model; use /ready for that."""
        return {"status": "ok"}

    @app.get("/ready", tags=["ops"], responses={503: {"model": ErrorResponse}})
    def ready():
        if not service.ready:
            return _not_ready(service)
        return {"ready": True, "model_version": service.version}

    @app.get("/model-info", tags=["model"], responses={503: {"model": ErrorResponse}})
    def model_info() -> Any:
        if not service.ready:
            return _not_ready(service)
        return {**service.info(), "max_batch_size": settings.max_batch_size}

    @app.post(
        "/predict",
        tags=["predict"],
        response_model=Prediction,
        responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    def predict(customer: CustomerFeatures):  # type: ignore[valid-type]
        if not service.ready:
            return _not_ready(service)
        p = service.predict_proba([customer.model_dump()])[0]
        return Prediction(
            churn_probability=round(p, 4),
            risk_label=service.label(p),
            threshold=service.threshold,
            model_version=service.version,
        )

    @app.post(
        "/predict/batch",
        tags=["predict"],
        response_model=BatchPrediction,
        responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    def predict_batch(batch: BatchRequest):  # type: ignore[valid-type]
        if not service.ready:
            return _not_ready(service)
        probs = service.predict_proba([c.model_dump() for c in batch.customers])
        return BatchPrediction(
            predictions=[
                BatchPredictionItem(
                    index=i, churn_probability=round(p, 4), risk_label=service.label(p)
                )
                for i, p in enumerate(probs)
            ],
            threshold=service.threshold,
            model_version=service.version,
        )

    return app


app = create_app()
