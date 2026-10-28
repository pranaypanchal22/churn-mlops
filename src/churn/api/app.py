"""FastAPI prediction service.

Run locally:
    uvicorn churn.api.app:app --port 8000

Request bodies are never logged. Logs are JSON lines with request id, method, route, status,
and latency. Prometheus metrics are served at /metrics; aggregate drift at /monitoring/drift.
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
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from churn.api.observability import ApiMetrics, configure_logging
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
from churn.monitoring.drift import load_monitoring_config
from churn.schema.config import load_schema

log = logging.getLogger("churn.api")


def _not_ready(service: ModelService) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content=ErrorResponse(detail=f"Model not ready: {service.not_ready_reason}").model_dump(),
    )


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    settings = settings or ApiSettings()
    configure_logging(settings.log_level)
    schema = load_schema(settings.schema_path)
    monitoring_cfg = load_monitoring_config(settings.monitoring_config_path)
    metrics = ApiMetrics()
    CustomerFeatures = build_customer_model(schema)
    BatchRequest = build_batch_model(CustomerFeatures, settings.max_batch_size)
    service = ModelService()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service.load(settings.registry_dir, schema, settings.model_version, monitoring_cfg)
        metrics.model_ready.set(1 if service.ready else 0)
        if service.ready:
            metrics.model_info.labels(
                model_version=service.version, schema_version=schema.schema_version
            ).set(1)
        yield

    app = FastAPI(
        title="Churn Prediction API",
        version="0.1.0",
        description="Estimates churn probability for one telecom customer profile. "
        "Portfolio project; estimates are not guarantees.",
        lifespan=lifespan,
    )
    app.state.service = service
    app.state.metrics = metrics

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["x-request-id"] = request_id
        # Route template (e.g. /predict), never the raw URL, keeps metric labels bounded.
        route = getattr(request.scope.get("route"), "path", "unmatched")
        metrics.requests.labels(request.method, route, str(response.status_code)).inc()
        metrics.latency.labels(route).observe(elapsed_ms / 1000)
        log.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "route": route,
                "status": response.status_code,
                "latency_ms": round(elapsed_ms, 2),
                "model_version": service.version,
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
        for e in errors:
            # Strip list indices so batch errors map onto the same field label.
            field = ".".join(p for p in e["field"].split(".") if not p.isdigit())
            metrics.validation_errors.labels(field=field).inc()
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(detail="Invalid request", errors=errors).model_dump(),
        )

    def _observe(probs: list[float]) -> None:
        for p in probs:
            metrics.predictions.labels(risk_label=service.label(p)).inc()
            metrics.scores.observe(p)

    @app.get("/metrics", tags=["ops"], include_in_schema=False)
    def prometheus_metrics() -> Response:
        metrics.update_drift(service.monitor.report())
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/monitoring/drift", tags=["ops"], responses={503: {"model": ErrorResponse}})
    def monitoring_drift() -> Any:
        """Aggregate input/score drift since this process started (no row-level data)."""
        if not service.ready:
            return _not_ready(service)
        report = service.monitor.report()
        if report is None:
            return JSONResponse(
                status_code=404,
                content=ErrorResponse(
                    detail=f"No reference profile for model {service.version}; retrain to "
                    "enable drift monitoring."
                ).model_dump(),
            )
        return {"model_version": service.version, "window": "since process start", **report}

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
        _observe([p])
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
        _observe(probs)
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
