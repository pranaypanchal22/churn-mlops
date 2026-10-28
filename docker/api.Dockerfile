# API image: FastAPI service plus the training/promotion scripts (same Python package).
# Model artifacts are NOT baked in; models/ is mounted at runtime so promotion never
# requires a rebuild and the image contains no data.

ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# Dependencies first so code edits don't invalidate this layer.
COPY pyproject.toml README.md ./
COPY requirements/constraints.txt requirements/constraints.txt
COPY src ./src
RUN pip install -c requirements/constraints.txt .

FROM python:${PYTHON_VERSION}-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    CHURN_REGISTRY_DIR=/app/models/registry \
    CHURN_SCHEMA_PATH=/app/configs/schema.yaml \
    MLFLOW_DISABLE_AGENT_HINT=1
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app --shell /usr/sbin/nologin app
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY configs ./configs
COPY scripts ./scripts
RUN mkdir -p models data reports build && chown -R app:app models data reports build
USER app

EXPOSE 8000
# python instead of curl: keeps curl out of the image.
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"]
CMD ["uvicorn", "churn.api.app:app", "--host", "0.0.0.0", "--port", "8000", \
     "--timeout-keep-alive", "5", "--timeout-graceful-shutdown", "10", "--no-server-header"]
