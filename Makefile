# Task shortcuts for Linux/macOS/Git Bash with make installed. Every target is a plain
# command; docs/operations_runbook.md lists the same commands for shells without make.

PY ?= python
VERSION ?=
REASON ?=

.PHONY: help setup profile validate train promote list api ui test lint format smoke \
        docker-build up down logs docker-train docker-promote

help:
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-15s %s\n", $$1, $$2}'

setup: ## Install the package with dev, tracking and frontend extras (pinned)
	$(PY) -m pip install -c requirements/constraints.txt -e ".[dev,tracking,frontend]"

profile: ## Profile data/raw/CustomerChurn.csv
	$(PY) scripts/profile_data.py

validate: ## Map + validate raw data into data/processed
	$(PY) scripts/prepare_data.py

train: ## Train and register a candidate (not served)
	$(PY) scripts/train.py --config configs/train.yaml

list: ## List registered versions (* = served)
	$(PY) scripts/promote.py --list

promote: ## Promote: make promote VERSION=... REASON="..."
	$(PY) scripts/promote.py --version "$(VERSION)" --reason "$(REASON)"

api: ## Run the API locally on :8000
	$(PY) -m uvicorn churn.api.app:app --port 8000

ui: ## Run the Streamlit UI locally on :8501
	$(PY) -m streamlit run frontend/app.py

test: ## Run tests with coverage
	$(PY) -m pytest --cov=churn --cov=frontend

lint: ## Lint and format check
	$(PY) -m ruff check . && $(PY) -m ruff format --check .

format: ## Apply formatting
	$(PY) -m ruff format . && $(PY) -m ruff check --fix .

smoke: ## Synthetic end-to-end training (no dataset needed)
	$(PY) scripts/make_synthetic_data.py --rows 1000
	$(PY) scripts/train.py --config configs/train_smoke.yaml

docker-build: ## Build API and frontend images
	docker compose build

up: ## Start API + frontend in Docker
	docker compose up --build -d

down: ## Stop the stack
	docker compose down

logs: ## Follow container logs
	docker compose logs -f

docker-train: ## Train inside Docker using data/raw
	docker compose --profile tools run --rm trainer

docker-promote: ## Promote inside Docker: make docker-promote VERSION=... REASON="..."
	docker compose --profile tools run --rm trainer python scripts/promote.py --version "$(VERSION)" --reason "$(REASON)"
