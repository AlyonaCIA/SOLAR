.PHONY: help install install-dev install-api install-all sync lint format type-check test test-cov clean run-api run-pipeline docker-api docker-scheduler

# ─── Default ──────────────────────────────────────────────────────────────────

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ─── Installation ────────────────────────────────────────────────────────────

install: ## Install core dependencies
	uv sync

install-dev: ## Install with dev tools (ruff, mypy, pytest)
	uv sync --extra dev

install-api: ## Install with API dependencies
	uv sync --extra api

install-all: ## Install all dependency groups
	uv sync --all-extras

sync: ## Sync environment and lock file
	uv sync --all-extras
	uv lock

# ─── Code Quality ────────────────────────────────────────────────────────────

lint: ## Run Ruff linter
	uv run ruff check .

format: ## Run Ruff formatter (auto-fix)
	uv run ruff format .
	uv run ruff check --fix .

type-check: ## Run mypy type checker
	uv run mypy src/

check: lint type-check ## Run all code quality checks

# ─── Testing ─────────────────────────────────────────────────────────────────

test: ## Run tests
	uv run pytest

test-cov: ## Run tests with coverage report
	uv run pytest --cov=src --cov-report=term-missing --cov-report=html

# ─── Run Services ────────────────────────────────────────────────────────────

run-api: ## Start FastAPI development server
	cd API && uv run uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

run-pipeline: ## Run the anomaly detection pipeline (pass ARGS="--data_dir ...")
	uv run python src/solar/run_kmeans_pipeline.py $(ARGS)

# ─── Docker ──────────────────────────────────────────────────────────────────

docker-api: ## Build API Docker image
	docker build -t solar-api ./API

docker-scheduler: ## Build scheduled processing Docker image
	docker build -t solar-scheduler ./scheduled_processing

# ─── Utilities ───────────────────────────────────────────────────────────────

clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage

logs: ## Tail API logs (Docker)
	docker logs -f solar-api
