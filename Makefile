.PHONY: help start stop restart status logs health setup run worker format lint test test-cov clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

# Service management
start: ## Build and start all services
	docker compose up --build -d

stop: ## Stop all services (keeps data volumes)
	docker compose down

restart: ## Restart all services
	docker compose restart

status: ## Show service status
	docker compose ps

logs: ## Tail service logs
	docker compose logs -f

health: ## Check API readiness (all dependencies)
	@curl -s http://localhost:8000/api/v1/health | python3 -m json.tool || echo "API not responding"

# Local development (run against the dockerised infra)
setup: ## Install Python dependencies
	uv sync

run: ## Run the API locally with reload
	uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

worker: ## Run a Celery worker locally
	uv run celery -A src.worker.celery_app worker --loglevel=INFO --concurrency=2

format: ## Format code
	uv run ruff format src tests

lint: ## Lint code
	uv run ruff check --fix src tests

test: ## Run tests
	uv run pytest

test-cov: ## Run tests with coverage
	uv run pytest --cov=src --cov-report=html

# Cleanup
clean: ## Stop services and DELETE all data volumes
	docker compose down -v
