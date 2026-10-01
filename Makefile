.PHONY: help start stop restart status logs health setup run worker format lint eval eval-intents eval-cache eval-images eval-qa eval-agent eval-chat experiment-captions usage trace dashboards test test-cov clean

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
	@curl -s http://localhost:8080/api/v2/monitor/health | python3 -m json.tool || echo "Airflow not responding"

# Local development (run against the dockerised infra)
setup: ## Install Python dependencies
	uv sync

run: ## Run the API locally with reload
	uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

worker: ## Run a Celery worker locally
	uv run celery -A src.worker.celery_app worker --loglevel=INFO --concurrency=1

format: ## Format code
	uv run ruff format src tests

lint: ## Lint code
	uv run ruff check --fix src tests

eval: ## Measure search quality (keyword vs vector vs hybrid) on eval/queries.json
	uv run python scripts/evaluate_search.py --save eval/results/search.json

experiment-captions: ## Re-run the caption-embedding experiment (see docs/decisions/0001-no-caption-embeddings.md)
	uv run --with sentence-transformers python scripts/experiment_caption_embeddings.py --save eval/results/caption_embeddings.json

eval-intents: ## Request-understanding accuracy (LLM vs rules) on eval/intents.json
	uv run python scripts/evaluate_intents.py --save eval/results/intents.json

eval-images: ## Image queries: hit@1 and false answers per similarity cut-off (needs scripts/prepare_image_eval.py once)
	uv run python scripts/evaluate_images.py --save eval/results/images.json

eval-qa: ## Question answering: answered/not found, correct facts, citations (needs the sleep talk: scripts/make_talk_video.py)
	uv run python scripts/evaluate_qa.py --save eval/results/qa.json

eval-agent: ## Chat routing: rules vs model vs rules-first, on the labelled turns and both held-out sets
	for f in agent_turns agent_turns_heldout agent_turns_heldout2; do uv run python scripts/evaluate_agent_turns.py --turns eval/$$f.json --save eval/results/$$f.json; done

eval-chat: ## Whole conversations through /chat (routing + tools + memory), needs the stack running
	uv run python scripts/evaluate_conversations.py --save eval/results/conversations.json

eval-claude: ## The model evaluations again with Claude Haiku 4.5 (Anthropic API; ANTHROPIC_API_KEY in .env) → eval/results/claude/
	mkdir -p eval/results/claude
	LLM_PROVIDER=anthropic uv run python scripts/evaluate_intents.py --save eval/results/claude/intents.json
	for f in agent_turns agent_turns_heldout agent_turns_heldout2; do LLM_PROVIDER=anthropic uv run python scripts/evaluate_agent_turns.py --turns eval/$$f.json --save eval/results/claude/$$f.json; done
	docker compose -f compose.yml -f compose.claude.yml up -d --wait api worker
	uv run python scripts/evaluate_qa.py --save eval/results/claude/qa.json
	uv run python scripts/evaluate_conversations.py --save eval/results/claude/conversations.json
	docker compose up -d --wait api worker  # back to Ollama

eval-cache: ## What the understanding and answer caches save (needs the stack running)
	uv run python scripts/evaluate_cache.py --save eval/results/cache.json

usage: ## Tokens consumed by model calls and their estimated cost (from the llm_calls table)
	uv run python scripts/usage_report.py

dashboards: ## Regenerate the Grafana dashboards (infra/grafana/dashboards) from scripts/build_dashboards.py
	uv run python scripts/build_dashboards.py

trace: ## Recent traces, or one as a timeline: make trace ID=<X-Request-ID>
	uv run python scripts/show_trace.py $(ID)

test: ## Run tests
	uv run pytest

test-cov: ## Run tests with coverage
	uv run pytest --cov=src --cov-report=html

# Cleanup
clean: ## Stop services and DELETE all data volumes
	docker compose down -v
