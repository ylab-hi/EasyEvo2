.PHONY: clean lint format test test-all coverage build install dev dist venv pyz help \
       docker-build docker-build-light docker-run docker-run-light docker-push docker-push-light

.DEFAULT_GOAL := help

PACKAGE := easyevo2
TESTS_DIR := tests
GHCR_IMAGE := ghcr.io/ylab-hi/easyevo2

# Generate help text automatically from comments
help: ## Display this help message
	@echo "EasyEvo2 Makefile Commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

clean: ## Remove build artifacts and cache files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.pyc" -delete
	find . -type d -name "*.pyo" -delete
	find . -type d -name "*.pyd" -delete

lint: ## Run code linting (ruff)
	ruff check .
	ruff format --check .

format: ## Format code automatically with ruff
	ruff format .
	ruff check --fix .

test: ## Run tests (excluding slow tests)
	uv run pytest -v $(TESTS_DIR)

test-all: ## Run all tests including slow tests
	uv run pytest -v $(TESTS_DIR) -m ""

coverage: ## Run tests with coverage report
	uv run pytest --cov=$(PACKAGE) --cov-report=html --cov-report=term $(TESTS_DIR)

build: ## Build package
	uv build

install: ## Install package
	uv sync -U

dist: clean ## Create distribution packages
	$(PYTHON) -m build
	ls -l dist

venv: ## Create virtual environment
	$(PYTHON) -m venv .venv
	@echo "Virtual environment created. Activate it with: source .venv/bin/activate"

pyz: clean ## Create a self-contained Python executable
	uv run python -m zipapp $(PACKAGE)

docker-build: ## Build Docker image (full, all models, Hopper GPU)
	docker build --target full -t easyevo2:full -t $(GHCR_IMAGE):full .

docker-build-light: ## Build Docker image (light, 7B models, any CUDA GPU)
	docker build --target light -t easyevo2:light -t $(GHCR_IMAGE):latest .

docker-run: ## Run full image with GPU + CWD mounted
	docker run --rm --gpus all --shm-size=16g \
		-v ~/.cache/huggingface:/app/.cache/huggingface \
		-v "$(PWD)":/data -w /data easyevo2:full

docker-run-light: ## Run light image with GPU + CWD mounted
	docker run --rm --gpus all --shm-size=16g \
		-v ~/.cache/huggingface:/app/.cache/huggingface \
		-v "$(PWD)":/data -w /data easyevo2:light

docker-push: ## Push full image to GHCR (manual, requires login)
	docker push $(GHCR_IMAGE):full

docker-push-light: ## Push light image to GHCR (CI handles this automatically)
	docker push $(GHCR_IMAGE):latest
