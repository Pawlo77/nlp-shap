.DEFAULT_GOAL := help

.PHONY: help install update clean prek prek-all check tests coverage bench bench-regression docs notebooks build

help: ## Show available targets
	@printf "Available targets:\n"
	@grep -E '^[a-zA-Z0-9_-]+:.*##' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

install: ## Set up the development environment
	uv python install 3.12
	uv sync --all-groups
	uv run prek install

update: ## Update dependencies to their latest versions
	uv lock --upgrade
	uv sync --all-groups

clean: ## Remove virtual environment and cache files
	rm -rf .venv/
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d \( -name ".cache" -o -name "*__pycache__" -o -name ".*_cache" \) -exec rm -rf {} + 2>/dev/null || true

prek: ## Run git hook checks on changed files
	uv run prek run --fail-fast

prek-all: ## Run git hook checks on all files
	uv run prek run --all-files

check: ## Run tests and all quality hooks
	$(MAKE) tests
	$(MAKE) prek-all

tests: ## Run the test suite
	uv run pytest tests/

coverage: ## Run tests with coverage report
	uv run pytest --cov=./src/nlp_shap --cov-report=term-missing tests/

bench: ## Run benchmark tests with performance tracking
	uv run pytest tests/benchmarks/ -m bench --benchmark

bench-regression: ## Run benchmarks and fail on >10% regression vs baselines.json
	uv run pytest tests/benchmarks/ -m bench --benchmark --bench-regression

docs: ## Build the documentation
	uv run make -C docs html

notebooks: ## Execute all example notebooks in place (store outputs)
	@set -e; for nb in examples/*.ipynb; do \
		uv run jupyter nbconvert --to notebook --execute "$$nb" --inplace; \
	done

build: ## Build source and wheel distributions
	uv build
