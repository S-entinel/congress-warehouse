.PHONY: help install lint test download load transform check build dashboard

help:  ## Show available commands
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

install:  ## Install the project and development tools
	pip install -e ".[dev]"

lint:  ## Check code style
	ruff check src tests

test:  ## Run the Python tests
	pytest

download:  ## Download raw source files (not built yet)
	@echo "download: not built yet"

load:  ## Load raw files into DuckDB (not built yet)
	@echo "load: not built yet"

transform:  ## Build the dbt models (not built yet)
	@echo "transform: not built yet"

check:  ## Run the dbt data tests (not built yet)
	@echo "check: not built yet"

build: download load transform check  ## Run the whole pipeline

dashboard:  ## Launch the Streamlit dashboard (not built yet)
	@echo "dashboard: not built yet"