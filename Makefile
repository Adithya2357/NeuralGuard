# NeuralGuard developer tasks.  Run 'make' (or 'make help') for the list.
#
# Python targets use $(PYTHON); with a virtual environment activated that is the
# venv's interpreter. Override it explicitly with e.g. 'make test PYTHON=.venv/bin/python'.
# The model path, Kafka and Elasticsearch addresses come from NEURALGUARD_* variables
# (defaults: models/threat_model.joblib, localhost:9092, http://localhost:9200).

PYTHON ?= python3
VENV ?= .venv
COMPOSE ?= docker compose

.DEFAULT_GOAL := help

.PHONY: help venv install test lint security train demo up down logs produce detect \
	docker-demo clean

help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

venv: ## Create a virtual environment (.venv) with the package and dev tools
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install -e ".[dev]"
	@echo "Activate it with: source $(VENV)/bin/activate"

install: ## Install the package and dev tools into the current environment
	$(PYTHON) -m pip install -e ".[dev]"

test: ## Run the test suite with coverage
	$(PYTHON) -m pytest --cov=neuralguard --cov-report=term-missing

lint: ## Lint with ruff
	$(PYTHON) -m ruff check neuralguard tests

security: ## Static security scan (bandit) and dependency vulnerability audit (pip-audit)
	$(PYTHON) -m bandit -r neuralguard -ll
	$(PYTHON) -m pip_audit --skip-editable

train: ## Train the threat model on simulated traffic (saved to $NEURALGUARD_MODEL_PATH)
	$(PYTHON) -m neuralguard train

demo: ## End-to-end demo on simulated traffic (no Kafka/Elasticsearch needed)
	$(PYTHON) -m neuralguard demo

up: ## Start Kafka, Elasticsearch and Grafana (http://127.0.0.1:3000)
	$(COMPOSE) up -d kafka elasticsearch grafana

down: ## Stop and remove all containers (data volumes are kept)
	$(COMPOSE) --profile demo down

logs: ## Follow the logs of all containers
	$(COMPOSE) --profile demo logs -f --tail=100

produce: ## Publish simulated traffic to Kafka on localhost:9092
	$(PYTHON) -m neuralguard produce --source simulate

detect: ## Consume traffic from Kafka and index alerts into Elasticsearch
	$(PYTHON) -m neuralguard detect

docker-demo: ## Build the app image and run the whole stack plus detector and simulator
	$(COMPOSE) --profile demo up --build

clean: ## Remove caches, coverage and build artifacts (keeps models and the venv)
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache .coverage .coverage.* \
		coverage.xml htmlcov
	find . -path ./$(VENV) -prune -o -type d -name __pycache__ -prune -exec rm -rf {} +
