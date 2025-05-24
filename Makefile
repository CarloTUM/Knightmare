.PHONY: help install lint format test cov docs docker clean

PY ?= python
PIP ?= $(PY) -m pip

help:
	@echo "Targets:"
	@echo "  install   editable install with dev + docs extras"
	@echo "  lint      ruff + black --check + mypy"
	@echo "  format    ruff --fix + black"
	@echo "  test      pytest"
	@echo "  cov       pytest with coverage report"
	@echo "  docs      build mkdocs site"
	@echo "  docker    build CPU docker image"
	@echo "  clean     remove caches and build artefacts"

install:
	$(PIP) install -e ".[dev,docs]"

lint:
	$(PY) -m ruff check src tests
	$(PY) -m black --check src tests
	$(PY) -m mypy src

format:
	$(PY) -m ruff check --fix src tests
	$(PY) -m black src tests

test:
	$(PY) -m pytest

cov:
	$(PY) -m pytest --cov=selfrl_chess --cov-report=term-missing --cov-report=xml

docs:
	$(PY) -m mkdocs build --strict

docker:
	docker build -t knightmare:latest .

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache .coverage coverage.xml htmlcov site
	find . -type d -name __pycache__ -exec rm -rf {} +
