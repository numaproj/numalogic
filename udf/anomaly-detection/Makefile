# Check Python
PYTHON:=$(shell command -v python 2> /dev/null)
ifndef PYTHON
PYTHON:=$(shell command -v python3 2> /dev/null)
endif
ifndef PYTHON
$(error "Python is not available, please install.")
endif

clean:
	@rm -rf build dist .eggs *.egg-info
	@rm -rf .benchmarks .coverage coverage.xml htmlcov report.xml .tox
	@find . -type d -name '.mypy_cache' -exec rm -rf {} +
	@find . -type d -name '__pycache__' -exec rm -rf {} +
	@find . -type d -name '*pytest_cache*' -exec rm -rf {} +
	@find . -type f -name "*.py[co]" -exec rm -rf {} +

format: clean
	poetry run black src/ tests/
	poetry run black starter.py

lint: format
	poetry run flake8 .

# install all dependencies
setup:
	poetry install --with dev --all-extras

test:
	poetry run pytest -v tests/

requirements:
	poetry export -f requirements.txt --output requirements.txt --without-hashes

requirements-dev:
	poetry export -f requirements.txt --with dev --output requirements-dev.txt --without-hashes

pipeline:
	kustomize build manifests/prerequisites | kubectl apply -f -
	kustomize build manifests/ | kubectl apply -f -