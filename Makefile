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
	poetry run black numalogic/ examples/ tests/

lint: format
	poetry run ruff check --fix .

# install all dependencies
setup:
	poetry install --with dev,torch --all-extras

# test your application (tests in the tests/ directory)
test:
	poetry run pytest -v tests/

publish:
	@rm -rf dist
	poetry build
	poetry publish

requirements:
	poetry export -f requirements.txt --output requirements.txt --without-hashes

/usr/local/bin/mkdocs:
	$(PYTHON) -m pip install mkdocs==1.3.0 mkdocs_material==8.3.9

# docs

.PHONY: docs
docs: /usr/local/bin/mkdocs
	mkdocs build

.PHONY: docs-serve
docs-serve: docs
	mkdocs serve
