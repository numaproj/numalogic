clean:
	@rm -rf build dist .eggs *.egg-info
	@rm -rf .benchmarks .coverage coverage.xml htmlcov report.xml .tox
	@find . -type d -name '.mypy_cache' -exec rm -rf {} +
	@find . -type d -name '__pycache__' -exec rm -rf {} +
	@find . -type d -name '*pytest_cache*' -exec rm -rf {} +
	@find . -type f -name "*.py[co]" -exec rm -rf {} +

format: clean
	black libs/ apps/ examples/ tests/

lint: format
	ruff check --fix .

setup:
	PKG_NAME=numalogic pip install -v -e '.[dev,jupyter]' --config-settings editable_mode=strict
	PKG_NAME=numalogic-connectors pip install -v -e '.[dev]' --config-settings editable_mode=strict
	PKG_NAME=numalogic-registry pip install -v -e '.[dev]' --config-settings editable_mode=strict

# test your application (tests in the tests/ directory)
test:
	poetry run pytest -v tests/

publish:
	@rm -rf dist
	poetry build
	poetry publish

requirements:
	poetry export -f requirements.txt --output requirements.txt --without-hashes

tag:
	VERSION=v$(shell poetry version -s)
	@echo "Tagging version $(VERSION)"
	git tag -s -a $(VERSION) -m "Release $(VERSION)"

/usr/local/bin/mkdocs:
	$(PYTHON) -m pip install mkdocs==1.3.0 mkdocs_material==8.3.9

# docs

.PHONY: docs
docs: /usr/local/bin/mkdocs
	mkdocs build

.PHONY: docs-serve
docs-serve: docs
	mkdocs serve
