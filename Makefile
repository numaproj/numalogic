clean:
	@rm -rf build dist .eggs *.egg-info
	@rm -rf .benchmarks .coverage coverage.xml htmlcov report.xml .tox
	@find . -type d -name '.mypy_cache' -exec rm -rf {} +
	@find . -type d -name '__pycache__' -exec rm -rf {} +
	@find . -type d -name '*pytest_cache*' -exec rm -rf {} +
	@find . -type f -name "*.py[co]" -exec rm -rf {} +

format: clean
	poetry run black numalogic/

# install all dependencies
setup:
	poetry install -v

# test your application (tests in the tests/ directory)
test:
	poetry run pytest numalogic/tests/

build:
	poetry build

requirements:
	poetry export -f requirements.txt --output requirements.txt --without-hashes
