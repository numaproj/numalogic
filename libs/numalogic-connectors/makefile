clean:
	@rm -rf build dist .eggs *.egg-info
	@rm -rf .benchmarks .coverage coverage.xml htmlcov report.xml .tox
	@find . -type d -name '.mypy_cache' -exec rm -rf {} +
	@find . -type d -name '__pycache__' -exec rm -rf {} +
	@find . -type d -name '*pytest_cache*' -exec rm -rf {} +
	@find . -type f -name "*.py[co]" -exec rm -rf {} +

format: clean
	black nlconnectors/ tests/

lint: format
	ruff check --fix .

setup:
	pip install ../numalogic-core
	pip install -r requirements/requirements.txt
	pip install -r requirements/requirements-dev.txt

# test your application (tests in the tests/ directory)
test:
	pytest -v tests/
