    clean:
	@rm -rf build dist .eggs *.egg-info
	@rm -rf .benchmarks .coverage coverage.xml htmlcov report.xml .tox
	@find . -type d -name '.mypy_cache' -exec rm -rf {} +
	@find . -type d -name '__pycache__' -exec rm -rf {} +
	@find . -type d -name '*pytest_cache*' -exec rm -rf {} +
	@find . -type f -name "*.py[co]" -exec rm -rf {} +

format: clean
	@POETRY run black ml_steps/

# install all dependencies
setup:
	@POETRY install -v

test:
	@POETRY run python -m unittest discover

requirements:
	@POETRY export -f requirements.txt --output requirements.txt --without-hashes
