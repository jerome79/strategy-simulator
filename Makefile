.PHONY: format lint test precommit coverage

format:
	black .
	ruff check . --fix

lint:
	ruff check .

test:
	pytest -q

coverage:
	pytest --cov=src --cov-report=term-missing --cov-report=html

precommit:
	pre-commit run --all-files
