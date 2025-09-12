.PHONY: format lint test precommit coverage

format:
	black .
	ruff check . --fix

lint:
	ruff check .

test:
	pytest -q

precommit:
	pre-commit run --all-files
