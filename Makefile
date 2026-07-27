.PHONY: all install run debug clean lint lint-strict

all: install

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	find src -type d -name "__pycache__" -exec rm -rf {} +
	find src -type f -name "*.pyc" -delete
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf .venv

lint:
	flake8 src
	mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs src

lint-strict:
	flake8 src
	mypy --strict src