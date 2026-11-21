# Common tasks. Everything runs through uv, so no virtual environment needs to be activated.

UV ?= uv

.PHONY: help install data test lint reproduce

help:
	@echo install    Install dependencies and the pre-commit hooks
	@echo data       Download HateXplain and HateCheck and rebuild the original split
	@echo test       Run the test suite
	@echo lint       Run ruff
	@echo reproduce  Rerun systems 1 to 3 and check them against the archived results

install:
	$(UV) sync --frozen
	$(UV) run pre-commit install

data:
	$(UV) run python -m scripts.prepare_data

test:
	$(UV) run python -m pytest

lint:
	$(UV) run ruff check .

reproduce:
	$(UV) run python -m scripts.check_reproduction
