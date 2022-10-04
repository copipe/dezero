.PHONY: black-check
black-check:
	poetry run black --check dezero steps tests

.PHONY: black
black:
	poetry run black dezero steps tests

.PHONY: flake8
flake8:
	poetry run flake8 dezero steps tests

.PHONY: isort-check
isort-check:
	poetry run isort --check-only dezero steps tests

.PHONY: isort
isort:
	poetry run isort dezero steps tests

.PHONY: mypy
mypy:
	poetry run mypy dezero

.PHONY: test
test:
	poetry run pytest tests --cov=src --cov-report term-missing --durations 5

.PHONY: format
format:
	$(MAKE) black
	$(MAKE) isort

.PHONY: lint
lint:
	$(MAKE) black-check
	$(MAKE) isort-check
	$(MAKE) flake8
	$(MAKE) mypy

.PHONY: test-all
test-all:
	$(MAKE) black
	$(MAKE) flake8
	$(MAKE) isort
	$(MAKE) mypy
	$(MAKE) test