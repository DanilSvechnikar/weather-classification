#* Variables
SHELL := /usr/bin/env bash
PYTHON ?= python3

ifeq ($(OS),Windows_NT)
	detected_OS := Windows
	MKDIR_CMD := mkdir
else
	detected_OS := $(shell uname -s)
	MKDIR_CMD := mkdir -p
endif


#* Installation for development
.PHONY: project-init-dev
project-init-dev: poetry-install-dev tools-install

.PHONY: poetry-install-dev
poetry-install-dev:
	poetry install --no-interaction

.PHONY: tools-install
tools-install:
	poetry run pre-commit install --hook-type prepare-commit-msg --hook-type pre-commit
	poetry run nbdime config-git --enable
	poetry run mypy --install-types --non-interactive ./


#* Installation Not for development
.PHONY: project-init
project-init:
	poetry install --no-interaction --without dev


#* Pip tools
.PHONY: pip-install
pip-install: poetry-export
	pip3 install --no-cache-dir --upgrade pip && \
	pip3 install --no-cache-dir -r requirements.txt

.PHONY: poetry-export
poetry-export:
	poetry lock -n && poetry export --without-hashes > requirements.txt

.PHONY: poetry-export-dev
poetry-export-dev:
	poetry lock -n && poetry export --with dev --without-hashes > requirements.dev.txt


#* Cleaning
.PHONY: clean-trash
clean-trash: pycache-remove build-remove pip-cache-clear

.PHONY: pycache-remove
pycache-remove:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf
	find . | grep -E "(.ipynb_checkpoints$$)" | xargs rm -rf

.PHONY: build-remove
build-remove:
	rm -rf build/

.PHONY: pip-cache-clear
pip-cache-clear:
	pip cache purge

.PHONY: poetry-cache-clear
poetry-cache-clear:
	poetry cache clear --all . -n


#* Tests
PYTHONHASHSEED ?= 123456789

PYTEST_USER_OPTS ?=
PYTEST_USE_COLOR ?= yes
PYTEST_OPTS ?= -v --durations=10 --color=${PYTEST_USE_COLOR} ${PYTEST_USER_OPTS}

TEST_START_CMD = poetry run pytest

COVERAGE_CACHE_DIRECTORY = tests_cache
PYTEST_LOGS_DIR ?= logs

.PHONY: tests-logs-dir
tests-logs-dir:
	${MKDIR_CMD} ${PYTEST_LOGS_DIR}

.PHONY: tests-unit
tests-unit: tests-logs-dir
	COVERAGE_FILE=${COVERAGE_CACHE_DIRECTORY}/.coverage PYTHONHASHSEED=${PYTHONHASHSEED} ${TEST_START_CMD} -m unit ${PYTEST_OPTS}

.PHONY: tests-interface
tests-interface: tests-logs-dir
	COVERAGE_FILE=${COVERAGE_CACHE_DIRECTORY}/.coverage PYTHONHASHSEED=${PYTHONHASHSEED} ${TEST_START_CMD} -m interface ${PYTEST_OPTS}

.PHONY: run-tests
run-tests: tests-unit tests-interface
