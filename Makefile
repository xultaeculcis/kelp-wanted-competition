.DEFAULT_GOAL := help
isort = isort .
black = black .
flake8 = flake8 .
autoflake8 = autoflake8 .
mypy = mypy .
pre-commit = pre-commit run --all-files

DATA_DIR=data
SHELL=/bin/bash
# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^\.PHONY: ([0-9a-zA-Z_-]+).*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-45s - %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

.PHONY: help  ## Prints help message
help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

.PHONY: conda-lock  ## Creates conda-lock file
conda-lock:
	conda-lock --mamba -f ./env.yaml -p linux-64

.PHONY: conda-install  ## Creates env from conda-lock file
conda-install:
	conda-lock install --mamba -n kelp-wanted-competition conda-lock.yml

.PHONY: setup-pre-commit  ## Installs pre-commit hooks
setup-pre-commit:
	$(CONDA_ACTIVATE) kelp-wanted-competition ; pre-commit install

.PHONY: setup-editable  ## Installs the project in an editable mode
setup-editable:
	$(CONDA_ACTIVATE) kelp-wanted-competition ; pip install -e .

.PHONY: setup-local-env  ## Creates local environment and installs pre-commit hooks
setup-local-env: conda-install setup-pre-commit setup-editable

.PHONY: format  ## Runs code formatting (isort, black, flake8)
format:
	$(isort)
	$(black)
	$(flake8)

.PHONY: type-check  ## Runs type checking with mypy
type-check:
	pre-commit run --all-files mypy

.PHONY: test  ## Runs pytest
test:
	pytest -v tests/

.PHONY: testcov  ## Runs tests and generates coverage reports
testcov:
	@rm -rf htmlcov
	pytest -v --cov-report html --cov-report xml --cov=zf-powerbi-mongo tests/

.PHONY: mpc  ## Runs manual pre-commit stuff
mpc: format type-check test

.PHONY: docs  ## Build the documentation
docs:
	mkdocs build

.PHONY: pc  ## Runs pre-commit hooks
pc:
	$(pre-commit)

.PHONY: clean  ## Cleans artifacts
clean:
	rm -rf `find . -name __pycache__`
	rm -f `find . -type f -name '*.py[co]' `
	rm -f `find . -type f -name '*~' `
	rm -f `find . -type f -name '.*~' `
	rm -rf .cache
	rm -rf flame
	rm -rf htmlcov
	rm -rf .pytest_cache
	rm -rf *.egg-info
	rm -f .coverage
	rm -f .coverage.*
	rm -f coverage.*
	rm -rf build
	rm -rf perf.data*
	rm -rf zf-powerbi-mongo/*.so
	rm -rf .mypy_cache
	rm -rf .benchmark
	rm -rf .hypothesis
	rm -rf docs-site
