.DEFAULT_GOAL := help
isort = isort .
black = black .
flake8 = flake8 .
autoflake8 = autoflake8 .
mypy = mypy .
pre-commit = pre-commit run --all-files

DATA_DIR=data
SHELL=/bin/bash
SPLIT=0
RUN_DIR=mlruns/256237887236640917/2da570bb563e4172b329ef7d50d986e1
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

.PHONY: lock-env  ## Creates conda-lock file
lock-env:
	conda-lock --mamba -f ./env.yaml -p linux-64

.PHONY: create-env  ## Creates env from conda-lock file
create-env:
	conda-lock install --mamba -n kelp conda-lock.yml

.PHONY: setup-pre-commit  ## Installs pre-commit hooks
setup-pre-commit:
	$(CONDA_ACTIVATE) kelp ; pre-commit install

.PHONY: setup-editable  ## Installs the project in an editable mode
setup-editable:
	$(CONDA_ACTIVATE) kelp ; pip install -e .

.PHONY: configure-torch-ort  ## Configures torch-ort
configure-torch-ort:
	$(CONDA_ACTIVATE) kelp ; python -m torch_ort.configure

.PHONY: setup-local-env  ## Creates local environment and installs pre-commit hooks
setup-local-env: create-env setup-pre-commit setup-editable

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

.PHONY: sample-plotting  ## Runs tile plotting
sample-plotting:
	python ./kelp/entrypoints/sample_plotting.py \
 		--data_dir data/raw \
		--metadata_fp data/raw/metadata_fTq0l2T.csv \
		--output_dir data/processed

.PHONY: aoi-grouping  ## Runs AOI grouping
aoi-grouping:
	python ./kelp/entrypoints/aoi_grouping.py \
		--dem_dir data/processed/dem \
 		--output_dir data/processed/grouped_aoi_results \
 		--metadata_fp data/processed/stats/metadata_fTq0l2T.csv \
 		--batch_size 128 \
 		--similarity_threshold 0.97

.PHONY: eda  ## Runs EDA
eda:
	python ./kelp/entrypoints/eda.py \
 		--data_dir data/raw \
		--metadata_fp data/processed/grouped_aoi_results/metadata.parquet \
		--output_dir data/processed

.PHONY: calculate-band-stats  ## Runs band statistics calculation
calculate-band-stats:
	python ./kelp/entrypoints/calculate_band_stats.py \
 		--data_dir data/raw \
		--output_dir data/processed

.PHONY: train-val-test-split  ## Runs train-val-test split
train-val-test-split:
	python ./kelp/entrypoints/train_val_test_split.py \
		--dataset_metadata_fp data/processed/stats/dataset_stats.parquet \
		--seed 42 \
		--splits 10 \
		--output_dir data/processed

.PHONY: train-single-split  ## Trains single CV split
train-single-split:
	python ./kelp/entrypoints/train.py \
 		--data_dir data/raw \
		--output_dir mlruns \
		--metadata_fp data/processed/train_val_test_dataset.parquet \
		--cv_split $(SPLIT) \
		--batch_size 32 \
		--num_workers 6 \
		--image_size 352 \
		--normalization_strategy quantile \
		--architecture unet \
		--encoder resnet50 \
		--pretrained \
		--encoder_weights imagenet \
		--optimizer adamw \
		--weight_decay 1e-4 \
		--lr_scheduler onecycle \
		--pct_start 0.3 \
		--div_factor 2 \
		--final_div_factor 1e2 \
		--strategy no-freeze \
		--lr 3e-4 \
		--monitor_metric val/dice \
		--save_top_k 1 \
		--early_stopping_patience 7 \
		--precision 16-mixed \
		--epochs 10 \
		--band_order 2,3,4,0,1,5,6

.PHONY: train-all-splits  ## Trains on all splits
train-all-splits:
	make train-single-split SPLIT=0
	make train-single-split SPLIT=1
	make train-single-split SPLIT=2
	make train-single-split SPLIT=3
	make train-single-split SPLIT=4
	make train-single-split SPLIT=5
	make train-single-split SPLIT=6
	make train-single-split SPLIT=7
	make train-single-split SPLIT=8
	make train-single-split SPLIT=9

.PHONY: predict  ## Runs prediction
predict:
	python ./kelp/entrypoints/predict.py \
		--data_dir data/raw/test/images \
		--output_dir data/predictions \
		--model_checkpoint $(RUN_DIR)/artifacts/model \
		--original_training_config_fp $(RUN_DIR)/artifacts/config.yaml

.PHONY: submission  ## Generates submission file
submission:
	python ./kelp/entrypoints/submission.py \
		--predictions_dir data/predictions \
		--output_dir data/submissions

.PHONY: predict-and-submit  ## Runs inference and generates submission file
predict-and-submit:
	python ./kelp/entrypoints/submission.py \
		--data_dir data/raw/test/images \
		--output_dir data/submissions \
		--run_dir $(RUN_DIR)
