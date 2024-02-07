.DEFAULT_GOAL := help
isort = isort .
black = black .
flake8 = flake8 .
autoflake8 = autoflake8 .
mypy = mypy .
pre-commit = pre-commit run --all-files

DATA_DIR=data
PREDS_OUTPUT_DIR=data/predictions
SHELL=/bin/bash
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
	python ./kelp/data_prep/sample_plotting.py \
 		--data_dir data/raw \
		--metadata_fp data/raw/metadata_fTq0l2T.csv \
		--output_dir data/processed

.PHONY: aoi-grouping  ## Runs AOI grouping
aoi-grouping:
	python ./kelp/data_prep/aoi_grouping.py \
		--dem_dir data/processed/dem \
 		--output_dir data/processed/grouped_aoi_results \
 		--metadata_fp data/processed/stats/metadata_fTq0l2T.csv \
 		--batch_size 128 \
 		--similarity_threshold 0.97

.PHONY: eda  ## Runs EDA
eda:
	python ./kelp/data_prep/eda.py \
 		--data_dir data/raw \
		--metadata_fp data/processed/grouped_aoi_results/metadata.parquet \
		--output_dir data/processed

.PHONY: calculate-band-stats  ## Runs band statistics calculation
calculate-band-stats:
	python ./kelp/data_prep/calculate_band_stats.py \
 		--data_dir data/raw \
		--output_dir data/processed

.PHONY: train-val-test-split-cv  ## Runs train-val-test split using cross validation
train-val-test-split-cv:
	python ./kelp/data_prep/train_val_test_split.py \
		--dataset_metadata_fp data/processed/stats/dataset_stats.parquet \
		--split_strategy cross_val \
		--seed 42 \
		--splits 10 \
		--output_dir data/processed

.PHONY: train-val-test-split-random  ## Runs train-val-test split using random split
train-val-test-split-random:
	python ./kelp/data_prep/train_val_test_split.py \
		--dataset_metadata_fp data/processed/stats/dataset_stats.parquet \
		--split_strategy random_split \
		--random_split_train_size 0.98 \
		--seed 42 \
		--output_dir data/processed

.PHONY: train  ## Trains single CV split
train:
	python ./kelp/nn/training/train.py \
 		--data_dir data/raw \
		--output_dir mlruns \
		--metadata_fp data/processed/train_val_test_dataset.parquet \
		--dataset_stats_fp data/processed/2023-12-31T20:30:39-stats-fill_value=nan-mask_using_qa=True-mask_using_water_mask=True.json \
		--cv_split 8 \
		--batch_size 32 \
		--num_workers 4 \
		--bands R,G,B,SWIR,NIR,QA,DEM \
		--spectral_indices DEMWM,NDVI,ATSAVI,AVI,CI,ClGreen,GBNDVI,GVMI,IPVI,KIVU,MCARI,MVI,NormNIR,PNDVI,SABI,WDRVI,mCRIG \
		--image_size 352 \
		--resize_strategy pad \
		--interpolation nearest \
		--fill_missing_pixels_with_torch_nan \
		--mask_using_qa \
		--mask_using_water_mask \
		--use_weighted_sampler \
		--samples_per_epoch 10240 \
		--has_kelp_importance_factor 2.0 \
		--kelp_pixels_pct_importance_factor 0.5 \
		--qa_ok_importance_factor 0.5 \
		--qa_corrupted_pixels_pct_importance_factor -0.5 \
		--almost_all_water_importance_factor -1.0 \
		--dem_nan_pixels_pct_importance_factor 0.0 \
		--dem_zero_pixels_pct_importance_factor -0.25 \
		--normalization_strategy quantile \
		--architecture unet \
		--encoder tu-efficientnet_b5 \
		--pretrained \
		--encoder_weights imagenet \
		--lr 3e-4 \
		--optimizer adamw \
		--weight_decay 1e-4 \
		--lr_scheduler onecycle \
		--onecycle_pct_start 0.1 \
		--onecycle_div_factor 2 \
		--onecycle_final_div_factor 1e2 \
		--loss dice \
		--tta \
		--tta_merge_mode max \
		--monitor_metric val/dice \
		--save_top_k 1 \
		--early_stopping_patience 7 \
		--precision bf16-mixed \
		--epochs 10

.PHONY: predict  ## Runs prediction
predict:
	python ./kelp/nn/inference/predict.py \
		--data_dir data/raw/test/images \
		--dataset_stats_dir=data/processed \
		--output_dir $(PREDS_OUTPUT_DIR) \
		--run_dir $(RUN_DIR) \
		--soft_labels \
		--precision bf16-mixed

.PHONY: submission  ## Generates submission file
submission:
	python ./kelp/core/submission.py \
		--predictions_dir data/predictions \
		--output_dir data/submissions

.PHONY: predict-and-submit  ## Runs inference and generates submission file
predict-and-submit:
	python ./kelp/nn/inference/predict_and_submit.py \
		--data_dir data/raw/test/images \
		--dataset_stats_dir=data/processed \
		--output_dir data/submissions \
		--run_dir $(RUN_DIR) \
		--decision_threshold 0.48 \
		--precision bf16-mixed

.PHONY: eval  ## Runs evaluation for selected run
eval:
	python ./kelp/nn/training/eval.py \
		--data_dir data/raw \
		--metadata_dir data/processed \
		--dataset_stats_dir data/processed \
		--run_dir $(RUN_DIR) \
		--output_dir mlruns \
		--precision bf16-mixed \
		--decision_threshold=0.48 \
		--experiment_name model-eval-exp

.PHONY: average-predictions  ## Runs prediction averaging
average-predictions:
	python ./kelp/nn/inference/average_predictions.py \
		--predictions_dir=data/predictions/v4 \
		--output_dir=data/submissions/avg \
		--decision_threshold=0.48 \
		--fold_0_weight=0.666 \
		--fold_1_weight=0.0 \
		--fold_2_weight=0.666 \
		--fold_3_weight=0.88 \
		--fold_4_weight=0.637 \
		--fold_5_weight=0.59 \
		--fold_6_weight=0.733 \
		--fold_7_weight=0.63 \
		--fold_8_weight=1.0 \
		--fold_9_weight=0.0 \
		--preview_submission \
		--test_data_dir=data/raw/test/images \
		--preview_first_n=10

.PHONY: cv-predict  ## Runs inference on specified folds, averages the predictions and generates submission file
cv-predict:
	make predict RUN_DIR=data/aml/Job_sad_pummelo_nv069lvn_OutputsAndLogs PREDS_OUTPUT_DIR=data/predictions/v5/fold=0
	make predict RUN_DIR=data/aml/Job_silver_oyster_yppwcpr4_OutputsAndLogs PREDS_OUTPUT_DIR=data/predictions/v5/fold=1
	make predict RUN_DIR=data/aml/Job_hungry_loquat_qkrw2n2p_OutputsAndLogs PREDS_OUTPUT_DIR=data/predictions/v5/fold=2
	make predict RUN_DIR=data/aml/Job_elated_atemoya_31s98pwg_OutputsAndLogs PREDS_OUTPUT_DIR=data/predictions/v5/fold=3
	make predict RUN_DIR=data/aml/Job_brave_loquat_w4lm7093_OutputsAndLogs PREDS_OUTPUT_DIR=data/predictions/v5/fold=4
	make predict RUN_DIR=data/aml/Job_gentle_stamp_wry90x9f_OutputsAndLogs PREDS_OUTPUT_DIR=data/predictions/v5/fold=5
	make predict RUN_DIR=data/aml/Job_model_training_exp_67_OutputsAndLogs PREDS_OUTPUT_DIR=data/predictions/v5/fold=6
	make predict RUN_DIR=data/aml/Job_model_training_exp_65_OutputsAndLogs PREDS_OUTPUT_DIR=data/predictions/v5/fold=7
	make predict RUN_DIR=data/aml/Job_gentle_eagle_qwsnx2hc_OutputsAndLogs PREDS_OUTPUT_DIR=data/predictions/v5/fold=8
	make predict RUN_DIR=data/aml/Job_sharp_iron_dfcsht2c_OutputsAndLogs PREDS_OUTPUT_DIR=data/predictions/v5/fold=9
	make average-predictions PREDS_OUTPUT_DIR=data/predictions/v5

eval-many:
	make eval RUN_DIR=data/aml/Job_frank_key_k8b7jv40_OutputsAndLogs
	make eval RUN_DIR=data/aml/Job_bold_street_rcrzx0xq_OutputsAndLogs
	make eval RUN_DIR=data/aml/Job_green_soca_fxt5lbcm_OutputsAndLogs
	make eval RUN_DIR=data/aml/Job_neat_snake_bgbxxg7d_OutputsAndLogs
	make eval RUN_DIR=data/aml/Job_brave_loquat_w4lm7093_OutputsAndLogs
	make eval RUN_DIR=data/aml/Job_salmon_worm_fkc38xhc_OutputsAndLogs
	make eval RUN_DIR=data/aml/Job_kind_sugar_xmpt108y_OutputsAndLogs
	make eval RUN_DIR=data/aml/Job_bubbly_store_bdp54r2f_OutputsAndLogs
	make eval RUN_DIR=data/aml/Job_gentle_eagle_qwsnx2hc_OutputsAndLogs
	make eval RUN_DIR=data/aml/Job_sharp_iron_dfcsht2c_OutputsAndLogs
