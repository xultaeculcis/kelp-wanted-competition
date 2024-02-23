.DEFAULT_GOAL := help
isort = isort .
black = black .
flake8 = flake8 .
autoflake8 = autoflake8 .
mypy = mypy .
pre-commit = pre-commit run --all-files

DATA_DIR=data
PREDS_INPUT_DIR=data/raw/test/images
PREDS_OUTPUT_DIR=data/predictions
SHELL=/bin/bash
RUN_DIR=mlruns/256237887236640917/2da570bb563e4172b329ef7d50d986e1

AVG_PREDS_VERSION=v19
AVG_PREDS_OUTPUT_DIR=data/submissions/avg

FOLD_0_RUN_DIR=data/aml/Job_strong_door_yrq9zpmd_OutputsAndLogs
FOLD_1_RUN_DIR=data/aml/Job_keen_evening_3xnlbrsr_OutputsAndLogs
FOLD_2_RUN_DIR=data/aml/Job_hungry_loquat_qkrw2n2p_OutputsAndLogs
FOLD_3_RUN_DIR=data/aml/Job_elated_atemoya_31s98pwg_OutputsAndLogs
FOLD_4_RUN_DIR=data/aml/Job_nice_cheetah_grnc5x72_OutputsAndLogs
FOLD_5_RUN_DIR=data/aml/Job_willing_pin_72ss6cnc_OutputsAndLogs
FOLD_6_RUN_DIR=data/aml/Job_model_training_exp_67_OutputsAndLogs
FOLD_7_RUN_DIR=data/aml/Job_model_training_exp_65_OutputsAndLogs
FOLD_8_RUN_DIR=data/aml/Job_yellow_evening_cmy9cnv7_OutputsAndLogs
FOLD_9_RUN_DIR=data/aml/Job_icy_market_4l11bvw2_OutputsAndLogs

BEST_SINGLE_MODEL_RUN_DIR=data/aml/Job_sincere_tangelo_dm0xsbhc_OutputsAndLogs

FOLD_NUMBER=8
CHECKPOINT=best

OLD_FOLD_0_WEIGHT=0.666
OLD_FOLD_1_WEIGHT=0.5
OLD_FOLD_2_WEIGHT=0.666
OLD_FOLD_3_WEIGHT=0.88
OLD_FOLD_4_WEIGHT=0.637
OLD_FOLD_5_WEIGHT=0.59
OLD_FOLD_6_WEIGHT=0.733
OLD_FOLD_7_WEIGHT=0.63
OLD_FOLD_8_WEIGHT=1.0
OLD_FOLD_9_WEIGHT=0.2

FOLD_0_WEIGHT=1
FOLD_1_WEIGHT=1
FOLD_2_WEIGHT=1
FOLD_3_WEIGHT=1
FOLD_4_WEIGHT=1
FOLD_5_WEIGHT=1
FOLD_6_WEIGHT=1
FOLD_7_WEIGHT=1
FOLD_8_WEIGHT=1
FOLD_9_WEIGHT=1

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

.PHONY: lock  ## Creates conda-lock file
lock:
	conda-lock --mamba -f ./env.yaml -p linux-64

.PHONY: env  ## Creates env from conda-lock file
env:
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

.PHONY: local-env  ## Creates local environment and installs pre-commit hooks
local-env: env setup-pre-commit setup-editable

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
 		--output_dir data/processed/grouped_aoi_results/sim_th=0.97 \
 		--metadata_fp data/raw/metadata_fTq0l2T.csv \
 		--batch_size 128 \
 		--similarity_threshold 0.97

.PHONY: eda  ## Runs EDA
eda:
	python ./kelp/data_prep/eda.py \
 		--data_dir data/raw \
		--metadata_fp data/processed/grouped_aoi_results/sim_th=0.97/metadata_similarity_threshold=0.97.parquet \
		--output_dir data/processed/stats_97

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
		--cv_split $(FOLD_NUMBER) \
		--batch_size 32 \
		--num_workers 4 \
		--bands R,G,B,SWIR,NIR,QA,DEM \
		--spectral_indices DEMWM,NDVI,ATSAVI,AVI,CI,ClGreen,GBNDVI,GVMI,IPVI,KIVU,MCARI,MVI,NormNIR,PNDVI,SABI,WDRVI,mCRIG \
		--image_size 352 \
		--resize_strategy pad \
		--interpolation nearest \
		--fill_missing_pixels_with_torch_nan True \
		--mask_using_qa True \
		--mask_using_water_mask True \
		--use_weighted_sampler True \
		--samples_per_epoch 10240 \
		--has_kelp_importance_factor 3 \
		--kelp_pixels_pct_importance_factor 0.2 \
		--qa_ok_importance_factor 0 \
		--qa_corrupted_pixels_pct_importance_factor -1 \
		--almost_all_water_importance_factor 0.5 \
		--dem_nan_pixels_pct_importance_factor 0.25 \
		--dem_zero_pixels_pct_importance_factor -1 \
		--normalization_strategy quantile \
		--architecture unet \
		--encoder tu-efficientnet_b5 \
		--pretrained True \
		--encoder_weights imagenet \
		--lr 3e-4 \
		--optimizer adamw \
		--weight_decay 1e-4 \
		--loss dice \
		--monitor_metric val/dice \
		--save_top_k 1 \
		--early_stopping_patience 50 \
		--precision bf16-mixed \
		--epochs 50 \
		--swa False

.PHONY: predict  ## Runs prediction
predict:
	python ./kelp/nn/inference/predict.py \
		--data_dir $(PREDS_INPUT_DIR) \
		--dataset_stats_dir=data/processed \
		--output_dir $(PREDS_OUTPUT_DIR) \
		--run_dir $(RUN_DIR) \
		--use_checkpoint $(CHECKPOINT) \
		--decision_threshold 0.48 \
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
		--output_dir data/submissions/single-model \
		--run_dir $(BEST_SINGLE_MODEL_RUN_DIR) \
		--preview_submission \
		--decision_threshold 0.45 \
		--precision bf16-mixed

.PHONY: eval  ## Runs evaluation for selected run
eval:
	python ./kelp/nn/training/eval.py \<<<
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
		--predictions_dirs \
			data/predictions/$(AVG_PREDS_VERSION)/fold=0 \
			data/predictions/$(AVG_PREDS_VERSION)/fold=1 \
			data/predictions/$(AVG_PREDS_VERSION)/fold=2 \
			data/predictions/$(AVG_PREDS_VERSION)/fold=3 \
			data/predictions/$(AVG_PREDS_VERSION)/fold=4 \
			data/predictions/$(AVG_PREDS_VERSION)/fold=5 \
			data/predictions/$(AVG_PREDS_VERSION)/fold=6 \
			data/predictions/$(AVG_PREDS_VERSION)/fold=7 \
			data/predictions/$(AVG_PREDS_VERSION)/fold=8 \
			data/predictions/$(AVG_PREDS_VERSION)/fold=9 \
		--weights \
			$(FOLD_0_WEIGHT) \
			$(FOLD_1_WEIGHT) \
			$(FOLD_2_WEIGHT) \
			$(FOLD_3_WEIGHT) \
			$(FOLD_4_WEIGHT) \
			$(FOLD_5_WEIGHT) \
			$(FOLD_6_WEIGHT) \
			$(FOLD_7_WEIGHT) \
			$(FOLD_8_WEIGHT) \
			$(FOLD_9_WEIGHT) \
		--output_dir=$(AVG_PREDS_OUTPUT_DIR) \
		--decision_threshold=0.48 \
		--test_data_dir=$(PREDS_INPUT_DIR) \
		--preview_submission \
		--preview_first_n=10

.PHONY: cv-predict  ## Runs inference on specified folds, averages the predictions and generates submission file
cv-predict:
	make predict RUN_DIR=$(FOLD_0_RUN_DIR) PREDS_OUTPUT_DIR=data/predictions/$(AVG_PREDS_VERSION)/fold=0
	make predict RUN_DIR=$(FOLD_1_RUN_DIR) PREDS_OUTPUT_DIR=data/predictions/$(AVG_PREDS_VERSION)/fold=1
	make predict RUN_DIR=$(FOLD_2_RUN_DIR) PREDS_OUTPUT_DIR=data/predictions/$(AVG_PREDS_VERSION)/fold=2
	make predict RUN_DIR=$(FOLD_3_RUN_DIR) PREDS_OUTPUT_DIR=data/predictions/$(AVG_PREDS_VERSION)/fold=3
	make predict RUN_DIR=$(FOLD_4_RUN_DIR) PREDS_OUTPUT_DIR=data/predictions/$(AVG_PREDS_VERSION)/fold=4
	make predict RUN_DIR=$(FOLD_5_RUN_DIR) PREDS_OUTPUT_DIR=data/predictions/$(AVG_PREDS_VERSION)/fold=5
	make predict RUN_DIR=$(FOLD_6_RUN_DIR) PREDS_OUTPUT_DIR=data/predictions/$(AVG_PREDS_VERSION)/fold=6
	make predict RUN_DIR=$(FOLD_7_RUN_DIR) PREDS_OUTPUT_DIR=data/predictions/$(AVG_PREDS_VERSION)/fold=7
	make predict RUN_DIR=$(FOLD_8_RUN_DIR) PREDS_OUTPUT_DIR=data/predictions/$(AVG_PREDS_VERSION)/fold=8
	make predict RUN_DIR=$(FOLD_9_RUN_DIR) PREDS_OUTPUT_DIR=data/predictions/$(AVG_PREDS_VERSION)/fold=9
	make average-predictions

.PHONY: eval-many  ## Runs evaluation for specified runs
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

.PHONY: eval-from-folders  ## Runs evaluation by comparing predictions to ground truth mask
eval-from-folders:
	python kelp/nn/training/eval_from_folders.py \
		--gt_dir=$(GT_DIR) \
		--preds_dir=$(PREDS_DIR) \
		--tags fold_0_run_dir=$(FOLD_0_RUN_DIR) \
			fold_1_run_dir=$(FOLD_1_RUN_DIR) \
			fold_2_run_dir=$(FOLD_2_RUN_DIR) \
			fold_3_run_dir=$(FOLD_3_RUN_DIR) \
			fold_4_run_dir=$(FOLD_4_RUN_DIR) \
			fold_5_run_dir=$(FOLD_5_RUN_DIR) \
			fold_6_run_dir=$(FOLD_6_RUN_DIR) \
			fold_7_run_dir=$(FOLD_7_RUN_DIR) \
			fold_8_run_dir=$(FOLD_8_RUN_DIR) \
			fold_9_run_dir=$(FOLD_9_RUN_DIR) \
			fold_0_weight=$(FOLD_0_WEIGHT) \
			fold_1_weight=$(FOLD_1_WEIGHT) \
			fold_2_weight=$(FOLD_2_WEIGHT) \
			fold_3_weight=$(FOLD_3_WEIGHT) \
			fold_4_weight=$(FOLD_4_WEIGHT) \
			fold_5_weight=$(FOLD_5_WEIGHT) \
			fold_6_weight=$(FOLD_6_WEIGHT) \
			fold_7_weight=$(FOLD_7_WEIGHT) \
			fold_8_weight=$(FOLD_8_WEIGHT) \
			fold_9_weight=$(FOLD_9_WEIGHT) \
			soft_labels=True \
			split_decision_threshold=None \
			decision_threshold=0.48 \
			tta=False \
			tta_merge_mode=mean \
			precision=bf16-mixed

.PHONY: eval-ensemble  ## Runs ensemble evaluation
eval-ensemble:
	rm -rf data/predictions/eval_results
	make cv-predict AVG_PREDS_VERSION=eval PREDS_INPUT_DIR=data/raw/splits/split_8/images AVG_PREDS_OUTPUT_DIR=data/predictions/eval_results
	make average-predictions AVG_PREDS_VERSION=eval PREDS_INPUT_DIR=data/raw/splits/split_8/images AVG_PREDS_OUTPUT_DIR=data/predictions/eval_results
	make eval-from-folders GT_DIR=data/raw/splits/split_8/masks PREDS_DIR=data/predictions/eval_results

.PHONY: train-all-folds  ## Trains all CV folds
train-all-folds:
	make train FOLD_NUMBER=0
	make train FOLD_NUMBER=1
	make train FOLD_NUMBER=2
	make train FOLD_NUMBER=3
	make train FOLD_NUMBER=4
	make train FOLD_NUMBER=5
	make train FOLD_NUMBER=6
	make train FOLD_NUMBER=7
	make train FOLD_NUMBER=8
	make train FOLD_NUMBER=9
