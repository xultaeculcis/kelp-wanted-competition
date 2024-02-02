import argparse

from kelp import consts
from kelp.xgb.training.cfg import TrainConfig


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--dataset_fp", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--spectral_indices", type=str)
    parser.add_argument("--sample_size", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=consts.reproducibility.SEED)
    parser.add_argument("--plot_n_samples", type=int, default=10)
    parser.add_argument("--experiment", type=str, default="train-tree-clf-exp")
    parser.add_argument("--explain_model", action="store_true")
    args = parser.parse_args()
    cfg = TrainConfig(**vars(args))
    cfg.log_self()
    cfg.output_dir.mkdir(exist_ok=True, parents=True)
    return cfg
