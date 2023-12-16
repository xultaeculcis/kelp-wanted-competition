import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from kelp import consts
from kelp.core.configs import ConfigBase


class TrainTestSplitConfig(ConfigBase):
    dataset_metadata_fp: Path
    seed: int = consts.reproducibility.SEED
    splits: int = 5
    output_dir: Path


def parse_args() -> TrainTestSplitConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_metadata_fp",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=consts.reproducibility.SEED,
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    cfg = TrainTestSplitConfig(**vars(args))
    cfg.log_self()
    return cfg


def load_data(fp: Path) -> pd.DataFrame:
    return pd.read_parquet(fp).rename(columns={"split": "original_split"})


def make_stratification_column(df: pd.DataFrame) -> pd.DataFrame:
    df["stratification"] = df.apply(
        lambda row: f"{row['qa_ok']}-{row['has_kelp']}-{row['dem_has_nans']}-{row['high_corrupted_pixels_pct']}",
        axis=1,
    ).astype("category")
    return df


def k_fold_split(df: pd.DataFrame, splits: int = 5, seed: int = consts.reproducibility.SEED) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)

    for i in range(splits):
        df[f"split_{i}"] = "train"

    for i, (_, val_idx) in enumerate(skf.split(df, df["stratification"])):
        df.loc[val_idx, f"split_{i}"] = "val"

    return df


def split_dataset(df: pd.DataFrame, splits: int = 5, seed: int = consts.reproducibility.SEED) -> pd.DataFrame:
    train_samples = df[df["split"] == "train"].copy()
    test_samples = df[df["split"] == "test"].copy()
    train_val_samples = k_fold_split(train_samples, splits=splits, seed=seed)
    for split in range(splits):
        test_samples[f"split_{split}"] = "test"
    return pd.concat([train_val_samples, test_samples])


def save_data(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    df.to_parquet(output_dir / "train_val_test_dataset.parquet", index=False)
    return df


def main() -> None:
    cfg = parse_args()
    data = pd.read_parquet(cfg.dataset_metadata_fp)
    (
        data.pipe(make_stratification_column)
        .pipe(split_dataset, splits=cfg.splits, seed=cfg.seed)
        .pipe(save_data, output_dir=cfg.output_dir)
    )


if __name__ == "__main__":
    main()
