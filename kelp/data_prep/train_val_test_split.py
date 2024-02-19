from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Literal

import pandas as pd
from pydantic import field_validator
from sklearn.model_selection import StratifiedKFold, train_test_split

from kelp import consts
from kelp.core.configs import ConfigBase
from kelp.utils.logging import timed


class TrainTestSplitConfig(ConfigBase):
    dataset_metadata_fp: Path
    stratification_columns: List[str]
    split_strategy: Literal["cross_val", "random"] = "cross_val"
    random_split_train_size: float = 0.95
    seed: int = consts.reproducibility.SEED
    splits: int = 5
    output_dir: Path

    @field_validator("stratification_columns", mode="before")
    def validate_stratification_columns(cls, val: str) -> List[str]:
        return [s.strip() for s in val.split(",")]


def parse_args() -> TrainTestSplitConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_metadata_fp",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--stratification_columns",
        type=str,
        default="has_kelp,almost_all_water,qa_ok,high_corrupted_pixels_pct",
    )
    parser.add_argument(
        "--split_strategy",
        type=str,
        choices=["cross_val", "random"],
        default="cross_val",
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
        "--random_split_train_size",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    cfg = TrainTestSplitConfig(**vars(args))
    cfg.output_dir.mkdir(exist_ok=True, parents=True)
    cfg.log_self()
    return cfg


@timed
def load_data(fp: Path) -> pd.DataFrame:
    return pd.read_parquet(fp).rename(columns={"split": "original_split"})


@timed
def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["high_kelp_pixels_pct"].isin([False, None])]
    return df


@timed
def make_stratification_column(df: pd.DataFrame, stratification_columns: List[str]) -> pd.DataFrame:
    def make_stratification_key(series: pd.Series) -> str:
        vals = [f"{col}={str(series[col])}" for col in stratification_columns]
        return "-".join(vals)

    df["stratification"] = df.apply(lambda row: make_stratification_key(row), axis=1).astype("category")

    return df


@timed
def k_fold_split(df: pd.DataFrame, splits: int = 5, seed: int = consts.reproducibility.SEED) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)

    for i in range(splits):
        df[f"split_{i}"] = "train"

    for i, (_, val_idx) in enumerate(skf.split(df, df["stratification"])):
        df.loc[val_idx, f"split_{i}"] = "val"

    return df


@timed
def run_cross_val_split(
    train_samples: pd.DataFrame,
    test_samples: pd.DataFrame,
    splits: int = 5,
    seed: int = consts.reproducibility.SEED,
) -> pd.DataFrame:
    results = []
    for aoi_id, frame in train_samples[["aoi_id", "stratification"]].groupby("aoi_id"):
        results.append((aoi_id, frame["stratification"].value_counts().reset_index().iloc[0]["stratification"]))
    results_df = pd.DataFrame(results, columns=["aoi_id", "stratification"])
    train_val_samples = k_fold_split(results_df, splits=splits, seed=seed)
    train_samples = train_samples.drop("stratification", axis=1)
    train_samples = train_samples.merge(train_val_samples, how="inner", left_on="aoi_id", right_on="aoi_id")
    for split in range(splits):
        test_samples[f"split_{split}"] = "test"
    return train_samples


@timed
def run_random_split(
    train_samples: pd.DataFrame,
    test_samples: pd.DataFrame,
    random_split_train_size: float = 0.95,
    seed: int = consts.reproducibility.SEED,
) -> pd.DataFrame:
    X_train, X_val = train_test_split(
        train_samples,
        train_size=random_split_train_size,
        stratify=train_samples["stratification"],
        shuffle=True,
        random_state=seed,
    )
    X_train["split_0"] = "train"
    X_val["split_0"] = "val"
    test_samples["split_0"] = "test"
    return pd.concat([X_train, X_val])


@timed
def split_dataset(
    df: pd.DataFrame,
    split_strategy: Literal["cross_val", "random"] = "cross_val",
    random_split_train_size: float = 0.95,
    splits: int = 5,
    seed: int = consts.reproducibility.SEED,
) -> pd.DataFrame:
    train_samples = df[df["original_split"] == "train"].copy()
    test_samples = df[df["original_split"] == "test"].copy()
    if split_strategy == "cross_val":
        train_samples = run_cross_val_split(
            train_samples=train_samples,
            test_samples=test_samples,
            splits=splits,
            seed=seed,
        )
    elif split_strategy == "random":
        train_samples = run_random_split(
            train_samples=train_samples,
            test_samples=test_samples,
            random_split_train_size=random_split_train_size,
            seed=seed,
        )
    else:
        raise ValueError(f"{split_strategy=} is not supported")
    return pd.concat([train_samples, test_samples])


@timed
def save_data(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    df.to_parquet(output_path, index=False)
    return df


@timed
def main() -> None:
    cfg = parse_args()
    (
        load_data(cfg.dataset_metadata_fp)
        .pipe(filter_data)
        .pipe(make_stratification_column, stratification_columns=cfg.stratification_columns)
        .pipe(
            split_dataset,
            split_strategy=cfg.split_strategy,
            splits=cfg.splits,
            random_split_train_size=cfg.random_split_train_size,
            seed=cfg.seed,
        )
        .pipe(save_data, output_path=cfg.output_dir / f"train_val_test_dataset_strategy={cfg.split_strategy}.parquet")
    )


if __name__ == "__main__":
    main()
