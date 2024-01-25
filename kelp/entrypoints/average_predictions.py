import argparse
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import rasterio
import yaml
from tqdm import tqdm

from kelp.core.configs import ConfigBase
from kelp.entrypoints.submission import create_submission_tar

FOLDS = 10


class AveragePredictionsConfig(ConfigBase):
    predictions_dir: Path
    output_dir: Path
    decision_threshold: float = 0.5
    fold_0_weight: float = 1.0
    fold_1_weight: float = 1.0
    fold_2_weight: float = 1.0
    fold_3_weight: float = 1.0
    fold_4_weight: float = 1.0
    fold_5_weight: float = 1.0
    fold_6_weight: float = 1.0
    fold_7_weight: float = 1.0
    fold_8_weight: float = 1.0
    fold_9_weight: float = 1.0

    @property
    def weights(self) -> List[float]:
        return [
            self.fold_0_weight,
            self.fold_1_weight,
            self.fold_2_weight,
            self.fold_3_weight,
            self.fold_4_weight,
            self.fold_5_weight,
            self.fold_6_weight,
            self.fold_7_weight,
            self.fold_8_weight,
            self.fold_9_weight,
        ]


def parse_args() -> AveragePredictionsConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--decision_threshold", type=float, default=0.5)
    parser.add_argument("--fold_0_weight", type=float, default=1.0)
    parser.add_argument("--fold_1_weight", type=float, default=1.0)
    parser.add_argument("--fold_2_weight", type=float, default=1.0)
    parser.add_argument("--fold_3_weight", type=float, default=1.0)
    parser.add_argument("--fold_4_weight", type=float, default=1.0)
    parser.add_argument("--fold_5_weight", type=float, default=1.0)
    parser.add_argument("--fold_6_weight", type=float, default=1.0)
    parser.add_argument("--fold_7_weight", type=float, default=1.0)
    parser.add_argument("--fold_8_weight", type=float, default=1.0)
    parser.add_argument("--fold_9_weight", type=float, default=1.0)
    args = parser.parse_args()
    cfg = AveragePredictionsConfig(**vars(args))
    cfg.log_self()
    cfg.output_dir.mkdir(exist_ok=True, parents=True)
    return cfg


def average_predictions(
    preds_dir: Path,
    output_dir: Path,
    weights: List[float],
    decision_threshold: float = 0.5,
) -> None:
    if len(weights) != FOLDS:
        raise ValueError("Number of weights must match the number of folds")
    if len(list(preds_dir.iterdir())) != FOLDS:
        raise ValueError("Number of prediction dirs must match the number of folds")

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = {}
    fold_dirs = [preds_dir / f"fold={i}" for i in range(FOLDS)]

    for fold_dir, weight in zip(fold_dirs, weights):
        for pred_file in tqdm(sorted(list(fold_dir.glob("*.tif"))), desc=f"Processing files for {fold_dir.name}"):
            file_name = pred_file.name
            with rasterio.open(pred_file) as src:
                pred_array = src.read(1) * weight
                if file_name not in predictions:
                    predictions[file_name] = {
                        "data": np.zeros_like(pred_array, dtype=np.float32),
                        "profile": src.profile,
                        "count": 1,
                    }
                predictions[file_name]["data"] += pred_array
                predictions[file_name]["count"] += 1

    for file_name, content in tqdm(predictions.items(), desc="Saving predictions"):
        content["data"] = content["data"] / content["count"]
        content["data"] = np.where(content["data"] >= decision_threshold, 1, 0).astype(np.uint8)
        output_file = output_dir / file_name
        profile = content["profile"]
        with rasterio.open(output_file, "w", **profile) as dst:
            dst.write(content["data"].astype(rasterio.uint8), 1)


def main() -> None:
    cfg = parse_args()
    now = datetime.utcnow().isoformat()
    out_dir = cfg.output_dir / now
    preds_dir = cfg.output_dir / now / "predictions"
    preds_dir.mkdir(exist_ok=False, parents=True)
    avg_preds_config = cfg.model_dump(mode="json")
    (out_dir / "predict_config.yaml").write_text(yaml.dump(avg_preds_config))
    average_predictions(
        preds_dir=cfg.predictions_dir,
        output_dir=preds_dir,
        weights=cfg.weights,
        decision_threshold=cfg.decision_threshold,
    )
    create_submission_tar(
        preds_dir=preds_dir,
        output_dir=out_dir,
    )


if __name__ == "__main__":
    main()
