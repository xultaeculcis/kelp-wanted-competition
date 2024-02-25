import argparse
import warnings
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import rasterio
from pydantic import field_validator
from rasterio.errors import NotGeoreferencedWarning
from tqdm import tqdm

from kelp.core.configs import ConfigBase
from kelp.utils.plotting import plot_sample

warnings.filterwarnings(
    action="ignore",
    category=NotGeoreferencedWarning,
)


class PreviewSubmissionConfig(ConfigBase):
    """Config for previewing the submission files."""

    test_data_dir: Path
    output_dir: Path
    submission_dir: Path
    first_n: int = 10

    @field_validator("submission_dir", mode="before")
    def validate_submission_dir(cls, value: Union[str, Path]) -> Union[str, Path]:
        if value == "latest":
            submission_dir = sorted([d for d in Path("data/submissions").iterdir() if d.is_dir()])[-1]
            return submission_dir
        return value


def parse_args() -> PreviewSubmissionConfig:
    """
    Parse command line arguments.

    Returns: An instance of PreviewSubmissionConfig.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--submission_dir", type=str, required=True)
    parser.add_argument("--first_n", type=int, default=10)
    args = parser.parse_args()
    cfg = PreviewSubmissionConfig(**vars(args))
    cfg.log_self()
    cfg.output_dir.mkdir(exist_ok=True, parents=True)
    return cfg


def plot_first_n_samples(data_dir: Path, submission_dir: Path, output_dir: Path, n: int = 10) -> None:
    """
    Plots first N samples from the submission directory.

    Args:
        data_dir: The path to the data directory with files used to generate predictions in the submission dir.
        submission_dir: The path to the submission directory.
        output_dir: The path to the output directory.
        n: The number of samples to plot.

    """
    output_dir.mkdir(exist_ok=True, parents=True)
    for fp in tqdm(sorted(list(data_dir.glob("*.tif")))[:n], "Plotting predictions"):
        tile = fp.stem.split("_")[0]
        with rasterio.open(data_dir / f"{tile}_satellite.tif") as src:
            input = src.read()
        with rasterio.open(submission_dir / "predictions" / f"{tile}_kelp.tif") as src:
            prediction = src.read(1)
        fig = plot_sample(
            input_arr=input,
            predictions_arr=prediction,
            suptitle=tile,
        )
        fig.savefig(output_dir / f"{fp.stem}.png")
        plt.close(fig)


def main() -> None:
    """Main entrypoint for plotting sample predictions from submission directory."""
    cfg = parse_args()
    plot_first_n_samples(
        data_dir=cfg.test_data_dir,
        submission_dir=cfg.submission_dir,
        output_dir=cfg.output_dir / cfg.submission_dir.name,
        n=cfg.first_n,
    )


if __name__ == "__main__":
    main()
