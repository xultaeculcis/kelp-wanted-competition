import argparse
import tarfile
from pathlib import Path

from kelp.core.configs import ConfigBase
from kelp.utils.logging import get_logger

_logger = get_logger(__name__)


class SubmissionConfig(ConfigBase):
    predictions_dir: Path
    output_dir: Path


def parse_args() -> SubmissionConfig:
    """Parse command line arguments for making a submission file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    cfg = SubmissionConfig(**vars(args))
    cfg.log_self()
    return cfg


def create_submission_tar(preds_dir: Path, output_dir: Path, submission_file_name: str = "submission.tar") -> None:
    """
    Creates submission TAR archive.

    Args:
        preds_dir: The directory with predictions.
        output_dir: The output directory where the submission file will be saved.
        submission_file_name: The name of the submission file.

    """
    # Create a TAR file
    with tarfile.open(output_dir / submission_file_name, "w") as tar:
        for fp in preds_dir.glob("*_kelp.tif"):
            tar.add(fp, arcname=fp.name)

    _logger.info(f"Submission TAR file '{(output_dir / submission_file_name).as_posix()}' created successfully.")


def main() -> None:
    """Main entrypoint for creating submission from predictions directory."""
    cfg = parse_args()
    cfg.output_dir.mkdir(exist_ok=True, parents=True)
    create_submission_tar(preds_dir=cfg.predictions_dir, output_dir=cfg.output_dir)


if __name__ == "__main__":
    main()
