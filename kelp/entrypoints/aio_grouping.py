import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm

from kelp.core.configs import ConfigBase


class AOIGroupingConfig(ConfigBase):
    dem_dir: Path
    metadata_fp: Path
    output_dir: Path
    batch_size: int = 32
    similarity_threshold: float = 0.95


class ImageDataset(Dataset):
    def __init__(self, fps: list[Path], transform: Callable[[Any], Tensor]) -> None:
        self.fps = fps
        self.transform = transform

    def __getitem__(self, idx: int) -> Tensor:
        with open(self.fps[idx], "rb") as f:
            img = Image.open(f)
            sample = img.convert("RGB")
        sample = self.transform(sample)
        return sample

    def __len__(self) -> int:
        return len(self.fps)


def parse_args() -> AOIGroupingConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dem_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--metadata_fp",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.8,
    )
    args = parser.parse_args()
    cfg = AOIGroupingConfig(**vars(args))
    cfg.log_self()
    cfg.output_dir.mkdir(exist_ok=True, parents=True)
    return cfg


def generate_embeddings(
    data_folder: Path,
    batch_size: int = 32,
    num_workers: int = 6,
) -> tuple[np.ndarray, ImageDataset]:  # type: ignore[type-arg]
    fps = sorted(list(data_folder.glob("*.png")))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pre-trained ResNet-50 model
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.eval()
    model.to(device)

    # Transformation for the input images
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load images from folder
    dataset = ImageDataset(fps=fps, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    # Extract features
    features = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating embeddings"):
            outputs: Tensor = model(batch.to(device))
            features.append(outputs.detach().cpu())

    # Convert to numpy array
    features_arr = torch.cat(features, dim=0).numpy()

    return features_arr, dataset


def find_similar_images(
    data_folder: Path,
    threshold: float = 0.95,
    batch_size: int = 32,
    num_workers: int = 6,
) -> dict[str, list[str]]:
    # Generate embeddings
    features, dataset = generate_embeddings(data_folder=data_folder, batch_size=batch_size, num_workers=num_workers)

    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(features)

    # Group similar images
    groups = {}
    for i in tqdm(range(len(similarity_matrix)), desc="Grouping similar images", total=len(similarity_matrix)):
        similar_images = []
        for j in range(len(similarity_matrix[i])):
            if i != j and similarity_matrix[i][j] >= threshold:
                similar_images.append(dataset.fps[j].as_posix())  # Add image path
        if similar_images:
            groups[dataset.fps[i].as_posix()] = similar_images  # Key image path

    return groups


def group_aoi(
    dem_dir: Path,
    output_dir: Path,
    batch_size: int = 32,
    num_workers: int = 6,
    similarity_threshold: float = 0.95,
) -> None:
    groups = find_similar_images(
        data_folder=dem_dir, threshold=similarity_threshold, batch_size=batch_size, num_workers=num_workers
    )

    # Save to JSON file
    with open(output_dir / "image_groups.json", "w") as file:
        json.dump(groups, file, indent=4)


def main() -> None:
    cfg = parse_args()
    group_aoi(
        dem_dir=cfg.dem_dir,
        output_dir=cfg.output_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        similarity_threshold=cfg.similarity_threshold,
    )


if __name__ == "__main__":
    main()
