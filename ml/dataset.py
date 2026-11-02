from __future__ import annotations

import argparse
import os
from collections import Counter
from functools import lru_cache

import numpy as np
import torch
from torchvision.transforms import InterpolationMode, RandomAffine
from torch.utils.data import Dataset

from ml.config import CLASS_NAMES, DATA_ROOT


class PneumoniaTensorDataset(Dataset):
    """MedMNIST split converted to [-1, 1] tensors for GAN/CNN training."""

    def __init__(self, split: str, augment: bool = False, download: bool = True):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        try:
            from medmnist import PneumoniaMNIST
        except ImportError as exc:
            raise RuntimeError("Install project dependencies, including medmnist, first") from exc
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        self.dataset = PneumoniaMNIST(split=split, root=str(DATA_ROOT), download=download)
        self.images = np.asarray(self.dataset.imgs)
        self.labels = np.asarray(self.dataset.labels).reshape(-1).astype(np.int64)
        self.augment = augment
        self.augmentation = (
            RandomAffine(degrees=8, translate=(0.08, 0.08),
                         interpolation=InterpolationMode.BILINEAR, fill=0.0)
            if augment else None
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = torch.as_tensor(self.images[index], dtype=torch.float32)
        if image.ndim == 2:
            image = image.unsqueeze(0)
        else:
            image = image.permute(2, 0, 1)[:1]
        image = image / 127.5 - 1.0
        if self.augmentation:
            image = self.augmentation(image)
        return image, int(self.labels[index])


def save_distribution_artifacts(splits):
    """Save a measured CSV and plot for the official class/split counts."""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(DATA_ROOT / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    rows = [
        {"split": split["name"], "class_label": int(label),
         "class_name": CLASS_NAMES[int(label)], "count": int(count)}
        for split in splits for label, count in split["classes"].items()
    ]
    frame = pd.DataFrame(rows)
    csv_path = DATA_ROOT / "pneumoniamnist-distribution.csv"
    image_path = DATA_ROOT / "pneumoniamnist-distribution.png"
    frame.to_csv(csv_path, index=False)
    grouped = frame.pivot(index="split", columns="class_name", values="count").reindex(["train", "val", "test"])
    fig, axis = plt.subplots(figsize=(7.2, 3.6), layout="constrained")
    grouped.plot(kind="bar", ax=axis, color={"Normal": "#2eaaa2", "Pneumonia": "#596ed1"}, width=0.72)
    axis.set_title("PneumoniaMNIST class counts by official split", loc="left", fontsize=11, color="#27434d")
    axis.set_xlabel("")
    axis.set_ylabel("Images")
    axis.tick_params(axis="x", rotation=0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#e7eeee", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, ncol=2, loc="upper left")
    fig.savefig(image_path, dpi=150, facecolor="white")
    plt.close(fig)
    return csv_path, image_path


@lru_cache(maxsize=3)
def get_raw_split(split: str):
    """Load a MedMNIST split once for API exploration; returns (dataset, counts)."""
    from medmnist import PneumoniaMNIST

    dataset = PneumoniaMNIST(split=split, root=str(DATA_ROOT), download=True)
    labels = np.asarray(dataset.labels).reshape(-1).astype(int)
    counts = {label: int(Counter(labels).get(label, 0)) for label in CLASS_NAMES}
    return dataset, counts


def main():
    parser = argparse.ArgumentParser(description="Download and summarize PneumoniaMNIST")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    args = parser.parse_args()
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for split in splits:
        _, counts = get_raw_split(split)
        print(f"{split}: total={sum(counts.values())}, " + ", ".join(f"{CLASS_NAMES[k]}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
