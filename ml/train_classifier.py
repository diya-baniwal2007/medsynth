from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from ml.classifier import ChestXrayCNN
from ml.config import CHECKPOINT_ROOT, EXPERIMENT_ROOT
from ml.dataset import PneumoniaTensorDataset
from ml.generate_images import generate


class TensorRows(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        image, label = self.rows[idx]
        return image, label


def train_classifier(variant="baseline", epochs=5, batch_size=64, seed=42, on_epoch=None, run_id=None):
    if variant not in {"baseline", "conventional", "gan"}:
        raise ValueError("variant must be baseline, conventional, or gan")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = PneumoniaTensorDataset("train", augment=variant == "conventional")
    validation = PneumoniaTensorDataset("val")
    test = PneumoniaTensorDataset("test")
    synthetic_count = 0
    generator_checkpoint = None
    if variant == "gan":
        from ml.config import latest_checkpoint
        checkpoint = latest_checkpoint()
        if checkpoint is None:
            raise FileNotFoundError("GAN-augmented experiment needs a trained cGAN checkpoint")
        generator_checkpoint = str(checkpoint)
        labels = np.asarray(train.labels)
        counts = np.bincount(labels, minlength=2)
        minority = int(np.argmin(counts))
        extra = int(counts.max() - counts.min())
        if extra:
            artifact_key = f"run-{run_id}" if run_id is not None else f"seed-{seed}"
            synthetic = generate(checkpoint, minority, extra, EXPERIMENT_ROOT / f"synthetic-{artifact_key}")
            synthetic_count = len(synthetic)
            rows = []
            for item in synthetic:
                image = np.asarray(Image.open(item["path"]).convert("L"), dtype=np.float32)
                tensor = torch.from_numpy(image.copy()).unsqueeze(0) / 127.5 - 1
                rows.append((tensor, minority))
            train = ConcatDataset([train, TensorRows(rows)])
    loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(validation, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0)
    model = ChestXrayCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()
    train_history, val_history = [], []
    best_val = float("inf")
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    run_suffix = f"-run{run_id}" if run_id is not None else ""
    saved_path = EXPERIMENT_ROOT / f"classifier-{variant}-seed{seed}{run_suffix}.pt"
    for epoch in range(epochs):
        model.train()
        train_loss_sum = seen = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(labels)
            seen += len(labels)
        train_loss = train_loss_sum / max(seen, 1)
        validation_loss = _loss(model, val_loader, criterion, device)
        train_history.append(train_loss)
        val_history.append(validation_loss)
        if validation_loss < best_val:
            best_val = validation_loss
            torch.save({"model_state_dict": model.state_dict(), "variant": variant,
                        "epoch": epoch + 1, "seed": seed, "validation_loss": validation_loss}, saved_path)
        if on_epoch:
            on_epoch(epoch + 1, epochs, train_loss, validation_loss)
    best_state = torch.load(saved_path, map_location=device, weights_only=True)
    model.load_state_dict(best_state["model_state_dict"])
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            predictions = model(images.to(device)).argmax(dim=1).cpu().tolist()
            y_pred.extend(predictions)
            y_true.extend(labels.tolist())
    return {
        "variant": variant,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "training_loss": train_history,
        "validation_loss": val_history,
        "checkpoint_path": str(saved_path),
        "generator_checkpoint": generator_checkpoint,
        "synthetic_images_added": synthetic_count,
        "best_epoch": int(best_state["epoch"]),
        "device": str(device),
    }


def _loss(model, loader, criterion, device):
    model.eval()
    total = seen = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device, dtype=torch.long)
            total += float(criterion(model(images), labels)) * len(labels)
            seen += len(labels)
    return total / max(seen, 1)


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate a MedSynth CNN configuration")
    parser.add_argument("--variant", choices=["baseline", "conventional", "gan"], default="baseline")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(train_classifier(**vars(args)))


if __name__ == "__main__":
    main()
