from __future__ import annotations

import argparse
import random
import time
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ml.config import CHECKPOINT_ROOT, CLASS_NAMES
from ml.dataset import PneumoniaTensorDataset
from ml.discriminator import ConditionalDiscriminator
from ml.generator import ConditionalGenerator


def train_gan(config: dict, on_epoch=None, should_stop=None, resume_path=None):
    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    latent_dim = int(config.get("latent_dim", 100))
    generator = ConditionalGenerator(latent_dim).to(device)
    discriminator = ConditionalDiscriminator().to(device)
    if resume_path:
        state = torch.load(resume_path, map_location=device, weights_only=True)
        generator.load_state_dict(state["generator_state_dict"])
        discriminator.load_state_dict(state["discriminator_state_dict"])
    dataset = PneumoniaTensorDataset("train")
    loader = DataLoader(dataset, batch_size=int(config.get("batch_size", 64)), shuffle=True,
                        num_workers=0, drop_last=True)
    criterion = nn.BCEWithLogitsLoss()
    lr = float(config.get("learning_rate", 0.0002))
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    start_epoch = int(state.get("epoch", 0)) if resume_path else 0
    total_epochs = int(config.get("epochs", 5))
    history = []
    for epoch in range(start_epoch, total_epochs):
        if should_stop and should_stop():
            break
        generator.train()
        discriminator.train()
        sum_g = sum_d = 0.0
        seen = 0
        for real_images, labels in loader:
            if should_stop and should_stop():
                break
            real_images, labels = real_images.to(device), labels.to(device, dtype=torch.long)
            batch = real_images.size(0)
            valid = torch.full((batch, 1), 0.9, device=device)
            fake_target = torch.zeros((batch, 1), device=device)

            optimizer_d.zero_grad(set_to_none=True)
            real_logits = discriminator(real_images, labels)
            noise = torch.randn(batch, latent_dim, device=device)
            fake_images = generator(noise, labels)
            fake_logits = discriminator(fake_images.detach(), labels)
            d_loss = (criterion(real_logits, valid) + criterion(fake_logits, fake_target)) / 2
            d_loss.backward()
            optimizer_d.step()

            optimizer_g.zero_grad(set_to_none=True)
            generated_logits = discriminator(fake_images, labels)
            g_loss = criterion(generated_logits, torch.ones((batch, 1), device=device))
            g_loss.backward()
            optimizer_g.step()
            sum_g += float(g_loss.detach()) * batch
            sum_d += float(d_loss.detach()) * batch
            seen += batch

        if seen == 0:
            break
        record = {
            "epoch": epoch + 1,
            "generator_loss": sum_g / seen,
            "discriminator_loss": sum_d / seen,
            "samples": seen,
        }
        history.append(record)
        CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
        checkpoint_path = CHECKPOINT_ROOT / f"cgan-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.pt"
        torch.save({
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "latent_dim": latent_dim,
            "epoch": epoch + 1,
            "config": {**config, "device": str(device)},
            "history": history,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }, checkpoint_path)
        _save_epoch_grid(generator, device, latent_dim, epoch + 1)
        if on_epoch:
            on_epoch(record, str(checkpoint_path))
    return {
        "history": history,
        "checkpoint_path": str(checkpoint_path) if history else None,
        "completed_epochs": (history[-1]["epoch"] if history else start_epoch),
        "device": str(device),
        "stopped": bool(should_stop and should_stop()),
    }


def _save_epoch_grid(generator, device, latent_dim, epoch):
    from PIL import Image, ImageDraw
    generator.eval()
    tiles = []
    with torch.no_grad():
        for label in (0, 1):
            labels = torch.full((8,), label, device=device, dtype=torch.long)
            samples = generator(torch.randn(8, latent_dim, device=device), labels).cpu()
            for sample in samples:
                pixels = ((sample[0] + 1) * 127.5).clamp(0, 255).byte().numpy()
                tiles.append((Image.fromarray(pixels, mode="L").resize((112, 112)), label))
    canvas = Image.new("RGB", (4 * 128, 4 * 140), "#edf3f5")
    draw = ImageDraw.Draw(canvas)
    for index, (tile, label) in enumerate(tiles):
        x, y = (index % 4) * 128, (index // 4) * 140
        canvas.paste(tile, (x + 8, y + 4))
        draw.text((x + 8, y + 118), f"{CLASS_NAMES[label]} · epoch {epoch}", fill="#173246")
    canvas.save(CHECKPOINT_ROOT / f"samples-epoch-{epoch:04d}.png")


def main():
    parser = argparse.ArgumentParser(description="Train the MedSynth conditional GAN")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str)
    args = parser.parse_args()
    start = time.perf_counter()
    result = train_gan(vars(args), resume_path=args.resume)
    print(result)
    print(f"Elapsed: {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
