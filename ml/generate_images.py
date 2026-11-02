from __future__ import annotations

import argparse
import uuid
from datetime import datetime, timezone

import torch
from PIL import Image

from ml.config import CLASS_NAMES, GENERATED_ROOT, latest_checkpoint
from ml.generator import ConditionalGenerator


def generate(checkpoint_path, class_label: int, count: int, output_dir=GENERATED_ROOT):
    if not checkpoint_path or not checkpoint_path.exists():
        raise FileNotFoundError("No trained cGAN checkpoint exists. Start and complete GAN training first.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    latent_dim = int(state["latent_dim"])
    generator = ConditionalGenerator(latent_dim).to(device)
    generator.load_state_dict(state["generator_state_dict"])
    generator.eval()
    stamp = datetime.now(timezone.utc)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with torch.no_grad():
        labels = torch.full((count,), int(class_label), device=device, dtype=torch.long)
        images = generator(torch.randn(count, latent_dim, device=device), labels).cpu()
    for image in images:
        pixels = ((image[0] + 1) * 127.5).clamp(0, 255).byte().numpy()
        image_id = uuid.uuid4().hex
        path = output_dir / f"{stamp.strftime('%Y%m%dT%H%M%SZ')}-{image_id}.png"
        Image.fromarray(pixels, mode="L").save(path)
        results.append({"path": path, "image_id": image_id, "class_label": class_label,
                        "class_name": CLASS_NAMES[class_label], "created_at": stamp.isoformat(),
                        "checkpoint": str(checkpoint_path), "model_version": checkpoint_path.stem})
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate images from a trained MedSynth cGAN")
    parser.add_argument("--class-label", type=int, choices=[0, 1], required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--checkpoint", type=str)
    args = parser.parse_args()
    checkpoint = __import__("pathlib").Path(args.checkpoint) if args.checkpoint else latest_checkpoint()
    print(generate(checkpoint, args.class_label, args.count))


if __name__ == "__main__":
    main()
