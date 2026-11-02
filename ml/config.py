import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
_storage = Path(os.getenv("MEDSYNTH_STORAGE", "storage"))
if not _storage.is_absolute():
    _storage = (PROJECT_ROOT / _storage).resolve()
DATA_ROOT = _storage / "datasets"
CHECKPOINT_ROOT = _storage / "checkpoints"
EXPERIMENT_ROOT = _storage / "experiments"
GENERATED_ROOT = _storage / "generated"

DATASET_NAME = "PneumoniaMNIST"
CLASS_NAMES = {0: "Normal", 1: "Pneumonia"}
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_CLASSES = 2


def latest_checkpoint() -> Path | None:
    paths = sorted(CHECKPOINT_ROOT.glob("cgan-*.pt"), key=lambda item: item.stat().st_mtime)
    return paths[-1] if paths else None
