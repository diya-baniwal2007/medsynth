from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

storage_setting = os.getenv("MEDSYNTH_STORAGE", "storage")
STORAGE_ROOT = Path(storage_setting)
if not STORAGE_ROOT.is_absolute():
    # Resolve from the project root, including values such as ../storage when
    # the API is launched from its backend folder.
    STORAGE_ROOT = (PROJECT_ROOT / storage_setting).resolve()

DATA_ROOT = STORAGE_ROOT / "datasets"
CHECKPOINT_ROOT = STORAGE_ROOT / "checkpoints"
GENERATED_ROOT = STORAGE_ROOT / "generated"
EXPERIMENT_ROOT = STORAGE_ROOT / "experiments"

DATABASE_URL = os.getenv(
    "DATABASE_URL", f"sqlite:///{(STORAGE_ROOT / 'medsynth.db').as_posix()}"
)
CORS_ORIGINS = [
    value.strip()
    for value in os.getenv(
        "MEDSYNTH_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if value.strip()
]

for directory in (DATA_ROOT, CHECKPOINT_ROOT, GENERATED_ROOT, EXPERIMENT_ROOT):
    directory.mkdir(parents=True, exist_ok=True)
