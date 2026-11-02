from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    epochs: int = Field(default=5, ge=1, le=500)
    batch_size: int = Field(default=64, ge=8, le=256)
    latent_dim: int = Field(default=100, ge=32, le=512)
    learning_rate: float = Field(default=0.0002, gt=0, le=0.01)
    seed: int = Field(default=42, ge=0)
    resume: bool = False


class GenerationRequest(BaseModel):
    class_label: int = Field(ge=0, le=1)
    count: int = Field(default=8, ge=1, le=64)


class ExperimentRequest(BaseModel):
    epochs: int = Field(default=5, ge=1, le=100)
    batch_size: int = Field(default=64, ge=8, le=256)
    seed: int = Field(default=42, ge=0)
    repeats: int = Field(default=1, ge=1, le=10)
    variants: list[Literal["baseline", "conventional", "gan"]] = Field(
        default_factory=lambda: ["baseline", "conventional", "gan"]
    )


class TrainingStatus(BaseModel):
    id: int
    status: str
    epochs: int
    completed_epochs: int
    config: dict
    losses: list
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class GenerationResult(BaseModel):
    id: int
    class_label: int
    model_version: str
    image_url: str
    download_url: str
    created_at: datetime


class ExperimentResult(BaseModel):
    id: int
    name: str
    augmentation_type: str
    status: str
    config: dict
    error: str | None = None
    created_at: datetime
    completed_at: datetime | None = None
    metrics: list[dict] = Field(default_factory=list)
