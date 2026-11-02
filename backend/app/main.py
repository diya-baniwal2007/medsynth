from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.core.config import CORS_ORIGINS
from app.database import Base, engine
from app.models import *  # noqa: F403 - load all mapped tables before create_all
from app.routes import dashboard, dataset, experiments, generation, models, training

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medsynth")

app = FastAPI(
    title="MedSynth AI API",
    description="Research prototype API for conditional chest X-ray synthesis and augmentation experiments. Not for clinical use.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

for router in (dashboard.router, dataset.router, generation.router, training.router,
               experiments.router, models.router):
    app.include_router(router)


@app.on_event("startup")
def initialize_database():
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized (%s)", engine.url.get_backend_name())


@app.get("/health", tags=["health"])
def health():
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception as exc:
        logger.exception("Database health check failed")
        raise HTTPException(status_code=503, detail="Database unavailable") from exc
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "unavailable"
    return {"status": "ok", "database": engine.url.get_backend_name(), "device": device,
            "clinical_use": False}
