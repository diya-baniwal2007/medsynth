import threading
from datetime import datetime, timezone
from fastapi.responses import FileResponse
from pathlib import Path

import torch
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.config import CHECKPOINT_ROOT
from app.database import SessionLocal, get_db
from app.models import ModelRecord, TrainingRun
from app.schemas.api import TrainingConfig
from app.services.serializers import training_payload
from ml.config import latest_checkpoint
from ml.train_gan import train_gan

router = APIRouter(prefix="/api/training", tags=["training"])
_ACTIVE: dict[int, threading.Event] = {}
_LOCK = threading.Lock()


def _train_worker(run_id: int, model_id: int, config: dict, stop_event: threading.Event, resume):
    db = SessionLocal()
    try:
        run = db.get(TrainingRun, run_id)
        run.status = "running"
        run.started_at = datetime.now(timezone.utc)
        db.commit()

        def save_epoch(epoch, checkpoint):
            row = db.get(TrainingRun, run_id)
            model = db.get(ModelRecord, model_id)
            row.completed_epochs = epoch["epoch"]
            row.generator_loss = epoch["generator_loss"]
            row.discriminator_loss = epoch["discriminator_loss"]
            row.losses = (row.losses or []) + [epoch]
            model.checkpoint_path = checkpoint
            model.version = f"cgan-{run_id}-e{epoch['epoch']}"
            model.status = "ready"
            db.commit()

        result = train_gan(config, on_epoch=save_epoch, should_stop=stop_event.is_set, resume_path=resume)
        run = db.get(TrainingRun, run_id)
        model = db.get(ModelRecord, model_id)
        run.completed_epochs = max(run.completed_epochs, result["completed_epochs"])
        run.status = "stopped" if result["stopped"] else "completed"
        run.completed_at = datetime.now(timezone.utc)
        run.config = {**(run.config or {}), "device": result["device"]}
        model.config = {**(model.config or {}), "device": result["device"],
                        "completed_epochs": result["completed_epochs"]}
        if result["checkpoint_path"]:
            model.checkpoint_path = result["checkpoint_path"]
            model.status = "ready"
        elif model.checkpoint_path is None:
            model.status = "failed"
        db.commit()
    except Exception as exc:
        db.rollback()
        run = db.get(TrainingRun, run_id)
        model = db.get(ModelRecord, model_id)
        if run:
            run.status = "failed"
            run.error = f"{type(exc).__name__}: {exc}"
            run.completed_at = datetime.now(timezone.utc)
        if model and not model.checkpoint_path:
            model.status = "failed"
        db.commit()
    finally:
        db.close()
        with _LOCK:
            _ACTIVE.pop(run_id, None)


@router.post("/start", status_code=202)
def start_training(config: TrainingConfig, db: Session = Depends(get_db)):
    active = db.query(TrainingRun).filter(TrainingRun.status.in_(["queued", "running"])).first()
    if active:
        raise HTTPException(status_code=409, detail=f"Training run {active.id} is already {active.status}")
    resume = latest_checkpoint() if config.resume else None
    if config.resume and resume is None:
        raise HTTPException(status_code=409, detail="No checkpoint exists to resume. Start a new cGAN run first.")
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if int(state.get("latent_dim", -1)) != config.latent_dim:
            raise HTTPException(status_code=422, detail="Latent dimension must match the checkpoint to resume training.")
    model = ModelRecord(name="MedSynth Conditional GAN", model_type="cGAN",
                        version="pending", status="training", config=config.model_dump())
    db.add(model)
    db.flush()
    run = TrainingRun(model_id=model.id, epochs=config.epochs, config=config.model_dump(), status="queued")
    db.add(run)
    db.commit()
    db.refresh(run)
    stop_event = threading.Event()
    with _LOCK:
        _ACTIVE[run.id] = stop_event
    thread = threading.Thread(target=_train_worker,
                              args=(run.id, model.id, config.model_dump(), stop_event, resume), daemon=True)
    thread.start()
    return training_payload(run)


@router.post("/stop", status_code=202)
def stop_training(db: Session = Depends(get_db)):
    run = db.query(TrainingRun).filter(TrainingRun.status.in_(["queued", "running"])).order_by(TrainingRun.id.desc()).first()
    if not run:
        raise HTTPException(status_code=409, detail="There is no active training run")
    event = _ACTIVE.get(run.id)
    if event:
        event.set()
    return {"id": run.id, "status": "stop_requested"}


@router.get("/status")
def training_status(db: Session = Depends(get_db)):
    run = db.query(TrainingRun).order_by(TrainingRun.id.desc()).first()
    return training_payload(run) if run else {"status": "idle", "epochs": 0, "completed_epochs": 0, "losses": []}


@router.get("/history")
def training_history(db: Session = Depends(get_db)):
    return [training_payload(row) for row in db.query(TrainingRun).order_by(TrainingRun.id.desc()).limit(50).all()]


@router.get("/losses")
def training_losses(db: Session = Depends(get_db)):
    run = db.query(TrainingRun).order_by(TrainingRun.id.desc()).first()
    return {"run_id": run.id if run else None, "losses": run.losses or [] if run else []}


@router.get("/samples")
def training_samples():
    samples = []
    for path in sorted(CHECKPOINT_ROOT.glob("samples-epoch-*.png")):
        try:
            epoch = int(path.stem.rsplit("-", 1)[-1])
        except ValueError:
            continue
        samples.append({"epoch": epoch, "image_url": f"/api/training/samples/{epoch}",
                        "created_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)})
    return samples[-12:]


@router.get("/samples/{epoch}")
def training_sample(epoch: int):
    if epoch < 1:
        raise HTTPException(status_code=404, detail="Training sample not found")
    path = CHECKPOINT_ROOT / f"samples-epoch-{epoch:04d}.png"
    if not path.is_file() or not path.resolve().is_relative_to(CHECKPOINT_ROOT.resolve()):
        raise HTTPException(status_code=404, detail="Training sample not found")
    return FileResponse(path, media_type="image/png", filename=path.name)
