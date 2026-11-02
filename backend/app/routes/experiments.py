import threading
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import SessionLocal, get_db
from app.models import Experiment, ExperimentMetric, ModelRecord
from app.schemas.api import ExperimentRequest
from app.services.serializers import experiment_payload
from ml.config import latest_checkpoint
from ml.train_classifier import train_classifier

router = APIRouter(prefix="/api/experiments", tags=["experiments"])
_LOCK = threading.Lock()
_ACTIVE = False


def _worker(experiment_ids: list[int]):
    global _ACTIVE
    db = SessionLocal()
    try:
        for experiment_id in experiment_ids:
            try:
                experiment = db.get(Experiment, experiment_id)
                if experiment is None:
                    continue
                variant = experiment.augmentation_type
                config = experiment.config or {}
                if variant == "gan" and latest_checkpoint() is None:
                    experiment.status = "blocked"
                    experiment.error = "Train the conditional GAN before running the GAN-augmented classifier."
                    experiment.completed_at = datetime.now(timezone.utc)
                    db.commit()
                    continue
                experiment.status = "running"
                db.commit()

                def on_epoch(epoch, total, train_loss, validation_loss):
                    row = db.get(Experiment, experiment_id)
                    row.config = {**(row.config or {}), "progress": {"epoch": epoch, "total_epochs": total,
                                                                      "training_loss": train_loss,
                                                                      "validation_loss": validation_loss}}
                    db.commit()

                result = train_classifier(variant=variant, epochs=config["epochs"],
                                          batch_size=config["batch_size"], seed=config["seed"],
                                          run_id=experiment_id, on_epoch=on_epoch)
                model = ModelRecord(name=f"MedSynth CNN · {variant}", model_type="CNN",
                                    version=f"cnn-{variant}-seed{config['seed']}-run{experiment_id}",
                                    checkpoint_path=result["checkpoint_path"], status="ready",
                                    config={"variant": variant, "epochs": config["epochs"],
                                            "batch_size": config["batch_size"], "seed": config["seed"],
                                            "repeat": config.get("repeat"), "batch_id": config.get("batch_id"),
                                            "best_epoch": result["best_epoch"], "device": result["device"]})
                db.add(model)
                db.flush()
                row = db.get(Experiment, experiment_id)
                row.model_id = model.id
                row.status = "completed"
                row.completed_at = datetime.now(timezone.utc)
                row.config = {**(row.config or {}),
                              "generator_checkpoint": result.get("generator_checkpoint"),
                              "synthetic_images_added": result.get("synthetic_images_added", 0),
                              "best_epoch": result["best_epoch"], "device": result["device"]}
                metric = ExperimentMetric(
                    experiment_id=experiment_id,
                    accuracy=result["accuracy"], precision=result["precision"], recall=result["recall"],
                    macro_f1=result["macro_f1"], confusion_matrix=result["confusion_matrix"],
                    training_loss=result["training_loss"], validation_loss=result["validation_loss"],
                )
                db.add(metric)
                db.commit()
            except Exception as exc:
                db.rollback()
                row = db.get(Experiment, experiment_id)
                if row and row.status in {"queued", "running"}:
                    row.status = "failed"
                    row.error = f"{type(exc).__name__}: {exc}"
                    row.completed_at = datetime.now(timezone.utc)
                    db.commit()
    finally:
        db.close()
        with _LOCK:
            _ACTIVE = False

@router.post("/run", status_code=202)
def run_experiments(request: ExperimentRequest, db: Session = Depends(get_db)):
    global _ACTIVE
    if not request.variants:
        raise HTTPException(status_code=422, detail="Select at least one experiment variant")
    with _LOCK:
        if _ACTIVE:
            raise HTTPException(status_code=409, detail="An experiment batch is already running")
        _ACTIVE = True
    batch_id = str(uuid4())
    created = []
    for repeat in range(request.repeats):
        seed = request.seed + repeat
        for variant in request.variants:
            row = Experiment(name={"baseline": "Baseline CNN", "conventional": "Conventional augmentation CNN",
                                  "gan": "GAN-augmented CNN"}[variant], augmentation_type=variant,
                             status="queued", config={"epochs": request.epochs,
                                                       "batch_size": request.batch_size, "seed": seed,
                                                       "repeat": repeat + 1, "repeats": request.repeats,
                                                       "batch_id": batch_id})
            db.add(row)
            created.append(row)
    db.commit()
    for row in created:
        db.refresh(row)
    thread = threading.Thread(target=_worker, args=([row.id for row in created],), daemon=True)
    thread.start()
    return {"items": [experiment_payload(row) for row in created]}


@router.get("")
def list_experiments(db: Session = Depends(get_db)):
    return [experiment_payload(row) for row in db.query(Experiment).order_by(Experiment.id.desc()).limit(100).all()]


@router.get("/compare")
def compare_experiments(db: Session = Depends(get_db)):
    rows = db.query(Experiment).filter(Experiment.status == "completed").order_by(Experiment.id.desc()).all()
    return [{"id": row.id, "variant": row.augmentation_type, "name": row.name,
             "metrics": experiment_payload(row)["metrics"][-1] if row.metrics else None} for row in rows]


@router.get("/{experiment_id}")
def get_experiment(experiment_id: int, db: Session = Depends(get_db)):
    row = db.get(Experiment, experiment_id)
    if not row:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment_payload(row)
