from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Experiment, GeneratedImage, ModelRecord, TrainingRun
from app.routes.dataset import summarize_dataset
from app.services.serializers import experiment_payload, training_payload

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/summary")
def dashboard_summary(db: Session = Depends(get_db)):
    try:
        dataset = summarize_dataset(db)
    except Exception as exc:
        dataset = {"available": False, "error": str(exc), "total_images": 0, "splits": []}
    latest_training = db.query(TrainingRun).order_by(TrainingRun.id.desc()).first()
    latest_experiment = (db.query(Experiment).filter(Experiment.status == "completed")
                         .order_by(Experiment.id.desc()).first())
    return {
        "dataset": dataset,
        "total_generated": db.query(GeneratedImage).count(),
        "trained_models": db.query(ModelRecord).filter(ModelRecord.status == "ready").count(),
        "completed_experiments": db.query(Experiment).filter(Experiment.status == "completed").count(),
        "training": training_payload(latest_training) if latest_training else None,
        "latest_experiment": experiment_payload(latest_experiment) if latest_experiment else None,
        "as_of": datetime.now(timezone.utc),
    }


@router.get("/activity")
def dashboard_activity(db: Session = Depends(get_db)):
    events = []
    for run in db.query(TrainingRun).order_by(TrainingRun.id.desc()).limit(8).all():
        events.append({"type": "training", "id": run.id, "title": "cGAN training run",
                       "status": run.status, "date": run.started_at or run.completed_at})
    for experiment in db.query(Experiment).order_by(Experiment.id.desc()).limit(8).all():
        events.append({"type": "experiment", "id": experiment.id, "title": experiment.name,
                       "status": experiment.status, "date": experiment.completed_at or experiment.created_at})
    for image in db.query(GeneratedImage).order_by(GeneratedImage.id.desc()).limit(8).all():
        events.append({"type": "generation", "id": image.id,
                       "title": f"{('Normal', 'Pneumonia')[image.class_label]} image generated",
                       "status": "completed", "date": image.created_at})
    events.sort(key=lambda event: event["date"].timestamp() if event["date"] else 0, reverse=True)
    return events[:12]
