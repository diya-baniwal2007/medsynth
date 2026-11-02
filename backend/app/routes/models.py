from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import ModelRecord

router = APIRouter(prefix="/api/models", tags=["models"])


def model_payload(row: ModelRecord):
    return {"id": row.id, "name": row.name, "model_type": row.model_type,
            "version": row.version, "checkpoint_path": row.checkpoint_path,
            "status": row.status, "config": row.config or {}, "created_at": row.created_at}


@router.get("")
def list_models(db: Session = Depends(get_db)):
    return [model_payload(row) for row in db.query(ModelRecord).order_by(ModelRecord.id.desc()).all()]


@router.get("/{model_id}")
def get_model(model_id: int, db: Session = Depends(get_db)):
    row = db.get(ModelRecord, model_id)
    if not row:
        raise HTTPException(status_code=404, detail="Model not found")
    return model_payload(row)
