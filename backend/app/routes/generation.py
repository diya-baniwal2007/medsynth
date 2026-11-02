from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.config import GENERATED_ROOT
from app.database import get_db
from app.models import GeneratedImage, ModelRecord
from app.schemas.api import GenerationRequest
from app.services.serializers import generated_payload
from ml.config import latest_checkpoint
from ml.generate_images import generate

router = APIRouter(prefix="/api/generation", tags=["generation"])


@router.post("/generate", status_code=201)
def generate_images(request: GenerationRequest, db: Session = Depends(get_db)):
    checkpoint = latest_checkpoint()
    if checkpoint is None:
        raise HTTPException(status_code=409, detail="A trained cGAN checkpoint is required. Start GAN training first.")
    model = db.query(ModelRecord).filter(ModelRecord.checkpoint_path == str(checkpoint)).first()
    if model is None:
        model = ModelRecord(name="MedSynth Conditional GAN", model_type="cGAN",
                            version=checkpoint.stem, checkpoint_path=str(checkpoint), status="ready")
        db.add(model)
        db.flush()
    try:
        files = generate(checkpoint, request.class_label, request.count)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {exc}") from exc
    rows = []
    for file in files:
        item = GeneratedImage(model_id=model.id, class_label=request.class_label,
                              image_path=str(Path(file["path"]).resolve()),
                              generation_config={"count": request.count, "checkpoint": checkpoint.name,
                                                 "latent_dim": int(__import__("torch").load(checkpoint, map_location="cpu", weights_only=True)["latent_dim"])})
        db.add(item)
        rows.append(item)
    db.commit()
    for item in rows:
        db.refresh(item)
    return {"items": [generated_payload(item, model) for item in rows], "count": len(rows)}


@router.get("/history")
def generation_history(class_label: int | None = None, limit: int = 50, db: Session = Depends(get_db)):
    query = db.query(GeneratedImage).order_by(GeneratedImage.id.desc())
    if class_label is not None:
        query = query.filter(GeneratedImage.class_label == class_label)
    rows = query.limit(min(max(limit, 1), 200)).all()
    models = {item.id: item for item in db.query(ModelRecord).all()}
    return [generated_payload(row, models.get(row.model_id)) for row in rows]


@router.get("/{image_id}")
def get_generated_image(image_id: int, db: Session = Depends(get_db)):
    item = db.get(GeneratedImage, image_id)
    if not item:
        raise HTTPException(status_code=404, detail="Generated image not found")
    return generated_payload(item, db.get(ModelRecord, item.model_id))


@router.get("/{image_id}/download")
def download_image(image_id: int, db: Session = Depends(get_db)):
    item = db.get(GeneratedImage, image_id)
    if not item:
        raise HTTPException(status_code=404, detail="Generated image not found")
    path = Path(item.image_path)
    if not path.is_file() or not path.resolve().is_relative_to(GENERATED_ROOT.resolve()):
        raise HTTPException(status_code=404, detail="Generated image file is missing")
    return FileResponse(path, media_type="image/png", filename=path.name)
