from collections import Counter

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from PIL import Image
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.config import DATA_ROOT
from app.database import get_db
from app.models import DatasetRecord, GeneratedImage, ModelRecord
from app.services.serializers import generated_payload
from ml.config import CLASS_NAMES, DATASET_NAME
from ml.dataset import get_raw_split, save_distribution_artifacts

router = APIRouter(prefix="/api/dataset", tags=["dataset"])


def summarize_dataset(db: Session):
    splits = []
    totals = {0: 0, 1: 0}
    split_counts = {}
    for split in ("train", "val", "test"):
        try:
            _, counts = get_raw_split(split)
        except Exception as exc:
            raise RuntimeError(f"Could not load PneumoniaMNIST {split} split: {exc}") from exc
        split_counts[split] = {str(label): count for label, count in counts.items()}
        for label, count in counts.items():
            totals[label] += count
        splits.append({"name": split, "total": sum(counts.values()), "classes": counts})
    distribution_csv = DATA_ROOT / "pneumoniamnist-distribution.csv"
    distribution_plot = DATA_ROOT / "pneumoniamnist-distribution.png"
    if not distribution_csv.is_file() or not distribution_plot.is_file():
        save_distribution_artifacts(splits)
    record = db.query(DatasetRecord).filter(DatasetRecord.name == DATASET_NAME).first()
    payload_counts = {str(label): count for label, count in totals.items()}
    if record is None:
        record = DatasetRecord(name=DATASET_NAME,
                               description="28 × 28 grayscale PneumoniaMNIST benchmark with official train/validation/test splits.",
                               total_images=sum(totals.values()), split_counts=split_counts)
        db.add(record)
    else:
        record.total_images = sum(totals.values())
        record.split_counts = split_counts
    db.commit()
    return {
        "available": True,
        "name": DATASET_NAME,
        "description": record.description,
        "total_images": record.total_images,
        "image_size": "28 × 28",
        "channels": 1,
        "classes": [{"label": label, "name": name, "count": totals[label]} for label, name in CLASS_NAMES.items()],
        "splits": splits,
    }


@router.get("/summary")
def dataset_summary(db: Session = Depends(get_db)):
    try:
        return summarize_dataset(db)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/distribution")
def dataset_distribution(db: Session = Depends(get_db)):
    summary = dataset_summary(db)
    return {"splits": summary["splits"], "classes": summary["classes"]}


@router.get("/distribution/plot")
def dataset_distribution_plot(db: Session = Depends(get_db)):
    path = DATA_ROOT / "pneumoniamnist-distribution.png"
    if not path.is_file():
        dataset_summary(db)
    if not path.is_file():
        raise HTTPException(status_code=503, detail="Dataset distribution plot is unavailable")
    return FileResponse(path, media_type="image/png", filename=path.name)


@router.get("/distribution.csv")
def dataset_distribution_csv(db: Session = Depends(get_db)):
    path = DATA_ROOT / "pneumoniamnist-distribution.csv"
    if not path.is_file():
        dataset_summary(db)
    if not path.is_file():
        raise HTTPException(status_code=503, detail="Dataset distribution CSV is unavailable")
    return FileResponse(path, media_type="text/csv", filename=path.name)


@router.get("/images")
def dataset_images(
    split: str = Query("train", pattern="^(train|val|test)$"),
    class_label: int | None = Query(None, ge=0, le=1),
    source: str = Query("real", pattern="^(real|synthetic)$"),
    limit: int = Query(32, ge=1, le=64),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    if source == "synthetic":
        query = db.query(GeneratedImage).order_by(GeneratedImage.id.desc())
        if class_label is not None:
            query = query.filter(GeneratedImage.class_label == class_label)
        rows = query.offset(offset).limit(limit).all()
        models = {item.id: item for item in db.query(ModelRecord).all()}
        return {"items": [generated_payload(row, models.get(row.model_id)) for row in rows],
                "total": query.count(), "source": source, "split": None}
    try:
        dataset, _ = get_raw_split(split)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Dataset unavailable: {exc}") from exc
    labels = np.asarray(dataset.labels).reshape(-1).astype(int)
    indices = np.flatnonzero(labels == class_label) if class_label is not None else np.arange(len(labels))
    page = indices[offset:offset + limit]
    return {
        "source": "real",
        "split": split,
        "total": int(len(indices)),
        "items": [{"index": int(index), "class_label": int(labels[index]),
                   "class_name": CLASS_NAMES[int(labels[index])],
                   "image_url": f"/api/dataset/images/{split}/{int(index)}"} for index in page],
    }


@router.get("/images/{split}/{index}")
def dataset_image(split: str, index: int):
    if split not in {"train", "val", "test"}:
        raise HTTPException(status_code=404, detail="Unknown split")
    try:
        dataset, _ = get_raw_split(split)
        if index < 0 or index >= len(dataset.imgs):
            raise HTTPException(status_code=404, detail="Image index out of range")
        image = Image.fromarray(np.asarray(dataset.imgs[index]).astype(np.uint8), mode="L")
        from io import BytesIO
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return Response(content=buffer.getvalue(), media_type="image/png",
                        headers={"Cache-Control": "public, max-age=3600"})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Dataset unavailable: {exc}") from exc
