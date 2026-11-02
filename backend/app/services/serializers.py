from app.models import Experiment, GeneratedImage, ModelRecord, TrainingRun


def training_payload(run: TrainingRun):
    return {
        "id": run.id,
        "model_id": run.model_id,
        "status": run.status,
        "epochs": run.epochs,
        "completed_epochs": run.completed_epochs,
        "generator_loss": run.generator_loss,
        "discriminator_loss": run.discriminator_loss,
        "losses": run.losses or [],
        "config": run.config or {},
        "model_version": run.model.version if run.model else None,
        "checkpoint_path": run.model.checkpoint_path if run.model else None,
        "error": run.error,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
    }


def generated_payload(item: GeneratedImage, model: ModelRecord | None):
    return {
        "id": item.id,
        "class_label": item.class_label,
        "class_name": "Normal" if item.class_label == 0 else "Pneumonia",
        "model_version": model.version if model else "unknown",
        "image_url": f"/api/generation/{item.id}/download",
        "download_url": f"/api/generation/{item.id}/download",
        "created_at": item.created_at,
        "generation_config": item.generation_config or {},
    }


def experiment_payload(experiment: Experiment):
    return {
        "id": experiment.id,
        "name": experiment.name,
        "augmentation_type": experiment.augmentation_type,
        "status": experiment.status,
        "config": experiment.config or {},
        "error": experiment.error,
        "created_at": experiment.created_at,
        "completed_at": experiment.completed_at,
        "metrics": [
            {
                "accuracy": row.accuracy,
                "precision": row.precision,
                "recall": row.recall,
                "macro_f1": row.macro_f1,
                "confusion_matrix": row.confusion_matrix,
                "training_loss": row.training_loss,
                "validation_loss": row.validation_loss,
                "created_at": row.created_at,
            }
            for row in experiment.metrics
        ],
    }
