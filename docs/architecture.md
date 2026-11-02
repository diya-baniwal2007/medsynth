# Architecture

```text
React + Vite
  └─ REST/JSON (Axios, /api)
      └─ FastAPI routers
          ├─ SQLAlchemy → PostgreSQL (or SQLite development fallback)
          ├─ background threads → cGAN / classifier training
          └─ ML package → MedMNIST, PyTorch, local files
              ├─ storage/datasets
              ├─ storage/checkpoints
              ├─ storage/generated
              └─ storage/experiments
```

The API keeps request handlers responsive during training. Each job owns its database session, persists status between epochs, and stores a checkpoint before exposing generated samples. The frontend polls the run status and reads persisted values; it does not simulate metrics or generation.

The database layer has dataset, model, training run, generated image, experiment, and experiment metric records. `DATABASE_URL` selects PostgreSQL; the default SQLite database lives in `storage/medsynth.db`. Alembic has an initial schema migration; development startup also creates missing tables for a simple first run.
