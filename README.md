# MedSynth AI

**MedSynth: Conditional GAN-Based Medical Image Synthesis and Data Augmentation for Imbalanced Chest X-Ray Classification**

MedSynth is an academic research prototype for exploring class-conditional image generation and its effect on a chest X-ray classifier. It is **not a clinical diagnostic system**. Synthetic images and model scores are research artifacts and must not be used for patient care.

## What is implemented

- A React/Vite research dashboard backed by FastAPI.
- PneumoniaMNIST dataset discovery and separate train, validation, and test splits.
- A trainable, label-conditioned DCGAN for 28 × 28 grayscale images. Training is a background job, checkpoints are persisted, and generation is rejected until a trained checkpoint exists.
- Local generated image files with database metadata.
- A separate CNN with baseline, conventional augmentation, and GAN-augmented experiment runs. Metrics are written only after an actual evaluation on the official test split.
- Repeatable experiment batches across one, three, or five seeds, with per-variant means and sample standard deviations for completed runs.
- SQLAlchemy persistence, PostgreSQL by configuration, and SQLite for local zero-setup development.

## Windows PowerShell setup

Python 3.12 is recommended on Windows because it is supported by the current PyTorch Windows wheels. The app detects CUDA at runtime and falls back to CPU.

```powershell
cd C:\Users\diyab\OneDrive\Desktop\medsynth
py -3.12 -m venv .venv
\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
```

On Windows, PyPI's default PyTorch wheel is CPU-only. For this workspace's NVIDIA GPU, install the official CUDA 13.0 build (the installed driver reports CUDA 13.0 support):

```powershell
python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

For CPU-only use, run `python -m pip install -r ml\requirements.txt`. Check the [official PyTorch installer](https://pytorch.org/get-started/locally/) for another supported version/compute platform combination. Keeping PyTorch separate avoids downloading the CPU wheel before the selected GPU wheel.

The default database is SQLite at `storage/medsynth.db`; no database service is needed. To use PostgreSQL, create a database and set `DATABASE_URL` in a root `.env` file, for example:

```dotenv
DATABASE_URL=postgresql+psycopg://medsynth:change-me@localhost:5432/medsynth
```

If the local PostgreSQL service is not configured yet, create a dedicated login and database from PowerShell (replace the example password before using it):

```powershell
psql -U postgres -h localhost -c "CREATE ROLE medsynth LOGIN PASSWORD 'replace-with-a-local-password';"
psql -U postgres -h localhost -c "CREATE DATABASE medsynth OWNER medsynth;"
```

Create the local environment file if you need to override settings. The default uses SQLite and needs no edits:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Do not commit `.env`.

### Start the backend

```powershell
cd backend
..\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload
```

Open Swagger at <http://127.0.0.1:8000/docs> and health at <http://127.0.0.1:8000/health>.

### Start the frontend

In another PowerShell terminal:

```powershell
cd C:\Users\diyab\OneDrive\Desktop\medsynth\frontend
npm.cmd install
npm.cmd run dev
```

The Vite dev server proxies `/api` to `http://127.0.0.1:8000`. Set `VITE_API_URL` only when the backend is hosted elsewhere.

## Dataset and model workflow

The dataset downloads automatically from MedMNIST the first time it is requested. The data is cached under `storage/datasets`; train, validation, and test partitions remain separate. The first dataset summary also saves a measured split/class CSV and PNG distribution plot there; the API serves them at `/api/dataset/distribution.csv` and `/api/dataset/distribution/plot`.

1. Start both servers and open the Dataset Explorer to trigger metadata loading/download.
2. In GAN Training Monitor, start a run. The default is a short 5-epoch run intended to validate the pipeline; choose more epochs for research. Stop is cooperative and takes effect between batches.
3. Once training has saved a checkpoint, generate class-conditional images in Synthetic Image Studio.
4. In Experiment Lab, choose one, three, or five seed repeats. Each seed runs the three configurations with the same seed across variants; choose more repeats for a more stable comparison. The GAN-augmented experiment requires a trained generator. Every run uses the same untouched official test split, and the latest batch shows means and sample standard deviations across its completed seeds.

The CLI equivalents (from the project root with `.venv` active) are:

```powershell
python -m ml.dataset
python -m ml.train_gan --epochs 5 --batch-size 64
python -m ml.generate_images --class-label 1 --count 8
python -m ml.train_classifier --variant baseline --epochs 5
python -m ml.train_classifier --variant conventional --epochs 5
python -m ml.train_classifier --variant gan --epochs 5
```

To apply the SQLAlchemy/Alembic schema explicitly (from `backend`):

```powershell
..\.venv\Scripts\python.exe -m alembic -c alembic.ini upgrade head
```

Use `--help` on each command for options. GPU memory is constrained by available hardware; reduce batch size to 32 or 16 if CUDA runs out of memory. CPU training works but is slower.

## Architecture

See [docs/architecture.md](docs/architecture.md), [docs/methodology.md](docs/methodology.md), and [docs/results.md](docs/results.md). The frontend calls the REST API; FastAPI starts long work in a background thread; the ML package owns dataset, GAN, synthesis, and classifier logic; SQLAlchemy stores run metadata; image files and checkpoints stay in `storage/`.

## Limitations

PneumoniaMNIST is a compact 28 × 28 benchmark derived from chest X-rays. A small GAN training run is for pipeline validation, not evidence of image quality or clinical value. GAN augmentation may help, hurt, or have no measurable effect; use repeated seeds and a prespecified evaluation design for a defensible study. The test split is never used for training or model selection. PostgreSQL is the primary production database; SQLite is the local development fallback.
"# medsynth" 
