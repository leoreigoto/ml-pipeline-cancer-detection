# Breast Cancer Diagnosis from Thermographic Images — MLOps Pipeline (UFF Master's Thesis)

This repository contains the full MLOps pipeline developed for a Master's thesis at **UFF (Universidade Federal Fluminense)**. It trains, tracks, serves, and exposes through a web UI a deep learning model that classifies breast thermography images as **healthy/sick** (or **benign/malignant**, depending on the dataset variant used), with **Grad-CAM** visual explanations for each prediction.

The pipeline covers the full lifecycle of the model:

- **Training & evaluation** with PyTorch Lightning (EfficientNet-B0 backbone)
- **Experiment tracking & model registry** with MLflow
- **Orchestration** of training/evaluation/export jobs with Apache Airflow
- **Inference API** with FastAPI, serving predictions and Grad-CAM heatmaps
- **User interface** with Gradio, for doctors/researchers to upload an image and get a diagnosis + heatmap
- **Export** of the trained model to the Hugging Face Hub

> **Thermographic images used to train and evaluate this model are available at the UFF DMI (Database for Mastology Research with Infrared image) repository: https://visual.ic.uff.br/dmi/**
> This repository does **not** ship the dataset — see [Dataset](#dataset) below for how to obtain and place the images.

---

## Architecture

```
                        ┌───────────────┐
                        │   Airflow     │  orchestrates training/eval/export DAGs
                        └───────┬───────┘
                                │ triggers (Docker containers)
                                ▼
┌───────────┐   logs/artifacts  ┌───────────────┐
│  Training │ ────────────────► │  MLflow Server │◄────────┐
│  (Model_*)│                   └───────┬────────┘         │
└───────────┘                           │ loads "production" model
                                         ▼                  │
                                  ┌─────────────┐    predict/health/info
                                  │  FastAPI    │◄───────────┘
                                  │  (API_main) │
                                  └──────┬──────┘
                                         │ REST call (predict_with_gradcam)
                                         ▼
                                  ┌─────────────┐
                                  │  Gradio UI  │  doctors upload an image here
                                  └─────────────┘
```

All services run as separate Docker containers, orchestrated with `docker-compose`.

---

## Repository structure

```
.
├── API_main.py                    # FastAPI app: /info, /predict, /predict_with_gradcam, /health
├── API_config_loader.py           # API configuration and response schema
├── API_security_key.py            # API key validation (header: UFF-API-KEY)
├── API_loggers.py                 # API logging setup
│
├── MODEL_main.py                  # Entry point for a training run
├── MODEL_config_loader.py         # Data/training/MLflow configuration (dataclasses)
├── MODEL_data_loader.py           # PyTorch Lightning DataModule (ImageFolder-based)
├── MODEL_mlflow_train.py          # Training/evaluation loop, logs to MLflow
├── MODEL_evaluate_and_register.py # Evaluates a trained run and registers it in MLflow
├── MODEL_export_from_local_weights.py
├── MODEL_export_to_hf.py          # Exports the "production" model to Hugging Face Hub
├── MODEL_register_model.py
├── models_custom_module.py        # Model architecture (EfficientNet-B0) + Grad-CAM hooks
│
├── gradio_app.py                  # Gradio UI, calls the FastAPI /predict_with_gradcam endpoint
├── create_logger.py               # Shared logging utility
│
├── airflow/
│   └── dags/
│       ├── dag_train_eval.py
│       ├── dag_train_eval_with_gpu.py
│       ├── dag_train_eval_with_gpu_and_export_HF.py
│       └── dag_export_to_huggingface.py
│
├── UTILS/                         # One-off / helper scripts
│
├── DATA/                          # Expected dataset layout (empty — see Dataset section)
│   ├── breast_thermal_frontal/
│   │   ├── Train/{healthy,sick}
│   │   └── Test/{healthy,sick}
│   ├── breast_thermal_frontal_SANITY_CHECK/
│   │   └── ... (same layout, small subset for smoke tests)
│   └── breast_thermal_segmented/
│       ├── Train/{Benign,Malignant}
│       ├── Test/{Benign,Malignant}
│       └── Evaluate/{Benign,Malignant}
│
├── file/                          # Static assets used by the UI (logos)
│
├── Dockerfile.api.cpu / .gpu      # Inference API image
├── Dockerfile.ui                  # Gradio UI image
├── Dockerfile.mlflow              # MLflow tracking server image
├── Dockerfile.model_train_eval.cpu / .gpu   # Training image (used by Airflow's DockerOperator)
├── Dockerfile.export_to_hf        # Hugging Face export image
│
├── docker-compose.yml             # Main stack: Postgres, Airflow, MLflow, API, UI
├── docker-compose.gpu.yml         # GPU override
├── docker-compose.single_gpu_train.yml
│
├── iniciar_com_build.bat / iniciar_com_build_GPU.bat   # Windows: first run (build + start)
├── iniciar_sem_build.bat                                # Windows: subsequent runs (start only)
├── encerrar.bat / encerrar_com_GPU.bat                  # Windows: stop everything
└── COMMANDS.txt                   # Quick reference for service URLs and docker-compose commands
```

---

## Services & ports

| Service              | URL                          | Purpose                                                    |
|----------------------|-------------------------------|-------------------------------------------------------------|
| Gradio UI            | http://localhost:7860        | Upload a thermography image, get diagnosis + Grad-CAM       |
| Inference API (FastAPI) | http://localhost:8000     | `/info`, `/predict`, `/predict_with_gradcam`, `/health`     |
| MLflow               | http://localhost:5000        | Experiment tracking, model registry, artifacts              |
| Airflow              | http://localhost:8080        | DAGs for training, evaluation, and Hugging Face export      |

---

## Getting started

### Requirements

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (with Docker Compose)
- ~8 GB free RAM for the full stack (Airflow + Postgres + MLflow + API + UI)
- Optional: an NVIDIA GPU + drivers if you want to use the `.gpu` Dockerfiles/compose overrides

### 1. Clone the repository

```bash
git clone <this-repo-url>
cd <this-repo>
```

### 2. Add the dataset

See [Dataset](#dataset) below — download the images from the UFF DMI database and place them under `DATA/` following the existing folder structure.

### 3. Configure secrets

Create a `.env` file in the project root (see [Configuration & secrets](#configuration--secrets)) with, at minimum:

```
API_KEY=<choose-a-strong-random-key>
```

### 4. Start the stack

**Linux/macOS:**
```bash
docker-compose up --build   # first run
docker-compose up           # subsequent runs
docker-compose down         # stop everything
```

**Windows:** double-click `iniciar_com_build.bat` on the first run, `iniciar_sem_build.bat` on subsequent runs, and `encerrar.bat` to stop. A terminal window will handle the setup, and your browser will open automatically on the Gradio UI once the services are ready.

### 5. Use the UI

Open http://localhost:7860, log in, upload a breast thermography image, and the app will return a diagnosis and a Grad-CAM heatmap overlay.

More service-by-service commands (rebuilding a single container, checking logs, etc.) are listed in `COMMANDS.txt`.

---

## Dataset

This repository ships **only the expected folder structure** under `DATA/` — the actual thermographic images are **not included** in this repository.

Images used for training and evaluation come from the **UFF DMI (Database for Mastology Research with Infrared image)**:

**https://visual.ic.uff.br/dmi/**

To reproduce training, download the images from the link above and organize them to match the structure already present under `DATA/`:

```
DATA/breast_thermal_frontal/Train/healthy
DATA/breast_thermal_frontal/Train/sick
DATA/breast_thermal_frontal/Test/healthy
DATA/breast_thermal_frontal/Test/sick
```

(and analogously for `breast_thermal_frontal_SANITY_CHECK` and `breast_thermal_segmented`, which uses `Benign`/`Malignant` labels and an additional `Evaluate` split). The loader (`MODEL_data_loader.py`) uses `torchvision.datasets.ImageFolder`, so each class must be a subfolder containing its images.

Which dataset variant is used for a given run is controlled in `MODEL_config_loader.py` (`DataConfiguration._base_root`).

---

## Model & training

- **Architecture:** EfficientNet-B0 (`models_custom_module.py`), fine-tuned with a configurable multi-step unfreezing schedule (train head only → progressively unfreeze deeper blocks).
- **Framework:** PyTorch Lightning.
- **Grad-CAM:** hooks are registered on the last convolutional block (`self.model.features[8]`) to produce class-activation heatmaps at inference time (`forward_with_gradcam_heatmap`).
- **Tracking:** every run (metrics, confusion matrices, model weights, preprocessing pipeline) is logged to MLflow. A model promoted to the `production` alias in the MLflow Model Registry is automatically picked up by the inference API.
- **Orchestration:** Airflow DAGs (`airflow/dags/`) trigger training/evaluation as Docker containers via the `DockerOperator`, and can optionally export the resulting model to the Hugging Face Hub.

To run a training job directly (outside Airflow):

```bash
python MODEL_main.py
```

This expects an MLflow tracking server reachable at the URI configured in `MODEL_config_loader.py` / `MLFLOW_TRACKING_URI`.

---

## Inference API

Base URL: `http://localhost:8000`. All endpoints require the `UFF-API-KEY` header (see [Configuration & secrets](#configuration--secrets)).

| Endpoint                | Method | Description                                                  |
|--------------------------|--------|----------------------------------------------------------------|
| `/info`                 | GET    | Returns the name/version of the currently loaded model        |
| `/health`                | GET    | Returns API and model status                                  |
| `/predict`               | POST   | Runs inference on a base64-encoded image, returns the prediction |
| `/predict_with_gradcam`  | POST   | Same as above, plus a base64-encoded Grad-CAM heatmap (PNG)   |

Example request body for `/predict` and `/predict_with_gradcam`:

```json
{
  "inputs": "<base64-encoded image bytes>"
}
```

The model is reloaded automatically in the background whenever a new version is promoted to the `production` alias in MLflow — no API restart needed.

---

Suggested `.gitignore` entries:

```
mlflow_data/
postgres_data/
airflow/logs/
.env
__pycache__/
*.pyc
```

---

## Exporting to Hugging Face

`MODEL_export_to_hf.py` (triggered via `dag_export_to_huggingface.py` or manually) pushes the current `production` model from MLflow to the Hugging Face Hub. It requires an `HF_TOKEN` environment variable / Airflow Variable with a valid Hugging Face access token.

---

## Notes

- GPU variants of the Docker images/compose files (`*.gpu.yml`, `Dockerfile.*.gpu`) are provided for training and serving with an NVIDIA GPU; the CPU variants work out of the box on any machine with Docker.
