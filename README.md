# NVMe Drive Failure Predictor

A full-stack predictive maintenance system that estimates the probability of an NVMe SSD failing based on its SMART telemetry, served through a REST API and visualized in a fleet-monitoring dashboard.

## Overview

Enterprise and datacenter storage teams rely on early warning signs to replace drives before they fail in production. This project trains a machine learning model on real NVMe SMART telemetry (power-on hours, wear level, error counters, temperature, etc.) to predict failure risk, then exposes that model through a FastAPI backend and a Next.js dashboard that supports both single-drive predictions and fleet-wide analytics.

**Core problem characteristics tackled:**
- Rare-event classification (only ~1.9% of drives in the dataset actually failed)
- Avoiding data leakage from features that are effectively derived from the label
- Producing model explanations without a heavyweight library dependency
- Serving a stateful ML pipeline reliably in a real API, not just a notebook

## Tech stack

| Layer | Technologies |
|---|---|
| **Machine learning** | Python, pandas, scikit-learn (RandomForestClassifier, Pipeline, ColumnTransformer, OneHotEncoder, RandomizedSearchCV, StratifiedKFold), NumPy, joblib |
| **Backend API** | FastAPI, Pydantic, Uvicorn, Docker |
| **Frontend** | Next.js 16, React 19, TypeScript, Tailwind CSS v4, Recharts, Lucide icons |
| **Deployment** | Vercel (frontend), Docker on Render/Railway/Fly.io (backend) |

## File structure

```
ml/                          Training pipeline
  data/                       Source dataset (SMART telemetry, 10,000 drives)
  features.py                 Shared data cleaning + feature engineering
  train.py                    Trains, tunes, evaluates, and saves the model
  artifacts/                  model_pipeline.joblib + metrics.json (generated)

backend/                     FastAPI service
  app/
    main.py                   /predict, /model-info, /fleet-overview, /health endpoints
    schemas.py                 Pydantic request/response models
    features.py                 Mirrors ml/features.py for consistent inference
  model/                      Deployed copy of the trained model + metrics
  Dockerfile

frontend/                    Next.js (TypeScript + Tailwind) dashboard
  src/app/                     Page + layout
  src/components/              TelemetryForm, ResultPanel, FleetDashboard, ModelInfoStrip
  src/lib/                     API client + shared types
```

## Machine learning approach

### Feature engineering
Raw SMART counters are cumulative (e.g. total errors), so they mostly just reflect drive age rather than actual risk. The pipeline derives rate-based features that normalize for usage:

- **TBW/TBR per hour** — write/read intensity
- **Wear rate per 1,000 hours** — how fast a drive is burning through its rated endurance
- **Errors per 1,000 hours** — media + CRC error rate, normalized by operating time
- **Unsafe shutdown rate**, **read/write ratio**, **media error ratio**

### Handling data leakage
Two raw dataset columns were identified as directly encoding the target label (a warning flag and a failure-mode code that are only ever non-zero for already-failed drives) and were deliberately excluded from the feature set, along with the row identifier. This is a key design decision worth discussing in interviews: naively including "leaky" columns can make a model look artificially perfect (99%+ accuracy) while learning nothing predictive.

### Model
A **Random Forest classifier** inside a scikit-learn `Pipeline`, with a `ColumnTransformer` one-hot encoding categorical fields (vendor, model, firmware) and passing engineered numeric features through untouched.

### Hyperparameter tuning
`RandomizedSearchCV` (18 candidates × 5-fold stratified CV) searches over tree count, max depth, leaf/split sizes, feature sampling, and class weighting. Two choices stand out:

- **Scored on PR-AUC (average precision), not accuracy** — with a 98%/2% class split, a model that always predicts "healthy" scores 98% accuracy while catching zero failures. PR-AUC forces the search to actually reward catching the rare positive class.
- **`class_weight` balancing instead of oversampling** — `balanced` / `balanced_subsample` reweight the loss for the minority class rather than synthetically duplicating rows.
- **Bounded max depth** — unconstrained trees can carve out a leaf for nearly every failure row in training, inflating in-sample performance without improving generalization. Depth is intentionally capped.

### Validation strategy
- **Stratified 5-fold cross-validation** on the training set, preserving the ~1.9% failure rate in every fold.
- **80/20 stratified holdout split**, set aside before any tuning and never touched until final evaluation.
- **CV-vs-holdout comparison** as a memorization check: two independent out-of-sample estimates should roughly agree if the model generalizes. This is computed and reported explicitly.

### Held-out test results (2,000 drives never used in training or tuning)

| Metric | Value |
|---|---|
| Accuracy | 99.0% |
| Precision | 69.4% |
| Recall | 87.2% |
| F1 | 0.773 |
| ROC-AUC | 0.997 |
| PR-AUC | 0.912 |

The model catches roughly 9 in 10 failing drives, with a modest false-alarm rate. Precision/recall are reported instead of leaning on accuracy alone, since accuracy is not meaningful on this class imbalance.

### Explainability without SHAP
Each prediction is accompanied by a lightweight, dependency-free explanation: for every input feature, the model computes how many standard deviations that value sits from the healthy-population mean, weights it by the feature's global importance and failure direction, and surfaces the top contributing factors — giving interpretable "why" signals without adding a SHAP/treeinterpreter dependency.

## API endpoints

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Takes a drive's telemetry, returns failure probability, risk level, and top contributing factors |
| `GET /model-info` | Returns model metadata: training metrics, feature importances, dataset stats |
| `GET /fleet-overview` | Aggregate fleet statistics: failure rates by vendor/firmware, wear distribution, temperature trends |
| `GET /health` | Liveness check |

## Running locally

```bash
# 1. Train the model
cd ml
pip install -r requirements.txt
python train.py
cp artifacts/model_pipeline.joblib artifacts/metrics.json ../backend/model/

# 2. Run the backend
cd ../backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# 3. Run the frontend
cd ../frontend
npm install
npm run dev
```

## Deployment

The backend and frontend are deployed as two separate services since the API is a stateful process (loads pandas + a scikit-learn pipeline into memory) rather than a serverless function:

- **Frontend** deploys to Vercel (root directory `frontend`).
- **Backend** runs from the provided `Dockerfile` on Render, Railway, or Fly.io, with `ALLOWED_ORIGINS` configured for CORS once the frontend URL is known.
