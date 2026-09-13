# NVMe Drive Failure Predictor

A predictive-maintenance tool for NVMe SSD fleets: a Random Forest model
trained on SMART telemetry predicts the probability that a drive is
heading toward failure, served through a FastAPI backend and a Next.js
dashboard where you can browse fleet-wide stats or run a prediction on a
single drive.

Live pieces:

- **`ml/`** -- trains the model from raw telemetry and writes out the
  fitted pipeline + evaluation metrics
- **`backend/`** -- FastAPI service that loads the trained pipeline and
  serves predictions, fleet stats, and model-info over HTTP
- **`frontend/`** -- Next.js (TypeScript + Tailwind) console-style
  dashboard that talks to the backend

---

## Why this exists

The first pass at this dataset scored 99.95% accuracy -- which turned out
to be a bug, not a result. The dataset has a column,
`SMART_Warning_Flag`, that is set to 1 if and only if `Failure_Flag` is 1
for every one of the 10,000 rows. It's the label, renamed. A model that
sees it "predicts" failure by reading the answer off the same row.
`Failure_Mode` has the same problem (it's only ever non-zero once a drive
has already failed).

This version:

- Trains only on the original, non-synthetic 10,000-row dataset (194 real
  failures, 1.94%)
- Drops `Drive_ID`, `SMART_Warning_Flag`, and `Failure_Mode` as inputs
- Adds rate-based engineered features (errors per 1,000 power-on hours,
  wear rate, write/read intensity, read/write ratio) instead of relying
  only on raw cumulative counters, which mostly just encode drive age
- Uses stratified 5-fold cross-validation and an 80/20 held-out test
  split, tuned on PR-AUC rather than accuracy, since accuracy is
  meaningless on a ~98%/2% class split
- Reports precision, recall, F1, ROC-AUC, and PR-AUC, and explicitly
  checks the CV-vs-holdout gap to catch overfitting rather than reporting
  a single flattering number

### Honest results (2,000-drive held-out test set)

| Metric    | Value |
|-----------|-------|
| Accuracy  | 99.0% |
| Precision | 69.4% |
| Recall    | 87.2% |
| F1        | 0.773 |
| ROC-AUC   | 0.997 |
| PR-AUC    | 0.912 |

In plain terms: the model catches roughly 9 in 10 drives that actually
fail, at the cost of about 15 false alarms per 1,961 healthy drives.
Full per-fold numbers and the confusion matrix are written to
`ml/artifacts/metrics.json` after training.

---

## How it works

1. **Training** (`ml/train.py`) cleans the raw CSV, engineers rate-based
   features, fits a `ColumnTransformer` (one-hot encoding for
   Vendor/Model/Firmware) into a `RandomForestClassifier`, tunes
   hyperparameters with `RandomizedSearchCV` against PR-AUC, and saves
   the fitted pipeline as `model_pipeline.joblib` alongside a
   `metrics.json` report.
2. **Serving** (`backend/app/main.py`) loads that pipeline once at
   startup. `POST /predict` takes raw SMART telemetry for one drive,
   applies the *same* feature-engineering function used at training time
   (kept in `backend/app/features.py`, a manually-synced copy of
   `ml/features.py`), and returns a failure probability plus a
   lightweight, dependency-free explanation of which telemetry values are
   pushing the score up.
3. **Dashboard** (`frontend/`) shows fleet-wide failure stats pulled
   straight from the training CSV (`GET /fleet-overview`), the model's
   own reported metrics (`GET /model-info`), and a form for running
   `/predict` against a single drive, including a few preset example
   drives (healthy, controller-surge, power-event-storm, firmware-watch)
   to try out the model's behavior quickly.

---

## Project layout

```
ml/
  data/                    Original dataset only (no synthetic rows)
  features.py               Data cleaning + feature engineering
  train.py                   Trains, tunes, evaluates, saves the model
  artifacts/                  model_pipeline.joblib + metrics.json (generated)

backend/
  app/
    main.py                  /predict, /fleet-overview, /model-info, /health
    schemas.py                 Pydantic request/response models
    features.py                 Copy of ml/features.py (keep in sync manually)
  model/                     model_pipeline.joblib + metrics.json go here
  requirements.txt
  Dockerfile                 For deploying to Render / Railway / Fly.io

frontend/
  src/app/                  Page + layout
  src/components/            Telemetry form, result panel, fleet dashboard
  src/lib/                    API client + shared TypeScript types
  .env.example
```

---

## Tech stack

- **ML**: Python, pandas, scikit-learn (`RandomForestClassifier`,
  `ColumnTransformer`, `RandomizedSearchCV`), joblib
- **Backend**: FastAPI, Pydantic, uvicorn
- **Frontend**: Next.js 16 (App Router, Turbopack), React 19, TypeScript,
  Tailwind CSS 4, Recharts, lucide-react

---

## Running it locally

You need Python 3.10+ and Node 18+.

### 1. Train the model

```bash
cd ml
pip install -r requirements.txt
python train.py
```

This runs a hyperparameter search (a few minutes) and writes
`artifacts/model_pipeline.joblib` and `artifacts/metrics.json`. Copy both
into the backend:

```bash
cp artifacts/model_pipeline.joblib artifacts/metrics.json ../backend/model/
```

> A pre-trained pipeline and metrics file are already included under
> `backend/model/`, so you can skip this step and go straight to step 2
> if you just want to run the app as-is.

### 2. Run the backend

```bash
cd ../backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for interactive Swagger docs, or
`http://localhost:8000/health` to confirm it's up.

### 3. Run the frontend

```bash
cd ../frontend
npm install
npm run dev
```

Visit `http://localhost:3000`. By default it talks to the backend at
`http://localhost:8000` (see `NEXT_PUBLIC_API_URL` in `.env.example` --
copy it to `.env.local` if you need to point at a different backend URL).

---

## API reference

| Endpoint          | Method | Description                                                        |
|--------------------|--------|--------------------------------------------------------------------|
| `/health`           | GET    | Liveness check                                                    |
| `/predict`          | POST   | Failure probability + risk level + explanation for one drive       |
| `/fleet-overview`   | GET    | Aggregate stats across the training dataset (for the dashboard)    |
| `/model-info`       | GET    | Training metadata: metrics, dropped columns, feature importances   |

Full request/response schemas are in `backend/app/schemas.py`, or browse
them live at `/docs` once the backend is running.

---

## Deploying

Vercel is a great fit for the **frontend**, but not for the **backend**
-- it's built for Next.js, not for hosting a long-lived Python process
that loads a ~4MB scikit-learn pipeline and pandas into memory. So this
ships as two pieces.

### Backend -- Render, Railway, or Fly.io

`backend/Dockerfile` builds the API as a container. Build context must be
the **repo root**, not `backend/`, since the image also needs `ml/data/`
(used by `/fleet-overview`):

```bash
docker build -f backend/Dockerfile -t nvme-backend .
```

On Render: **New > Web Service** -> connect the repo -> Runtime: Docker
-> Dockerfile Path: `backend/Dockerfile` -> Build Context Directory: `.`
-> free instance is fine to start.

Once your frontend is deployed (next section) and you know its URL, set
this environment variable on the backend service to lock down CORS:

```
ALLOWED_ORIGINS=https://your-app.vercel.app
```

Leaving it unset defaults to allowing any origin, which is fine for local
development but shouldn't stay that way once the backend is public.

### Frontend -- Vercel

Import the repo in Vercel, set **Root Directory** to `frontend` (Vercel
tries to build the whole monorepo otherwise), and add:

```
NEXT_PUBLIC_API_URL=https://your-backend-host.onrender.com
```

Redeploy after adding it -- `NEXT_PUBLIC_*` variables are baked in at
build time, not read at runtime.

Render's free tier spins down after 15 minutes idle, so the first request
after a lull can take 30-50 seconds while it wakes back up. That's
expected on the free tier, not a bug.

---

## Retraining / extending

- All feature engineering lives in `features.py`, duplicated once into
  `backend/app/features.py` so the API doesn't depend on the `ml/`
  package at runtime -- keep the two in sync if you change either.
- To change the decision threshold (default 0.5), edit
  `DECISION_THRESHOLD` in `backend/app/main.py`, or expose it as a
  request parameter if you want callers to tune the precision/recall
  tradeoff themselves.
- `ml/train.py` deliberately avoids `max_depth=None` in the
  hyperparameter search space -- unconstrained trees can perfectly fit
  the ~200 failure rows in training without generalizing better. If you
  widen the search space, watch the CV-vs-holdout gap printed at the end
  of training; a gap that grows past a few percentage points means the
  model has started memorizing rather than learning.
