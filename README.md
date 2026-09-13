# NVMe Failure Predictor (rebuilt)

A from-scratch rebuild of the NVMe drive failure predictor: FastAPI backend
serving a Random Forest model, Next.js frontend, and an ML pipeline that
fixes the data leakage responsible for the original project's 99.95%
accuracy.

## Why the old model was wrong

The original project trained on `NVMe_Drive_Failure_Dataset_Augmented.csv`
(synthetic rows added) and included `SMART_Warning_Flag` as an input
feature. That column is set to 1 if and only if `Failure_Flag` is 1, for
every one of the 10,000 real rows -- it's the label, renamed. Any model
that sees it will report ~100% accuracy without learning anything about
actual drive telemetry. `Failure_Mode` has the same problem (it's only
non-zero when the drive already failed).

This rebuild:

- Trains only on `NVMe_Drive_Failure_Dataset.csv`, the **original,
  non-synthetic** dataset (10,000 rows, 194 real failures, 1.94%).
- Drops `Drive_ID`, `SMART_Warning_Flag`, and `Failure_Mode` as model
  inputs.
- Adds real feature engineering: rate-based features (errors per 1,000
  power-on hours, wear rate, write/read intensity, read/write ratio)
  instead of only raw cumulative counters, which materially improved
  recall on the rare failure class.
- Uses stratified 5-fold cross-validation plus an 80/20 held-out test
  split, and tunes hyperparameters on **PR-AUC** (average precision)
  rather than accuracy, because accuracy is meaningless on a 98%/2%
  class split (always predicting "healthy" already scores 98%).
- Reports precision, recall, F1, ROC-AUC, and PR-AUC -- and explicitly
  compares cross-validation scores against the held-out test set to show
  the model generalizes rather than memorizes (the largest metric gap is
  7.0 percentage points between CV and holdout).

### Honest results (held-out 20% test set, 2,000 drives never used for
### training or tuning)

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 99.0%  |
| Precision | 69.4%  |
| Recall    | 87.2%  |
| F1        | 0.773  |
| ROC-AUC   | 0.997  |
| PR-AUC    | 0.912  |

This is a real, harder number than 99.95%: the model catches roughly 9 in
10 failing drives, with 15 false alarms out of 1,961 healthy drives in the
test set. Precision of ~69% reflects the tradeoff for catching more of the
rare failures. Full details, the confusion
matrix, and per-fold numbers are in `ml/artifacts/metrics.json` after
training.

## Project layout

```
ml/                    Training pipeline (run this first)
  data/                 Original dataset only (no synthetic data)
  features.py           Shared data cleaning + feature engineering
  train.py               Trains, tunes, evaluates, and saves the model
  artifacts/              model_pipeline.joblib + metrics.json (generated)

backend/                FastAPI service
  app/
    main.py               /predict, /model-info, /health
    schemas.py             Pydantic request/response models
    features.py             Copy of ml/features.py (kept in sync manually)
  model/                  Copy model_pipeline.joblib + metrics.json here

frontend/               Next.js (TypeScript + Tailwind) console UI
  src/app/                Page + layout
  src/components/         Telemetry form, result panel, model-info panel
  src/lib/                API client + shared types
```

## Running it locally

### 1. Train the model

```bash
cd ml
pip install -r requirements.txt
python train.py
```

This takes a few minutes (hyperparameter search). It writes
`artifacts/model_pipeline.joblib` and `artifacts/metrics.json`.

Copy both files into the backend:

```bash
cp artifacts/model_pipeline.joblib artifacts/metrics.json ../backend/model/
```

### 2. Run the backend

```bash
cd ../backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` for interactive API docs.

### 3. Run the frontend

```bash
cd ../frontend
npm install
npm run dev
```

Visit `http://localhost:3000`. It talks to the backend at
`http://localhost:8000` by default (see `.env.local`,
`NEXT_PUBLIC_API_URL`).

## Deploying

Vercel only runs the **frontend** well here -- it's built for Next.js, not
for hosting a long-lived Python process that loads a ~4MB scikit-learn
pipeline and pandas into memory. So this deploys as two pieces:

### 1. Backend (FastAPI) -- Render, Railway, or Fly.io

Any of these can run `backend/Dockerfile` directly (build context must be
the **repo root**, since the image also needs `ml/data/` for the
`/fleet-overview` endpoint):

```bash
docker build -f backend/Dockerfile -t nvme-backend .
```

On Render: New > Web Service > connect the repo > set "Dockerfile path" to
`backend/Dockerfile` and "Docker build context" to the repo root. Add an
environment variable once you know your Vercel URL:

```
ALLOWED_ORIGINS=https://your-app.vercel.app
```

(Leave `ALLOWED_ORIGINS` unset locally -- it defaults to allowing any
origin, which is fine for `localhost` development but should always be
locked down once the backend is public.)

Note the deployed backend URL (e.g. `https://nvme-backend.onrender.com`)
-- you need it for step 2.

### 2. Frontend (Next.js) -- Vercel

```bash
cd frontend
vercel
```

Or via the Vercel dashboard: New Project > import the repo > set the
**root directory to `frontend`** (important -- otherwise Vercel tries to
build the whole monorepo). Then add the environment variable from
`.env.example`:

```
NEXT_PUBLIC_API_URL=https://nvme-backend.onrender.com
```

Redeploy after adding it (Vercel only bakes `NEXT_PUBLIC_*` vars in at
build time). Once both are live, open the Vercel URL -- it now calls your
Render/Railway backend instead of `localhost:8000`.

## Retraining / extending

- All feature engineering lives in `features.py` (duplicated once into
  `backend/app/features.py` so the API doesn't depend on the `ml/`
  package at runtime -- keep the two in sync if you change it).
- To change the decision threshold (default 0.5), edit
  `DECISION_THRESHOLD` in `backend/app/main.py`, or expose it as a
  request parameter if you want to tune precision/recall tradeoffs
  per use case (e.g. lower it to catch more failures at the cost of more
  false alarms).
- `ml/train.py` explicitly avoids `max_depth=None` in the hyperparameter
  search space -- unconstrained trees can perfectly fit the ~200 failure
  rows in the training set without improving true generalization. If you
  widen the search space, keep an eye on the CV-vs-holdout gap printed at
  the end of training; a gap that grows past a few percentage points is
  the tell that the model has started memorizing rather than learning.
