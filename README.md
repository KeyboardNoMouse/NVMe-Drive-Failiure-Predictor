# NVMe Drive Failure Predictor & Fleet Dashboard

**SMART Telemetry Analysis · Random Forest Classifier · Flask REST API · 99.95% Accuracy**

A full-stack NVMe drive health monitoring system: a Flask backend serving a 14-endpoint REST API and a redesigned cyberpunk-style analytics dashboard.

---

## Quick Start

```bash
# 1. Install dependencies
pip install flask scikit-learn pandas numpy

# 2. Run the server
python server.py

# 3. Open your browser
#    Dashboard  →  http://localhost:5000
#    Predictor  →  http://localhost:5000/predictor
```

---

## Project Files

| File | Description |
|------|-------------|
| `server.py` | Flask backend — trains RF model, serves API + pages |
| `dashboard.html` | Multi-page analytics dashboard (served at `/`) |
| `nvme_failure_predictor.html` | Single-drive predictor (served at `/predictor`) |
| `NVMe_Drive_Failure_Dataset.csv` | Training dataset (10,000 drive snapshots) |
| `NVMe_Drives.docx` | Problem statement & dataset documentation |
| `rf_model.json` | Exported RF model for standalone HTML use |
| `model_data.json` | Feature statistics & mode profiles |
| `train_model.py` | Standalone Python training script |

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| **Overview** | Fleet KPIs with animated counters, failure mode donut, vendor bar, temp/life distributions, scatter plot |
| **Fleet Health** | Vendor & model matrix tables, firmware failure rates |
| **Error Analysis** | Radar chart (with Unsafe_Shutdowns fix), read/write error bars, media/CRC by vendor |
| **Alerts** | SMART-flagged drives sorted by criticality |
| **Drive Explorer** | Paginated, filterable, sortable full drive table |
| **Predictor** | Opens `/predictor` in a new tab (single-drive ML inference) |
| **Model Info** | RF config, feature importances, full API reference |

---

## Changes & Bug Fixes

### Bug Fixes (`server.py`)
- **Column mutation bug**: `temperature_dist` and `life_dist` were adding `_tb`/`_lb` columns directly to the shared `df` dataframe on every request. Fixed by operating on the `pd.cut` series directly instead.
- **Missing `Unsafe_Shutdowns` in error API**: `error_by_vendor` endpoint now returns `Unsafe_Shutdowns` mean so the radar chart renders all 5 axes correctly.
- **`NaN` serialization**: `/api/drives` now replaces `NaN` with `None` before calling `jsonify` to prevent silent failures with certain datasets.
- **Batch predict error handling**: Each item in batch prediction is now wrapped in try/except so one bad record doesn't abort the entire batch.
- **JSON body validation**: `/api/predict` and `/api/predict/batch` now return proper 400/422 errors on missing or malformed bodies.
- **Unknown categoricals**: `encode_input` now defaults to `0` for unknown vendor/model/firmware strings instead of a potential `KeyError`.

### Dashboard Redesign (`dashboard.html`)
- Complete visual overhaul: cyberpunk/ops-center aesthetic with Space Mono, Rajdhani & Barlow fonts
- Animated KPI counters on load
- Neon glow accents, scanline overlay, grid background
- Removed duplicate Predictor page — nav item now opens `/predictor` directly in a new tab
- All chart colors updated to match new palette with proper border glow
- Improved error handling — all data loads wrapped in try/catch with user-visible error states
- Vendor names color-coded consistently across tables and charts

---

## REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/fleet/summary` | Fleet KPI summary |
| GET | `/api/fleet/failure_modes` | Failure mode distribution |
| GET | `/api/fleet/by_vendor` | Stats grouped by vendor |
| GET | `/api/fleet/by_model` | Stats grouped by model |
| GET | `/api/fleet/by_firmware` | Stats grouped by firmware |
| GET | `/api/fleet/temperature_dist` | Temperature distribution buckets |
| GET | `/api/fleet/life_dist` | Life-used distribution buckets |
| GET | `/api/fleet/scatter` | Scatter plot data (POH vs life) |
| GET | `/api/fleet/error_by_vendor` | Error metrics by vendor (incl. Unsafe_Shutdowns) |
| GET | `/api/drives` | Paginated drive list (filter/sort params) |
| GET | `/api/alerts` | Top SMART-flagged alert drives |
| POST | `/api/predict` | Single drive failure prediction |
| POST | `/api/predict/batch` | Batch prediction (up to 500 drives) |
| GET | `/api/model/info` | Model metadata & feature importances |

---

## Model Details

| Property | Value |
|----------|-------|
| Algorithm | Random Forest Classifier |
| Trees | 100 (sklearn) · 20 exported to JS |
| Max Depth | 15 |
| Class Weighting | Balanced |
| Test Accuracy | **99.95%** |
| Training Samples | 10,000 |
| Features | 14 |

### Failure Modes

| Mode | Label |
|------|-------|
| 0 | Healthy |
| 1 | Wear-Out Failure (high TBW / life %) |
| 4 | Controller / Firmware Failure |
| 5 | Rapid Error Accumulation (Early-Life) |

---

*Flask · scikit-learn · Chart.js · HTML/CSS/JS*
