# NVMe Drive Failure Predictor & Fleet Dashboard

**SMART Telemetry Analysis · Random Forest Classifier · Flask REST API · 99.95% Accuracy**

A full-stack NVMe drive health monitoring system: a Flask backend serving a 14-endpoint REST API and a rich multi-page analytics dashboard.

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
| **Overview** | Fleet KPIs, failure mode donut, vendor bar, temp/life distributions, scatter plot |
| **Fleet Health** | Vendor & model matrix tables, firmware failure rates |
| **Error Analysis** | Radar chart, read/write error bars, media/CRC by vendor |
| **Alerts** | SMART-flagged drives sorted by criticality |
| **Drive Explorer** | Paginated, filterable, sortable full drive table |
| **Predictor** | Embedded single-drive ML inference tool |
| **Model Info** | RF config, feature importances, full API reference |

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
| GET | `/api/fleet/error_by_vendor` | Error metrics by vendor |
| GET | `/api/drives` | Paginated drive list (filter/sort params) |
| GET | `/api/alerts` | Top SMART-flagged alert drives |
| POST | `/api/predict` | Single drive failure prediction |
| POST | `/api/predict/batch` | Batch prediction (up to 500 drives) |
| GET | `/api/model/info` | Model metadata & feature importances |

### Example: Predict a drive
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Power_On_Hours": 1200, "Total_TBW_TB": 95, "Total_TBR_TB": 80,
    "Temperature_C": 41, "Percent_Life_Used": 24, "Media_Errors": 1,
    "Unsafe_Shutdowns": 3, "CRC_Errors": 1, "Read_Error_Rate": 15.5,
    "Write_Error_Rate": 10.0, "SMART_Warning_Flag": 1,
    "Vendor": "VendorA", "Model": "Model-X2", "Firmware_Version": "FW1.0"
  }'
```

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
