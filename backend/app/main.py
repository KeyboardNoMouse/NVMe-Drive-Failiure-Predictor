"""
NVMe Drive Failure Predictor -- FastAPI backend.

Loads the trained scikit-learn pipeline (preprocessing + Random Forest)
and serves predictions over HTTP for the Next.js frontend.
"""
import json
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.features import ENGINEERED_COLS, FEATURE_COLS, RAW_NUMERIC_COLS, engineer_features
from app.schemas import (
    DriveTelemetry,
    FleetOverviewResponse,
    ModelInfoResponse,
    PredictionResponse,
)

MODEL_DIR = Path(__file__).parent.parent / "model"
PIPELINE_PATH = MODEL_DIR / "model_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
DATA_PATH = Path(__file__).parent.parent.parent / "ml" / "data" / "NVMe_Drive_Failure_Dataset.csv"

DECISION_THRESHOLD = 0.5  # probability >= threshold => predicted "failure"

app = FastAPI(
    title="NVMe Drive Failure Predictor API",
    description=(
        "Predicts probability of NVMe drive failure from SMART telemetry. "
        "Trained on the original (non-synthetic) 10,000-drive dataset with "
        "SMART_Warning_Flag and Failure_Mode excluded as leaky features."
    ),
    version="1.0.0",
)

# Comma-separated list of allowed frontend origins, e.g.
# "https://your-app.vercel.app,https://your-app-git-main.vercel.app"
# Falls back to "*" (any origin) for local development.
_allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = (
    ["*"] if _allowed_origins_env.strip() == "*"
    else [o.strip() for o in _allowed_origins_env.split(",") if o.strip()]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline = None
_metrics = None
_fleet = None

FAILURE_MODE_LABELS = {
    1: "Wear-out failure",
    4: "Controller / firmware",
    5: "Early-life failure",
}


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        if not PIPELINE_PATH.exists():
            raise RuntimeError(
                f"Model file not found at {PIPELINE_PATH}. Run ml/train.py "
                f"and copy artifacts/model_pipeline.joblib here first."
            )
        _pipeline = joblib.load(PIPELINE_PATH)
    return _pipeline


def get_metrics():
    global _metrics
    if _metrics is None:
        if not METRICS_PATH.exists():
            raise RuntimeError(f"Metrics file not found at {METRICS_PATH}.")
        with open(METRICS_PATH) as f:
            _metrics = json.load(f)
    return _metrics


def get_fleet():
    global _fleet
    if _fleet is None:
        if not DATA_PATH.exists():
            raise RuntimeError(f"Fleet dataset not found at {DATA_PATH}.")
        _fleet = pd.read_csv(DATA_PATH)
    return _fleet


def risk_bucket(prob: float) -> str:
    if prob < 0.2:
        return "low"
    if prob < 0.6:
        return "moderate"
    return "high"


def top_contributing_factors(row: dict, n: int = 5) -> list[dict]:
    """Lightweight, dependency-free explanation: for each numeric feature,
    score = global_importance * direction * how many std-devs this value
    sits above/below the healthy population mean. Ranks the features
    pushing hardest toward "failure" for this specific input."""
    metrics = get_metrics()
    ref = metrics.get("feature_reference_stats", {})
    scored = []
    for col in RAW_NUMERIC_COLS + ENGINEERED_COLS:
        if col not in ref or col not in row:
            continue
        stats = ref[col]
        z = (row[col] - stats["healthy_mean"]) / stats["healthy_std"]
        # positive "push" means this value moves the drive toward the
        # failed population relative to healthy, weighted by how much the
        # model actually relies on this feature
        push = max(z * stats["direction"], 0.0) * stats["importance"]
        if push > 0:
            scored.append({
                "feature": col,
                "value": round(row[col], 3),
                "healthy_typical": round(stats["healthy_mean"], 3),
                "push_score": round(float(push), 4),
            })
    scored.sort(key=lambda d: -d["push_score"])
    return scored[:n]


def failure_attribution(telemetry: DriveTelemetry, probability: float) -> list[dict]:
    """Return a transparent signal attribution, not a causal diagnosis.

    Failure_Mode is excluded from the model because it leaks the label. These
    buckets show which telemetry family is most suggestive for this drive,
    while the healthy share is the model's complement probability.
    """
    firmware_rates = get_fleet().groupby("Firmware_Version")["Failure_Flag"].mean()
    controller = (
        telemetry.media_errors * 1.5
        + telemetry.read_error_rate
        + telemetry.write_error_rate
        + max(telemetry.temperature_c - 55, 0) / 10
    )
    power = (
        telemetry.unsafe_shutdowns * 1.4
        + telemetry.crc_errors * 1.1
        + max(telemetry.power_on_hours - 30000, 0) / 10000
    )
    firmware = float(firmware_rates.get(telemetry.firmware_version, 0.02)) * 100
    signal_total = max(controller + power + firmware, 0.001)
    failed_share = probability * 100
    return [
        {"label": "Media / error signals", "percentage": round(failed_share * controller / signal_total, 1), "kind": "failure"},
        {"label": "Shutdown / interface signals", "percentage": round(failed_share * power / signal_total, 1), "kind": "failure"},
        {"label": "Firmware cohort signal", "percentage": round(failed_share * firmware / signal_total, 1), "kind": "failure"},
        {"label": "Healthy drive", "percentage": round((1 - probability) * 100, 1), "kind": "healthy"},
    ]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/fleet-overview", response_model=FleetOverviewResponse)
def fleet_overview():
    df = get_fleet()
    failures = df[df["Failure_Flag"] == 1]
    mode_counts = failures["Failure_Mode"].value_counts().sort_index()
    mode_total = max(len(failures), 1)

    vendor_group = df.groupby("Vendor").agg(drives=("Failure_Flag", "size"), failures=("Failure_Flag", "sum"))
    firmware_group = df.groupby("Firmware_Version").agg(drives=("Failure_Flag", "size"), failures=("Failure_Flag", "sum"))

    bins = [0, 20, 40, 60, 80, 100, 125, 200]
    labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%", "100-125%", "125%+"]
    banded = pd.cut(df["Percent_Life_Used"], bins=bins, labels=labels, include_lowest=True, right=False)
    life_bands = []
    for label in labels:
        rows = df[banded == label]
        life_bands.append({
            "label": label,
            "healthy": int((rows["Failure_Flag"] == 0).sum()),
            "failed": int((rows["Failure_Flag"] == 1).sum()),
        })

    sample_step = max(len(df) // 280, 1)
    telemetry_points = [
        {"temperature": round(float(row.Temperature_C), 1), "life_used": round(float(row.Percent_Life_Used), 1), "failed": int(row.Failure_Flag)}
        for row in df.iloc[::sample_step].itertuples()
    ]

    return FleetOverviewResponse(
        total_drives=int(len(df)),
        healthy_drives=int((df["Failure_Flag"] == 0).sum()),
        failed_drives=int(len(failures)),
        failure_rate=round(float(df["Failure_Flag"].mean()), 4),
        average_temperature=round(float(df["Temperature_C"].mean()), 1),
        average_life_used=round(float(df["Percent_Life_Used"].mean()), 1),
        average_power_on_hours=round(float(df["Power_On_Hours"].mean()), 0),
        failure_modes=[
            {"code": int(code), "label": FAILURE_MODE_LABELS.get(int(code), f"Mode {int(code)}"), "count": int(count), "percentage": round(float(count / mode_total * 100), 1)}
            for code, count in mode_counts.items()
        ],
        vendor_stats=[
            {"label": label, "drives": int(row.drives), "failures": int(row.failures), "failure_rate": round(float(row.failures / row.drives * 100), 2)}
            for label, row in vendor_group.sort_index().iterrows()
        ],
        firmware_stats=[
            {"label": label, "drives": int(row.drives), "failures": int(row.failures), "failure_rate": round(float(row.failures / row.drives * 100), 2)}
            for label, row in firmware_group.sort_index().iterrows()
        ],
        life_bands=life_bands,
        telemetry_points=telemetry_points,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    m = get_metrics()
    top_fi = dict(list(m["feature_importances"].items())[:10])
    return ModelInfoResponse(
        trained_at=m["trained_at"],
        dataset=m["dataset"],
        dropped_leaky_columns=m["dropped_leaky_columns"],
        holdout_test_metrics=m["holdout_test_metrics"],
        cv_metrics_train=m["cv_metrics_train"],
        cv_vs_holdout_gap=m["cv_vs_holdout_gap"],
        top_feature_importances=top_fi,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(telemetry: DriveTelemetry):
    try:
        pipeline = get_pipeline()
        raw = {
            "Vendor": telemetry.vendor,
            "Model": telemetry.model,
            "Firmware_Version": telemetry.firmware_version,
            "Power_On_Hours": telemetry.power_on_hours,
            "Total_TBW_TB": telemetry.total_tbw_tb,
            "Total_TBR_TB": telemetry.total_tbr_tb,
            "Temperature_C": telemetry.temperature_c,
            "Percent_Life_Used": telemetry.percent_life_used,
            "Media_Errors": telemetry.media_errors,
            "Unsafe_Shutdowns": telemetry.unsafe_shutdowns,
            "CRC_Errors": telemetry.crc_errors,
            "Read_Error_Rate": telemetry.read_error_rate,
            "Write_Error_Rate": telemetry.write_error_rate,
        }
        df = pd.DataFrame([raw])
        df = engineer_features(df)
        X = df[FEATURE_COLS]

        proba = float(pipeline.predict_proba(X)[0, 1])
        label = "failure" if proba >= DECISION_THRESHOLD else "healthy"

        row_for_explain = df.iloc[0].to_dict()
        factors = top_contributing_factors(row_for_explain)

        return PredictionResponse(
            failure_probability=round(proba, 4),
            risk_level=risk_bucket(proba),
            predicted_label=label,
            decision_threshold=DECISION_THRESHOLD,
            top_contributing_factors=factors,
            failure_attribution=failure_attribution(telemetry, proba),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
