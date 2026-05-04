"""
NVMe Drive Failure Predictor — Flask Backend Server
====================================================
Serves the dashboard UI and exposes REST API endpoints for:
  - Fleet statistics & chart data
  - Live drive failure prediction (Random Forest)
  - Drive listing with filtering
  - Alert feed

Requirements:
    pip install flask scikit-learn pandas numpy

Usage:
    python server.py
    → Open http://localhost:5000 in your browser
"""

from flask import Flask, jsonify, request, send_from_directory, render_template_string
import pandas as pd
import numpy as np
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ── App setup ──────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load & prepare data ────────────────────────────────────────────────────
AUGMENTED_PATH = os.path.join(BASE_DIR, "NVMe_Drive_Failure_Dataset_Augmented.csv")
ORIGINAL_PATH = os.path.join(BASE_DIR, "NVMe_Drive_Failure_Dataset.csv")
CSV_PATH = AUGMENTED_PATH if os.path.exists(AUGMENTED_PATH) else ORIGINAL_PATH
df_raw = pd.read_csv(CSV_PATH)

le_vendor = LabelEncoder()
le_model  = LabelEncoder()
le_fw     = LabelEncoder()

df = df_raw.copy()
df["Vendor_enc"] = le_vendor.fit_transform(df["Vendor"])
df["Model_enc"]  = le_model.fit_transform(df["Model"])
df["FW_enc"]     = le_fw.fit_transform(df["Firmware_Version"])

FEATURE_COLS = [
    "Power_On_Hours", "Total_TBW_TB", "Total_TBR_TB", "Temperature_C",
    "Percent_Life_Used", "Media_Errors", "Unsafe_Shutdowns", "CRC_Errors",
    "Read_Error_Rate", "Write_Error_Rate", "SMART_Warning_Flag",
    "Vendor_enc", "Model_enc", "FW_enc",
]

FAIL_LABELS = {
    0: "Healthy",
    1: "Wear-Out Failure",
    2: "Thermal Failure",
    3: "Power-Related Failure",
    4: "Controller / Firmware Failure",
    5: "Rapid Error Accumulation (Early-Life)",
}

# ── Train model ────────────────────────────────────────────────────────────
print("Training Random Forest model …")
X = df[FEATURE_COLS]
y = df["Failure_Mode"]

rf = RandomForestClassifier(
    n_estimators=100, max_depth=15, min_samples_split=5,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
rf.fit(X, y)
print(f"Model ready — classes: {rf.classes_.tolist()}")

IMPORTANCES = {col: float(rf.feature_importances_[i]) for i, col in enumerate(FEATURE_COLS)}

def encode_input(d: dict) -> list:
    """Convert raw input dict → feature vector."""
    vendor_map = {v: int(i) for i, v in enumerate(le_vendor.classes_)}
    model_map  = {v: int(i) for i, v in enumerate(le_model.classes_)}
    fw_map     = {v: int(i) for i, v in enumerate(le_fw.classes_)}
    return [
        float(d.get("Power_On_Hours", 0)),
        float(d.get("Total_TBW_TB", 0)),
        float(d.get("Total_TBR_TB", 0)),
        float(d.get("Temperature_C", 40)),
        float(d.get("Percent_Life_Used", 0)),
        float(d.get("Media_Errors", 0)),
        float(d.get("Unsafe_Shutdowns", 0)),
        float(d.get("CRC_Errors", 0)),
        float(d.get("Read_Error_Rate", 0)),
        float(d.get("Write_Error_Rate", 0)),
        float(d.get("SMART_Warning_Flag", 0)),
        vendor_map.get(d.get("Vendor", "VendorA"), 0),
        model_map.get(d.get("Model", "Model-PRO"), 0),
        fw_map.get(d.get("Firmware_Version", "FW2.0"), 0),
    ]


# ══════════════════════════════════════════════════════════════════════════
# ROUTES — Static pages
# ══════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Serve the dashboard."""
    dashboard_path = os.path.join(BASE_DIR, "dashboard.html")
    with open(dashboard_path, "r", encoding="utf-8") as f:
        return f.read()


@app.route("/predictor")
def predictor():
    """Serve the original single-drive predictor."""
    path = os.path.join(BASE_DIR, "nvme_failure_predictor.html")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ══════════════════════════════════════════════════════════════════════════
# API — Fleet Analytics
# ══════════════════════════════════════════════════════════════════════════

@app.route("/api/fleet/summary")
def fleet_summary():
    return jsonify({
        "total":         int(len(df)),
        "healthy":       int((df["Failure_Flag"] == 0).sum()),
        "failed":        int((df["Failure_Flag"] == 1).sum()),
        "smart_flagged": int(df["SMART_Warning_Flag"].sum()),
        "high_temp":     int((df["Temperature_C"] > 55).sum()),
        "high_life":     int((df["Percent_Life_Used"] > 80).sum()),
        "failure_rate":  round(df["Failure_Flag"].mean() * 100, 2),
        "avg_temp":      round(df["Temperature_C"].mean(), 1),
        "avg_life":      round(df["Percent_Life_Used"].mean(), 1),
    })


@app.route("/api/fleet/failure_modes")
def failure_modes():
    counts = df[df["Failure_Flag"] == 1]["Failure_Mode"].value_counts().to_dict()
    return jsonify({
        str(k): {"count": int(v), "label": FAIL_LABELS.get(k, "Unknown")}
        for k, v in counts.items()
    })


@app.route("/api/fleet/by_vendor")
def by_vendor():
    grp = df.groupby("Vendor").agg(
        total=("Drive_ID", "count"),
        failed=("Failure_Flag", "sum"),
        avg_temp=("Temperature_C", "mean"),
        avg_life=("Percent_Life_Used", "mean"),
        avg_read_err=("Read_Error_Rate", "mean"),
    ).reset_index()
    grp["failure_rate"] = (grp["failed"] / grp["total"] * 100).round(2)
    grp["avg_temp"]     = grp["avg_temp"].round(1)
    grp["avg_life"]     = grp["avg_life"].round(1)
    grp["avg_read_err"] = grp["avg_read_err"].round(2)
    return jsonify(grp.to_dict("records"))


@app.route("/api/fleet/by_model")
def by_model():
    grp = df.groupby("Model").agg(
        total=("Drive_ID", "count"),
        failed=("Failure_Flag", "sum"),
        avg_life=("Percent_Life_Used", "mean"),
    ).reset_index()
    grp["failure_rate"] = (grp["failed"] / grp["total"] * 100).round(2)
    grp["avg_life"]     = grp["avg_life"].round(1)
    return jsonify(grp.to_dict("records"))


@app.route("/api/fleet/by_firmware")
def by_firmware():
    grp = df.groupby("Firmware_Version").agg(
        total=("Drive_ID", "count"),
        failed=("Failure_Flag", "sum"),
    ).reset_index()
    grp["failure_rate"] = (grp["failed"] / grp["total"] * 100).round(2)
    return jsonify(grp.to_dict("records"))


@app.route("/api/fleet/temperature_dist")
def temperature_dist():
    bins   = [20, 30, 40, 50, 60, 70, 80]
    labels = ["20-30", "30-40", "40-50", "50-60", "60-70", "70+"]
    df["_tb"] = pd.cut(df["Temperature_C"], bins=bins, labels=labels, right=False)
    dist = df["_tb"].value_counts().sort_index()
    return jsonify({"labels": labels, "counts": [int(dist.get(l, 0)) for l in labels]})


@app.route("/api/fleet/life_dist")
def life_dist():
    bins   = [0, 20, 40, 60, 80, 100, 130]
    labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%", ">100%"]
    df["_lb"] = pd.cut(df["Percent_Life_Used"], bins=bins, labels=labels, right=False)
    dist = df["_lb"].value_counts().sort_index()
    return jsonify({"labels": labels, "counts": [int(dist.get(l, 0)) for l in labels]})


@app.route("/api/fleet/scatter")
def scatter():
    failed  = df[df["Failure_Flag"] == 1][
        ["Power_On_Hours", "Percent_Life_Used", "Failure_Mode", "Vendor"]
    ].to_dict("records")
    healthy = df[df["Failure_Flag"] == 0][
        ["Power_On_Hours", "Percent_Life_Used", "Failure_Mode", "Vendor"]
    ].sample(400, random_state=42).to_dict("records")
    return jsonify({"failed": failed, "healthy": healthy})


@app.route("/api/fleet/error_by_vendor")
def error_by_vendor():
    grp = df.groupby("Vendor").agg(
        Read_Error_Rate=("Read_Error_Rate", "mean"),
        Write_Error_Rate=("Write_Error_Rate", "mean"),
        Media_Errors=("Media_Errors", "mean"),
        CRC_Errors=("CRC_Errors", "mean"),
    ).round(3).reset_index()
    return jsonify(grp.to_dict("records"))


# ══════════════════════════════════════════════════════════════════════════
# API — Drives listing
# ══════════════════════════════════════════════════════════════════════════

@app.route("/api/drives")
def drives():
    page     = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 50))
    vendor   = request.args.get("vendor")
    model    = request.args.get("model")
    status   = request.args.get("status")   # "healthy" | "failed"
    sort_by  = request.args.get("sort", "Drive_ID")
    order    = request.args.get("order", "asc")

    q = df_raw.copy()
    if vendor: q = q[q["Vendor"] == vendor]
    if model:  q = q[q["Model"] == model]
    if status == "failed":  q = q[q["Failure_Flag"] == 1]
    if status == "healthy": q = q[q["Failure_Flag"] == 0]

    if sort_by in q.columns:
        q = q.sort_values(sort_by, ascending=(order == "asc"))

    total = len(q)
    start = (page - 1) * per_page
    page_data = q.iloc[start:start + per_page].to_dict("records")

    return jsonify({
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": (total + per_page - 1) // per_page,
        "data": page_data,
    })


# ══════════════════════════════════════════════════════════════════════════
# API — Alerts
# ══════════════════════════════════════════════════════════════════════════

@app.route("/api/alerts")
def alerts():
    flagged = df_raw[df_raw["SMART_Warning_Flag"] == 1].copy()
    flagged["severity"] = flagged["Failure_Flag"].map({0: "warning", 1: "critical"})
    top = flagged.sort_values("Percent_Life_Used", ascending=False).head(20)
    return jsonify(top[[
        "Drive_ID", "Vendor", "Model", "Firmware_Version",
        "Temperature_C", "Percent_Life_Used", "Failure_Mode",
        "Failure_Flag", "severity",
    ]].to_dict("records"))


# ══════════════════════════════════════════════════════════════════════════
# API — Prediction
# ══════════════════════════════════════════════════════════════════════════

@app.route("/api/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True)
    features = [encode_input(body)]
    proba    = rf.predict_proba(features)[0]
    pred     = int(rf.predict(features)[0])

    class_probs = {
        int(cls): round(float(prob), 4)
        for cls, prob in zip(rf.classes_, proba)
    }
    failure_prob = round(1.0 - class_probs.get(0, 0.0), 4)

    return jsonify({
        "prediction":     pred,
        "label":          FAIL_LABELS.get(pred, "Unknown"),
        "failure":        pred != 0,
        "failure_prob":   failure_prob,
        "confidence":     round(float(max(proba)), 4),
        "class_probs":    class_probs,
        "importances":    IMPORTANCES,
    })


@app.route("/api/predict/batch", methods=["POST"])
def predict_batch():
    body  = request.get_json(force=True)
    items = body if isinstance(body, list) else body.get("drives", [])
    results = []
    for item in items[:500]:   # cap at 500
        features = [encode_input(item)]
        proba = rf.predict_proba(features)[0]
        pred  = int(rf.predict(features)[0])
        results.append({
            "drive_id":     item.get("Drive_ID", "?"),
            "prediction":   pred,
            "label":        FAIL_LABELS.get(pred, "Unknown"),
            "failure":      pred != 0,
            "failure_prob": round(1.0 - float(dict(zip(rf.classes_, proba)).get(0, 0)), 4),
        })
    return jsonify(results)


# ══════════════════════════════════════════════════════════════════════════
# API — Model metadata
# ══════════════════════════════════════════════════════════════════════════

@app.route("/api/model/info")
def model_info():
    return jsonify({
        "algorithm":   "Random Forest Classifier",
        "n_estimators": rf.n_estimators,
        "max_depth":   rf.max_depth,
        "classes":     [int(c) for c in rf.classes_],
        "features":    FEATURE_COLS,
        "importances": IMPORTANCES,
        "vendors":     le_vendor.classes_.tolist(),
        "models":      le_model.classes_.tolist(),
        "firmwares":   le_fw.classes_.tolist(),
        "fail_labels": {str(k): v for k, v in FAIL_LABELS.items()},
    })


# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  NVMe Failure Predictor Server")
    print("  Dashboard  →  http://localhost:5000")
    print("  Predictor  →  http://localhost:5000/predictor")
    print("  API docs   →  http://localhost:5000/api/model/info")
    print("="*55 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
