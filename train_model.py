"""
NVMe Drive Failure Predictor - Model Training Script
=====================================================
Trains a Random Forest classifier on the NVMe Drive Failure Dataset.
Exports the model as JSON for use in the HTML frontend.

Requirements:
    pip install scikit-learn pandas numpy

Usage:
    python train_model.py
"""

import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ── Load Data ──────────────────────────────────────────────────────────────
df = pd.read_csv("NVMe_Drive_Failure_Dataset.csv")
print(f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"\nFailure Mode Distribution:\n{df['Failure_Mode'].value_counts().to_string()}")

# ── Feature Engineering ────────────────────────────────────────────────────
le_vendor = LabelEncoder()
le_model  = LabelEncoder()
le_fw     = LabelEncoder()

df['Vendor_enc'] = le_vendor.fit_transform(df['Vendor'])
df['Model_enc']  = le_model.fit_transform(df['Model'])
df['FW_enc']     = le_fw.fit_transform(df['Firmware_Version'])

FEATURE_COLS = [
    'Power_On_Hours', 'Total_TBW_TB', 'Total_TBR_TB', 'Temperature_C',
    'Percent_Life_Used', 'Media_Errors', 'Unsafe_Shutdowns', 'CRC_Errors',
    'Read_Error_Rate', 'Write_Error_Rate', 'SMART_Warning_Flag',
    'Vendor_enc', 'Model_enc', 'FW_enc'
]

X      = df[FEATURE_COLS]
y_mode = df['Failure_Mode']   # multiclass (0,1,4,5)
y_flag = df['Failure_Flag']   # binary

# ── Train / Evaluate ───────────────────────────────────────────────────────
X_tr, X_te, ym_tr, ym_te = train_test_split(
    X, y_mode, test_size=0.2, random_state=42, stratify=y_mode)

rf = RandomForestClassifier(
    n_estimators=100, max_depth=15, min_samples_split=5,
    class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_tr, ym_tr)

ym_pred = rf.predict(X_te)
print(f"\nTest Accuracy : {accuracy_score(ym_te, ym_pred):.4f}")
print("\nClassification Report:")
print(classification_report(ym_te, ym_pred))

# ── Retrain on Full Dataset & Export ──────────────────────────────────────
rf.fit(X, y_mode)

def export_tree(tree):
    t = tree.tree_
    classes = [int(c) for c in rf.classes_]
    def recurse(node):
        if t.children_left[node] == -1:
            counts = t.value[node][0].tolist()
            total  = sum(counts)
            return {'leaf': True, 'probs': [c/total for c in counts], 'classes': classes}
        return {
            'leaf': False,
            'feature':   int(t.feature[node]),
            'threshold': float(t.threshold[node]),
            'left':  recurse(t.children_left[node]),
            'right': recurse(t.children_right[node])
        }
    return recurse(0)

trees = [export_tree(est) for est in rf.estimators_[:20]]

model_js = {
    'trees':        trees,
    'classes':      [int(c) for c in rf.classes_],
    'feature_cols': FEATURE_COLS,
    'importances':  {col: float(rf.feature_importances_[i])
                     for i, col in enumerate(FEATURE_COLS)},
    'encoders': {
        'Vendor':   {v: int(i) for i, v in enumerate(le_vendor.classes_)},
        'Model':    {v: int(i) for i, v in enumerate(le_model.classes_)},
        'Firmware': {v: int(i) for i, v in enumerate(le_fw.classes_)}
    },
    'failure_labels': {
        0: 'Healthy',
        1: 'Wear-Out Failure',
        2: 'Thermal Failure',
        3: 'Power-Related Failure',
        4: 'Controller/Firmware Failure',
        5: 'Rapid Error Accumulation (Early-Life Failure)'
    }
}

with open("rf_model.json", "w") as f:
    json.dump(model_js, f)

print(f"\nModel exported → rf_model.json  ({len(json.dumps(model_js))//1024} KB)")
print("Done! Open nvme_failure_predictor.html in any browser to run predictions.")
