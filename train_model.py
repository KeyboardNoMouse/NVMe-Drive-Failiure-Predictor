"""
NVMe Drive Failure Predictor — Model Training Script
=====================================================
Trains a Random Forest classifier on the augmented NVMe dataset that
includes synthetic Mode 2 (Thermal) and Mode 3 (Power-Related) samples.

If the augmented CSV is not found, generate_synthetic.py is run first.

Requirements:
    pip install scikit-learn pandas numpy

Usage:
    python train_model.py
"""

import os
import subprocess
import sys
import json

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ── Dataset path resolution ────────────────────────────────────────────────
AUGMENTED_CSV = "NVMe_Drive_Failure_Dataset_Augmented.csv"

if not os.path.exists(AUGMENTED_CSV):
    print(f"[INFO] {AUGMENTED_CSV} not found — running generate_synthetic.py first...")
    subprocess.run([sys.executable, "generate_synthetic.py"], check=True)
    print()

df = pd.read_csv(AUGMENTED_CSV)
print(f"Dataset : {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"\nFailure Mode Distribution:")
vc = df['Failure_Mode'].value_counts().sort_index()
labels = {0:'Healthy', 1:'Wear-Out', 2:'Thermal', 3:'Power-Related',
          4:'Controller/FW', 5:'Early-Life'}
for mode, count in vc.items():
    bar = 'X' * min(count // 50, 60)
    print(f"  Mode {mode} ({labels.get(mode,'?'):18s}): {count:5d}  {bar}")

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
y_mode = df['Failure_Mode']

# ── Train / Evaluate ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("  TRAINING — Hold-out evaluation (80/20 stratified split)")
print("="*60)

X_tr, X_te, ym_tr, ym_te = train_test_split(
    X, y_mode, test_size=0.2, random_state=42, stratify=y_mode)

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_tr, ym_tr)
ym_pred = rf.predict(X_te)

print(f"\nHold-out Accuracy : {accuracy_score(ym_te, ym_pred):.4f}")
print("\nClassification Report:")
mode_names = [f"Mode {m} ({labels[m]})" for m in sorted(labels)]
print(classification_report(ym_te, ym_pred, target_names=mode_names))

print("\nConfusion Matrix (rows=actual, cols=predicted):")
cm = confusion_matrix(ym_te, ym_pred, labels=sorted(labels))
mode_keys = sorted(labels)
print("       " + "  ".join(f"M{m}" for m in mode_keys))
for i, row_mode in enumerate(mode_keys):
    print(f"  M{row_mode}   " + "   ".join(f"{cm[i][j]:2d}" for j in range(len(mode_keys))))

# ── Cross-validation ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("  5-FOLD CROSS-VALIDATION")
print("="*60)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X, y_mode, cv=cv, scoring='accuracy', n_jobs=-1)
print(f"\n  Accuracy per fold : {[f'{s:.4f}' for s in cv_scores]}")
print(f"  Mean +/- Std      : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

# ── Feature Importance ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("  TOP FEATURE IMPORTANCES")
print("="*60)
fi = sorted(zip(FEATURE_COLS, rf.feature_importances_), key=lambda x: -x[1])
for feat, imp in fi[:8]:
    print(f"  {feat:25s} {imp:.4f}")

# ── Retrain on Full Dataset & Export ──────────────────────────────────────
print("\n" + "="*60)
print("  RETRAINING ON FULL DATASET & EXPORTING")
print("="*60)
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
            'leaf':      False,
            'feature':   int(t.feature[node]),
            'threshold': float(t.threshold[node]),
            'left':      recurse(t.children_left[node]),
            'right':     recurse(t.children_right[node])
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
        str(k): v for k, v in {
            0: 'Healthy',
            1: 'Wear-Out Failure',
            2: 'Thermal Failure',
            3: 'Power-Related Failure',
            4: 'Controller/Firmware Failure',
            5: 'Rapid Error Accumulation (Early-Life Failure)'
        }.items()
    },
    'cv_accuracy': float(cv_scores.mean()),
    'cv_std':      float(cv_scores.std()),
    'n_samples':   int(len(df)),
    'n_modes':     6
}

with open("rf_model.json", "w") as f:
    json.dump(model_js, f)

size_kb = len(json.dumps(model_js)) // 1024
print(f"\nModel exported -> rf_model.json  ({size_kb} KB)")
print(f"  Classes in model : {model_js['classes']}")
print(f"  CV accuracy      : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
print("\nDone! Open nvme_failure_predictor.html in any browser to run predictions.")
