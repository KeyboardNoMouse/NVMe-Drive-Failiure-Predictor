"""
NVMe Drive Failure Predictor -- training script (rebuilt from scratch)
========================================================================

Trains a Random Forest classifier on the ORIGINAL (non-synthetic) NVMe
dataset to predict Failure_Flag (binary: will this drive fail).

Why this is not the old 99.95%-accuracy model
-----------------------------------------------
The original project trained on an augmented dataset and included
`SMART_Warning_Flag` as an input feature. That flag is set if and only if
Failure_Flag == 1 for every single row in the source data -- it IS the
label, just renamed. Any model that sees it will trivially get ~100%
accuracy without learning anything. This script drops that column (and
`Failure_Mode`, `Drive_ID`, both leaky/uninformative for the binary task)
and only uses the 10,000-row original dataset, not the synthetic one.

Because only ~1.9% of drives in the dataset actually failed, plain
accuracy is a misleading metric on its own (predicting "healthy" for
every drive already scores ~98%). This script reports precision, recall,
F1, ROC-AUC and PR-AUC (average precision) for the failure class, uses
stratified 5-fold cross-validation, and tunes the model with a
recall/precision-aware scorer instead of accuracy.

Usage:
    pip install -r requirements.txt
    python train.py
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from features import (
    CATEGORICAL_COLS, FEATURE_COLS, RAW_NUMERIC_COLS, ENGINEERED_COLS,
    TARGET_COL, build_feature_frame,
)

DATA_PATH = Path(__file__).parent / "data" / "NVMe_Drive_Failure_Dataset.csv"
ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def build_pipeline(rf_params: dict) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ],
        remainder="passthrough",  # numeric + engineered columns pass through untouched
    )
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **rf_params)
    return Pipeline([("preprocess", preprocessor), ("model", rf)])


def evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("NVMe Drive Failure Predictor -- Training (leakage-free rebuild)")
    print("=" * 70)

    raw = pd.read_csv(DATA_PATH)
    print(f"\nLoaded original dataset: {raw.shape[0]:,} rows x {raw.shape[1]} columns")
    print("(Synthetic/augmented dataset is intentionally NOT used for training.)")

    df = build_feature_frame(raw)
    print(f"After cleaning + feature engineering: {df.shape[0]:,} rows")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    fail_rate = y.mean()
    print(f"\nClass balance: {int(y.sum())} failures / {len(y)} drives "
          f"({fail_rate:.2%} positive class)")
    print("This is a rare-event problem -- accuracy alone is not a "
          "meaningful metric here (always predicting 'healthy' already "
          f"scores {1 - fail_rate:.2%}).")

    # ---- Hold-out split, stratified so the rare positive class is
    # ---- represented proportionally in both halves ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nTrain/test split: {len(X_train)} train / {len(X_test)} test "
          f"({y_train.sum()} / {y_test.sum()} failures respectively)")

    # ---- Hyperparameter search, scored on average precision (PR-AUC) ----
    # PR-AUC is the right metric to optimize for a rare positive class --
    # unlike accuracy or ROC-AUC it isn't dominated by the majority class.
    # NOTE: max_depth is intentionally bounded (no "None"/unlimited option).
    # Unlimited trees can grow a leaf for nearly every minority-class row,
    # which drives train-set metrics to a trivial 1.0 without improving
    # true generalization (verified: this raised the train/CV gap without
    # improving held-out recall or PR-AUC). Capping depth trades a little
    # training-set fit for a model that relies on general patterns.
    param_dist = {
        "model__n_estimators": [200, 300, 400],
        "model__max_depth": [5, 6, 8, 10, 12],
        "model__min_samples_leaf": [2, 4, 8, 12],
        "model__min_samples_split": [4, 8, 10, 16],
        "model__max_features": ["sqrt", "log2"],
        "model__class_weight": ["balanced", "balanced_subsample"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    base_pipe = build_pipeline({})
    search = RandomizedSearchCV(
        base_pipe, param_distributions=param_dist, n_iter=18,
        scoring="average_precision", cv=cv, random_state=RANDOM_STATE,
        n_jobs=-1, refit=True,
    )
    print("\nRunning randomized hyperparameter search "
          "(5-fold CV x 18 candidates, scored on PR-AUC)...")
    search.fit(X_train, y_train)
    best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    print(f"Best params: {best_params}")
    print(f"Best CV PR-AUC: {search.best_score_:.4f}")

    best_pipe = search.best_estimator_

    # ---- 5-fold CV on the training set with the tuned params, reporting
    # ---- the full metric suite (not just the scorer used for tuning) ----
    cv_results = cross_validate(
        best_pipe, X_train, y_train, cv=cv,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"],
        n_jobs=-1,
    )
    RENAME = {"average_precision": "pr_auc"}  # align sklearn scorer name with evaluate()'s key
    cv_summary = {
        RENAME.get(metric.replace("test_", ""), metric.replace("test_", "")): {
            "mean": float(np.mean(cv_results[metric])),
            "std": float(np.std(cv_results[metric])),
            "folds": [round(float(v), 4) for v in cv_results[metric]],
        }
        for metric in cv_results if metric.startswith("test_")
    }
    print("\n" + "-" * 70)
    print("5-fold cross-validation on training data (tuned model):")
    for metric, stats in cv_summary.items():
        print(f"  {metric:12s}: {stats['mean']:.4f} +/- {stats['std']:.4f}  "
              f"(folds: {stats['folds']})")

    # ---- Held-out test evaluation (data the model never touched) ----
    best_pipe.fit(X_train, y_train)
    y_pred = best_pipe.predict(X_test)
    y_proba = best_pipe.predict_proba(X_test)[:, 1]
    holdout_metrics = evaluate(y_test, y_pred, y_proba)

    print("\n" + "-" * 70)
    print("Held-out test set (20%, never seen during training or tuning):")
    for k, v in holdout_metrics.items():
        print(f"  {k:12s}: {v:.4f}")
    print("\nClassification report (test set):")
    print(classification_report(y_test, y_pred, target_names=["Healthy", "Failure"], zero_division=0))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix (rows=actual, cols=predicted) [Healthy, Failure]:")
    print(cm)

    # ---- Sanity check: memorization probe ----
    # In-sample (train-set) metrics are NOT a fair memorization check for a
    # random forest -- with enough leaves it can fit training data almost
    # perfectly regardless of whether it generalizes. The meaningful
    # comparison is between two *out-of-sample* estimates that should agree
    # if the model is learning real patterns rather than getting lucky on
    # one split: the 5-fold CV mean (computed on the training set, never
    # touching the test fold within each split) vs. the held-out test set
    # (never touched during training or tuning at all).
    y_train_pred = best_pipe.predict(X_train)
    y_train_proba = best_pipe.predict_proba(X_train)[:, 1]
    train_metrics = evaluate(y_train, y_train_pred, y_train_proba)
    print("\n" + "-" * 70)
    print("In-sample train-set metrics (context only, not a generalization check):")
    for k, v in train_metrics.items():
        print(f"  {k:12s}: {v:.4f}")

    cv_vs_holdout_gap = {
        metric: round(abs(cv_summary[metric]["mean"] - holdout_metrics[metric]), 4)
        for metric in holdout_metrics
    }
    print("\nCross-validation mean vs. held-out test set (both out-of-sample; "
          "should agree if the model generalizes rather than got lucky):")
    for metric, gap_val in cv_vs_holdout_gap.items():
        print(f"  {metric:12s}: CV={cv_summary[metric]['mean']:.4f}  "
              f"holdout={holdout_metrics[metric]:.4f}  gap={gap_val:.4f}")

    # ---- Refit on ALL data for the final deployed model ----
    # Standard practice: once architecture/hyperparameters are chosen and
    # validated on the held-out split above, refit on the full dataset so
    # the shipped model has seen as much data as possible. All reported
    # metrics above come from data this final refit did not use for
    # hyperparameter selection.
    final_pipe = build_pipeline(best_params)
    final_pipe.fit(X, y)

    # Feature importances (mapped back from one-hot columns to readable names)
    ohe = final_pipe.named_steps["preprocess"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(CATEGORICAL_COLS))
    passthrough_names = RAW_NUMERIC_COLS + ENGINEERED_COLS
    all_names = cat_names + passthrough_names
    importances = final_pipe.named_steps["model"].feature_importances_
    fi_sorted = sorted(zip(all_names, importances), key=lambda x: -x[1])

    print("\n" + "-" * 70)
    print("Top 15 feature importances (final model, trained on all data):")
    for name, imp in fi_sorted[:15]:
        print(f"  {name:28s} {imp:.4f}")

    # ---- Reference stats for lightweight per-prediction explanations ----
    # For each numeric feature, record the healthy-population mean/std and
    # whether failed drives trend higher or lower than healthy ones. The
    # backend uses this (deviation from healthy mean, weighted by global
    # feature importance and direction) to surface plausible "why" factors
    # for a single prediction, without adding a SHAP/treeinterpreter
    # dependency.
    numeric_cols_for_stats = RAW_NUMERIC_COLS + ENGINEERED_COLS
    healthy = df[df[TARGET_COL] == 0]
    failed = df[df[TARGET_COL] == 1]
    reference_stats = {}
    for col in numeric_cols_for_stats:
        h_mean, h_std = float(healthy[col].mean()), float(healthy[col].std() or 1.0)
        f_mean = float(failed[col].mean())
        reference_stats[col] = {
            "healthy_mean": h_mean,
            "healthy_std": h_std if h_std > 1e-9 else 1.0,
            "failed_mean": f_mean,
            "direction": 1 if f_mean >= h_mean else -1,  # 1 = higher is worse, -1 = lower is worse
            "importance": float(dict(fi_sorted).get(col, 0.0)),
        }

    # ---- Persist artifacts ----
    import joblib
    joblib.dump(final_pipe, ARTIFACT_DIR / "model_pipeline.joblib")

    metrics_out = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {
            "source": "NVMe_Drive_Failure_Dataset.csv (original, non-synthetic)",
            "n_rows": int(len(df)),
            "n_failures": int(y.sum()),
            "failure_rate": float(fail_rate),
        },
        "dropped_leaky_columns": ["SMART_Warning_Flag", "Failure_Mode", "Drive_ID"],
        "feature_columns": FEATURE_COLS,
        "best_hyperparameters": best_params,
        "cv_metrics_train": cv_summary,
        "holdout_test_metrics": holdout_metrics,
        "in_sample_train_metrics": train_metrics,
        "cv_vs_holdout_gap": cv_vs_holdout_gap,
        "confusion_matrix_holdout": {
            "labels": ["Healthy", "Failure"],
            "matrix": cm.tolist(),
        },
        "feature_importances": {name: float(imp) for name, imp in fi_sorted},
        "feature_reference_stats": reference_stats,
    }
    with open(ARTIFACT_DIR / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    print(f"\nSaved model pipeline -> {ARTIFACT_DIR / 'model_pipeline.joblib'}")
    print(f"Saved metrics report -> {ARTIFACT_DIR / 'metrics.json'}")
    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
