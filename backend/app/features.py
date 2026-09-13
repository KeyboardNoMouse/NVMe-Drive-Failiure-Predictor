"""
Feature engineering for the NVMe drive failure model.

This module is imported by both train.py (training) and the FastAPI
backend (inference), so the exact same transformation is applied in
both places -- no train/serve skew.
"""
import pandas as pd

RAW_NUMERIC_COLS = [
    "Power_On_Hours", "Total_TBW_TB", "Total_TBR_TB", "Temperature_C",
    "Percent_Life_Used", "Media_Errors", "Unsafe_Shutdowns", "CRC_Errors",
    "Read_Error_Rate", "Write_Error_Rate",
]
CATEGORICAL_COLS = ["Vendor", "Model", "Firmware_Version"]

ENGINEERED_COLS = [
    "TBW_per_hour", "TBR_per_hour", "read_write_ratio",
    "wear_rate_per_1000h", "errors_per_1000h", "unsafe_shutdown_rate",
    "total_error_rate", "media_error_ratio",
]

# Columns that are excluded from the model on purpose:
#   Drive_ID            -> identifier, no predictive signal, would let the
#                           model memorize row identity instead of learning.
#   SMART_Warning_Flag   -> DATA LEAKAGE. In the source dataset this flag is
#                           set if and only if Failure_Flag == 1 (a perfect
#                           1:1 correspondence, verified on all 10,000 rows).
#                           It is effectively the label in disguise, which is
#                           why the original model reported 99.95% accuracy.
#                           A real SMART warning flag would be observed at
#                           the same time as the failure, not usable to
#                           predict it in advance -- so it must not be a
#                           model input.
#   Failure_Mode         -> only defined when Failure_Flag == 1 (0 for every
#                           healthy row), so it also leaks the label and is
#                           dropped from this binary predictor.
DROPPED_COLS = ["Drive_ID", "SMART_Warning_Flag", "Failure_Mode"]

FEATURE_COLS = RAW_NUMERIC_COLS + ENGINEERED_COLS + CATEGORICAL_COLS
TARGET_COL = "Failure_Flag"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic data cleaning: dedupe, enforce dtypes, clip physically
    impossible values. Returns a new dataframe."""
    df = df.copy()

    # Drop exact duplicate rows (ignoring the identifier column)
    id_col = "Drive_ID" if "Drive_ID" in df.columns else None
    subset = [c for c in df.columns if c != id_col]
    df = df.drop_duplicates(subset=subset).reset_index(drop=True)

    # Numeric columns that must be non-negative
    non_negative_cols = [
        "Power_On_Hours", "Total_TBW_TB", "Total_TBR_TB", "Media_Errors",
        "Unsafe_Shutdowns", "CRC_Errors", "Read_Error_Rate", "Write_Error_Rate",
    ]
    for col in non_negative_cols:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    # Percent_Life_Used is a percentage; a handful of rows exceed 100
    # (drives run past rated endurance). That's plausible telemetry, not
    # corrupt data, so we keep it but cap absurd outliers (>200%) which
    # would more likely be a sensor/logging fault.
    if "Percent_Life_Used" in df.columns:
        df["Percent_Life_Used"] = df["Percent_Life_Used"].clip(lower=0, upper=200)

    # Drop rows missing the target or any raw feature -- with ~10k rows and
    # no missingness in the source file this is a no-op safety net.
    required = [c for c in RAW_NUMERIC_COLS + CATEGORICAL_COLS + [TARGET_COL] if c in df.columns]
    df = df.dropna(subset=required).reset_index(drop=True)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features that describe usage intensity and error rates
    rather than raw cumulative counters. Cumulative counters (e.g. total
    errors) mostly just encode drive age; rates normalize for that and
    give the model a genuine wear/anomaly signal instead."""
    df = df.copy()

    hours = df["Power_On_Hours"].replace(0, 1)
    hours_k = (df["Power_On_Hours"] / 1000.0) + 1  # +1 avoids div-by-zero for new drives

    df["TBW_per_hour"] = df["Total_TBW_TB"] / hours
    df["TBR_per_hour"] = df["Total_TBR_TB"] / hours
    df["read_write_ratio"] = df["Total_TBR_TB"] / (df["Total_TBW_TB"] + 1)
    df["wear_rate_per_1000h"] = df["Percent_Life_Used"] / hours_k
    df["errors_per_1000h"] = (df["Media_Errors"] + df["CRC_Errors"]) / hours_k
    df["unsafe_shutdown_rate"] = df["Unsafe_Shutdowns"] / hours_k
    df["total_error_rate"] = df["Read_Error_Rate"] + df["Write_Error_Rate"]
    df["media_error_ratio"] = df["Media_Errors"] / (df["Media_Errors"] + df["CRC_Errors"] + 1)

    return df


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: clean -> engineer -> select model input columns.
    Does NOT touch the target column."""
    df = clean(df)
    df = engineer_features(df)
    return df
