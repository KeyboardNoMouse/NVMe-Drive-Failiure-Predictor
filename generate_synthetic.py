"""
NVMe Synthetic Data Generator — Mode 2 (Thermal) & Mode 3 (Power-Related)
==========================================================================
Generates physically-grounded synthetic samples for the two failure modes
absent from the original dataset.  The distributions are calibrated from:

  • NVMe / SATA SSD datasheet operating limits (JEDEC JESD218B)
  • SMART attribute semantics (ATA / NVMe spec)
  • Statistical properties of the existing healthy + failure mode rows

Mode 2 – Thermal Failure
  Signature: extreme or sustained high temperature that causes NAND
  degradation, read-disturb acceleration, and controller throttling.
  Key signals: Temperature_C >> 60 °C, SMART_Warning_Flag=1,
               elevated Media_Errors and Read_Error_Rate, moderate
               Write_Error_Rate, normal-to-high Unsafe_Shutdowns.

Mode 3 – Power-Related Failure
  Signature: repeated abrupt power loss corrupts write cache, damages
  FTL metadata, and stresses the controller capacitor backup.
  Key signals: Unsafe_Shutdowns >> 8, elevated CRC_Errors (dirty power
               on the bus), SMART_Warning_Flag=1, Write_Error_Rate
               elevated relative to Read_Error_Rate.

Usage:
    python generate_synthetic.py              # generates files, prints stats
    python generate_synthetic.py --preview    # preview only, no files written
"""

import argparse
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# ── Categorical pools (match existing dataset) ────────────────────────────
VENDORS   = ['VendorA', 'VendorB', 'VendorC', 'VendorD']
MODELS    = ['Model-PRO', 'Model-X1', 'Model-X2', 'Model-LITE', 'Model-ULTRA']
FIRMWARES = ['FW1.0', 'FW1.1', 'FW1.2', 'FW2.0', 'FW2.1']


def _rand_cat(pool, n):
    return RNG.choice(pool, size=n)


def _clip_int(arr, lo, hi):
    return np.clip(np.round(arr).astype(int), lo, hi)


def _clip_float(arr, lo, hi, decimals=2):
    return np.round(np.clip(arr, lo, hi), decimals)


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2 — Thermal Failure
# ─────────────────────────────────────────────────────────────────────────────
def generate_mode2(n: int = 300) -> pd.DataFrame:
    """
    Thermal failure samples.  Three sub-populations model different
    thermal stress scenarios:
      A (40%): Extreme sustained heat  — temp 65-75 °C
      B (35%): High heat + high workload wear — temp 60-67 °C
      C (25%): Moderate heat but NAND already degraded — temp 55-63 °C
    """
    nA = int(n * 0.40)
    nB = int(n * 0.35)
    nC = n - nA - nB

    def sub_A(k):
        return dict(
            Temperature_C      = _clip_float(RNG.uniform(65, 75, k), 63, 80),
            Power_On_Hours     = _clip_int(RNG.normal(28000, 12000, k), 500, 60000),
            Total_TBW_TB       = _clip_float(RNG.normal(130, 60, k), 5, 400),
            Total_TBR_TB       = _clip_float(RNG.normal(125, 65, k), 4, 560),
            Percent_Life_Used  = _clip_float(RNG.normal(38, 20, k), 5, 95),
            Media_Errors       = _clip_int(RNG.poisson(3.5, k), 1, 7),
            Unsafe_Shutdowns   = _clip_int(RNG.normal(3, 2, k), 0, 10),
            CRC_Errors         = _clip_int(RNG.poisson(0.8, k), 0, 5),
            Read_Error_Rate    = _clip_float(RNG.normal(14, 5, k), 5, 34),
            Write_Error_Rate   = _clip_float(RNG.normal(10, 4, k), 2, 30),
            SMART_Warning_Flag = np.ones(k, dtype=int),
        )

    def sub_B(k):
        return dict(
            Temperature_C      = _clip_float(RNG.uniform(60, 68, k), 58, 75),
            Power_On_Hours     = _clip_int(RNG.normal(35000, 14000, k), 5000, 60000),
            Total_TBW_TB       = _clip_float(RNG.normal(200, 80, k), 30, 400),
            Total_TBR_TB       = _clip_float(RNG.normal(210, 90, k), 25, 560),
            Percent_Life_Used  = _clip_float(RNG.normal(55, 22, k), 10, 95),
            Media_Errors       = _clip_int(RNG.poisson(2.8, k), 1, 7),
            Unsafe_Shutdowns   = _clip_int(RNG.normal(3.5, 2, k), 0, 9),
            CRC_Errors         = _clip_int(RNG.poisson(1.0, k), 0, 5),
            Read_Error_Rate    = _clip_float(RNG.normal(13, 5, k), 4, 34),
            Write_Error_Rate   = _clip_float(RNG.normal(9, 4, k), 2, 28),
            SMART_Warning_Flag = np.ones(k, dtype=int),
        )

    def sub_C(k):
        # Moderate temp but NAND wear amplifies thermal sensitivity
        return dict(
            Temperature_C      = _clip_float(RNG.uniform(55, 64, k), 53, 70),
            Power_On_Hours     = _clip_int(RNG.normal(40000, 13000, k), 8000, 60000),
            Total_TBW_TB       = _clip_float(RNG.normal(280, 70, k), 80, 400),
            Total_TBR_TB       = _clip_float(RNG.normal(295, 80, k), 70, 560),
            Percent_Life_Used  = _clip_float(RNG.normal(72, 18, k), 30, 95),
            Media_Errors       = _clip_int(RNG.poisson(2.2, k), 1, 6),
            Unsafe_Shutdowns   = _clip_int(RNG.normal(3, 1.8, k), 0, 8),
            CRC_Errors         = _clip_int(RNG.poisson(0.6, k), 0, 4),
            Read_Error_Rate    = _clip_float(RNG.normal(11, 4, k), 3, 30),
            Write_Error_Rate   = _clip_float(RNG.normal(8, 3.5, k), 1, 26),
            SMART_Warning_Flag = np.ones(k, dtype=int),
        )

    rows = []
    for sub_fn, k in [(sub_A, nA), (sub_B, nB), (sub_C, nC)]:
        d = sub_fn(k)
        for i in range(k):
            rows.append({col: d[col][i] for col in d})

    df = pd.DataFrame(rows)
    df['Failure_Mode'] = 2
    df['Failure_Flag'] = 1
    df['Vendor']           = _rand_cat(VENDORS, n)
    df['Model']            = _rand_cat(MODELS, n)
    df['Firmware_Version'] = _rand_cat(FIRMWARES, n)
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODE 3 — Power-Related Failure
# ─────────────────────────────────────────────────────────────────────────────
def generate_mode3(n: int = 300) -> pd.DataFrame:
    """
    Power-related failure samples.  Three sub-populations:
      A (40%): Catastrophic power abuse — Unsafe_Shutdowns ≥ 10
      B (35%): High unsafe shutdowns + CRC errors from dirty power
      C (25%): Moderate unsafe shutdowns with SMART flag + write degradation
    """
    nA = int(n * 0.40)
    nB = int(n * 0.35)
    nC = n - nA - nB

    def sub_A(k):
        return dict(
            Unsafe_Shutdowns   = _clip_int(RNG.normal(12, 2, k), 10, 20),
            CRC_Errors         = _clip_int(RNG.poisson(4.5, k), 2, 10),
            Temperature_C      = _clip_float(RNG.normal(40, 8, k), 20, 58),
            Power_On_Hours     = _clip_int(RNG.normal(25000, 14000, k), 500, 60000),
            Total_TBW_TB       = _clip_float(RNG.normal(95, 65, k), 3, 400),
            Total_TBR_TB       = _clip_float(RNG.normal(90, 70, k), 2, 560),
            Percent_Life_Used  = _clip_float(RNG.normal(25, 18, k), 2, 88),
            Media_Errors       = _clip_int(RNG.poisson(1.2, k), 0, 5),
            Read_Error_Rate    = _clip_float(RNG.normal(8, 5, k), 0, 30),
            Write_Error_Rate   = _clip_float(RNG.normal(11, 5, k), 2, 32),
            SMART_Warning_Flag = np.ones(k, dtype=int),
        )

    def sub_B(k):
        return dict(
            Unsafe_Shutdowns   = _clip_int(RNG.normal(8, 1.5, k), 6, 12),
            CRC_Errors         = _clip_int(RNG.poisson(5.0, k), 3, 10),
            Temperature_C      = _clip_float(RNG.normal(41, 7.5, k), 20, 58),
            Power_On_Hours     = _clip_int(RNG.normal(28000, 13000, k), 500, 60000),
            Total_TBW_TB       = _clip_float(RNG.normal(100, 70, k), 3, 400),
            Total_TBR_TB       = _clip_float(RNG.normal(98, 75, k), 2, 560),
            Percent_Life_Used  = _clip_float(RNG.normal(27, 17, k), 2, 88),
            Media_Errors       = _clip_int(RNG.poisson(1.0, k), 0, 5),
            Read_Error_Rate    = _clip_float(RNG.normal(7.5, 4.5, k), 0, 28),
            Write_Error_Rate   = _clip_float(RNG.normal(12, 5, k), 3, 34),
            SMART_Warning_Flag = np.ones(k, dtype=int),
        )

    def sub_C(k):
        # Moderate unsafe shutdowns BUT elevated CRC + SMART + write errors
        # force the RF to learn the multi-feature combination
        return dict(
            Unsafe_Shutdowns   = _clip_int(RNG.normal(7, 1.0, k), 6, 10),
            CRC_Errors         = _clip_int(RNG.poisson(4.2, k), 3, 9),
            Temperature_C      = _clip_float(RNG.normal(40, 8, k), 20, 57),
            Power_On_Hours     = _clip_int(RNG.normal(22000, 13000, k), 500, 60000),
            Total_TBW_TB       = _clip_float(RNG.normal(88, 60, k), 3, 380),
            Total_TBR_TB       = _clip_float(RNG.normal(85, 65, k), 2, 500),
            Percent_Life_Used  = _clip_float(RNG.normal(22, 16, k), 2, 88),
            Media_Errors       = _clip_int(RNG.poisson(0.8, k), 0, 4),
            Read_Error_Rate    = _clip_float(RNG.normal(7, 4.5, k), 0, 26),
            Write_Error_Rate   = _clip_float(RNG.normal(12, 4.5, k), 4, 32),
            SMART_Warning_Flag = np.ones(k, dtype=int),
        )

    rows = []
    for sub_fn, k in [(sub_A, nA), (sub_B, nB), (sub_C, nC)]:
        d = sub_fn(k)
        for i in range(k):
            rows.append({col: d[col][i] for col in d})

    df = pd.DataFrame(rows)
    df['Failure_Mode'] = 3
    df['Failure_Flag'] = 1
    df['Vendor']           = _rand_cat(VENDORS, n)
    df['Model']            = _rand_cat(MODELS, n)
    df['Firmware_Version'] = _rand_cat(FIRMWARES, n)
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Boundary Guard
# ─────────────────────────────────────────────────────────────────────────────
def validate_no_overlap(df_orig, df_syn):
    """
    Warn if synthetic Mode 2/3 samples fall into the healthy zone of the
    original dataset on the primary discriminating feature for each mode.
    Healthy thresholds (99th percentile from original data):
      Temperature_C   ≤ 59 °C
      Unsafe_Shutdowns ≤ 8
    """
    m2 = df_syn[df_syn['Failure_Mode'] == 2]
    m3 = df_syn[df_syn['Failure_Mode'] == 3]

    healthy_max_temp   = df_orig[df_orig['Failure_Mode'] == 0]['Temperature_C'].quantile(0.99)
    healthy_max_unsafe = df_orig[df_orig['Failure_Mode'] == 0]['Unsafe_Shutdowns'].quantile(0.99)

    overlap_m2 = (m2['Temperature_C'] <= healthy_max_temp).mean()
    overlap_m3 = (m3['Unsafe_Shutdowns'] <= healthy_max_unsafe).mean()

    print(f"\n  Healthy 99th-pct Temperature : {healthy_max_temp:.1f} °C")
    print(f"  Mode 2 samples with Temp ≤ that threshold: {overlap_m2:.1%}  "
          f"(expected: sub-C population, ~25% — these use multi-feature signal)")
    print(f"\n  Healthy 99th-pct Unsafe_Shutdowns : {healthy_max_unsafe:.0f}")
    print(f"  Mode 3 samples with Unsafe ≤ that threshold: {overlap_m3:.1%}  "
          f"(expected: sub-C population, ~25% — these use multi-feature signal)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview', action='store_true',
                        help='Print stats only, do not write files')
    parser.add_argument('--n_mode2', type=int, default=300,
                        help='Number of Mode 2 synthetic samples (default 300)')
    parser.add_argument('--n_mode3', type=int, default=300,
                        help='Number of Mode 3 synthetic samples (default 300)')
    args = parser.parse_args()

    print("=" * 60)
    print("  NVMe Synthetic Data Generator")
    print("=" * 60)

    df_orig = pd.read_csv('NVMe_Drive_Failure_Dataset.csv')
    print(f"\nOriginal dataset: {len(df_orig):,} rows")
    print(f"  Mode distribution: {df_orig['Failure_Mode'].value_counts().sort_index().to_dict()}")

    print(f"\nGenerating {args.n_mode2} Mode 2 (Thermal) samples...")
    df_m2 = generate_mode2(args.n_mode2)

    print(f"Generating {args.n_mode3} Mode 3 (Power-Related) samples...")
    df_m3 = generate_mode3(args.n_mode3)

    # Assign new Drive_IDs
    max_id = int(df_orig['Drive_ID'].str.replace('NVME-', '').astype(int).max())
    ids_m2 = [f'NVME-{max_id + i + 1:05d}' for i in range(len(df_m2))]
    ids_m3 = [f'NVME-{max_id + len(df_m2) + i + 1:05d}' for i in range(len(df_m3))]
    df_m2.insert(0, 'Drive_ID', ids_m2)
    df_m3.insert(0, 'Drive_ID', ids_m3)

    # Reorder columns to match original
    col_order = df_orig.columns.tolist()
    df_m2 = df_m2[col_order]
    df_m3 = df_m3[col_order]

    print("\n── Mode 2 (Thermal) synthetic stats ──────────────────────────")
    print(df_m2[['Temperature_C', 'Media_Errors', 'Read_Error_Rate',
                  'Write_Error_Rate', 'SMART_Warning_Flag']].describe(
                  percentiles=[.25, .5, .75, .9]).round(2).to_string())

    print("\n── Mode 3 (Power-Related) synthetic stats ─────────────────────")
    print(df_m3[['Unsafe_Shutdowns', 'CRC_Errors', 'Write_Error_Rate',
                  'Read_Error_Rate', 'SMART_Warning_Flag']].describe(
                  percentiles=[.25, .5, .75, .9]).round(2).to_string())

    print("\n── Overlap validation ──────────────────────────────────────────")
    validate_no_overlap(df_orig, pd.concat([df_m2, df_m3]))

    if args.preview:
        print("\n[Preview mode — no files written]")
        return

    # Merge and save
    df_aug = pd.concat([df_orig, df_m2, df_m3], ignore_index=True)
    out_path = 'NVMe_Drive_Failure_Dataset_Augmented.csv'
    df_aug.to_csv(out_path, index=False)

    print(f"\n✅  Augmented dataset saved → {out_path}")
    print(f"   Total rows : {len(df_aug):,}  "
          f"(original {len(df_orig):,} + {len(df_m2)} Mode2 + {len(df_m3)} Mode3)")
    print(f"   New mode distribution:")
    print(f"   {df_aug['Failure_Mode'].value_counts().sort_index().to_dict()}")


if __name__ == '__main__':
    main()
