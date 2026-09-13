export type Vendor = "VendorA" | "VendorB" | "VendorC" | "VendorD";
export type DriveModel = "Model-LITE" | "Model-PRO" | "Model-ULTRA" | "Model-X1" | "Model-X2";
export type Firmware = "FW1.0" | "FW1.1" | "FW1.2" | "FW2.0" | "FW2.1";

export interface DriveTelemetry {
  vendor: Vendor;
  model: DriveModel;
  firmware_version: Firmware;
  power_on_hours: number;
  total_tbw_tb: number;
  total_tbr_tb: number;
  temperature_c: number;
  percent_life_used: number;
  media_errors: number;
  unsafe_shutdowns: number;
  crc_errors: number;
  read_error_rate: number;
  write_error_rate: number;
}

export interface ContributingFactor {
  feature: string;
  value: number;
  healthy_typical: number;
  push_score: number;
}

export interface PredictionResponse {
  failure_probability: number;
  risk_level: "low" | "moderate" | "high";
  predicted_label: "healthy" | "failure";
  decision_threshold: number;
  top_contributing_factors: ContributingFactor[];
  failure_attribution: {
    label: string;
    percentage: number;
    kind: "failure" | "healthy";
  }[];
}

export interface FleetOverviewResponse {
  total_drives: number;
  healthy_drives: number;
  failed_drives: number;
  failure_rate: number;
  average_temperature: number;
  average_life_used: number;
  average_power_on_hours: number;
  failure_modes: { code: number; label: string; count: number; percentage: number }[];
  vendor_stats: { label: string; drives: number; failures: number; failure_rate: number }[];
  firmware_stats: { label: string; drives: number; failures: number; failure_rate: number }[];
  life_bands: { label: string; healthy: number; failed: number }[];
  telemetry_points: { temperature: number; life_used: number; failed: number }[];
}

export interface ModelInfoResponse {
  trained_at: string;
  dataset: {
    source: string;
    n_rows: number;
    n_failures: number;
    failure_rate: number;
  };
  dropped_leaky_columns: string[];
  holdout_test_metrics: Record<string, number>;
  cv_metrics_train: Record<string, { mean: number; std: number; folds: number[] }>;
  cv_vs_holdout_gap: Record<string, number>;
  top_feature_importances: Record<string, number>;
}
