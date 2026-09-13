from typing import Literal

from pydantic import BaseModel, Field

Vendor = Literal["VendorA", "VendorB", "VendorC", "VendorD"]
Model_ = Literal["Model-LITE", "Model-PRO", "Model-ULTRA", "Model-X1", "Model-X2"]
Firmware = Literal["FW1.0", "FW1.1", "FW1.2", "FW2.0", "FW2.1"]


class DriveTelemetry(BaseModel):
    """Raw SMART / usage telemetry for a single drive, as you'd read it
    directly off the device. No engineered or leaky fields here -- the
    backend derives features and never accepts SMART_Warning_Flag or
    Failure_Mode as input."""

    vendor: Vendor = Field(..., description="Drive vendor")
    model: Model_ = Field(..., description="Drive model line")
    firmware_version: Firmware = Field(..., description="Firmware version")
    power_on_hours: float = Field(..., ge=0, description="Total powered-on hours")
    total_tbw_tb: float = Field(..., ge=0, description="Total terabytes written")
    total_tbr_tb: float = Field(..., ge=0, description="Total terabytes read")
    temperature_c: float = Field(..., ge=-40, le=125, description="Current drive temperature, Celsius")
    percent_life_used: float = Field(..., ge=0, le=200, description="Rated endurance used, percent")
    media_errors: float = Field(..., ge=0, description="Cumulative media error count")
    unsafe_shutdowns: float = Field(..., ge=0, description="Cumulative unsafe shutdown count")
    crc_errors: float = Field(..., ge=0, description="Cumulative interface CRC error count")
    read_error_rate: float = Field(..., ge=0, description="Read error rate")
    write_error_rate: float = Field(..., ge=0, description="Write error rate")

    model_config = {
        "json_schema_extra": {
            "example": {
                "vendor": "VendorB",
                "model": "Model-ULTRA",
                "firmware_version": "FW2.0",
                "power_on_hours": 24744,
                "total_tbw_tb": 228.6,
                "total_tbr_tb": 221.5,
                "temperature_c": 45.5,
                "percent_life_used": 47.5,
                "media_errors": 0,
                "unsafe_shutdowns": 1,
                "crc_errors": 0,
                "read_error_rate": 5.27,
                "write_error_rate": 2.21,
            }
        }
    }


class PredictionResponse(BaseModel):
    failure_probability: float = Field(..., description="Predicted probability the drive will fail (0-1)")
    risk_level: Literal["low", "moderate", "high"]
    predicted_label: Literal["healthy", "failure"]
    decision_threshold: float
    top_contributing_factors: list[dict]
    failure_attribution: list[dict]


class FleetOverviewResponse(BaseModel):
    total_drives: int
    healthy_drives: int
    failed_drives: int
    failure_rate: float
    average_temperature: float
    average_life_used: float
    average_power_on_hours: float
    failure_modes: list[dict]
    vendor_stats: list[dict]
    firmware_stats: list[dict]
    life_bands: list[dict]
    telemetry_points: list[dict]


class ModelInfoResponse(BaseModel):
    trained_at: str
    dataset: dict
    dropped_leaky_columns: list[str]
    holdout_test_metrics: dict
    cv_metrics_train: dict
    cv_vs_holdout_gap: dict
    top_feature_importances: dict
