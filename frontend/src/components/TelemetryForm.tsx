"use client";

import { DriveTelemetry, DriveModel, Firmware, Vendor } from "@/lib/types";
import { NumberField, Panel, SelectField } from "./fields";

const VENDORS: Vendor[] = ["VendorA", "VendorB", "VendorC", "VendorD"];
const MODELS: DriveModel[] = ["Model-LITE", "Model-PRO", "Model-ULTRA", "Model-X1", "Model-X2"];
const FIRMWARES: Firmware[] = ["FW1.0", "FW1.1", "FW1.2", "FW2.0", "FW2.1"];

export const EXAMPLE_TELEMETRY: DriveTelemetry = {
  vendor: "VendorB",
  model: "Model-ULTRA",
  firmware_version: "FW2.0",
  power_on_hours: 24744,
  total_tbw_tb: 228.6,
  total_tbr_tb: 221.5,
  temperature_c: 45.5,
  percent_life_used: 47.5,
  media_errors: 0,
  unsafe_shutdowns: 1,
  crc_errors: 0,
  read_error_rate: 5.27,
  write_error_rate: 2.21,
};

const EXAMPLE_DRIVES: { name: string; tag: string; description: string; tone: string; telemetry: DriveTelemetry }[] = [
  {
    name: "Quiet baseline",
    tag: "HEALTHY",
    description: "Low-stress drive with clean signals",
    tone: "teal",
    telemetry: {
      vendor: "VendorB", model: "Model-LITE", firmware_version: "FW2.0", power_on_hours: 22000,
      total_tbw_tb: 110, total_tbr_tb: 105, temperature_c: 44, percent_life_used: 28,
      media_errors: 0, unsafe_shutdowns: 0, crc_errors: 0, read_error_rate: 3, write_error_rate: 3,
    },
  },
  {
    name: "Controller surge",
    tag: "HIGH RISK",
    description: "Media errors and thermal wear stacking up",
    tone: "coral",
    telemetry: {
      vendor: "VendorC", model: "Model-LITE", firmware_version: "FW1.0", power_on_hours: 16000,
      total_tbw_tb: 320, total_tbr_tb: 300, temperature_c: 58, percent_life_used: 86,
      media_errors: 9, unsafe_shutdowns: 6, crc_errors: 3, read_error_rate: 25, write_error_rate: 24,
    },
  },
  {
    name: "Power event storm",
    tag: "HIGH RISK",
    description: "Shutdowns and interface errors dominate",
    tone: "gold",
    telemetry: {
      vendor: "VendorD", model: "Model-PRO", firmware_version: "FW2.0", power_on_hours: 36000,
      total_tbw_tb: 220, total_tbr_tb: 230, temperature_c: 55, percent_life_used: 68,
      media_errors: 4, unsafe_shutdowns: 120, crc_errors: 80, read_error_rate: 18, write_error_rate: 15,
    },
  },
  {
    name: "Firmware watch",
    tag: "WATCH",
    description: "FW1.0 cohort with emerging wear signals",
    tone: "sky",
    telemetry: {
      vendor: "VendorC", model: "Model-LITE", firmware_version: "FW1.0", power_on_hours: 9000,
      total_tbw_tb: 130, total_tbr_tb: 120, temperature_c: 48, percent_life_used: 38,
      media_errors: 1, unsafe_shutdowns: 3, crc_errors: 1, read_error_rate: 13, write_error_rate: 12,
    },
  },
];

export function TelemetryForm({
  value,
  onChange,
  onSubmit,
  loading,
}: {
  value: DriveTelemetry;
  onChange: (v: DriveTelemetry) => void;
  onSubmit: () => void;
  loading: boolean;
}) {
  const set = <K extends keyof DriveTelemetry>(key: K, v: DriveTelemetry[K]) =>
    onChange({ ...value, [key]: v });

  return (
    <form
      className="flex flex-col gap-4"
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit();
      }}
    >
      <Panel title="Drive identity" subtitle="Vendor, model line, and firmware revision">
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <SelectField label="Vendor" value={value.vendor} options={VENDORS} onChange={(v) => set("vendor", v)} />
          <SelectField label="Model" value={value.model} options={MODELS} onChange={(v) => set("model", v)} />
          <SelectField
            label="Firmware"
            value={value.firmware_version}
            options={FIRMWARES}
            onChange={(v) => set("firmware_version", v)}
          />
        </div>
      </Panel>

      <Panel title="Usage" subtitle="Cumulative operating counters read from SMART">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <NumberField
            label="Power-on hours"
            value={value.power_on_hours}
            min={0}
            onChange={(v) => set("power_on_hours", v)}
          />
          <NumberField
            label="Percent life used"
            unit="%"
            value={value.percent_life_used}
            min={0}
            max={200}
            onChange={(v) => set("percent_life_used", v)}
          />
          <NumberField
            label="Total written"
            unit="TB"
            value={value.total_tbw_tb}
            min={0}
            onChange={(v) => set("total_tbw_tb", v)}
          />
          <NumberField
            label="Total read"
            unit="TB"
            value={value.total_tbr_tb}
            min={0}
            onChange={(v) => set("total_tbr_tb", v)}
          />
        </div>
      </Panel>

      <Panel title="Health signals" subtitle="Error counters and temperature">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <NumberField
            label="Temperature"
            unit="°C"
            value={value.temperature_c}
            min={-40}
            max={125}
            onChange={(v) => set("temperature_c", v)}
          />
          <NumberField
            label="Media errors"
            value={value.media_errors}
            min={0}
            onChange={(v) => set("media_errors", v)}
          />
          <NumberField
            label="Unsafe shutdowns"
            value={value.unsafe_shutdowns}
            min={0}
            onChange={(v) => set("unsafe_shutdowns", v)}
          />
          <NumberField
            label="CRC errors"
            value={value.crc_errors}
            min={0}
            onChange={(v) => set("crc_errors", v)}
          />
          <NumberField
            label="Read error rate"
            value={value.read_error_rate}
            min={0}
            onChange={(v) => set("read_error_rate", v)}
          />
          <NumberField
            label="Write error rate"
            value={value.write_error_rate}
            min={0}
            onChange={(v) => set("write_error_rate", v)}
          />
        </div>
      </Panel>

      <div className="example-dock">
        <div className="example-dock-heading">
          <div><span className="eyebrow">Scenario deck</span><p>Load a known drive profile to compare model behavior.</p></div>
          <span className="font-mono-data text-[10px] text-ink-dim">{EXAMPLE_DRIVES.length} READY</span>
        </div>
        <div className="example-grid">
          {EXAMPLE_DRIVES.map((example) => (
            <button key={example.name} type="button" className={`example-card ${example.tone}`} onClick={() => onChange(example.telemetry)}>
              <span className="example-card-top"><span>{example.tag}</span><span className="example-arrow">↗</span></span>
              <strong>{example.name}</strong>
              <small>{example.description}</small>
            </button>
          ))}
        </div>
      </div>

      <div className="predictor-actions flex items-center gap-3">
        <button
          type="submit"
          disabled={loading}
          className="bg-teal text-[#05100d] font-medium text-[14px] px-5 py-2.5 rounded-sm hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? "Running diagnostic..." : "Run diagnostic"}
        </button>
        <button type="button" onClick={() => onChange(EXAMPLE_TELEMETRY)} className="text-[13px] text-ink-dim hover:text-ink transition-colors underline underline-offset-4 decoration-border-console">Reset default</button>
      </div>
    </form>
  );
}
