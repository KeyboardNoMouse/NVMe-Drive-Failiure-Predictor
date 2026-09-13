"use client";

import { PredictionResponse } from "@/lib/types";

const RISK_COPY: Record<PredictionResponse["risk_level"], { label: string; color: string }> = {
  low: { label: "Low risk", color: "var(--accent-teal)" },
  moderate: { label: "Moderate risk", color: "var(--accent-amber)" },
  high: { label: "High risk", color: "var(--accent-red)" },
};

function featureLabel(feature: string): string {
  const map: Record<string, string> = {
    Power_On_Hours: "Power-on hours",
    Total_TBW_TB: "Total written",
    Total_TBR_TB: "Total read",
    Temperature_C: "Temperature",
    Percent_Life_Used: "Percent life used",
    Media_Errors: "Media errors",
    Unsafe_Shutdowns: "Unsafe shutdowns",
    CRC_Errors: "CRC errors",
    Read_Error_Rate: "Read error rate",
    Write_Error_Rate: "Write error rate",
    TBW_per_hour: "Write intensity (TB/hr)",
    TBR_per_hour: "Read intensity (TB/hr)",
    read_write_ratio: "Read/write ratio",
    wear_rate_per_1000h: "Wear rate per 1k hours",
    errors_per_1000h: "Errors per 1k hours",
    unsafe_shutdown_rate: "Unsafe shutdown rate",
    total_error_rate: "Combined error rate",
    media_error_ratio: "Media error share",
  };
  return map[feature] ?? feature;
}

export function ResultPanel({ result }: { result: PredictionResponse | null }) {
  if (!result) {
    return (
      <section className="border border-border-console bg-surface rounded-sm p-6 flex flex-col items-center justify-center text-center min-h-[280px]">
        <p className="text-ink-dim text-[14px] max-w-[32ch]">
          Fill in the telemetry panel and run a diagnostic to see the failure-risk readout here.
        </p>
      </section>
    );
  }

  const risk = RISK_COPY[result.risk_level];
  const pct = Math.round(result.failure_probability * 1000) / 10;
  const thresholdPct = result.decision_threshold * 100;

  return (
    <section className="border border-border-console bg-surface rounded-sm">
      <header className="px-5 pt-4 pb-3 border-b border-border-console flex items-baseline justify-between">
        <h2 className="text-[15px] font-medium text-ink">Diagnostic result</h2>
        <span
          className="text-[13px] font-mono-data px-2 py-0.5 rounded-sm border"
          style={{ color: risk.color, borderColor: risk.color }}
        >
          {risk.label}
        </span>
      </header>

      <div className="p-5 flex flex-col gap-6">
        <div>
          <div className="flex items-end gap-2">
            <span className="font-mono-data text-[44px] leading-none" style={{ color: risk.color }}>
              {pct.toFixed(1)}
            </span>
            <span className="text-[16px] text-ink-dim mb-1">% failure probability</span>
          </div>

          <div className="mt-3 h-2 bg-surface-raised rounded-full relative overflow-hidden">
            <div
              className="h-full rounded-full transition-[width]"
              style={{ width: `${Math.min(pct, 100)}%`, background: risk.color }}
            />
            <div
              className="absolute top-0 h-full w-px bg-ink-dim/60"
              style={{ left: `${thresholdPct}%` }}
              title={`Decision threshold: ${thresholdPct}%`}
            />
          </div>
          <p className="text-[12px] text-ink-dim mt-1.5">
            Predicted label: <span className="font-mono-data text-ink">{result.predicted_label}</span>{" "}
            &middot; decision threshold {thresholdPct.toFixed(0)}%
          </p>
        </div>

        <div className="attribution-box">
          <div className="attribution-heading">
            <div>
              <h3>What is driving the readout?</h3>
              <p>Signal attribution, not a causal diagnosis</p>
            </div>
            <span className="font-mono-data text-[11px] text-ink-dim">100% OUTCOME</span>
          </div>
          <div className="attribution-bars">
            {result.failure_attribution.map((item) => (
              <div className="attribution-row" key={item.label}>
                <div className="flex items-center justify-between text-[12px] mb-1">
                  <span className={item.kind === "healthy" ? "text-teal" : "text-ink"}>{item.label}</span>
                  <span className="font-mono-data text-ink-dim">{item.percentage.toFixed(1)}%</span>
                </div>
                <div className="attribution-track"><div className={`attribution-fill ${item.kind}`} style={{ width: `${item.percentage}%` }} /></div>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h3 className="text-[13px] text-ink-dim mb-2">Top contributing factors</h3>
          {result.top_contributing_factors.length === 0 ? (
            <p className="text-[13px] text-ink-dim">
              No telemetry values sit meaningfully above the healthy-population range.
            </p>
          ) : (
            <ul className="flex flex-col gap-2">
              {result.top_contributing_factors.map((f) => (
                <li
                  key={f.feature}
                  className="flex items-center justify-between text-[13px] border-b border-border-console/60 pb-2 last:border-0"
                >
                  <span className="text-ink">{featureLabel(f.feature)}</span>
                  <span className="font-mono-data text-ink-dim">
                    {f.value} <span className="text-ink-dim/60">(typical {f.healthy_typical})</span>
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </section>
  );
}
