"use client";

import { useEffect, useState } from "react";
import { getFleetOverview, getModelInfo, predictFailure, ApiError, API_BASE } from "@/lib/api";
import { DriveTelemetry, FleetOverviewResponse, ModelInfoResponse, PredictionResponse } from "@/lib/types";
import { TelemetryForm, EXAMPLE_TELEMETRY } from "@/components/TelemetryForm";
import { ResultPanel } from "@/components/ResultPanel";
import { ModelInfoStrip } from "@/components/ModelInfoStrip";
import { FleetDashboard } from "@/components/FleetDashboard";

export default function Home() {
  const [telemetry, setTelemetry] = useState<DriveTelemetry>(EXAMPLE_TELEMETRY);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [fleet, setFleet] = useState<FleetOverviewResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([getModelInfo(), getFleetOverview()])
      .then(([model, overview]) => {
        setModelInfo(model);
        setFleet(overview);
      })
      .catch(() => {
        setModelInfo(null);
        setFleet(null);
      });
  }, []);

  async function handleSubmit() {
    setLoading(true);
    setError(null);
    try {
      const res = await predictFailure(telemetry);
      setResult(res);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Something went wrong running the diagnostic.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="topbar-inner">
          <a className="brand" href="#top"><span className="brand-pulse" />NVMe Fleet Dashboard</a>
          <span className="topbar-title">Failure monitoring and diagnostics</span>
          <nav><a href="#fleet">Overview</a><a href="#predictor">Predictor</a><a href={`${API_BASE}/docs`} target="_blank" rel="noreferrer">API docs</a><span className="live-pill"><span /> MODEL ONLINE</span></nav>
        </div>
      </header>

      <main id="top" className="page-shell">
        <section className="hero-intro">
          <div>
            <p className="eyebrow">Fleet overview / September 2026</p>
            <h1>NVMe drive<br /><em>health overview</em></h1>
            <p className="hero-copy">Review failure patterns across the fleet and run a prediction for an individual drive using its SMART telemetry.</p>
          </div>
          <div className="hero-signal"><div className="signal-ring"><span>{fleet ? (100 - fleet.failure_rate * 100).toFixed(1) : "--"}<small>%</small></span></div><p>healthy drives</p></div>
        </section>

        <div id="fleet"><FleetDashboard data={fleet} /></div>

        <section id="predictor" className="predictor-section">
          <div className="section-heading">
            <div><p className="eyebrow">Individual prediction</p><h2>Check a drive</h2></div>
            <span className="data-stamp">Random forest · SMART telemetry</span>
          </div>
          <div className="predictor-layout">
            <TelemetryForm value={telemetry} onChange={setTelemetry} onSubmit={handleSubmit} loading={loading} />
            <div className="result-column">
              {error && <div className="error-banner">{error}</div>}
              <ResultPanel result={result} />
            </div>
          </div>
          <div className="model-summary"><ModelInfoStrip info={modelInfo} /></div>
        </section>
      </main>

      <footer className="site-footer"><span>NVMe Fleet Dashboard</span><span>For fleet monitoring guidance, not a substitute for vendor RMA diagnostics.</span></footer>
    </div>
  );
}
