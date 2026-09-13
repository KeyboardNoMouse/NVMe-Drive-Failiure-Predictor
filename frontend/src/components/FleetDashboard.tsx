"use client";

import { ReactNode } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { FleetOverviewResponse } from "@/lib/types";

const MODE_COLORS = ["#ff6b5f", "#f6bd60", "#4ecdc4"];
const CHART_GRID = "rgba(141, 164, 168, 0.14)";
const AXIS = "#83989d";
const TOOLTIP_STYLE = {
  background: "#142124",
  border: "1px solid #2b4146",
  borderRadius: "3px",
  color: "#e8f0ef",
  fontSize: "12px",
};

function pct(value: number) {
  return `${value.toFixed(1)}%`;
}

export function FleetDashboard({ data }: { data: FleetOverviewResponse | null }) {
  if (!data) {
    return <div className="dashboard-loading">Loading fleet telemetry...</div>;
  }

  return (
    <section className="fleet-dashboard" aria-label="Fleet overview">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Fleet overview</p>
          <h2>Current fleet health</h2>
        </div>
        <span className="data-stamp">Original dataset · 10,000 drives</span>
      </div>

      <div className="kpi-grid">
        <Kpi label="Drives observed" value={data.total_drives.toLocaleString()} note="real telemetry rows" tone="teal" />
        <Kpi label="Failure rate" value={pct(data.failure_rate * 100)} note={`${data.failed_drives} labeled failures`} tone="coral" />
        <Kpi label="Avg. life used" value={`${data.average_life_used.toFixed(1)}%`} note="fleet endurance consumed" tone="gold" />
        <Kpi label="Avg. temperature" value={`${data.average_temperature.toFixed(1)}°`} note={`${Math.round(data.average_power_on_hours).toLocaleString()} avg hours`} tone="sky" />
      </div>

      <div className="chart-grid chart-grid-primary">
        <ChartFrame title="Wear band / failure density" subtitle="Labeled health state by percent of life used" className="chart-span-two">
          <ResponsiveContainer width="100%" height={260}>
            <AreaChart data={data.life_bands} margin={{ top: 12, right: 10, left: -22, bottom: 0 }}>
              <defs>
                <linearGradient id="healthyFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#4ecdc4" stopOpacity={0.38} />
                  <stop offset="100%" stopColor="#4ecdc4" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="failedFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#ff6b5f" stopOpacity={0.38} />
                  <stop offset="100%" stopColor="#ff6b5f" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke={CHART_GRID} vertical={false} />
              <XAxis dataKey="label" stroke={AXIS} tickLine={false} axisLine={false} fontSize={11} />
              <YAxis stroke={AXIS} tickLine={false} axisLine={false} fontSize={11} />
              <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ stroke: "#2b4146" }} />
              <Legend verticalAlign="top" height={34} iconType="circle" />
              <Area type="monotone" dataKey="healthy" name="Healthy" stroke="#4ecdc4" fill="url(#healthyFill)" strokeWidth={2} />
              <Area type="monotone" dataKey="failed" name="Failed" stroke="#ff6b5f" fill="url(#failedFill)" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </ChartFrame>

        <ChartFrame title="Failure mode mix" subtitle="Observed labels among the 194 failures">
          <div className="donut-wrap">
            <ResponsiveContainer width="100%" height={205}>
              <PieChart>
                <Pie data={data.failure_modes} dataKey="count" nameKey="label" innerRadius={58} outerRadius={84} paddingAngle={3} stroke="none">
                  {data.failure_modes.map((mode, index) => <Cell key={mode.code} fill={MODE_COLORS[index % MODE_COLORS.length]} />)}
                </Pie>
                <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(value) => [`${value} drives`, "Observed"]} />
              </PieChart>
            </ResponsiveContainer>
            <div className="donut-center"><strong>{data.failed_drives}</strong><span>failures</span></div>
          </div>
          <div className="legend-stack">
            {data.failure_modes.map((mode, index) => (
              <div className="legend-row" key={mode.code}>
                <span className="legend-dot" style={{ background: MODE_COLORS[index % MODE_COLORS.length] }} />
                <span>{mode.label}</span><strong>{pct(mode.percentage)}</strong>
              </div>
            ))}
          </div>
        </ChartFrame>
      </div>

      <div className="chart-grid">
        <ChartFrame title="Vendor watchlist" subtitle="Failure rate by hardware vendor">
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={data.vendor_stats} layout="vertical" margin={{ top: 8, right: 18, left: 8, bottom: 0 }}>
              <CartesianGrid stroke={CHART_GRID} horizontal={false} />
              <XAxis type="number" stroke={AXIS} tickLine={false} axisLine={false} fontSize={11} unit="%" />
              <YAxis type="category" dataKey="label" stroke={AXIS} tickLine={false} axisLine={false} fontSize={11} width={62} />
              <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(value) => [`${value}%`, "Failure rate"]} />
              <Bar dataKey="failure_rate" fill="#f6bd60" radius={[0, 2, 2, 0]} barSize={18} />
            </BarChart>
          </ResponsiveContainer>
        </ChartFrame>

        <ChartFrame title="Firmware watchlist" subtitle="Failure rate by firmware family">
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={data.firmware_stats} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
              <CartesianGrid stroke={CHART_GRID} vertical={false} />
              <XAxis dataKey="label" stroke={AXIS} tickLine={false} axisLine={false} fontSize={11} />
              <YAxis stroke={AXIS} tickLine={false} axisLine={false} fontSize={11} unit="%" />
              <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(value) => [`${value}%`, "Failure rate"]} />
              <Bar dataKey="failure_rate" fill="#7aa7ff" radius={[2, 2, 0, 0]} barSize={24} />
            </BarChart>
          </ResponsiveContainer>
        </ChartFrame>

        <ChartFrame title="Thermal / endurance field" subtitle="Each point is a sampled drive; red points are failures" className="chart-span-two">
          <ResponsiveContainer width="100%" height={240}>
            <ScatterChart margin={{ top: 12, right: 18, left: -12, bottom: 0 }}>
              <CartesianGrid stroke={CHART_GRID} />
              <XAxis type="number" dataKey="temperature" name="Temperature" unit="°C" stroke={AXIS} tickLine={false} axisLine={false} fontSize={11} />
              <YAxis type="number" dataKey="life_used" name="Life used" unit="%" stroke={AXIS} tickLine={false} axisLine={false} fontSize={11} />
              <ZAxis range={[18, 18]} />
              <Tooltip contentStyle={TOOLTIP_STYLE} cursor={{ strokeDasharray: "3 3" }} />
              <Scatter name="Healthy drives" data={data.telemetry_points.filter((point) => point.failed === 0)} fill="#4ecdc4" fillOpacity={0.24} />
              <Scatter name="Failed drives" data={data.telemetry_points.filter((point) => point.failed === 1)} fill="#ff6b5f" />
            </ScatterChart>
          </ResponsiveContainer>
        </ChartFrame>
      </div>

      <div className="data-note">
        <span className="status-dot" />
        <span><strong>Reading the room:</strong> fleet charts show observed dataset patterns. Failure-mode labels are held out of the predictor to prevent leakage.</span>
      </div>
    </section>
  );
}

function Kpi({ label, value, note, tone }: { label: string; value: string; note: string; tone: string }) {
  return <div className={`kpi kpi-${tone}`}><span>{label}</span><strong>{value}</strong><small>{note}</small></div>;
}

function ChartFrame({ title, subtitle, children, className = "" }: { title: string; subtitle: string; children: ReactNode; className?: string }) {
  return <section className={`chart-frame ${className}`}><header><div><h3>{title}</h3><p>{subtitle}</p></div></header><div className="chart-body">{children}</div></section>;
}
