"use client";

import { ModelInfoResponse } from "@/lib/types";

function pct(v: number) {
  return `${(v * 100).toFixed(1)}%`;
}

export function ModelInfoStrip({ info }: { info: ModelInfoResponse | null }) {
  if (!info) return null;

  const m = info.holdout_test_metrics;

  return (
    <section className="border border-border-console bg-surface rounded-sm p-5">
      <h2 className="text-[15px] font-medium text-ink mb-1">About this model</h2>
      <p className="text-[13px] text-ink-dim max-w-[70ch] leading-relaxed">
        Trained on {info.dataset.n_rows.toLocaleString()} real drives (
        {info.dataset.n_failures} failures, {pct(info.dataset.failure_rate)} of the fleet) from the
        original dataset -- the synthetic/augmented file was not used. {" "}
        <span className="text-ink">{info.dropped_leaky_columns.join(", ")}</span> were excluded as
        inputs because they leak the label rather than describe drive condition.
      </p>

      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-4 mt-5">
        <Metric label="Accuracy" value={pct(m.accuracy)} note="misleading alone -- see recall" />
        <Metric label="Precision" value={pct(m.precision)} note="alerts that were failures" />
        <Metric label="Recall" value={pct(m.recall)} note="failed drives caught" />
        <Metric label="F1" value={pct(m.f1)} />
        <Metric label="ROC-AUC" value={m.roc_auc.toFixed(3)} />
        <Metric label="PR-AUC" value={m.pr_auc.toFixed(3)} />
      </div>

      <p className="text-[12px] text-ink-dim mt-4">
        Held out on a 20% test split the model never trained or tuned on. Cross-validation and
        held-out scores agree within{" "}
        {pct(Math.max(...Object.values(info.cv_vs_holdout_gap)))} on every metric, which is the
        signal that the model generalizes rather than memorizes.
      </p>
    </section>
  );
}

function Metric({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[12px] text-ink-dim">{label}</span>
      <span className="font-mono-data text-[20px] text-ink">{value}</span>
      {note && <span className="text-[11px] text-ink-dim/70">{note}</span>}
    </div>
  );
}
