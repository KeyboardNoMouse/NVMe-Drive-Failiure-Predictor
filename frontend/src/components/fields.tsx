"use client";

import { ReactNode } from "react";

export function Panel({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children: ReactNode;
}) {
  return (
    <section className="border border-border-console bg-surface rounded-sm">
      <header className="px-5 pt-4 pb-3 border-b border-border-console">
        <h2 className="text-[15px] font-medium text-ink">{title}</h2>
        {subtitle && <p className="text-[13px] text-ink-dim mt-0.5">{subtitle}</p>}
      </header>
      <div className="p-5">{children}</div>
    </section>
  );
}

export function NumberField({
  label,
  unit,
  value,
  min,
  max,
  step = "any",
  onChange,
}: {
  label: string;
  unit?: string;
  value: number;
  min?: number;
  max?: number;
  step?: number | "any";
  onChange: (v: number) => void;
}) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-[13px] text-ink-dim">
        {label}
        {unit && <span className="text-ink-dim/70"> ({unit})</span>}
      </span>
      <input
        type="number"
        inputMode="decimal"
        className="font-mono-data text-[14px] bg-surface-raised border border-border-console rounded-sm px-3 py-2 text-ink focus:border-teal transition-colors"
        value={Number.isFinite(value) ? value : ""}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(Number.isFinite(e.target.valueAsNumber) ? e.target.valueAsNumber : 0)}
      />
    </label>
  );
}

export function SelectField<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: readonly T[];
  onChange: (v: T) => void;
}) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-[13px] text-ink-dim">{label}</span>
      <select
        className="text-[14px] bg-surface-raised border border-border-console rounded-sm px-3 py-2 text-ink focus:border-teal transition-colors"
        value={value}
        onChange={(e) => onChange(e.target.value as T)}
      >
        {options.map((opt) => (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    </label>
  );
}
