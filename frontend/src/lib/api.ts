import { DriveTelemetry, FleetOverviewResponse, ModelInfoResponse, PredictionResponse } from "./types";

export const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch {
    throw new ApiError(
      `Could not reach the prediction API at ${API_BASE}. Is the FastAPI backend running?`
    );
  }
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new ApiError(body?.detail ?? `Request failed with status ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export function predictFailure(telemetry: DriveTelemetry) {
  return request<PredictionResponse>("/predict", {
    method: "POST",
    body: JSON.stringify(telemetry),
  });
}

export function getModelInfo() {
  return request<ModelInfoResponse>("/model-info");
}

export function getFleetOverview() {
  return request<FleetOverviewResponse>("/fleet-overview");
}
