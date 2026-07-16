import type { SolveRequest, SolveResponse } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export class ApiError extends Error {}

export async function solvePde(request: SolveRequest): Promise<SolveResponse> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/api/solve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
  } catch {
    throw new ApiError(
      `Could not reach the solver API at ${API_BASE}. Is the backend running?`
    );
  }

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") {
        detail = body.detail;
      } else if (Array.isArray(body.detail)) {
        detail = body.detail.map((d: { loc?: string[]; msg: string }) =>
          `${(d.loc ?? []).join(".")}: ${d.msg}`
        ).join("; ");
      }
    } catch {
      // response wasn't JSON; fall back to statusText
    }
    throw new ApiError(detail);
  }

  return res.json();
}
