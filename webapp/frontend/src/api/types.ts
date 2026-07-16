// Mirrors webapp/backend/schemas.py. Keep in sync with that file.

export type Dimension = "1D" | "2D";
export type TimeOrder = 0 | 1 | 2;

export interface GridSpec {
  dimension: Dimension;
  nx: number;
  ny: number;
  lx: number;
  ly: number;
}

export type TermSpec =
  | { type: "diffusion"; coeff: number }
  | { type: "advection"; velocity: number | [number, number]; scheme: "upwind" | "central" }
  | { type: "reaction"; coeff: number }
  | { type: "source"; expression: string }
  | { type: "function"; expression: string };

export type TermType = TermSpec["type"];

export type BCSpec =
  | { type: "dirichlet"; value: number }
  | { type: "dirichlet_box"; left: number; right: number; bottom: number; top: number }
  | { type: "neumann"; derivative: number }
  | { type: "periodic" };

export type BCType = BCSpec["type"];

export type InitialConditionShape = "uniform" | "gaussian" | "sine_wave" | "random" | "expression";

export interface InitialConditionSpec {
  shape: InitialConditionShape;
  intensity: number;
  center?: number[] | null;
  spread?: number | null;
  seed?: number | null;
  expression?: string | null;
}

export type IntegrationScheme = "euler" | "rk2" | "rk4" | "implicit_euler" | "crank_nicolson";

export interface IntegrationSpec {
  scheme: IntegrationScheme;
  dt: number;
  steps: number;
  record_every: number;
  v0_expression?: string | null;
}

export interface SolveRequest {
  grid: GridSpec;
  time_order: TimeOrder;
  terms: TermSpec[];
  bc: BCSpec;
  initial_condition: InitialConditionSpec;
  integration: IntegrationSpec | null;
}

export interface SolveResponse {
  dimension: Dimension;
  nx: number;
  ny: number;
  dx: number;
  dy: number | null;
  lx: number;
  ly: number;
  times: number[];
  history: number[][] | number[][][];
  vmin: number;
  vmax: number;
}
