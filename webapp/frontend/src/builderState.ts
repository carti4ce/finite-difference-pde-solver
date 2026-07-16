import type {
  BCSpec, BCType, Dimension, InitialConditionSpec, IntegrationSpec,
  SolveRequest, TermSpec, TermType, TimeOrder,
} from "./api/types";

/** A term with a stable client-side id, for React keys / add-remove. */
export type EditableTerm = TermSpec & { id: string };

export interface BuilderState {
  dimension: Dimension;
  nx: number;
  ny: number;
  lx: number;
  ly: number;
  timeOrder: TimeOrder;
  terms: EditableTerm[];
  bc: BCSpec;
  initialCondition: InitialConditionSpec;
  integration: IntegrationSpec;
}

let idCounter = 0;
export function nextId(): string {
  idCounter += 1;
  return `term-${idCounter}`;
}

export function defaultTerm(type: TermType): EditableTerm {
  const id = nextId();
  switch (type) {
    case "diffusion":
      return { id, type, coeff: 0.1 };
    case "advection":
      return { id, type, velocity: 1.0, scheme: "upwind" };
    case "reaction":
      return { id, type, coeff: -1.0 };
    case "source":
      return { id, type, expression: "sin(pi * x)" };
    case "function":
      return { id, type, expression: "u * (1 - u)" };
  }
}

export function defaultBc(type: BCType): BCSpec {
  switch (type) {
    case "dirichlet":
      return { type, value: 0.0 };
    case "dirichlet_box":
      return { type, left: 1.0, right: 0.0, bottom: 0.0, top: 0.0 };
    case "neumann":
      return { type, derivative: 0.0 };
    case "periodic":
      return { type };
  }
}

export const TERM_LABELS: Record<TermType, string> = {
  diffusion: "Diffusion",
  advection: "Advection",
  reaction: "Reaction",
  source: "Source",
  function: "Nonlinear",
};

export const TERM_TYPE_OPTIONS: { value: TermType; label: string }[] = (
  Object.keys(TERM_LABELS) as TermType[]
).map((value) => ({ value, label: TERM_LABELS[value] }));

export const BC_LABELS: Record<BCType, string> = {
  dirichlet: "Dirichlet",
  dirichlet_box: "Dirichlet (per edge)",
  neumann: "Neumann",
  periodic: "Periodic",
};

export function defaultBuilderState(): BuilderState {
  return {
    dimension: "1D",
    nx: 100,
    ny: 60,
    lx: 1.0,
    ly: 1.0,
    timeOrder: 1,
    terms: [defaultTerm("diffusion")],
    bc: defaultBc("dirichlet"),
    initialCondition: { shape: "gaussian", intensity: 1.0 },
    integration: { scheme: "crank_nicolson", dt: 1e-3, steps: 400, record_every: 10 },
  };
}

export function toSolveRequest(state: BuilderState): SolveRequest {
  return {
    grid: { dimension: state.dimension, nx: state.nx, ny: state.ny, lx: state.lx, ly: state.ly },
    time_order: state.timeOrder,
    terms: state.terms.map((t) => {
      const { id, ...term } = t;
      void id;
      return term as TermSpec;
    }),
    bc: state.bc,
    initial_condition: state.initialCondition,
    integration: state.timeOrder === 0 ? null : state.integration,
  };
}

// --- presets ---------------------------------------------------------------

export interface Preset {
  name: string;
  description: string;
  build: () => BuilderState;
}

export const PRESETS: Preset[] = [
  {
    name: "Heat diffusion",
    description: "u_t = 0.1 * laplacian(u), insulated ends",
    build: () => ({
      ...defaultBuilderState(),
      dimension: "1D",
      nx: 120,
      timeOrder: 1,
      terms: [defaultTerm("diffusion")],
      bc: defaultBc("neumann"),
      initialCondition: { shape: "gaussian", intensity: 1.0, spread: 0.05 },
      integration: { scheme: "crank_nicolson", dt: 1e-3, steps: 600, record_every: 15 },
    }),
  },
  {
    name: "Wave",
    description: "u_tt = laplacian(u), fixed ends",
    build: () => ({
      ...defaultBuilderState(),
      dimension: "1D",
      nx: 128,
      timeOrder: 2,
      terms: [{ ...defaultTerm("diffusion"), coeff: 1.0 }],
      bc: defaultBc("dirichlet"),
      initialCondition: { shape: "sine_wave", intensity: 1.0 },
      integration: { scheme: "rk4", dt: 5e-4, steps: 800, record_every: 20 },
    }),
  },
  {
    name: "Advection",
    description: "u_t = -v * u_x, periodic wrap",
    build: () => ({
      ...defaultBuilderState(),
      dimension: "1D",
      nx: 200,
      timeOrder: 1,
      terms: [{ ...defaultTerm("advection"), velocity: 1.0, scheme: "upwind" }],
      bc: defaultBc("periodic"),
      initialCondition: { shape: "gaussian", intensity: 1.0, spread: 0.03, center: [0.3] },
      integration: { scheme: "rk4", dt: 1e-3, steps: 500, record_every: 10 },
    }),
  },
  {
    name: "Fisher-KPP front",
    description: "u_t = u_xx + u(1-u), traveling front",
    build: () => ({
      ...defaultBuilderState(),
      dimension: "1D",
      nx: 200,
      lx: 40,
      timeOrder: 1,
      terms: [
        { ...defaultTerm("diffusion"), coeff: 1.0 },
        { ...defaultTerm("function"), expression: "u * (1 - u)" },
      ],
      bc: defaultBc("dirichlet"),
      initialCondition: { shape: "expression", intensity: 1.0, expression: "1 if abs(x - 10) < 1 else 0" },
      integration: { scheme: "rk4", dt: 5e-3, steps: 1200, record_every: 30 },
    }),
  },
  {
    name: "2D Poisson",
    description: "0 = laplacian(u) + source, steady",
    build: () => ({
      ...defaultBuilderState(),
      dimension: "2D",
      nx: 48,
      ny: 48,
      timeOrder: 0,
      terms: [
        { ...defaultTerm("diffusion"), coeff: 1.0 },
        { ...defaultTerm("source"), expression: "2 * pi**2 * sin(pi * x) * sin(pi * y)" },
      ],
      bc: defaultBc("dirichlet"),
      initialCondition: { shape: "uniform", intensity: 0.0 },
    }),
  },
  {
    name: "2D heated plate",
    description: "u_t = 0.1*laplacian(u) + heaters, cold edges",
    build: () => ({
      ...defaultBuilderState(),
      dimension: "2D",
      nx: 40,
      ny: 40,
      timeOrder: 1,
      terms: [
        { ...defaultTerm("diffusion"), coeff: 0.1 },
        {
          ...defaultTerm("source"),
          expression: "6*exp(-((x-0.3)**2+(y-0.3)**2)/0.01) + 4*exp(-((x-0.7)**2+(y-0.65)**2)/0.006)",
        },
      ],
      bc: defaultBc("dirichlet"),
      initialCondition: { shape: "uniform", intensity: 0.0 },
      integration: { scheme: "crank_nicolson", dt: 1e-2, steps: 300, record_every: 6 },
    }),
  },
];
