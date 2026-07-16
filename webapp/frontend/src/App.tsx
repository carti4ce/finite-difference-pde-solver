import { useState } from "react";
import "./app.css";
import { solvePde, ApiError } from "./api/client";
import type { SolveResponse } from "./api/types";
import { defaultBuilderState, toSolveRequest } from "./builderState";
import type { BuilderState } from "./builderState";
import { useTheme } from "./useTheme";

import { PresetsBar } from "./components/builder/PresetsBar";
import { GridSection } from "./components/builder/GridSection";
import { TimeOrderSection } from "./components/builder/TimeOrderSection";
import { TermsSection } from "./components/builder/TermsSection";
import { BoundaryConditionSection } from "./components/builder/BoundaryConditionSection";
import { InitialConditionSection } from "./components/builder/InitialConditionSection";
import { IntegrationSection } from "./components/builder/IntegrationSection";
import { Button } from "./components/ui/Button";
import { AnimationPlayer } from "./components/viewer/AnimationPlayer";

export default function App() {
  const [state, setState] = useState<BuilderState>(defaultBuilderState);
  const [response, setResponse] = useState<SolveResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { isDark, toggle } = useTheme();

  function patch(p: Partial<BuilderState>) {
    setState((s) => ({ ...s, ...p }));
  }

  async function handleSolve() {
    setLoading(true);
    setError(null);
    try {
      const result = await solvePde(toSolveRequest(state));
      setResponse(result);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Unexpected error while solving.");
      setResponse(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-title">
          <svg className="mark" viewBox="0 0 24 24" fill="none">
            <path d="M3 12c2-6 4-9 6-9s2 12 4 12 2-9 4-9 2 6 4 6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
          </svg>
          PDE Studio
          <span className="app-subtitle">custom finite-difference solver</span>
        </div>
        <Button variant="ghost" icon onClick={toggle} aria-label="Toggle theme">
          {isDark ? "☀" : "🌙"}
        </Button>
      </header>

      <div className="app-body">
        <div className="builder-pane">
          <PresetsBar onSelect={setState} />
          <GridSection state={state} onChange={patch} />
          <TimeOrderSection value={state.timeOrder} onChange={(timeOrder) => patch({ timeOrder })} />
          <TermsSection terms={state.terms} dimension={state.dimension} onChange={(terms) => patch({ terms })} />
          <BoundaryConditionSection bc={state.bc} dimension={state.dimension} onChange={(bc) => patch({ bc })} />
          <InitialConditionSection
            ic={state.initialCondition}
            dimension={state.dimension}
            onChange={(initialCondition) => patch({ initialCondition })}
          />
          <IntegrationSection
            integration={state.integration}
            timeOrder={state.timeOrder}
            terms={state.terms}
            onChange={(integration) => patch({ integration })}
          />

          <Button variant="primary" large block onClick={handleSolve} disabled={loading}>
            {loading ? (
              <>
                <span className="spinner" /> Solving…
              </>
            ) : (
              "Solve"
            )}
          </Button>
        </div>

        <div className="viewer-pane">
          {error && (
            <div className="alert error">
              <div className="alert-title">Solve failed</div>
              {error}
            </div>
          )}

          {response ? (
            <AnimationPlayer response={response} dark={isDark} />
          ) : (
            !error && (
              <div className="empty-state">
                <svg viewBox="0 0 24 24" fill="none">
                  <path d="M3 12c2-6 4-9 6-9s2 12 4 12 2-9 4-9 2 6 4 6" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
                </svg>
                <p>Build a PDE on the left, then Solve to see it animate here.</p>
              </div>
            )
          )}
        </div>
      </div>
    </div>
  );
}
