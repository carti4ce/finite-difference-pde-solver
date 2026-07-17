import type { EditableTerm } from "../../builderState";
import type { IntegrationScheme, IntegrationSpec, TimeOrder } from "../../api/types";
import { Card } from "../ui/Card";
import { Field, FieldRow } from "../ui/Field";
import { NumberField } from "../ui/NumberField";
import { Select } from "../ui/Select";

const ALL_SCHEMES: { value: IntegrationScheme; label: string }[] = [
  { value: "rk4", label: "RK4 (explicit)" },
  { value: "rk2", label: "RK2 (explicit)" },
  { value: "euler", label: "Euler (explicit)" },
  { value: "implicit_euler", label: "Implicit Euler" },
  { value: "crank_nicolson", label: "Crank-Nicolson" },
];

const MAX_FRAMES = 600;

export function IntegrationSection({
  integration,
  timeOrder,
  terms,
  onChange,
}: {
  integration: IntegrationSpec;
  timeOrder: TimeOrder;
  terms: EditableTerm[];
  onChange: (integration: IntegrationSpec) => void;
}) {
  if (timeOrder === 0) return null;

  const hasNonlinear = terms.some((t) => t.type === "function");
  const schemeOptions = hasNonlinear ? ALL_SCHEMES.filter((s) => !s.value.includes("implicit") && s.value !== "crank_nicolson") : ALL_SCHEMES;
  const numFrames = Math.floor(integration.steps / integration.record_every) + 1;
  const tooManyFrames = numFrames > MAX_FRAMES;

  return (
    <Card title="Time integration">
      {hasNonlinear && (
        <p className="field-hint" style={{ marginTop: 0 }}>
          Nonlinear terms require an explicit scheme.
        </p>
      )}
      <Field label="Scheme">
        <Select
          value={integration.scheme}
          onChange={(scheme) => onChange({ ...integration, scheme })}
          options={schemeOptions}
        />
      </Field>

      <div style={{ height: 10 }} />

      <FieldRow>
        <Field label="Time step (dt)">
          <NumberField value={integration.dt} step={0.0001} min={0} onChange={(dt) => onChange({ ...integration, dt })} />
        </Field>
        <Field label="Steps">
          <NumberField
            value={integration.steps}
            step={10}
            min={1}
            max={20000}
            onChange={(steps) => onChange({ ...integration, steps: Math.round(steps) })}
          />
        </Field>
      </FieldRow>

      <div style={{ height: 10 }} />

      <Field
        label="Record every N steps"
        error={tooManyFrames ? `${numFrames} frames requested, max ${MAX_FRAMES} — increase this` : undefined}
        hint={tooManyFrames ? undefined : `${numFrames} animation frames`}
      >
        <NumberField
          value={integration.record_every}
          step={1}
          min={1}
          onChange={(record_every) => onChange({ ...integration, record_every: Math.round(record_every) })}
        />
      </Field>

      {timeOrder === 2 && (
        <>
          <div style={{ height: 10 }} />
          <Field label="Initial velocity uₜ(x, 0)" hint="Optional expression; blank = 0">
            <textarea
              className="textarea mono"
              rows={2}
              value={integration.v0_expression ?? ""}
              onChange={(e) => onChange({ ...integration, v0_expression: e.target.value })}
            />
          </Field>
        </>
      )}
    </Card>
  );
}
