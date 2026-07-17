import type { BuilderState } from "../../builderState";
import { Card } from "../ui/Card";
import { Field, FieldRow } from "../ui/Field";
import { NumberField } from "../ui/NumberField";
import { SegmentedControl } from "../ui/SegmentedControl";

export function GridSection({
  state,
  onChange,
}: {
  state: BuilderState;
  onChange: (patch: Partial<BuilderState>) => void;
}) {
  return (
    <Card title="Domain & grid">
      <Field label="Dimension">
        <SegmentedControl
          value={state.dimension}
          onChange={(dimension) => onChange({ dimension })}
          options={[
            { value: "1D", label: "1D" },
            { value: "2D", label: "2D" },
          ]}
        />
      </Field>

      <div style={{ height: 12 }} />

      <FieldRow>
        <Field label="Points nx" hint="3–256">
          <NumberField value={state.nx} min={3} max={256} step={1} onChange={(nx) => onChange({ nx: Math.round(nx) })} />
        </Field>
        {state.dimension === "2D" && (
          <Field label="Points ny" hint="3–256">
            <NumberField value={state.ny} min={3} max={256} step={1} onChange={(ny) => onChange({ ny: Math.round(ny) })} />
          </Field>
        )}
      </FieldRow>

      <div style={{ height: 10 }} />

      <FieldRow>
        <Field label="Length Lx">
          <NumberField value={state.lx} min={0.001} step={0.1} onChange={(lx) => onChange({ lx })} />
        </Field>
        {state.dimension === "2D" && (
          <Field label="Length Ly">
            <NumberField value={state.ly} min={0.001} step={0.1} onChange={(ly) => onChange({ ly })} />
          </Field>
        )}
      </FieldRow>
    </Card>
  );
}
