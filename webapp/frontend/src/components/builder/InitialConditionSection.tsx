import type { Dimension, InitialConditionShape, InitialConditionSpec } from "../../api/types";
import { Card } from "../ui/Card";
import { Field, FieldRow } from "../ui/Field";
import { NumberField } from "../ui/NumberField";
import { Select } from "../ui/Select";

const SHAPE_OPTIONS: { value: InitialConditionShape; label: string }[] = [
  { value: "gaussian", label: "Gaussian bump" },
  { value: "sine_wave", label: "Sine wave" },
  { value: "uniform", label: "Uniform" },
  { value: "random", label: "Random" },
  { value: "expression", label: "Custom expression" },
];

export function InitialConditionSection({
  ic,
  dimension,
  onChange,
}: {
  ic: InitialConditionSpec;
  dimension: Dimension;
  onChange: (ic: InitialConditionSpec) => void;
}) {
  const centerLen = dimension === "1D" ? 1 : 2;
  const center = ic.center ?? Array(centerLen).fill(undefined as unknown as number);

  return (
    <Card title="Initial condition">
      <Field label="Shape">
        <Select value={ic.shape} onChange={(shape) => onChange({ ...ic, shape })} options={SHAPE_OPTIONS} />
      </Field>

      <div style={{ height: 10 }} />

      {ic.shape !== "expression" && (
        <Field label="Intensity">
          <NumberField value={ic.intensity} step={0.1} onChange={(intensity) => onChange({ ...ic, intensity })} />
        </Field>
      )}

      {ic.shape === "gaussian" && (
        <>
          <div style={{ height: 10 }} />
          <FieldRow>
            <Field label={dimension === "1D" ? "Center x" : "Center x"} hint="blank = domain center">
              <NumberField
                value={center[0] as number}
                step={0.05}
                onChange={(x) => onChange({ ...ic, center: [x, ...(center.slice(1) as number[])] })}
              />
            </Field>
            {dimension === "2D" && (
              <Field label="Center y" hint="blank = domain center">
                <NumberField
                  value={center[1] as number}
                  step={0.05}
                  onChange={(y) => onChange({ ...ic, center: [center[0] as number, y] })}
                />
              </Field>
            )}
          </FieldRow>
          <div style={{ height: 10 }} />
          <Field label="Spread (σ)" hint="blank = 5% of domain">
            <NumberField value={ic.spread as number} step={0.005} min={0} onChange={(spread) => onChange({ ...ic, spread })} />
          </Field>
        </>
      )}

      {ic.shape === "random" && (
        <>
          <div style={{ height: 10 }} />
          <Field label="Seed" hint="blank = non-reproducible">
            <NumberField value={ic.seed as number} step={1} onChange={(seed) => onChange({ ...ic, seed: Math.round(seed) })} />
          </Field>
        </>
      )}

      {ic.shape === "expression" && (
        <Field
          label={dimension === "1D" ? "u₀(x)" : "u₀(x, y)"}
          hint="Same syntax as term expressions (functions, pi/e, ** for power)."
        >
          <textarea
            className="textarea mono"
            rows={2}
            value={ic.expression ?? ""}
            onChange={(e) => onChange({ ...ic, expression: e.target.value })}
          />
        </Field>
      )}
    </Card>
  );
}
