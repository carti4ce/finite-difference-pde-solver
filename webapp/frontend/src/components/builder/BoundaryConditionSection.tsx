import type { BCSpec, BCType, Dimension } from "../../api/types";
import { BC_LABELS, defaultBc } from "../../builderState";
import { Card } from "../ui/Card";
import { Field, FieldRow } from "../ui/Field";
import { NumberField } from "../ui/NumberField";
import { Select } from "../ui/Select";

const BC_OPTIONS: { value: BCType; label: string }[] = (
  Object.keys(BC_LABELS) as BCType[]
).map((value) => ({ value, label: BC_LABELS[value] }));

export function BoundaryConditionSection({
  bc,
  dimension,
  onChange,
}: {
  bc: BCSpec;
  dimension: Dimension;
  onChange: (bc: BCSpec) => void;
}) {
  return (
    <Card title="Boundary condition">
      <Field label="Type">
        <Select value={bc.type} onChange={(type) => onChange(defaultBc(type))} options={BC_OPTIONS} />
      </Field>

      <div style={{ height: 10 }} />

      {bc.type === "dirichlet" && (
        <Field label="Value" hint="u = value on all edges">
          <NumberField value={bc.value} step={0.1} onChange={(value) => onChange({ ...bc, value })} />
        </Field>
      )}

      {bc.type === "dirichlet_box" && (
        <>
          <FieldRow>
            <Field label={dimension === "1D" ? "Left (x=0)" : "Left edge"}>
              <NumberField value={bc.left} step={0.1} onChange={(left) => onChange({ ...bc, left })} />
            </Field>
            <Field label={dimension === "1D" ? "Right (x=Lx)" : "Right edge"}>
              <NumberField value={bc.right} step={0.1} onChange={(right) => onChange({ ...bc, right })} />
            </Field>
          </FieldRow>
          {dimension === "2D" && (
            <>
              <div style={{ height: 10 }} />
              <FieldRow>
                <Field label="Bottom edge">
                  <NumberField value={bc.bottom} step={0.1} onChange={(bottom) => onChange({ ...bc, bottom })} />
                </Field>
                <Field label="Top edge">
                  <NumberField value={bc.top} step={0.1} onChange={(top) => onChange({ ...bc, top })} />
                </Field>
              </FieldRow>
            </>
          )}
        </>
      )}

      {bc.type === "neumann" && (
        <Field label="Derivative (flux)" hint="∂u/∂n at every edge; 0 = insulated">
          <NumberField value={bc.derivative} step={0.1} onChange={(derivative) => onChange({ ...bc, derivative })} />
        </Field>
      )}

      {bc.type === "periodic" && (
        <p className="field-hint" style={{ margin: 0 }}>
          The domain wraps around; no parameters needed.
        </p>
      )}
    </Card>
  );
}
