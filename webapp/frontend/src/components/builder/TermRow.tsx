import type { EditableTerm } from "../../builderState";
import { TERM_LABELS } from "../../builderState";
import type { Dimension } from "../../api/types";
import { Field, FieldRow } from "../ui/Field";
import { NumberField } from "../ui/NumberField";
import { Select } from "../ui/Select";
import { Button } from "../ui/Button";

const EXPRESSION_HINT = "Variables: x (, y in 2D), u for nonlinear terms. Functions: sin cos tan exp log sqrt abs tanh min max where. Use ** for power, pi and e are available.";

export function TermRow({
  term,
  dimension,
  onChange,
  onRemove,
  removable,
}: {
  term: EditableTerm;
  dimension: Dimension;
  onChange: (term: EditableTerm) => void;
  onRemove: () => void;
  removable: boolean;
}) {
  return (
    <div className="term-row">
      <div className="term-row-head">
        <span className="term-badge">{TERM_LABELS[term.type]}</span>
        <Button variant="ghost" icon type="button" onClick={onRemove} disabled={!removable} aria-label="Remove term">
          ✕
        </Button>
      </div>

      {term.type === "diffusion" && (
        <Field label="Coefficient (D)" hint="D · ∇²u">
          <NumberField value={term.coeff} step={0.01} onChange={(coeff) => onChange({ ...term, coeff })} />
        </Field>
      )}

      {term.type === "reaction" && (
        <Field label="Coefficient (k)" hint="k · u">
          <NumberField value={term.coeff} step={0.1} onChange={(coeff) => onChange({ ...term, coeff })} />
        </Field>
      )}

      {term.type === "advection" && (
        <>
          {dimension === "1D" ? (
            <Field label="Velocity (v)" hint="-v · uₓ">
              <NumberField
                value={typeof term.velocity === "number" ? term.velocity : term.velocity[0]}
                step={0.1}
                onChange={(v) => onChange({ ...term, velocity: v })}
              />
            </Field>
          ) : (
            <FieldRow>
              <Field label="Velocity x">
                <NumberField
                  value={typeof term.velocity === "number" ? term.velocity : term.velocity[0]}
                  step={0.1}
                  onChange={(vx) => {
                    const vy = typeof term.velocity === "number" ? term.velocity : term.velocity[1];
                    onChange({ ...term, velocity: [vx, vy] });
                  }}
                />
              </Field>
              <Field label="Velocity y">
                <NumberField
                  value={typeof term.velocity === "number" ? term.velocity : term.velocity[1]}
                  step={0.1}
                  onChange={(vy) => {
                    const vx = typeof term.velocity === "number" ? term.velocity : term.velocity[0];
                    onChange({ ...term, velocity: [vx, vy] });
                  }}
                />
              </Field>
            </FieldRow>
          )}
          <Field label="Scheme">
            <Select
              value={term.scheme}
              onChange={(scheme) => onChange({ ...term, scheme })}
              options={[
                { value: "upwind", label: "Upwind (stable, diffusive)" },
                { value: "central", label: "Central (2nd order, can oscillate)" },
              ]}
            />
          </Field>
        </>
      )}

      {term.type === "source" && (
        <Field label="Source expression s(x, y)" hint={EXPRESSION_HINT}>
          <textarea
            className="textarea mono"
            rows={2}
            value={term.expression}
            onChange={(e) => onChange({ ...term, expression: e.target.value })}
          />
        </Field>
      )}

      {term.type === "function" && (
        <Field label="Nonlinear expression g(u, x, y)" hint={EXPRESSION_HINT}>
          <textarea
            className="textarea mono"
            rows={2}
            value={term.expression}
            onChange={(e) => onChange({ ...term, expression: e.target.value })}
          />
        </Field>
      )}
    </div>
  );
}
