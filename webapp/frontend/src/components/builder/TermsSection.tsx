import { useState } from "react";
import type { EditableTerm } from "../../builderState";
import { defaultTerm, TERM_TYPE_OPTIONS } from "../../builderState";
import type { Dimension, TermType } from "../../api/types";
import { Card } from "../ui/Card";
import { Select } from "../ui/Select";
import { Button } from "../ui/Button";
import { TermRow } from "./TermRow";

const MAX_TERMS = 8;

export function TermsSection({
  terms,
  dimension,
  onChange,
}: {
  terms: EditableTerm[];
  dimension: Dimension;
  onChange: (terms: EditableTerm[]) => void;
}) {
  const [nextType, setNextType] = useState<TermType>("diffusion");

  return (
    <Card
      title="Terms"
      action={
        <span className="field-hint">
          f(u) = {terms.length ? terms.map((_, i) => `T${i + 1}`).join(" + ") : "0"}
        </span>
      }
    >
      <div className="term-list">
        {terms.map((term) => (
          <TermRow
            key={term.id}
            term={term}
            dimension={dimension}
            removable={terms.length > 1}
            onChange={(updated) => onChange(terms.map((t) => (t.id === term.id ? updated : t)))}
            onRemove={() => onChange(terms.filter((t) => t.id !== term.id))}
          />
        ))}
      </div>

      <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
        <Select value={nextType} onChange={setNextType} options={TERM_TYPE_OPTIONS} />
        <Button
          type="button"
          disabled={terms.length >= MAX_TERMS}
          onClick={() => onChange([...terms, defaultTerm(nextType)])}
        >
          + Add term
        </Button>
      </div>
    </Card>
  );
}
