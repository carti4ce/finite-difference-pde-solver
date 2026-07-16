export function NumberField({
  value,
  onChange,
  step,
  min,
  max,
  placeholder,
}: {
  value: number;
  onChange: (v: number) => void;
  step?: number;
  min?: number;
  max?: number;
  placeholder?: string;
}) {
  return (
    <input
      className="input"
      type="number"
      value={Number.isFinite(value) ? value : ""}
      step={step ?? "any"}
      min={min}
      max={max}
      placeholder={placeholder}
      onChange={(e) => {
        const v = e.target.valueAsNumber;
        onChange(Number.isNaN(v) ? 0 : v);
      }}
    />
  );
}
