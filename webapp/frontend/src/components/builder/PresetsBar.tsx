import { PRESETS } from "../../builderState";
import type { BuilderState } from "../../builderState";

export function PresetsBar({ onSelect }: { onSelect: (state: BuilderState) => void }) {
  return (
    <div className="preset-row">
      {PRESETS.map((preset) => (
        <button
          key={preset.name}
          type="button"
          className="preset-chip"
          title={preset.description}
          onClick={() => onSelect(preset.build())}
        >
          {preset.name}
        </button>
      ))}
    </div>
  );
}
