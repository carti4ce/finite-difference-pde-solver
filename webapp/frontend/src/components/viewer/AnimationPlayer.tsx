import { useEffect, useMemo, useState } from "react";
import type { SolveResponse } from "../../api/types";
import { Button } from "../ui/Button";
import { Select } from "../ui/Select";
import { Line1DCanvas } from "./Line1DCanvas";
import { Heatmap2DCanvas } from "./Heatmap2DCanvas";
import { gradientCss, makeColorScale } from "./colormap";

const SPEED_OPTIONS = ["0.25", "0.5", "1", "2", "4"] as const;

export function AnimationPlayer({ response, dark }: { response: SolveResponse; dark: boolean }) {
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(response.times.length > 1);
  const [speed, setSpeed] = useState<(typeof SPEED_OPTIONS)[number]>("1");

  const numFrames = response.times.length;
  const isSteady = numFrames <= 1;

  useEffect(() => {
    setFrame(0);
    setPlaying(response.times.length > 1);
  }, [response]);

  useEffect(() => {
    if (!playing || isSteady) return;
    // setInterval rather than requestAnimationFrame: rAF stops entirely in
    // hidden/background tabs, which would freeze the loop state.
    const framesPerSecond = 8 * parseFloat(speed);
    const id = setInterval(() => setFrame((f) => (f + 1) % numFrames), 1000 / framesPerSecond);
    return () => clearInterval(id);
  }, [playing, speed, numFrames, isSteady]);

  const scale = useMemo(
    () => makeColorScale(response.vmin, response.vmax, dark),
    [response.vmin, response.vmax, dark]
  );

  const x1d = useMemo(
    () => (response.dimension === "1D" ? Array.from({ length: response.nx }, (_, i) => (i + 1) * response.dx) : []),
    [response.dimension, response.nx, response.dx]
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div className="viewer-toolbar">
        <div className="viewer-stats">
          <div className="stat">
            <span className="stat-label">Time</span>
            <span className="stat-value">{response.times[frame]?.toPrecision(4)}</span>
          </div>
          <div className="stat">
            <span className="stat-label">Range</span>
            <span className="stat-value">
              [{response.vmin.toPrecision(3)}, {response.vmax.toPrecision(3)}]
            </span>
          </div>
          <div className="stat">
            <span className="stat-label">Grid</span>
            <span className="stat-value">
              {response.nx}
              {response.dimension === "2D" ? ` × ${response.ny}` : ""}
            </span>
          </div>
        </div>
        {!isSteady && (
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span className="field-hint">Speed</span>
            <Select value={speed} onChange={setSpeed} options={SPEED_OPTIONS.map((s) => ({ value: s, label: `${s}×` }))} />
          </div>
        )}
      </div>

      {response.dimension === "1D" ? (
        <Line1DCanvas
          x={x1d}
          y={response.history[frame] as number[]}
          vmin={response.vmin}
          vmax={response.vmax}
          dark={dark}
          seriesColor="var(--series-1)"
        />
      ) : (
        <>
          <Heatmap2DCanvas
            grid={response.history[frame] as number[][]}
            lx={response.lx}
            ly={response.ly}
            scale={scale}
          />
          <div className="colorbar">
            <span>{response.vmin.toPrecision(3)}</span>
            <div className="colorbar-gradient" style={{ background: gradientCss(scale) }} />
            <span>{response.vmax.toPrecision(3)}</span>
          </div>
        </>
      )}

      {!isSteady && (
        <div className="playback-bar">
          <Button variant="ghost" icon type="button" onClick={() => setPlaying((p) => !p)} aria-label={playing ? "Pause" : "Play"}>
            {playing ? "⏸" : "▶"}
          </Button>
          <input
            className="playback-slider"
            type="range"
            min={0}
            max={numFrames - 1}
            value={frame}
            onChange={(e) => {
              setPlaying(false);
              setFrame(Number(e.target.value));
            }}
          />
          <span className="frame-readout">
            frame {frame + 1} / {numFrames}
          </span>
        </div>
      )}
    </div>
  );
}
