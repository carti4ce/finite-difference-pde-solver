import { useEffect, useRef, useState } from "react";
import type { ColorScale } from "./colormap";

interface Props {
  grid: number[][]; // grid[i][j], i over x (nx), j over y (ny)
  lx: number;
  ly: number;
  scale: ColorScale;
}

const PAD = { left: 8, right: 8, top: 8, bottom: 8 };

export function Heatmap2DCanvas({ grid, lx, ly, scale }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 600, height: 480 });
  const [hover, setHover] = useState<{ px: number; py: number; x: number; y: number; u: number } | null>(null);

  const nx = grid.length;
  const ny = grid[0]?.length ?? 1;

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const box = entries[0].contentRect;
      setSize({ width: box.width, height: box.width * (ly / lx || 1) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [lx, ly]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Render the field at native grid resolution, then upscale — crisp cells,
    // cheap to redraw every frame.
    const off = document.createElement("canvas");
    off.width = nx;
    off.height = ny;
    const offCtx = off.getContext("2d");
    if (!offCtx) return;
    const img = offCtx.createImageData(nx, ny);
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < ny; j++) {
        const value = grid[i][j];
        const [r, g, b] = scale.toRgb(value);
        // flip vertically: j=0 (y=0) should be at the BOTTOM (origin='lower')
        const row = ny - 1 - j;
        const idx = (row * nx + i) * 4;
        img.data[idx] = r;
        img.data[idx + 1] = g;
        img.data[idx + 2] = b;
        img.data[idx + 3] = 255;
      }
    }
    offCtx.putImageData(img, 0, 0);

    const dpr = window.devicePixelRatio || 1;
    canvas.width = size.width * dpr;
    canvas.height = size.height * dpr;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.imageSmoothingEnabled = true;
    ctx.clearRect(0, 0, size.width, size.height);
    ctx.drawImage(
      off, 0, 0, nx, ny,
      PAD.left, PAD.top, size.width - PAD.left - PAD.right, size.height - PAD.top - PAD.bottom
    );
  }, [grid, nx, ny, scale, size]);

  function handleMove(e: React.MouseEvent<HTMLCanvasElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const plotW = size.width - PAD.left - PAD.right;
    const plotH = size.height - PAD.top - PAD.bottom;
    const tx = (px - PAD.left) / plotW;
    const ty = (py - PAD.top) / plotH;
    if (tx < 0 || tx > 1 || ty < 0 || ty > 1) {
      setHover(null);
      return;
    }
    const i = Math.min(nx - 1, Math.max(0, Math.floor(tx * nx)));
    const j = Math.min(ny - 1, Math.max(0, Math.floor((1 - ty) * ny)));
    setHover({ px, py, x: (i / (nx - 1 || 1)) * lx, y: (j / (ny - 1 || 1)) * ly, u: grid[i][j] });
  }

  return (
    <div ref={wrapRef} className="canvas-wrap" onMouseLeave={() => setHover(null)}>
      <canvas
        ref={canvasRef}
        style={{ width: size.width, height: size.height }}
        onMouseMove={handleMove}
      />
      {hover && (
        <div className="hover-tooltip" style={{ left: hover.px, top: hover.py }}>
          x = {hover.x.toPrecision(3)}, y = {hover.y.toPrecision(3)}, u = {hover.u.toPrecision(4)}
        </div>
      )}
    </div>
  );
}
