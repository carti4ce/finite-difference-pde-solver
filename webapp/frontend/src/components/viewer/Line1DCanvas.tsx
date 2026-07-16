import { useEffect, useRef, useState } from "react";

interface Props {
  x: number[];
  y: number[];
  vmin: number;
  vmax: number;
  dark: boolean;
  seriesColor: string;
}

const PAD = { left: 44, right: 16, top: 16, bottom: 28 };

function paddedDomain(vmin: number, vmax: number): [number, number] {
  const span = vmax - vmin || Math.max(Math.abs(vmax), 1);
  const pad = 0.1 * span;
  return [vmin - pad, vmax + pad];
}

export function Line1DCanvas({ x, y, vmin, vmax, dark, seriesColor }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<{ px: number; py: number; x: number; y: number } | null>(null);
  const [size, setSize] = useState({ width: 600, height: 320 });

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const box = entries[0].contentRect;
      setSize({ width: box.width, height: Math.max(240, box.width * 0.5) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size.width * dpr;
    canvas.height = size.height * dpr;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, size.width, size.height);

    const style = getComputedStyle(document.documentElement);
    const ink = dark ? "#c3c2b7" : "#52514e";
    const muted = "#898781";
    const gridline = style.getPropertyValue("--gridline").trim() || "#e1e0d9";

    const plotW = size.width - PAD.left - PAD.right;
    const plotH = size.height - PAD.top - PAD.bottom;
    const [ylo, yhi] = paddedDomain(vmin, vmax);
    const xlo = x[0] ?? 0;
    const xhi = x[x.length - 1] ?? 1;

    const toPx = (xv: number) => PAD.left + ((xv - xlo) / (xhi - xlo || 1)) * plotW;
    const toPy = (yv: number) => PAD.top + plotH - ((yv - ylo) / (yhi - ylo || 1)) * plotH;

    // gridlines + y ticks
    ctx.strokeStyle = gridline;
    ctx.fillStyle = muted;
    ctx.font = "11px system-ui, sans-serif";
    ctx.lineWidth = 1;
    const numTicks = 5;
    for (let i = 0; i < numTicks; i++) {
      const yv = ylo + ((yhi - ylo) * i) / (numTicks - 1);
      const py = toPy(yv);
      ctx.beginPath();
      ctx.moveTo(PAD.left, py);
      ctx.lineTo(size.width - PAD.right, py);
      ctx.stroke();
      ctx.fillText(yv.toPrecision(3), 4, py + 3);
    }
    // x ticks
    for (let i = 0; i < 5; i++) {
      const xv = xlo + ((xhi - xlo) * i) / 4;
      ctx.fillText(xv.toFixed(2), toPx(xv) - 10, size.height - 8);
    }

    // baseline (y=0) if in range
    if (ylo < 0 && yhi > 0) {
      ctx.strokeStyle = style.getPropertyValue("--baseline").trim() || "#c3c2b7";
      ctx.beginPath();
      ctx.moveTo(PAD.left, toPy(0));
      ctx.lineTo(size.width - PAD.right, toPy(0));
      ctx.stroke();
    }

    // line
    ctx.strokeStyle = seriesColor;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.beginPath();
    for (let i = 0; i < x.length; i++) {
      const px = toPx(x[i]);
      const py = toPy(y[i]);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();

    // hover crosshair
    if (hover) {
      ctx.strokeStyle = ink;
      ctx.globalAlpha = 0.4;
      ctx.beginPath();
      ctx.moveTo(hover.px, PAD.top);
      ctx.lineTo(hover.px, PAD.top + plotH);
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.beginPath();
      ctx.arc(hover.px, hover.py, 4, 0, Math.PI * 2);
      ctx.fillStyle = seriesColor;
      ctx.fill();
      ctx.strokeStyle = dark ? "#1a1a19" : "#fcfcfb";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [x, y, vmin, vmax, dark, seriesColor, size, hover?.px]);

  function handleMove(e: React.MouseEvent<HTMLCanvasElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const plotW = size.width - PAD.left - PAD.right;
    const xlo = x[0] ?? 0;
    const xhi = x[x.length - 1] ?? 1;
    const t = Math.max(0, Math.min(1, (mx - PAD.left) / plotW));
    const idx = Math.round(t * (x.length - 1));
    if (idx < 0 || idx >= x.length) return;
    const [ylo, yhi] = paddedDomain(vmin, vmax);
    const plotH = size.height - PAD.top - PAD.bottom;
    const py = PAD.top + plotH - ((y[idx] - ylo) / (yhi - ylo || 1)) * plotH;
    const px = PAD.left + ((x[idx] - xlo) / (xhi - xlo || 1)) * plotW;
    setHover({ px, py, x: x[idx], y: y[idx] });
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
          x = {hover.x.toPrecision(4)}, u = {hover.y.toPrecision(4)}
        </div>
      )}
    </div>
  );
}
