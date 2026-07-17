// Colormaps built from the documented palette (dataviz skill's palette.md).
// Sequential: single blue hue, light -> dark, for all-positive (or all-negative)
// fields. Diverging: blue <-> neutral gray <-> red, for fields that cross zero.
// Anchors are the exact documented hex steps; values between anchors are
// linearly interpolated in sRGB.

export type Rgb = [number, number, number];

function hexToRgb(hex: string): Rgb {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function lerpRgb(a: Rgb, b: Rgb, t: number): Rgb {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

function ramp(stops: string[], t: number): Rgb {
  const clamped = Math.max(0, Math.min(1, t));
  const scaled = clamped * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(scaled));
  return lerpRgb(hexToRgb(stops[i]), hexToRgb(stops[i + 1]), scaled - i);
}

const SEQUENTIAL_LIGHT = [
  "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
  "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
];
const SEQUENTIAL_DARK = SEQUENTIAL_LIGHT; // same ramp; anchored by usage, not surface

/** Sequential colormap for magnitude-only data (0 = lightest, 1 = darkest). */
export function sequentialColor(t: number, dark: boolean): Rgb {
  return ramp(dark ? SEQUENTIAL_DARK : SEQUENTIAL_LIGHT, t);
}

/** Diverging colormap: t in [-1, 1], 0 = neutral midpoint. */
export function divergingColor(t: number, dark: boolean): Rgb {
  const neg = dark ? "#3987e5" : "#256abf";
  const mid = dark ? "#383835" : "#f0efec";
  const pos = dark ? "#e66767" : "#e34948";
  const clamped = Math.max(-1, Math.min(1, t));
  if (clamped < 0) return lerpRgb(hexToRgb(mid), hexToRgb(neg), -clamped);
  return lerpRgb(hexToRgb(mid), hexToRgb(pos), clamped);
}

export interface ColorScale {
  /** Map a raw data value to an "rgb(r, g, b)" CSS color string. */
  toCss(value: number): string;
  toRgb(value: number): Rgb;
  readonly kind: "sequential" | "diverging";
  readonly domain: [number, number];
}

/** Pick sequential vs diverging based on whether the data crosses zero. */
export function makeColorScale(vmin: number, vmax: number, dark: boolean): ColorScale {
  const diverging = vmin < 0 && vmax > 0;
  const span = vmax - vmin || 1;
  const toRgb = diverging
    ? (v: number) => divergingColor(v / Math.max(Math.abs(vmin), Math.abs(vmax), 1e-12), dark)
    : (v: number) => sequentialColor((v - vmin) / span, dark);
  return {
    kind: diverging ? "diverging" : "sequential",
    domain: [vmin, vmax],
    toRgb,
    toCss: (v: number) => {
      const [r, g, b] = toRgb(v);
      return `rgb(${r | 0}, ${g | 0}, ${b | 0})`;
    },
  };
}

/** CSS linear-gradient() string for a colorbar legend, sampled across the domain. */
export function gradientCss(scale: ColorScale, steps = 16): string {
  const [lo, hi] = scale.domain;
  const stops: string[] = [];
  for (let i = 0; i <= steps; i++) {
    const v = lo + ((hi - lo) * i) / steps;
    stops.push(`${scale.toCss(v)} ${(i / steps) * 100}%`);
  }
  return `linear-gradient(to right, ${stops.join(", ")})`;
}
