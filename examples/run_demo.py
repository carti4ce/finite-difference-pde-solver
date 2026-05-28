"""Command-line demo runner for heat solver examples."""
import argparse
import sys

from examples.heat_solver import run_explicit, run_cn


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run heat equation demos")
    p.add_argument("--mode", choices=["explicit", "cn"], default="explicit",
                   help="Which integrator to run: explicit or cn (Crank-Nicolson)")
    p.add_argument("--nx", type=int, default=64)
    p.add_argument("--ny", type=int, default=64)
    p.add_argument("--alpha", type=float, default=1e-3)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--steps", type=int, default=100)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.mode == "explicit":
        dt = args.dt if args.dt is not None else 5e-5
        run_explicit(nx=args.nx, ny=args.ny, alpha=args.alpha, dt=dt, steps=args.steps)
    else:
        dt = args.dt if args.dt is not None else 1e-3
        run_cn(nx=args.nx, ny=args.ny, alpha=args.alpha, dt=dt, steps=args.steps)


if __name__ == "__main__":
    main()
