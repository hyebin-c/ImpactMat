"""Command line interface.

    python -m impactmat.cli fem      --obj-id 6 [--materials Ceramic Glass]
    python -m impactmat.cli features --obj-id 6
    python -m impactmat.cli synth    --obj-id 6
    python -m impactmat.cli all      --obj-id 6
    python -m impactmat.cli render   --obj-id 6 --material Steel --out steel.wav
    python -m impactmat.cli render   --obj-id 6 --params 7850 2e11 0.29 5 3e-8 --out custom.wav

Common options:  --config configs/default.yaml  --set audio.duration=1.0 hit.index=12
"""
from __future__ import annotations

import argparse

import numpy as np
import scipy.io.wavfile as wavfile

from . import pipeline
from .config import load_config
from .render import render_from_params


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="impactmat", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["fem", "features", "synth", "all", "render"])
    p.add_argument("--obj-id", required=True)
    p.add_argument("--config", default=None, help="YAML config (defaults are used if omitted)")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="config overrides, e.g. audio.duration=1.0")
    p.add_argument("--materials", nargs="*", default=None, help="subset of materials for fem/features/synth/all")
    p.add_argument("--material", default=None, help="render: material name")
    p.add_argument("--params", nargs=5, type=float, metavar=("RHO", "E", "NU", "ALPHA", "BETA"),
                   default=None, help="render: explicit material parameters")
    p.add_argument("--out", default="render.wav", help="render: output wav path")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    cfg = load_config(args.config, args.set)

    if args.command == "render":
        if (args.material is None) == (args.params is None):
            raise SystemExit("render needs exactly one of --material or --params")
        sound = render_from_params(pipeline.load_voxel(cfg, args.obj_id),
                                   args.material if args.material else args.params, cfg=cfg)
        wavfile.write(args.out, cfg.audio.sample_rate, (sound * 32767).astype(np.int16))
        print(f"saved {args.out}")
        return

    if args.command in ("fem", "all"):
        pipeline.run_fem(cfg, args.obj_id, args.materials)
    if args.command in ("features", "all") and cfg.fem.save_modal:
        pipeline.run_features(cfg, args.obj_id, args.materials)
    if args.command in ("synth", "all"):
        pipeline.run_synthesis(cfg, args.obj_id, args.materials)


if __name__ == "__main__":
    main()
