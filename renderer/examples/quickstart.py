"""Quick start: hear how material parameters change an impact sound.

    uv run python examples/quickstart.py            # writes wavs to outputs/quickstart/
    uv run python examples/quickstart.py --voxel outputs/6/voxel.npz     # your own object

No mesh needed: a procedural box is used unless --voxel is given.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile

from impactmat import DEFAULT_MATERIALS, Config, modal_features, resolve_material
from impactmat.geometry.primitives import box
from impactmat.synth import synthesize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voxel", default=None, help="voxel.npz of your own object (default: a procedural box)")
    ap.add_argument("--out", default="outputs/quickstart")
    ap.add_argument("--hit", type=int, default=0, help="strike point (index of a boundary voxel)")
    args = ap.parse_args()

    cfg = Config()
    cfg.voxel.object_length = 0.4      # a 40 cm object: pitches stay well inside the audible range
    voxel = np.load(args.voxel)["voxel"] if args.voxel else box()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def save(name: str, sound: np.ndarray) -> None:
        wavfile.write(out / f"{name}.wav", cfg.audio.sample_rate, (sound * 32767).astype(np.int16))

    def render(features, mat, **kw):
        return synthesize(features, mat.alpha, mat.beta, hit_index=args.hit,
                          duration=cfg.audio.duration, sample_rate=cfg.audio.sample_rate, **kw)

    # 1. Same object, the 8 base materials ---------------------------------------------------
    print("1) base materials (lowest mode frequency):")
    modes = {}
    for name, mat in DEFAULT_MATERIALS.items():
        modes[name] = modal_features(voxel, mat, cfg)
        save(f"material_{name}", render(modes[name], mat))
        print(f"   {name:<14} f1 = {modes[name].freqs[0]:8.1f} Hz")

    # 2. Young's modulus E -> pitch (f ~ sqrt(E / rho)) -----------------------------------------
    print("2) Young's modulus E scales the pitch (4x E -> 2x frequency):")
    base = DEFAULT_MATERIALS["Steel"]
    for scale in (0.25, 1.0, 4.0):
        mat = resolve_material({**base.__dict__, "E": base.E * scale})
        features = modes["Steel"] if scale == 1.0 else modal_features(voxel, mat, cfg)   # reuse step 1
        save(f"E_x{scale}", render(features, mat))
        print(f"   E x{scale:<5} f1 = {features.freqs[0]:8.1f} Hz")

    # 3. Damping (alpha, beta) -> decay. Modes do not depend on damping, so reuse the features.
    print("3) damping controls how fast the sound dies (modes are reused, no new FEM):")
    features = modes["Steel"]
    for label, alpha, beta in [("light", 1.0, 3e-9), ("medium", 20.0, 3e-8), ("heavy", 200.0, 3e-7)]:
        save(f"damping_{label}", synthesize(features, alpha, beta, hit_index=args.hit,
                                            duration=cfg.audio.duration, sample_rate=cfg.audio.sample_rate))
        print(f"   {label:<7} alpha={alpha:<6g} beta={beta:g}")

    print(f"\nwrote {len(list(out.glob('*.wav')))} wav files to {out.resolve()}")


if __name__ == "__main__":
    main()
