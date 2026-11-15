# ImpactMat

Modal impact-sound rendering from continuous material parameters
(density ρ, Young's modulus E, Poisson's ratio ν, Rayleigh damping α, β).

An object is voxelized, its vibration modes are computed with a hexahedral FEM,
and the sound of a strike is synthesized as a sum of damped sinusoids whose
decay depends on the material.

```
mesh ──voxelize──▶ voxel ──FEM (ρ, E, ν)──▶ modes ──features──▶ (freqs, mode shapes)
                                                                      │
                                                  damping (α, β), strike point
                                                                      ▼
                                                                   sound.wav
```

## Setup

We use [uv](https://docs.astral.sh/uv/) to manage the environment.

```bash
# 1. install uv (once)
curl -LsSf https://astral.sh/uv/install.sh | sh                 # macOS / Linux
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"      # Windows

# 2. clone and create the environment (Python 3.10+, installed automatically)
git clone https://github.com/<org>/impactmat.git
cd impactmat
uv sync
```

`uv sync` creates `.venv/` and installs the locked dependencies (PyTorch is large, so the first
install takes a few minutes; later ones reuse uv's cache). Run anything inside it with `uv run`,
e.g. `uv run python -m impactmat.cli --help`. If uv warns about hardlinks, set `UV_LINK_MODE=copy`.

Everything runs on CPU, including mesh voxelization (numba). Optionally, the original CUDA
voxelizer is available as a second backend; it gives the same grids but needs a GPU and the CUDA
toolkit at install time:

```bash
uv sync --extra cuda        # adds pycuda; then use --set voxel.backend=cuda
```

## Quick start

Hear how material parameters change an impact sound, on a built-in box:

```bash
uv run python examples/quickstart.py
```

It takes a few minutes on CPU (about 10 FEM solves; the first run also compiles numba kernels) and
writes 14 wav files to `outputs/quickstart/`:

| Files | What to listen for |
|---|---|
| `material_<name>.wav` (8 base materials) | the same object sounds different for each material |
| `E_x0.25.wav`, `E_x1.0.wav`, `E_x4.0.wav` | Young's modulus E sets the pitch (frequencies scale with √(E/ρ)) |
| `damping_light/medium/heavy.wav` | α and β set how quickly the ringing dies out |

The same thing from Python:

```python
import numpy as np
from impactmat import render_from_params
from impactmat.geometry.primitives import box          # or np.load("voxel.npz")["voxel"]

voxel = box()
wav = render_from_params(voxel, "Steel")                              # a named material
wav = render_from_params(voxel, (7850, 2.0e11, 0.29, 5, 3e-8))        # rho, E, nu, alpha, beta
wav = render_from_params(voxel, "Glass", hit_index=40, t_start=0.1)   # choose the strike point
```

`wav` is a float array in [-1, 1] at 44.1 kHz.

### Using your own mesh

Place an OBJ at `data/meshes/<obj_id>/model.obj` (meshes are not distributed with this
repository), then:

```bash
uv run python -m impactmat.cli all    --obj-id 6                                  # voxelize, FEM, features, wav
uv run python -m impactmat.cli render --obj-id 6 --material Steel --out steel.wav
uv run python -m impactmat.cli render --obj-id 6 --params 7850 2e11 0.29 5 3e-8 --out custom.wav
uv run python -m impactmat.cli all    --obj-id 6 --set audio.duration=1.0 hit.index=12
```

Results are written to `outputs/<obj_id>/` (`voxel.npz`, `modal/`, `feat/`, `wav/`).

## Configuration

Every setting lives in [`configs/default.yaml`](configs/default.yaml): paths, voxel resolution,
number of modes, audio, the strike point, and extra materials. Pass your own file with
`--config my.yaml` (only the values you change are needed) and override single values with
`--set section.key=value`.

The eigen-solver starts from random vectors, so repeated runs differ by tiny amounts; set
`fem.seed` (e.g. `--set fem.seed=0`) for bit-exact repeatability.

Materials are `(ρ [kg/m³], E [Pa], ν, α [1/s], β [s])`. A mode with angular frequency ω decays as
exp(−d t) with d = (α + β ω²) / 2.

## Layout

```
impactmat/
├── config.py        dataclass config + YAML loader
├── materials.py     Material, the 8 base materials
├── geometry/        OBJ loading, voxelizer (CPU / CUDA), boundary-voxel utilities, procedural shapes
├── fem/             stiffness/mass assembly, LOBPCG mode solver
├── modal/           frequencies and boundary mode shapes
├── synth/           modal synthesis
├── render.py        parameters + voxel -> waveform
├── pipeline.py      file-based steps (voxel/, modal/, feat/, wav/)
└── cli.py
examples/quickstart.py
configs/default.yaml
```

## Acknowledgements

The voxelizer (triangle–box test, CPU and CUDA), the hexahedral FEM assembly and the voxel/boundary
processing are based on [NeuralSound](https://github.com/hellojxt/NeuralSound)
(Jin et al., SIGGRAPH 2022), and have been modified for this project.
