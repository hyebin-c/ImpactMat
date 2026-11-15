"""Step-by-step, file-based pipeline.

Per object, the layout under ``{paths.out_dir}/{obj_id}/`` is:

    voxel.npz                 step 1  voxel grid
    modal/modal_{mat}.npz     step 1  raw FEM eigenpairs (optional, large)
    feat/feat_{mat}.npz       step 2  modal features
    wav/sound_{mat}.wav       step 3  rendered sound
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.io.wavfile as wavfile
from tqdm import tqdm

from .config import Config
from .fem import solve_modes
from .geometry.voxelize import voxelize_mesh
from .modal import ModalFeatures, compute_features
from .synth import synthesize


def object_dir(cfg: Config, obj_id: str | int) -> Path:
    return Path(cfg.paths.out_dir) / str(obj_id)


def _select(cfg: Config, names: Iterable[str] | None) -> dict:
    table = cfg.material_table()
    return table if not names else {n: table[n] for n in names}


def load_voxel(cfg: Config, obj_id: str | int) -> np.ndarray:
    return np.load(object_dir(cfg, obj_id) / "voxel.npz")["voxel"]


def run_fem(cfg: Config, obj_id: str | int, materials: Iterable[str] | None = None) -> None:
    """Step 1: voxelize the mesh and run FEM modal analysis for each material."""
    mesh_path = Path(cfg.paths.mesh_dir) / str(obj_id) / cfg.paths.mesh_file
    out = object_dir(cfg, obj_id)
    out.mkdir(parents=True, exist_ok=True)

    voxel = voxelize_mesh(mesh_path, cfg.voxel.resolution, cfg.voxel.backend)
    np.savez_compressed(out / "voxel.npz", voxel=voxel)             # material-independent

    if cfg.fem.save_modal:
        (out / "modal").mkdir(exist_ok=True)
    for name, mat in tqdm(_select(cfg, materials).items(), desc=f"FEM {obj_id}"):
        eigenvalues, eigenvectors = solve_modes(
            voxel, mat, num_modes=cfg.fem.num_modes, num_rigid_modes=cfg.fem.num_rigid_modes,
            object_length=cfg.voxel.object_length, seed=cfg.fem.seed)
        if cfg.fem.save_modal:
            np.savez_compressed(out / "modal" / f"modal_{name}.npz",
                                eigenvals=eigenvalues, eigenvecs=eigenvectors)
        else:                                                       # features straight from memory
            _save_features(cfg, out, name, eigenvalues, eigenvectors, voxel)


def run_features(cfg: Config, obj_id: str | int, materials: Iterable[str] | None = None) -> None:
    """Step 2: modal features from the saved eigenpairs."""
    out = object_dir(cfg, obj_id)
    voxel = load_voxel(cfg, obj_id)
    for name in tqdm(_select(cfg, materials), desc=f"features {obj_id}"):
        modal = np.load(out / "modal" / f"modal_{name}.npz")
        _save_features(cfg, out, name, modal["eigenvals"], modal["eigenvecs"], voxel)


def _save_features(cfg, out, name, eigenvalues, eigenvectors, voxel) -> None:
    (out / "feat").mkdir(exist_ok=True)
    compute_features(eigenvalues, eigenvectors, voxel,
                     cfg.audio.min_freq, cfg.audio.max_freq).save(out / "feat" / f"feat_{name}.npz")


def run_synthesis(cfg: Config, obj_id: str | int, materials: Iterable[str] | None = None) -> None:
    """Step 3: render one wav per material from the saved features."""
    out = object_dir(cfg, obj_id)
    (out / "wav").mkdir(exist_ok=True)
    for name, mat in tqdm(_select(cfg, materials).items(), desc=f"synthesis {obj_id}"):
        features = ModalFeatures.load(out / "feat" / f"feat_{name}.npz")
        sound = synthesize(features, mat.alpha, mat.beta, hit_index=cfg.hit.index,
                           t_start=cfg.hit.t_start, gain=cfg.hit.gain,
                           duration=cfg.audio.duration, sample_rate=cfg.audio.sample_rate)
        wavfile.write(out / "wav" / f"sound_{name}.wav", cfg.audio.sample_rate,
                      (sound * 32767).astype(np.int16))
