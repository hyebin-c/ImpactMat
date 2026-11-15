"""Material parameters + object -> impact sound, entirely in memory."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .config import Config
from .fem import solve_modes
from .materials import MaterialSpec, resolve_material
from .modal import ModalFeatures, compute_features
from .synth import synthesize


def modal_features(voxel: np.ndarray, material: MaterialSpec, cfg: Config | None = None) -> ModalFeatures:
    """FEM modal analysis of ``voxel`` made of ``material`` -> modal features."""
    cfg = cfg or Config()
    mat = resolve_material(material, cfg.material_table())
    eigenvalues, eigenvectors = solve_modes(
        voxel, mat, num_modes=cfg.fem.num_modes, num_rigid_modes=cfg.fem.num_rigid_modes,
        object_length=cfg.voxel.object_length, seed=cfg.fem.seed)
    return compute_features(eigenvalues, eigenvectors, voxel, cfg.audio.min_freq, cfg.audio.max_freq)


def render_from_params(voxel: np.ndarray, material: MaterialSpec, *, hit_index: int | None = None,
                       t_start: float | Sequence[float] | None = None, gain: float | None = None,
                       cfg: Config | None = None) -> np.ndarray:
    """Render the impact sound of an object with the given material parameters.

    ``material`` is a material name, a (rho, E, nu, alpha, beta) sequence, a dict
    with those keys, or a :class:`~impactmat.materials.Material`. Strike settings
    default to ``cfg.hit``.
    """
    cfg = cfg or Config()
    mat = resolve_material(material, cfg.material_table())
    features = modal_features(voxel, mat, cfg)
    return synthesize(
        features, mat.alpha, mat.beta,
        hit_index=cfg.hit.index if hit_index is None else hit_index,
        t_start=cfg.hit.t_start if t_start is None else t_start,
        gain=cfg.hit.gain if gain is None else gain,
        duration=cfg.audio.duration, sample_rate=cfg.audio.sample_rate)
