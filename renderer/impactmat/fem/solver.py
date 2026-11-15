"""Generalized eigenvalue problem  K u = lambda M u  for the vibration modes."""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from ..materials import Material, MaterialSpec, resolve_material
from . import assembler

Solver = Callable[..., tuple[np.ndarray, np.ndarray]]


def _to_torch_sparse(M) -> torch.Tensor:
    indices = torch.from_numpy(np.vstack((M.row, M.col))).long()
    values = torch.from_numpy(M.data)
    with torch.sparse.check_sparse_tensor_invariants(False):
        return torch.sparse_coo_tensor(indices, values, torch.Size(M.shape)).coalesce()


def lobpcg_solver(num_modes: int = 20, num_rigid_modes: int = 6, seed: int | None = None) -> Solver:
    """LOBPCG for the lowest modes of a symmetric positive-(semi)definite pair (K, M).

    The solve runs in float32, as in the dataset generation. The ``num_rigid_modes``
    lowest eigenpairs (rigid-body motion, ~0 Hz) are computed and then dropped.
    """
    def solve(stiff, mass):
        K = _to_torch_sparse(stiff).float()
        M = _to_torch_sparse(mass).float()
        if seed is not None:
            torch.manual_seed(seed)
        vals, vecs = torch.lobpcg(K, num_modes + num_rigid_modes, M, tracker=None, largest=False)
        return vals.cpu().numpy()[num_rigid_modes:], vecs.cpu().numpy()[:, num_rigid_modes:]
    return solve


def solve_modes(voxel: np.ndarray, material: MaterialSpec, *, num_modes: int = 20,
                num_rigid_modes: int = 6, object_length: float = 0.15,
                seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Vibration modes of a voxel object.

    Only ``rho``, ``E`` and ``nu`` affect the modes; the damping terms are applied
    later, at synthesis time.

    Returns:
        eigenvalues  (k,)     omega^2 of each mode [rad^2/s^2]
        eigenvectors (3N, k)  per-vertex displacement of each mode (3 DOFs per vertex)
    """
    mat: Material = resolve_material(material)
    voxel = np.asarray(voxel)
    length = object_length / voxel.shape[-1]                    # edge length of one voxel
    stiff = assembler.stiff_matrix(voxel, mat.E, mat.nu, length)
    mass = assembler.mass_matrix(voxel, mat.rho, length)
    return lobpcg_solver(num_modes, num_rigid_modes, seed)(stiff, mass)
