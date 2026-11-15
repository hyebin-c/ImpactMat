"""Modal features: natural frequencies and boundary mode shapes of an object."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..geometry import surface_voxels, vertex_to_voxel_matrix


@dataclass
class ModalFeatures:
    freqs: np.ndarray       # (k,)       natural frequencies [Hz]
    feats_in: np.ndarray    # (B, 3, k)  displacement of each mode at the B boundary voxels
    coords: np.ndarray      # (B, 3)     boundary-voxel coordinates
    surface: np.ndarray     # (B, 6)     faces open to the outside (+z,-z,+y,-y,+x,-x)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, coords=self.coords, feats_in=self.feats_in,
                            surface=self.surface, freqs=self.freqs)

    @classmethod
    def load(cls, path: str | Path) -> "ModalFeatures":
        d = np.load(path)
        return cls(freqs=d["freqs"], feats_in=d["feats_in"], coords=d["coords"], surface=d["surface"])


def eigenvalues_to_freqs(eigenvalues: np.ndarray, min_freq: float = 20.0,
                         max_freq: float = 20000.0) -> np.ndarray:
    """omega^2 -> frequency in Hz, clipped to the audible range.

    Modes outside [min_freq, max_freq] are clamped to the nearest bound rather than
    dropped; this matches how the ImpactMat dataset was generated.
    """
    return np.clip(eigenvalues ** 0.5 / (2 * np.pi), min_freq, max_freq)


def compute_features(eigenvalues: np.ndarray, eigenvectors: np.ndarray, voxel: np.ndarray,
                     min_freq: float = 20.0, max_freq: float = 20000.0) -> ModalFeatures:
    """Turn raw FEM eigenpairs into the features used for sound synthesis.

    Mode shapes live on mesh vertices; they are averaged onto voxels and kept only
    on the boundary voxels, where an object can be struck.
    """
    freqs = eigenvalues_to_freqs(eigenvalues, min_freq, max_freq)
    coords, index, surface = surface_voxels(voxel)

    num_vertices = eigenvectors.shape[0] // 3
    num_modes = len(freqs)
    to_voxel = vertex_to_voxel_matrix(voxel, rows=index)                        # (B, N)
    feats_in = (to_voxel @ eigenvectors.reshape(num_vertices, 3 * num_modes))   # (B, 3k)
    return ModalFeatures(freqs=freqs,
                         feats_in=feats_in.reshape(len(index), 3, num_modes),
                         coords=coords, surface=surface)
