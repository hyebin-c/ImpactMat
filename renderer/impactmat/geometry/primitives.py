"""Procedural voxel shapes, for trying the pipeline without any mesh (and for tests)."""
from __future__ import annotations

import numpy as np


def box(sizes: tuple[int, int, int] = (22, 14, 8), resolution: int = 32) -> np.ndarray:
    """A solid cuboid with the given edge lengths (in voxels), centred in the grid.

    Unequal edges avoid degenerate modes, so the spectrum has clearly separate peaks.
    """
    voxel = np.zeros((resolution,) * 3, dtype=np.int32)
    lo = [(resolution - s) // 2 for s in sizes]
    voxel[tuple(slice(l, l + s) for l, s in zip(lo, sizes))] = 1
    return voxel


def cylinder(radius: float = 9.0, height: int = 18, resolution: int = 32) -> np.ndarray:
    """A solid cylinder with its axis along z."""
    x, y = np.meshgrid(*[np.arange(resolution) - (resolution - 1) / 2] * 2, indexing="ij")
    disk = x ** 2 + y ** 2 <= radius ** 2
    z0 = (resolution - height) // 2
    voxel = np.zeros((resolution,) * 3, dtype=np.int32)
    voxel[:, :, z0:z0 + height] = disk[:, :, None]
    return voxel


def sphere(radius: float = 11.0, resolution: int = 32) -> np.ndarray:
    """A solid ball."""
    c = (resolution - 1) / 2
    x, y, z = np.meshgrid(*[np.arange(resolution) - c] * 3, indexing="ij")
    return (x ** 2 + y ** 2 + z ** 2 <= radius ** 2).astype(np.int32)
