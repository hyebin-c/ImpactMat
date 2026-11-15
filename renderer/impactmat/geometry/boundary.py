"""Voxel-grid topology helpers shared by voxelization and feature extraction.

Everything here is derived from one primitive, :func:`exterior_empty`: the empty
cells that are 6-connected to the outside of the object.
"""
from __future__ import annotations

from itertools import product

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix, csr_matrix

# Face-neighbour offsets; the order defines the bit order of the surface code:
# +z, -z, +y, -y, +x, -x
CUBE_NORMALS = np.array([
    [0, 0, 1], [0, 0, -1],
    [0, 1, 0], [0, -1, 0],
    [1, 0, 0], [-1, 0, 0],
])

_CORNERS = np.array(list(product((0, 1), repeat=3)))       # 8 corner offsets of a voxel


def exterior_empty(solid: np.ndarray) -> np.ndarray:
    """Empty cells connected (6-neighbourhood) to the outside of ``solid``.

    The grid is padded by one cell on every side, so the returned mask has shape
    ``(res + 2,) * 3``; index ``p + 1`` corresponds to voxel ``p``.
    """
    empty = ~np.pad(solid.astype(bool), 1)
    labels, _ = ndimage.label(empty)                        # default structure = 6-connectivity
    return labels == labels[0, 0, 0]


def fill_interior(shell: np.ndarray) -> np.ndarray:
    """Fill the cavity enclosed by a voxelized surface shell (int32 grid of 0/1)."""
    solid = shell != 0
    outside = exterior_empty(solid)[1:-1, 1:-1, 1:-1]
    return (solid | ~outside).astype(np.int32)


def surface_voxels(voxel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the boundary voxels of an occupancy grid.

    Returns:
        coords:  (B, 3) integer coordinates of voxels that touch the outside
        index:   (B,)   position of each boundary voxel in the list of all occupied voxels
        surface: (B, 6) which of the 6 faces (see ``CUBE_NORMALS``) open to the outside
    """
    solid = voxel != 0
    occupied = np.argwhere(solid)                           # (A, 3), raster order
    outside = exterior_empty(solid)
    neighbours = occupied[:, None, :] + 1 + CUBE_NORMALS[None]          # (A, 6, 3) in padded coords
    open_face = outside[neighbours[..., 0], neighbours[..., 1], neighbours[..., 2]]   # (A, 6)
    index = np.flatnonzero(open_face.any(axis=1))
    return occupied[index], index, open_face[index].astype(np.float64)


def vertex_to_voxel_matrix(voxel: np.ndarray, rows: np.ndarray | None = None) -> csr_matrix:
    """Sparse averaging operator from hexahedral-mesh vertices to voxels.

    Vertices are numbered in raster order over the (res+1)^3 corner grid, matching
    :mod:`impactmat.fem.assembler`. Row ``i`` averages the 8 corners of the i-th
    occupied voxel; pass ``rows`` to keep only a subset of voxels.
    """
    res = voxel.shape[0]
    occupied = np.argwhere(voxel != 0)                                   # (A, 3)
    corners = (occupied[:, None, :] + _CORNERS[None]).reshape(-1, 3)     # (8A, 3)
    corner_ix = tuple(corners.T)

    used = np.zeros((res + 1,) * 3, dtype=bool)
    used[corner_ix] = True
    vertex_id = np.cumsum(used.ravel()).reshape(used.shape) - 1          # raster-order numbering
    cols = vertex_id[corner_ix]

    num_voxels = len(occupied)
    mat = coo_matrix((np.full(cols.shape, 1.0 / 8.0),
                      (np.repeat(np.arange(num_voxels), 8), cols)),
                     shape=(num_voxels, int(used.sum()))).tocsr()
    return mat if rows is None else mat[rows]
