"""Stiffness / mass matrix assembly for a voxel grid of trilinear hexahedra.

Vertices are numbered in raster order over the (res+1)^3 corner grid and each
vertex carries 3 displacement DOFs, so DOF index = 3 * vertex + axis.
"""
from __future__ import annotations

import numpy as np
from numba import njit
from scipy.sparse import coo_matrix

# Corner order of a hexahedron (matches the shape-function layout below)
_CORNERS = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
])


def stiff_matrix(voxel: np.ndarray, youngs: float, poisson: float, length: float) -> coo_matrix:
    """Global stiffness matrix K. ``length`` is the edge length of one voxel [m]."""
    return _assemble(voxel, _stiff_element_matrix(youngs, poisson, length))


def mass_matrix(voxel: np.ndarray, density: float, length: float) -> coo_matrix:
    """Global (consistent) mass matrix M. ``length`` is the edge length of one voxel [m]."""
    return _assemble(voxel, _mass_element_matrix(density, length))


def _assemble(voxel: np.ndarray, element_matrix: np.ndarray) -> coo_matrix:
    values, rows, cols, m, n = _assemble_triplets(voxel, element_matrix, _CORNERS)
    return coo_matrix((values, (rows, cols)), shape=(m, n))


@njit(cache=True)
def _assemble_triplets(voxel, element_matrix, corners):
    voxel = voxel.copy()
    res = voxel.shape[-1]

    # number the occupied hexahedra and mark every vertex they touch
    vertex = np.zeros((res + 1, res + 1, res + 1))
    hex_num = 0
    for i in range(res):
        for j in range(res):
            for k in range(res):
                if voxel[i, j, k] == 1:
                    for c_idx in range(8):
                        c = corners[c_idx]
                        vertex[i + c[0], j + c[1], k + c[2]] = 1
                    hex_num += 1
                    voxel[i, j, k] = hex_num

    # number the vertices in raster order
    vertex_num = 0
    for i in range(res + 1):
        for j in range(res + 1):
            for k in range(res + 1):
                if vertex[i, j, k] == 1:
                    vertex[i, j, k] = vertex_num
                    vertex_num += 1
    m = n = 3 * vertex_num

    single_element_num = 0
    for i in range(24):
        for j in range(24):
            if element_matrix[i, j] != 0:
                single_element_num += 1

    element_num = hex_num * single_element_num
    values = np.zeros(element_num, dtype=np.float64)
    rows = np.zeros(element_num, dtype=np.int32)
    cols = np.zeros(element_num, dtype=np.int32)

    for i in range(res):
        for j in range(res):
            for k in range(res):
                if voxel[i, j, k] > 0:
                    idx = 0
                    offset = int(voxel[i, j, k] - 1) * single_element_num
                    for i_ in range(24):
                        for j_ in range(24):
                            if element_matrix[i_, j_] != 0:
                                c = corners[i_ // 3]
                                rows[offset + idx] = vertex[i + c[0], j + c[1], k + c[2]] * 3 + i_ % 3
                                c = corners[j_ // 3]
                                cols[offset + idx] = vertex[i + c[0], j + c[1], k + c[2]] * 3 + j_ % 3
                                values[offset + idx] = element_matrix[i_, j_]
                                idx += 1
    return values, rows, cols, m, n


def _mass_element_matrix(density: float, length: float) -> np.ndarray:
    M = density * length ** 3 / 216 * np.array([
        [4, 4, 2, 4, 4, 2, 1, 2],
        [0, 4, 4, 2, 2, 4, 2, 1],
        [0, 0, 4, 4, 1, 2, 4, 2],
        [0, 0, 0, 4, 2, 1, 2, 4],
        [0, 0, 0, 0, 4, 4, 2, 4],
        [0, 0, 0, 0, 0, 4, 4, 2],
        [0, 0, 0, 0, 0, 0, 4, 4],
        [0, 0, 0, 0, 0, 0, 0, 4],
    ])
    M = M + M.T
    M = M[..., np.newaxis, np.newaxis] * np.eye(3)               # one copy per displacement axis
    return np.transpose(M, [0, 2, 1, 3]).reshape(24, 24)


def _stiff_element_matrix(youngs: float, poisson: float, length: float) -> np.ndarray:
    E, nu = youngs, poisson
    C = E / ((1 + nu) * (1 - 2 * nu)) * np.array([               # isotropic elasticity (Voigt)
        [1 - nu, nu, nu, 0, 0, 0],
        [nu, 1 - nu, nu, 0, 0, 0],
        [nu, nu, 1 - nu, 0, 0, 0],
        [0, 0, 0, (1 - 2 * nu) / 2, 0, 0],
        [0, 0, 0, 0, (1 - 2 * nu) / 2, 0],
        [0, 0, 0, 0, 0, (1 - 2 * nu) / 2],
    ])
    gauss = [-1 / 3 ** 0.5, 1 / 3 ** 0.5]
    h = length / 2
    coordinates = np.array([
        [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
        [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h],
    ])
    K = np.zeros((24, 24))
    for xi1 in gauss:
        for xi2 in gauss:
            for xi3 in gauss:
                d_shape = (1 / 8) * np.array([
                    [-(1 - xi2) * (1 - xi3), (1 - xi2) * (1 - xi3),
                     (1 + xi2) * (1 - xi3), -(1 + xi2) * (1 - xi3),
                     -(1 - xi2) * (1 + xi3), (1 - xi2) * (1 + xi3),
                     (1 + xi2) * (1 + xi3), -(1 + xi2) * (1 + xi3)],
                    [-(1 - xi1) * (1 - xi3), -(1 + xi1) * (1 - xi3),
                     (1 + xi1) * (1 - xi3), (1 - xi1) * (1 - xi3),
                     -(1 - xi1) * (1 + xi3), -(1 + xi1) * (1 + xi3),
                     (1 + xi1) * (1 + xi3), (1 - xi1) * (1 + xi3)],
                    [-(1 - xi1) * (1 - xi2), -(1 + xi1) * (1 - xi2),
                     -(1 + xi1) * (1 + xi2), -(1 - xi1) * (1 + xi2),
                     (1 - xi1) * (1 - xi2), (1 + xi1) * (1 - xi2),
                     (1 + xi1) * (1 + xi2), (1 - xi1) * (1 + xi2)],
                ])
                jacobian = d_shape @ coordinates
                grad = np.linalg.inv(jacobian) @ d_shape          # (3, 8) shape-function gradients

                B = np.zeros((6, 24))                              # strain-displacement matrix
                for axis in range(3):
                    B[axis, axis::3] = grad[axis]                  # normal strains
                B[3, 0::3], B[3, 1::3] = grad[1], grad[0]          # xy shear
                B[4, 2::3], B[4, 1::3] = grad[1], grad[2]          # yz shear
                B[5, 0::3], B[5, 2::3] = grad[2], grad[0]          # xz shear

                K = K + B.T @ C @ B * np.linalg.det(jacobian)
    return K
