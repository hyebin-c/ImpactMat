"""CPU triangle/box voxelizer (numba): the same test as ``voxelize_kernel.cu``.

Each voxel is a box of half-size ``1.1 * voxel_size / 2`` around its centre; it is part of the
shell when the separating-axis test (Akenine-Möller) finds a triangle overlapping it. The
test is evaluated in float32 like the CUDA kernel. Instead of testing every voxel against
every triangle, each triangle is only tested against the voxels inside its bounding box.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

_BOX_SCALE = 1.1          # boxes are slightly enlarged so that thin surfaces leave no gaps


@njit(cache=True)
def _separated_x(a, b, fa, fb, ya, za, yb, zb, h):
    p0 = a * ya - b * za
    p1 = a * yb - b * zb
    rad = fa * h + fb * h
    return min(p0, p1) > rad or max(p0, p1) < -rad


@njit(cache=True)
def _separated_y(a, b, fa, fb, xa, za, xb, zb, h):
    p0 = -a * xa + b * za
    p1 = -a * xb + b * zb
    rad = fa * h + fb * h
    return min(p0, p1) > rad or max(p0, p1) < -rad


@njit(cache=True)
def _separated_z(a, b, fa, fb, xa, ya, xb, yb, h):
    p0 = a * xa - b * ya
    p1 = a * xb - b * yb
    rad = fa * h + fb * h
    return min(p0, p1) > rad or max(p0, p1) < -rad


@njit(cache=True)
def _tri_box_overlap(cx, cy, cz, h, t):
    """Separating-axis test between the box (centre c, half-size h) and triangle ``t`` (3, 3)."""
    v0x, v0y, v0z = t[0, 0] - cx, t[0, 1] - cy, t[0, 2] - cz
    v1x, v1y, v1z = t[1, 0] - cx, t[1, 1] - cy, t[1, 2] - cz
    v2x, v2y, v2z = t[2, 0] - cx, t[2, 1] - cy, t[2, 2] - cz
    e0x, e0y, e0z = v1x - v0x, v1y - v0y, v1z - v0z
    e1x, e1y, e1z = v2x - v1x, v2y - v1y, v2z - v1z
    e2x, e2y, e2z = v0x - v2x, v0y - v2y, v0z - v2z

    # 9 axes: edge x coordinate axis
    fx, fy, fz = abs(e0x), abs(e0y), abs(e0z)
    if _separated_x(e0z, e0y, fz, fy, v0y, v0z, v2y, v2z, h): return False
    if _separated_y(e0z, e0x, fz, fx, v0x, v0z, v2x, v2z, h): return False
    if _separated_z(e0y, e0x, fy, fx, v1x, v1y, v2x, v2y, h): return False
    fx, fy, fz = abs(e1x), abs(e1y), abs(e1z)
    if _separated_x(e1z, e1y, fz, fy, v0y, v0z, v2y, v2z, h): return False
    if _separated_y(e1z, e1x, fz, fx, v0x, v0z, v2x, v2z, h): return False
    if _separated_z(e1y, e1x, fy, fx, v0x, v0y, v1x, v1y, h): return False
    fx, fy, fz = abs(e2x), abs(e2y), abs(e2z)
    if _separated_x(e2z, e2y, fz, fy, v0y, v0z, v1y, v1z, h): return False
    if _separated_y(e2z, e2x, fz, fx, v0x, v0z, v1x, v1z, h): return False
    if _separated_z(e2y, e2x, fy, fx, v1x, v1y, v2x, v2y, h): return False

    # the 3 coordinate axes
    if min(v0x, v1x, v2x) > h or max(v0x, v1x, v2x) < -h: return False
    if min(v0y, v1y, v2y) > h or max(v0y, v1y, v2y) < -h: return False
    if min(v0z, v1z, v2z) > h or max(v0z, v1z, v2z) < -h: return False

    # the triangle plane
    nx = e0y * e1z - e0z * e1y
    ny = e0z * e1x - e0x * e1z
    nz = e0x * e1y - e0y * e1x
    vminx = (-h - v0x) if nx > 0 else (h - v0x)
    vminy = (-h - v0y) if ny > 0 else (h - v0y)
    vminz = (-h - v0z) if nz > 0 else (h - v0z)
    vmaxx = (h - v0x) if nx > 0 else (-h - v0x)
    vmaxy = (h - v0y) if ny > 0 else (-h - v0y)
    vmaxz = (h - v0z) if nz > 0 else (-h - v0z)
    if nx * vminx + ny * vminy + nz * vminz > 0: return False
    return nx * vmaxx + ny * vmaxy + nz * vmaxz >= 0


@njit(cache=True, parallel=True)
def _shell(vertices, triangles, res):
    shell = np.zeros((res, res, res), dtype=np.int32)
    size = np.float32(1.0) / np.float32(res)
    half = size / np.float32(2.0) * np.float32(_BOX_SCALE)
    for n in prange(len(triangles)):
        t = vertices[triangles[n]]                                  # (3, 3) float32
        lo = np.empty(3, dtype=np.int64)
        hi = np.empty(3, dtype=np.int64)
        for a in range(3):                                          # voxel range touched by the bounding box
            tmin, tmax = t[:, a].min(), t[:, a].max()
            lo[a] = max(int(np.floor((tmin - half + 0.5) / size - 0.5)) - 1, 0)
            hi[a] = min(int(np.ceil((tmax + half + 0.5) / size - 0.5)) + 1, res - 1)
        for i in range(lo[0], hi[0] + 1):
            cx = np.float32(-0.5) + (np.float32(i) + np.float32(0.5)) * size
            for j in range(lo[1], hi[1] + 1):
                cy = np.float32(-0.5) + (np.float32(j) + np.float32(0.5)) * size
                for k in range(lo[2], hi[2] + 1):
                    cz = np.float32(-0.5) + (np.float32(k) + np.float32(0.5)) * size
                    if _tri_box_overlap(cx, cy, cz, half, t):
                        shell[i, j, k] = 1
    return shell


def voxelize_shell_cpu(vertices: np.ndarray, triangles: np.ndarray, resolution: int = 32) -> np.ndarray:
    """Surface voxels (int32 0/1 grid) of a mesh normalized to [-0.5, 0.5]^3."""
    return _shell(np.ascontiguousarray(vertices, dtype=np.float32),
                  np.ascontiguousarray(triangles, dtype=np.int64), resolution)
