"""Mesh -> filled voxel grid.

Two backends produce the same grid:

* ``"cpu"``  (default) numba, no GPU needed -- :mod:`impactmat.geometry.voxelize_cpu`
* ``"cuda"`` the original CUDA kernel; needs ``pycuda`` and a GPU (``pip install impactmat[cuda]``)
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from .boundary import fill_interior
from .mesh import load_obj, normalize_vertices
from .voxelize_cpu import voxelize_shell_cpu

BACKENDS = ("cpu", "cuda")


@lru_cache(maxsize=1)
def _cuda_kernel():
    try:
        import pycuda.autoinit  # noqa: F401  (creates the CUDA context)
        import pycuda.driver as drv
        from pycuda.compiler import SourceModule
    except ImportError as e:
        raise ImportError("The CUDA backend requires pycuda and a CUDA GPU: pip install 'impactmat[cuda]'") from e
    source = (Path(__file__).parent / "voxelize_kernel.cu").read_text()
    return drv, SourceModule(source).get_function("voxelize")


def _shell_cuda(vertices: np.ndarray, triangles: np.ndarray, resolution: int) -> np.ndarray:
    if resolution > 32:
        raise ValueError(f"The CUDA backend supports resolution 1..32 (one thread block per slice), got {resolution}")
    drv, kernel = _cuda_kernel()
    shell = np.zeros(resolution ** 3, dtype=np.int32)
    kernel(drv.Out(shell), np.int32(resolution),
           drv.In(vertices.astype(np.float32)), drv.In(triangles.astype(np.int32)),
           np.int32(len(triangles)),
           block=(resolution, resolution, 1), grid=(resolution, 1))
    return shell.reshape(resolution, resolution, resolution)


def voxelize(vertices: np.ndarray, triangles: np.ndarray, resolution: int = 32,
             backend: str = "cpu") -> np.ndarray:
    """Voxelize a triangle mesh and fill its interior. Returns an int32 grid of 0/1.

    ``vertices`` must be normalized to [-0.5, 0.5]^3 (see :func:`normalize_vertices`).
    """
    if backend not in BACKENDS:
        raise ValueError(f"Unknown voxelizer backend {backend!r}; choose one of {BACKENDS}")
    if resolution < 1:
        raise ValueError(f"resolution must be positive, got {resolution}")
    shell = (voxelize_shell_cpu if backend == "cpu" else _shell_cuda)(vertices, triangles, resolution)
    return fill_interior(shell)


def voxelize_mesh(mesh_path: str | Path, resolution: int = 32, backend: str = "cpu") -> np.ndarray:
    """Load an OBJ, normalize it to the unit cube and voxelize it."""
    vertices, triangles = load_obj(mesh_path)
    return voxelize(normalize_vertices(vertices), triangles, resolution, backend)
