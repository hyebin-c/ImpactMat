"""Minimal Wavefront OBJ reader."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_obj(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read vertices (V, 3) and triangles (T, 3) from an OBJ file.

    Polygons with more than 3 corners are fan-triangulated. Texture/normal
    indices (``v/vt/vn``) are ignored.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mesh not found: {path}")

    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                vertices.append([float(x) for x in line[1:].split()[:3]])
            elif line.startswith("f"):
                face = [int(item.split("/")[0]) - 1 for item in line[1:].split()]
                for i in range(1, len(face) - 1):
                    triangles.append([face[0], face[i], face[i + 1]])
    return np.asarray(vertices), np.asarray(triangles, dtype=np.int64)


def normalize_vertices(vertices: np.ndarray) -> np.ndarray:
    """Center on the bounding box and scale the longest side to 1 (range [-0.5, 0.5])."""
    hi, lo = vertices.max(0), vertices.min(0)
    return (vertices - (hi + lo) / 2) / (hi - lo).max()
