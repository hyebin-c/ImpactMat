"""Configuration: nested dataclasses, loaded from YAML with dotted CLI overrides.

    cfg = load_config("configs/default.yaml", overrides=["audio.duration=1.0"])

Every field has a default, so ``Config()`` works without any file.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .materials import DEFAULT_MATERIALS, Material, resolve_material


@dataclass
class PathsCfg:
    mesh_dir: str = "data/meshes"      # {mesh_dir}/{obj_id}/{mesh_file}
    mesh_file: str = "model.obj"
    out_dir: str = "outputs"           # {out_dir}/{obj_id}/{voxel.npz, modal/, feat/, wav/}


@dataclass
class VoxelCfg:
    resolution: int = 32               # voxel grid is resolution^3
    backend: str = "cpu"               # voxelizer: "cpu" (numba) or "cuda" (pycuda, GPU)
    object_length: float = 0.15        # physical size of the normalized object [m]


@dataclass
class FemCfg:
    num_modes: int = 20                # modes kept after dropping the rigid-body modes
    num_rigid_modes: int = 6           # lowest (~zero-frequency) modes to discard
    seed: int | None = None            # LOBPCG starts from random vectors; set for repeatability
    save_modal: bool = True            # keep modal_{mat}.npz (raw eigenpairs, large)


@dataclass
class AudioCfg:
    sample_rate: int = 44100
    duration: float = 2.0              # [s]
    min_freq: float = 20.0             # mode frequencies are clipped to [min_freq, max_freq] [Hz]
    max_freq: float = 20000.0


@dataclass
class HitCfg:
    index: int = 0                     # boundary-voxel index of the strike point
    t_start: float | list[float] = 0.0 # strike time(s) within the clip [s]
    gain: float = 1.0                  # loudness scale applied after peak-normalization


@dataclass
class Config:
    paths: PathsCfg = field(default_factory=PathsCfg)
    voxel: VoxelCfg = field(default_factory=VoxelCfg)
    fem: FemCfg = field(default_factory=FemCfg)
    audio: AudioCfg = field(default_factory=AudioCfg)
    hit: HitCfg = field(default_factory=HitCfg)
    # extra / overriding materials: {name: {rho, E, nu, alpha, beta}}
    materials: dict[str, Any] = field(default_factory=dict)

    def material_table(self) -> dict[str, Material]:
        table = dict(DEFAULT_MATERIALS)
        for name, spec in self.materials.items():
            table[name] = resolve_material(spec)
        return table


def _build(cls, data: Mapping[str, Any]):
    """dict -> dataclass, rejecting unknown keys (catches config typos)."""
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise KeyError(f"Unknown config key(s) for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        factory = known[name].default_factory      # a dataclass type for nested sections
        if is_dataclass(factory) and isinstance(value, Mapping):
            ftype = factory
            kwargs[name] = _build(ftype, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _set_dotted(tree: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = tree
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def load_config(path: str | Path | None = None,
                overrides: Sequence[str] | None = None) -> Config:
    """Load a YAML file (optional) and apply ``section.key=value`` overrides."""
    tree: dict = {}
    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            tree = yaml.safe_load(f) or {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like section.key=value, got '{item}'")
        key, raw = item.split("=", 1)
        _set_dotted(tree, key.strip(), yaml.safe_load(raw))   # '1.0' -> float, '[0,0.5]' -> list
    return _build(Config, tree)
