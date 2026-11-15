"""Material parameters used by the modal sound model.

Units (SI):
    rho    density                          [kg/m^3]
    E      Young's modulus                  [Pa]
    nu     Poisson's ratio                  [-]
    alpha  Rayleigh damping, mass term      [1/s]
    beta   Rayleigh damping, stiffness term [s]

The damping matrix is C = alpha * M + beta * K, so a mode with angular frequency
omega decays as exp(-d t) with d = (alpha + beta * omega^2) / 2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Union


@dataclass(frozen=True)
class Material:
    rho: float
    E: float
    nu: float
    alpha: float
    beta: float

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return (self.rho, self.E, self.nu, self.alpha, self.beta)


MaterialSpec = Union[Material, str, Sequence[float], Mapping[str, float]]

DEFAULT_MATERIALS: dict[str, Material] = {
    #                      rho     E        nu     alpha  beta
    "Ceramic":       Material(2700, 7.2e10, 0.19,  6,   1e-7),
    "Glass":         Material(2600, 6.2e10, 0.20,  1,   1e-7),
    "Wood":          Material(750,  1.1e10, 0.25,  60,  2e-6),
    "Plastic":       Material(1070, 1.4e9,  0.35,  30,  1e-6),
    "Iron":          Material(8000, 2.1e11, 0.28,  5,   1e-7),
    "Polycarbonate": Material(1190, 2.4e9,  0.37,  0.5, 4e-7),
    "Steel":         Material(7850, 2.0e11, 0.29,  5,   3e-8),
    "Tin":           Material(7265, 5.0e10, 0.325, 2,   3e-8),
}


def resolve_material(spec: MaterialSpec,
                     table: Mapping[str, Material] | None = None) -> Material:
    """Accept a name, a (rho, E, nu, alpha, beta) sequence, a dict, or a Material."""
    if isinstance(spec, Material):
        return spec
    if isinstance(spec, str):
        table = DEFAULT_MATERIALS if table is None else table
        if spec not in table:
            raise KeyError(f"Unknown material '{spec}'. Available: {sorted(table)}")
        return table[spec]
    if isinstance(spec, Mapping):
        return Material(**{k: float(spec[k]) for k in ("rho", "E", "nu", "alpha", "beta")})
    values = [float(v) for v in spec]
    if len(values) != 5:
        raise ValueError("A material needs 5 values: rho, E, nu, alpha, beta")
    return Material(*values)
