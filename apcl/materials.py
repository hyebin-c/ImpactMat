from __future__ import annotations

import math
from typing import Sequence

import torch


MATERIAL_NAMES = [
    "Ceramic",
    "Glass",
    "Wood",
    "Plastic",
    "Iron",
    "Polycarbonate",
    "Steel",
    "Tin",
]
MATERIAL_TO_INDEX = {name: idx for idx, name in enumerate(MATERIAL_NAMES)}
TARGET_NAMES = ("log_E", "log_rho", "log_eta", "alpha", "beta")
SUPPORTED_TARGET_NAMES = (
    "log_E",
    "log_rho",
    "log_E_over_rho",
    "nu",
    "log_eta",
    "alpha",
    "beta",
    "log_alpha",
    "log_beta",
)

MATERIAL_PARAMS = {
    "Ceramic": (2700.0, 7.2e10, 0.19),
    "Glass": (2600.0, 6.2e10, 0.20),
    "Wood": (750.0, 1.1e10, 0.25),
    "Plastic": (1070.0, 1.4e9, 0.35),
    "Iron": (8000.0, 2.1e11, 0.28),
    "Polycarbonate": (1190.0, 2.4e9, 0.37),
    "Steel": (7850.0, 2.0e11, 0.29),
    "Tin": (7265.0, 5.0e10, 0.325),
}

MATERIAL_DAMPING_PARAMS = {
    "Ceramic": (6.0, 1e-7),
    "Glass": (1.0, 1e-7),
    "Wood": (60.0, 2e-6),
    "Plastic": (30.0, 1e-6),
    "Iron": (5.0, 1e-7),
    "Polycarbonate": (0.5, 4e-7),
    "Steel": (5.0, 3e-8),
    "Tin": (2.0, 3e-8),
}


def build_anchor_table(
    target_names: Sequence[str] = TARGET_NAMES,
    eta_reference_hz: float = 1000.0,
) -> torch.Tensor:
    target_names = tuple(target_names)
    unknown = sorted(set(target_names) - set(SUPPORTED_TARGET_NAMES))
    if unknown:
        raise ValueError(f"Unknown target names: {unknown}")

    omega = 2.0 * math.pi * float(eta_reference_hz)
    rows = []
    for material in MATERIAL_NAMES:
        rho, youngs, nu = MATERIAL_PARAMS[material]
        alpha, beta = MATERIAL_DAMPING_PARAMS[material]
        eta = alpha / omega + beta * omega
        values = {
            "log_E": math.log(youngs),
            "log_rho": math.log(rho),
            "log_E_over_rho": math.log(youngs / rho),
            "nu": nu,
            "log_eta": math.log(max(eta, 1e-12)),
            "alpha": alpha,
            "beta": beta,
            "log_alpha": math.log(max(alpha, 1e-12)),
            "log_beta": math.log(max(beta, 1e-12)),
        }
        rows.append([values[name] for name in target_names])
    return torch.tensor(rows, dtype=torch.float32)


def normalize_anchor_table(
    anchor_table: torch.Tensor,
    materials: Sequence[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if materials is None:
        selected = anchor_table
    else:
        indices = torch.tensor([MATERIAL_TO_INDEX[name] for name in materials], dtype=torch.long)
        selected = anchor_table.index_select(0, indices)
    mean = selected.mean(dim=0)
    std = selected.std(dim=0, unbiased=False).clamp_min(1e-8)
    return (anchor_table - mean) / std, mean, std
