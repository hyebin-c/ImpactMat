"""Modal sound synthesis: a sum of exponentially damped sinusoids."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..modal import ModalFeatures


def synthesize(features: ModalFeatures, alpha: float, beta: float, *, hit_index: int = 0,
               t_start: float | Sequence[float] = 0.0, gain: float = 1.0,
               duration: float = 2.0, sample_rate: int = 44100) -> np.ndarray:
    """Render the sound of one or more strikes at a boundary voxel.

    Mode k rings at its natural frequency f_k with amplitude |u_k(hit)| (the
    displacement magnitude at the strike point) and decays at rate
    d_k = (alpha + beta * omega_k^2) / 2 with omega_k = 2 pi f_k.

    The result is peak-normalized to ~1 and then scaled by ``gain``.

    Args:
        features:    modal features of the object (see :func:`impactmat.modal.compute_features`)
        alpha, beta: Rayleigh damping coefficients of the material
        hit_index:   index of the strike point among the boundary voxels
        t_start:     strike time(s) [s]; several times give several strikes
        gain:        output loudness scale
    """
    # NOTE: the grid includes the endpoint (dt = duration / (n - 1)); kept for
    # consistency with the ImpactMat dataset.
    t = np.linspace(0, duration, int(sample_rate * duration))
    omega = 2 * np.pi * features.freqs                                    # (k,)
    damping = (alpha + beta * omega ** 2) / 2                             # (k,)
    amplitude = np.linalg.norm(features.feats_in[hit_index], axis=0)      # (k,)

    starts = [t_start] if np.isscalar(t_start) else list(t_start)
    sound = np.zeros_like(t)
    for start in starts:
        mask = t >= start
        td = (t[mask] - start)[:, None]                                   # (T, 1)
        sound[mask] += (amplitude * np.exp(-damping * td) * np.sin(omega * td)).sum(axis=1)

    peak = np.max(np.abs(sound))
    if peak > 0:
        sound /= peak + 1e-6
    return sound * gain
