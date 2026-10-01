"""FB-GDM: variational conditional score for one DDPM reverse step."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from fb_gdm._host import preload as _preload

_preload()

import jax

jax.config.update("jax_enable_x64", True)

from fb_gdm.free_energy import (
    RATE_FLOOR,
    cavi_conjugate,
    free_energy_conjugate,
    free_energy_separable,
    gamma_entropy,
    separable_infer,
    separable_step,
)
from fb_gdm.sampler import (
    LinearOperator,
    conditional_score,
    corollary1,
    linear_beta_schedule,
    posterior_mean_cg,
    posterior_mean_dense,
    reverse_step,
    row_selection,
    xhat0_tweedie,
)

__all__ = [
    "RATE_FLOOR",
    "LinearOperator",
    "cavi_conjugate",
    "conditional_score",
    "corollary1",
    "free_energy_conjugate",
    "free_energy_separable",
    "gamma_entropy",
    "linear_beta_schedule",
    "posterior_mean_cg",
    "posterior_mean_dense",
    "reverse_step",
    "row_selection",
    "separable_infer",
    "separable_step",
    "xhat0_tweedie",
]
