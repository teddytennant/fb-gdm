"""Corollary 1 score and one FB-GDM reverse step (Algorithm 2)."""

from __future__ import annotations

import dataclasses
from typing import Callable, Optional

import jax
import jax.numpy as jnp

from fb_gdm.free_energy import separable_infer
from fb_gdm.operators import LinearOperator, as_linear_op, row_selection

Array = jnp.ndarray

__all__ = [
    "LinearOperator",
    "Schedule",
    "conditional_score",
    "corollary1",
    "linear_beta_schedule",
    "posterior_mean_cg",
    "posterior_mean_dense",
    "reverse_step",
    "row_selection",
    "xhat0_tweedie",
]


def xhat0_tweedie(xt, eps, alpha_bar) -> Array:
    """Tweedie estimate. ``eps`` is stopped so ∂xhat0/∂xt = I / sqrt(ᾱ)."""
    xt = jnp.asarray(xt, dtype=jnp.float64)
    eps = jax.lax.stop_gradient(jnp.asarray(eps, dtype=jnp.float64))
    alpha_bar = jnp.asarray(alpha_bar, dtype=jnp.float64)
    return (xt - jnp.sqrt(1.0 - alpha_bar) * eps) / jnp.sqrt(alpha_bar)


def posterior_mean_dense(xhat0, y, A, gamma_r, gamma_b) -> tuple[Array, Array]:
    """Dense Corollary 1 solve. Returns (μ_post, Σ_post).

    Σ_post⁻¹ = γ_r I + γ_b Aᵀ A
    μ_post = Σ_post (γ_r xhat0 + γ_b Aᵀ y)
    """
    A = jnp.asarray(A, dtype=jnp.float64)
    xhat0 = jnp.asarray(xhat0, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    gamma_r = jnp.asarray(gamma_r, dtype=jnp.float64)
    gamma_b = jnp.asarray(gamma_b, dtype=jnp.float64)
    n = A.shape[1]
    prec = gamma_r * jnp.eye(n, dtype=jnp.float64) + gamma_b * (A.T @ A)
    sigma = jnp.linalg.inv(prec)
    mu = sigma @ (gamma_r * xhat0 + gamma_b * (A.T @ y))
    return mu, sigma


def posterior_mean_cg(xhat0, y, A, gamma_r, gamma_b, *, tol: float = 1e-12, maxiter: Optional[int] = None) -> Array:
    """Equation (22) mean via conjugate gradient. Matvecs only, no n×n matrix.

    Solves (γ_r I + γ_b Aᵀ A) μ = γ_r xhat0 + γ_b Aᵀ y.
    """
    op = as_linear_op(A)
    xhat0 = jnp.asarray(xhat0, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    gamma_r = jnp.asarray(gamma_r, dtype=jnp.float64)
    gamma_b = jnp.asarray(gamma_b, dtype=jnp.float64)
    _m, n = op.shape
    rhs = gamma_r * xhat0 + gamma_b * op.rmatvec(y)

    def matvec(v):
        return gamma_r * v + gamma_b * op.rmatvec(op.matvec(v))

    mu, _info = jax.scipy.sparse.linalg.cg(
        matvec,
        rhs,
        tol=tol,
        maxiter=(4 * n) if maxiter is None else maxiter,
    )
    return mu


def conditional_score(mu_post, xhat0, alpha_bar) -> Array:
    """score(y | xt) = (μ_post - xhat0) / (sqrt(ᾱ) (1 - ᾱ))."""
    mu_post = jnp.asarray(mu_post, dtype=jnp.float64)
    xhat0 = jnp.asarray(xhat0, dtype=jnp.float64)
    alpha_bar = jnp.asarray(alpha_bar, dtype=jnp.float64)
    return (mu_post - xhat0) / (jnp.sqrt(alpha_bar) * (1.0 - alpha_bar))


def corollary1(xhat0, y, A, gamma_r, gamma_b, alpha_bar) -> tuple[Array, Array, Array]:
    """Dense Corollary 1: μ_post, Σ_post, and score(y | xt)."""
    mu, sigma = posterior_mean_dense(xhat0, y, A, gamma_r, gamma_b)
    score = conditional_score(mu, xhat0, alpha_bar)
    return mu, sigma, score


def prior_score(eps, alpha_bar) -> Array:
    """score(xt) = −ε_θ / sqrt(1 − ᾱ)."""
    eps = jnp.asarray(eps, dtype=jnp.float64)
    alpha_bar = jnp.asarray(alpha_bar, dtype=jnp.float64)
    return -eps / jnp.sqrt(1.0 - alpha_bar)


@dataclasses.dataclass
class Schedule:
    """1-based linear DDPM schedule. Index 0 of beta is unused; alpha_bar[0] = 1."""

    beta: Array
    alpha: Array
    alpha_bar: Array
    beta_tilde: Array
    num_steps: int


def linear_beta_schedule(num_steps: int, beta_start: float, beta_end: float) -> Schedule:
    """β_t = β_start + (t/T) (β_end − β_start) for t = 1..T.

    α_t = 1 − β_t, ᾱ_t = ∏_{s=1}^{t} α_s,
    β̃_t = β_t (1 − ᾱ_{t−1}) / (1 − ᾱ_t) for t > 1, and β̃_1 = β_1.
    """
    T = int(num_steps)
    t = jnp.arange(1, T + 1, dtype=jnp.float64)
    beta_body = beta_start + (t / T) * (beta_end - beta_start)
    alpha_body = 1.0 - beta_body
    alpha_bar_body = jnp.cumprod(alpha_body)
    alpha_bar_prev = jnp.concatenate([jnp.ones(1, dtype=jnp.float64), alpha_bar_body[:-1]])
    beta_tilde_body = beta_body * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_body)
    beta_tilde_body = beta_tilde_body.at[0].set(beta_body[0])
    # Leading slot so schedule.beta[t] is β_t for t = 1..T.
    beta = jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), beta_body])
    alpha = jnp.concatenate([jnp.ones(1, dtype=jnp.float64), alpha_body])
    alpha_bar = jnp.concatenate([jnp.ones(1, dtype=jnp.float64), alpha_bar_body])
    beta_tilde = jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), beta_tilde_body])
    return Schedule(beta=beta, alpha=alpha, alpha_bar=alpha_bar, beta_tilde=beta_tilde, num_steps=T)


def reverse_step(
    xt,
    y,
    A,
    eps_theta: Callable,
    t,
    alpha_bar,
    alpha,
    beta,
    beta_tilde,
    z,
    *,
    K: int = 5,
    cg_tol: float = 1e-12,
) -> Array:
    """Algorithm 2, one reverse step.

    Inner loop is the separable factorization (no n×n matrix). The score
    uses equation (22) at the inferred means ⟨γ_r⟩, ⟨γ_b⟩, solved by
    conjugate gradient so a masking operator never builds an n×n matrix.

        x_{t-1} = (x_t + β_t score(x_t | y)) / sqrt(α_t) + sqrt(β̃_t) z
    """
    xt = jnp.asarray(xt, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    z = jnp.asarray(z, dtype=jnp.float64)
    alpha_bar = jnp.asarray(alpha_bar, dtype=jnp.float64)
    alpha = jnp.asarray(alpha, dtype=jnp.float64)
    beta = jnp.asarray(beta, dtype=jnp.float64)
    beta_tilde = jnp.asarray(beta_tilde, dtype=jnp.float64)
    eps = jax.lax.stop_gradient(jnp.asarray(eps_theta(xt, t), dtype=jnp.float64))
    xhat0 = xhat0_tweedie(xt, eps, alpha_bar)
    n = xt.shape[0]
    sigma2_0 = (1.0 - alpha_bar) * jnp.ones(n, dtype=jnp.float64)
    fit = separable_infer(xhat0, y, A, K=K, mu=xhat0, sigma2=sigma2_0)
    mu_post = posterior_mean_cg(xhat0, y, A, fit.gamma_r, fit.gamma_b, tol=cg_tol)
    score_y = conditional_score(mu_post, xhat0, alpha_bar)
    score = prior_score(eps, alpha_bar) + score_y
    return (xt + beta * score) / jnp.sqrt(alpha) + jnp.sqrt(beta_tilde) * z
