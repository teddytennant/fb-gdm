"""Mean-field free energy and the conjugate / separable CAVI updates.

log p(v, y | xt) = (n/2) log γ_r - (γ_r / 2) ||x0 - xhat0||²
                 + (m/2) log γ_b - (γ_b / 2) ||y - A x0||²

F(q) = E_q[log p] + H[q(x0)] + H[q(γ_r)] + H[q(γ_b)].

Shapes are m/2 and n/2 (prior offsets a_b = b_b = a_r = b_r = 0). A rate
floor of 1e-12 keeps a perfect fit from sending the precision to infinity.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import jax.numpy as jnp
from jax.scipy.special import digamma, gammaln

from fb_gdm.operators import LinearOperator, as_linear_op

RATE_FLOOR = 1e-12

Array = jnp.ndarray


def gamma_entropy(a, b) -> Array:
    """Entropy of Gamma(shape=a, rate=b).

    a - log(b) + log Γ(a) + (1 - a) ψ(a).
    """
    a = jnp.asarray(a, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)
    return a - jnp.log(b) + gammaln(a) + (1.0 - a) * digamma(a)


def diagonal_gaussian_entropy(sigma2: Array) -> Array:
    """H[N(μ, diag(σ²))] = (n/2) log(2 π e) + (1/2) sum log σ²_i."""
    sigma2 = jnp.asarray(sigma2, dtype=jnp.float64)
    n = sigma2.shape[0]
    return 0.5 * n * jnp.log(2.0 * jnp.pi * jnp.e) + 0.5 * jnp.sum(jnp.log(sigma2))


def full_gaussian_entropy(sigma: Array) -> Array:
    """(1/2) log det(2 π e Σ)."""
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    n = sigma.shape[0]
    _sign, logdet = jnp.linalg.slogdet(sigma)
    return 0.5 * (n * jnp.log(2.0 * jnp.pi * jnp.e) + logdet)


def _elog(a, b) -> Array:
    """E[log γ] for Gamma(shape=a, rate=b) = ψ(a) - log(b)."""
    return digamma(jnp.asarray(a, dtype=jnp.float64)) - jnp.log(jnp.asarray(b, dtype=jnp.float64))


def _floor_rate(b) -> Array:
    return jnp.maximum(jnp.asarray(b, dtype=jnp.float64), RATE_FLOOR)


def expected_residuals_separable(mu, sigma2, xhat0, y, op: LinearOperator, H) -> tuple[Array, Array]:
    diff = mu - xhat0
    e_prior = jnp.dot(diff, diff) + jnp.sum(sigma2)
    resid = y - op.matvec(mu)
    e_obs = jnp.dot(resid, resid) + jnp.dot(H, sigma2)
    return e_prior, e_obs


def expected_residuals_conjugate(mu, sigma, xhat0, y, A) -> tuple[Array, Array]:
    diff = mu - xhat0
    e_prior = jnp.dot(diff, diff) + jnp.trace(sigma)
    resid = y - A @ mu
    e_obs = jnp.dot(resid, resid) + jnp.trace(A @ sigma @ A.T)
    return e_prior, e_obs


def _expected_log_joint(e_prior, e_obs, a_b, b_b, a_r, b_r, m: int, n: int) -> Array:
    gamma_b = jnp.asarray(a_b, dtype=jnp.float64) / jnp.asarray(b_b, dtype=jnp.float64)
    gamma_r = jnp.asarray(a_r, dtype=jnp.float64) / jnp.asarray(b_r, dtype=jnp.float64)
    return (
        (n / 2.0) * _elog(a_r, b_r)
        - (gamma_r / 2.0) * e_prior
        + (m / 2.0) * _elog(a_b, b_b)
        - (gamma_b / 2.0) * e_obs
    )


def free_energy_separable(mu, sigma2, a_b, b_b, a_r, b_r, xhat0, y, A) -> Array:
    """F(q) for q(x0) = N(μ, diag(σ²)), q(γ) = Gamma(ã, b̃)."""
    op = as_linear_op(A)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma2 = jnp.asarray(sigma2, dtype=jnp.float64)
    xhat0 = jnp.asarray(xhat0, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    H = op.column_squared_norms()
    m, n = op.shape
    e_prior, e_obs = expected_residuals_separable(mu, sigma2, xhat0, y, op, H)
    return (
        _expected_log_joint(e_prior, e_obs, a_b, b_b, a_r, b_r, m, n)
        + diagonal_gaussian_entropy(sigma2)
        + gamma_entropy(a_r, b_r)
        + gamma_entropy(a_b, b_b)
    )


def free_energy_conjugate(mu, sigma, a_b, b_b, a_r, b_r, xhat0, y, A) -> Array:
    """F(q) for q(x0) = N(μ, Σ) with a full covariance."""
    A = jnp.asarray(A, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    xhat0 = jnp.asarray(xhat0, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    m, n = A.shape
    e_prior, e_obs = expected_residuals_conjugate(mu, sigma, xhat0, y, A)
    return (
        _expected_log_joint(e_prior, e_obs, a_b, b_b, a_r, b_r, m, n)
        + full_gaussian_entropy(sigma)
        + gamma_entropy(a_r, b_r)
        + gamma_entropy(a_b, b_b)
    )


def gamma_rates_separable(mu, sigma2, xhat0, y, A) -> tuple[Array, Array]:
    """Equations (40)–(43): shapes are fixed by the caller; these are the rates."""
    op = as_linear_op(A)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma2 = jnp.asarray(sigma2, dtype=jnp.float64)
    xhat0 = jnp.asarray(xhat0, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    H = op.column_squared_norms()
    e_prior, e_obs = expected_residuals_separable(mu, sigma2, xhat0, y, op, H)
    return _floor_rate(0.5 * e_obs), _floor_rate(0.5 * e_prior)


def gamma_rates_conjugate(mu, sigma, xhat0, y, A) -> tuple[Array, Array]:
    A = jnp.asarray(A, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    xhat0 = jnp.asarray(xhat0, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    e_prior, e_obs = expected_residuals_conjugate(mu, sigma, xhat0, y, A)
    return _floor_rate(0.5 * e_obs), _floor_rate(0.5 * e_prior)


def separable_targets(mu, gamma_r, gamma_b, xhat0, A, At_y=None):
    """Corollary 2 mean-field target (36)–(37), matvecs only.

    μ_r,i = (⟨γ_r⟩ xhat0_i + ⟨γ_b⟩ (Aᵀ y)_i - ⟨γ_b⟩ (Aᵀ A μ)_i + ⟨γ_b⟩ H_i μ_i)
            / (⟨γ_r⟩ + ⟨γ_b⟩ H_i)
    σ_r,i² = 1 / (⟨γ_r⟩ + ⟨γ_b⟩ H_i)
    """
    op = as_linear_op(A)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    xhat0 = jnp.asarray(xhat0, dtype=jnp.float64)
    gamma_r = jnp.asarray(gamma_r, dtype=jnp.float64)
    gamma_b = jnp.asarray(gamma_b, dtype=jnp.float64)
    H = op.column_squared_norms()
    if At_y is None:
        # y is not available here; caller should pass Aᵀ y.
        raise TypeError("At_y is required")
    At_y = jnp.asarray(At_y, dtype=jnp.float64)
    At_A_mu = op.rmatvec(op.matvec(mu))
    denom = gamma_r + gamma_b * H
    mu_r = (
        gamma_r * xhat0 + gamma_b * At_y - gamma_b * At_A_mu + gamma_b * H * mu
    ) / denom
    sigma2_r = 1.0 / denom
    return mu_r, sigma2_r


def _line_free_energy(s, mu, delta_mu, sigma2, delta_s2, A_mu, A_delta, H, xhat0, y, a_b, b_b, a_r, b_r):
    """F along μ(s) = μ + s δ, σ²(s) = σ² + s δσ, Gammas held fixed."""
    s = jnp.asarray(s, dtype=jnp.float64)
    mu_s = mu + s * delta_mu
    s2_s = sigma2 + s * delta_s2
    resid = y - (A_mu + s * A_delta)
    e_obs = jnp.dot(resid, resid) + jnp.dot(H, s2_s)
    diff = mu_s - xhat0
    e_prior = jnp.dot(diff, diff) + jnp.sum(s2_s)
    m = y.shape[0]
    n = mu.shape[0]
    return (
        _expected_log_joint(e_prior, e_obs, a_b, b_b, a_r, b_r, m, n)
        + diagonal_gaussian_entropy(s2_s)
        + gamma_entropy(a_r, b_r)
        + gamma_entropy(a_b, b_b)
    )


def _line_derivatives(s, mu, delta_mu, sigma2, delta_s2, A_mu, A_delta, H, xhat0, y, a_b, b_b, a_r, b_r):
    """F'(s) and F''(s). F is strictly concave in s unless the step is zero."""
    s = jnp.asarray(s, dtype=jnp.float64)
    gamma_b = jnp.asarray(a_b, dtype=jnp.float64) / jnp.asarray(b_b, dtype=jnp.float64)
    gamma_r = jnp.asarray(a_r, dtype=jnp.float64) / jnp.asarray(b_r, dtype=jnp.float64)
    mu_s = mu + s * delta_mu
    s2_s = sigma2 + s * delta_s2
    resid = y - (A_mu + s * A_delta)
    q_prime = 2.0 * jnp.dot(mu_s - xhat0, delta_mu)
    v_prime = jnp.sum(delta_s2)
    r_prime = -2.0 * jnp.dot(resid, A_delta)
    w_prime = jnp.dot(H, delta_s2)
    h_prime = 0.5 * jnp.sum(delta_s2 / s2_s)
    d1 = -(gamma_r / 2.0) * (q_prime + v_prime) - (gamma_b / 2.0) * (r_prime + w_prime) + h_prime
    d2 = (
        -gamma_r * jnp.dot(delta_mu, delta_mu)
        - gamma_b * jnp.dot(A_delta, A_delta)
        - 0.5 * jnp.sum(delta_s2 * delta_s2 / (s2_s * s2_s))
    )
    return d1, d2


def optimal_step_size(mu, sigma2, mu_r, sigma2_r, xhat0, y, A, a_b, b_b, a_r, b_r) -> Array:
    """s in (0, 1] maximizing F along the relaxed Corollary 2 step.

    Newton on the strictly concave restriction, then a check against the
    endpoints. The closed form cited from Fraysse and Rodet is not in the
    paper; this is the 1D maximizer the spec asks for.
    """
    op = as_linear_op(A)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma2 = jnp.asarray(sigma2, dtype=jnp.float64)
    mu_r = jnp.asarray(mu_r, dtype=jnp.float64)
    sigma2_r = jnp.asarray(sigma2_r, dtype=jnp.float64)
    xhat0 = jnp.asarray(xhat0, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    H = op.column_squared_norms()
    delta_mu = mu_r - mu
    delta_s2 = sigma2_r - sigma2
    A_mu = op.matvec(mu)
    A_delta = op.matvec(delta_mu)
    args = (mu, delta_mu, sigma2, delta_s2, A_mu, A_delta, H, xhat0, y, a_b, b_b, a_r, b_r)

    # Already at the target: any s leaves the state unchanged.
    if float(jnp.linalg.norm(delta_mu) + jnp.linalg.norm(delta_s2)) < 1e-18:
        return jnp.asarray(1.0, dtype=jnp.float64)

    lo = 1e-8
    hi = 1.0
    d1_hi, _ = _line_derivatives(hi, *args)
    d1_lo, _ = _line_derivatives(0.0, *args)
    if float(d1_hi) >= 0.0:
        s = hi
    elif float(d1_lo) <= 0.0:
        # Maximizer of a concave function on (0, 1] sits at the open end.
        # Take a positive step small enough that F does not drop past 1e-9.
        slope = abs(float(d1_lo)) + 1e-12
        s = min(lo, 1e-10 / slope)
    else:
        s = 0.5
        for _ in range(40):
            d1, d2 = _line_derivatives(s, *args)
            d2f = float(d2)
            if abs(d2f) < 1e-18:
                break
            s_new = float(s) - float(d1) / d2f
            s_new = min(hi, max(lo, s_new))
            if abs(s_new - float(s)) < 1e-14:
                s = s_new
                break
            s = s_new
        s = float(s)

    F0 = float(_line_free_energy(0.0, *args))
    candidates = [float(s), hi]
    if float(d1_lo) > 0.0:
        candidates.append(lo)
    best = max(candidates, key=lambda c: float(_line_free_energy(c, *args)))
    if float(_line_free_energy(best, *args)) + 1e-9 < F0:
        slope = abs(float(d1_lo)) + 1e-12
        best = min(float(best), 1e-10 / slope)
        while best > 1e-18 and float(_line_free_energy(best, *args)) + 1e-9 < F0:
            best *= 0.1
    return jnp.asarray(best, dtype=jnp.float64)


@dataclasses.dataclass
class SeparableResult:
    mu: Array
    sigma2: Array
    a_b: Array
    b_b: Array
    a_r: Array
    b_r: Array
    gamma_b: Array
    gamma_r: Array
    free_energy: Array
    free_energies: list
    steps: list


def separable_step(mu, sigma2, xhat0, y, A, a_b, b_b, a_r, b_r) -> tuple:
    """One Corollary 2 sweep: target, s-relaxed update, then Gamma refresh.

    Returns (mu, sigma2, b_b, b_r, s, F_before, F_after_relax, F_after).
    F_after_relax holds Gammas fixed, so it is the quantity s maximizes.
    """
    op = as_linear_op(A)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma2 = jnp.asarray(sigma2, dtype=jnp.float64)
    xhat0 = jnp.asarray(xhat0, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    a_b = jnp.asarray(a_b, dtype=jnp.float64)
    b_b = jnp.asarray(b_b, dtype=jnp.float64)
    a_r = jnp.asarray(a_r, dtype=jnp.float64)
    b_r = jnp.asarray(b_r, dtype=jnp.float64)
    gamma_b = a_b / b_b
    gamma_r = a_r / b_r
    At_y = op.rmatvec(y)
    F_before = free_energy_separable(mu, sigma2, a_b, b_b, a_r, b_r, xhat0, y, op)
    mu_r, sigma2_r = separable_targets(mu, gamma_r, gamma_b, xhat0, op, At_y=At_y)
    s = optimal_step_size(mu, sigma2, mu_r, sigma2_r, xhat0, y, op, a_b, b_b, a_r, b_r)
    mu_new = (1.0 - s) * mu + s * mu_r
    sigma2_new = (1.0 - s) * sigma2 + s * sigma2_r
    F_relax = free_energy_separable(mu_new, sigma2_new, a_b, b_b, a_r, b_r, xhat0, y, op)
    b_b_new, b_r_new = gamma_rates_separable(mu_new, sigma2_new, xhat0, y, op)
    F_after = free_energy_separable(mu_new, sigma2_new, a_b, b_b_new, a_r, b_r_new, xhat0, y, op)
    return mu_new, sigma2_new, b_b_new, b_r_new, s, F_before, F_relax, F_after


def separable_infer(
    xhat0,
    y,
    A,
    *,
    K: int = 50,
    mu: Optional[Array] = None,
    sigma2: Optional[Array] = None,
    tol: float = 0.0,
) -> SeparableResult:
    """K sweeps of Corollary 2. Algorithm 2 initializes μ = xhat0.

    ``sigma2`` defaults to ones. Pass ``(1 - alpha_bar) * ones(n)`` to match
    Algorithm 2. Shapes stay at m/2 and n/2. ``tol`` stops early when the
    means move less than that absolute amount; 0 runs exactly K sweeps.
    """
    op = as_linear_op(A)
    xhat0 = jnp.asarray(xhat0, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    m, n = op.shape
    if mu is None:
        mu = xhat0
    mu = jnp.asarray(mu, dtype=jnp.float64)
    if sigma2 is None:
        sigma2 = jnp.ones(n, dtype=jnp.float64)
    sigma2 = jnp.asarray(sigma2, dtype=jnp.float64)
    a_b = jnp.asarray(m / 2.0, dtype=jnp.float64)
    a_r = jnp.asarray(n / 2.0, dtype=jnp.float64)
    b_b, b_r = gamma_rates_separable(mu, sigma2, xhat0, y, op)
    energies = [free_energy_separable(mu, sigma2, a_b, b_b, a_r, b_r, xhat0, y, op)]
    steps: list = []
    for _ in range(int(K)):
        mu_prev, sigma2_prev = mu, sigma2
        gamma_b_prev = a_b / b_b
        gamma_r_prev = a_r / b_r
        mu, sigma2, b_b, b_r, s, _F0, _Fr, F_after = separable_step(
            mu, sigma2, xhat0, y, op, a_b, b_b, a_r, b_r
        )
        energies.append(F_after)
        steps.append(s)
        moved = (
            jnp.linalg.norm(mu - mu_prev)
            + jnp.linalg.norm(sigma2 - sigma2_prev)
            + jnp.abs(a_b / b_b - gamma_b_prev)
            + jnp.abs(a_r / b_r - gamma_r_prev)
        )
        if tol > 0.0 and float(moved) < tol:
            break
    gamma_b = a_b / b_b
    gamma_r = a_r / b_r
    return SeparableResult(
        mu=mu,
        sigma2=sigma2,
        a_b=a_b,
        b_b=b_b,
        a_r=a_r,
        b_r=b_r,
        gamma_b=gamma_b,
        gamma_r=gamma_r,
        free_energy=energies[-1],
        free_energies=energies,
        steps=steps,
    )


def _gaussian_update(xhat0, y, A, gamma_r, gamma_b) -> tuple[Array, Array]:
    """Σ⁻¹ = ⟨γ_r⟩ I + ⟨γ_b⟩ Aᵀ A, μ = Σ (⟨γ_r⟩ xhat0 + ⟨γ_b⟩ Aᵀ y)."""
    A = jnp.asarray(A, dtype=jnp.float64)
    n = A.shape[1]
    prec = gamma_r * jnp.eye(n, dtype=jnp.float64) + gamma_b * (A.T @ A)
    chol = jnp.linalg.cholesky(prec)
    eye = jnp.eye(n, dtype=jnp.float64)
    sigma = jax_cho_solve(chol, eye)
    rhs = gamma_r * xhat0 + gamma_b * (A.T @ y)
    mu = jax_cho_solve(chol, rhs)
    return mu, sigma


def jax_cho_solve(chol: Array, rhs: Array) -> Array:
    from jax.scipy.linalg import cho_solve

    return cho_solve((chol, True), rhs)


@dataclasses.dataclass
class ConjugateResult:
    mu: Array
    sigma: Array
    a_b: Array
    b_b: Array
    a_r: Array
    b_r: Array
    gamma_b: Array
    gamma_r: Array
    free_energy: Array
    free_energies: list


def cavi_conjugate(
    xhat0,
    y,
    A,
    *,
    max_iter: int = 200,
    tol: float = 1e-12,
    mu: Optional[Array] = None,
    sigma: Optional[Array] = None,
) -> ConjugateResult:
    """Proposition 2. Alternate Gamma updates and the full Gaussian update.

    Each sweep is optimal for the other factors held fixed, so F does not
    decrease. Stops when the means move by less than ``tol``.
    """
    A = jnp.asarray(A, dtype=jnp.float64)
    xhat0 = jnp.asarray(xhat0, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    m, n = A.shape
    if mu is None:
        mu = xhat0
    mu = jnp.asarray(mu, dtype=jnp.float64)
    if sigma is None:
        sigma = jnp.eye(n, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    a_b = jnp.asarray(m / 2.0, dtype=jnp.float64)
    a_r = jnp.asarray(n / 2.0, dtype=jnp.float64)
    b_b, b_r = gamma_rates_conjugate(mu, sigma, xhat0, y, A)
    energies = [free_energy_conjugate(mu, sigma, a_b, b_b, a_r, b_r, xhat0, y, A)]
    for _ in range(int(max_iter)):
        gamma_b = a_b / b_b
        gamma_r = a_r / b_r
        mu_new, sigma_new = _gaussian_update(xhat0, y, A, gamma_r, gamma_b)
        b_b_new, b_r_new = gamma_rates_conjugate(mu_new, sigma_new, xhat0, y, A)
        F_new = free_energy_conjugate(mu_new, sigma_new, a_b, b_b_new, a_r, b_r_new, xhat0, y, A)
        energies.append(F_new)
        moved = (
            jnp.linalg.norm(mu_new - mu)
            + jnp.linalg.norm(sigma_new - sigma)
            + jnp.abs(a_b / b_b_new - gamma_b)
            + jnp.abs(a_r / b_r_new - gamma_r)
        )
        mu, sigma, b_b, b_r = mu_new, sigma_new, b_b_new, b_r_new
        if float(moved) < tol:
            break
    return ConjugateResult(
        mu=mu,
        sigma=sigma,
        a_b=a_b,
        b_b=b_b,
        a_r=a_r,
        b_r=b_r,
        gamma_b=a_b / b_b,
        gamma_r=a_r / b_r,
        free_energy=energies[-1],
        free_energies=energies,
    )
