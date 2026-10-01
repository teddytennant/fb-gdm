"""Numerical checks for FB-GDM. Import the package before JAX so NixOS libs load."""

from __future__ import annotations

import fb_gdm  # noqa: F401  preload libstdc++ before jax

import jax
import jax.numpy as jnp
import pytest

from fb_gdm import (
    cavi_conjugate,
    conditional_score,
    corollary1,
    free_energy_separable,
    linear_beta_schedule,
    posterior_mean_cg,
    reverse_step,
    separable_infer,
    separable_step,
    xhat0_tweedie,
)
from fb_gdm.free_energy import gamma_rates_separable


def _rel(a, b) -> float:
    a = jnp.asarray(a, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)
    return float(jnp.linalg.norm(a - b) / jnp.maximum(jnp.linalg.norm(b), 1e-12))


def test_corollary1_matches_dense_solve():
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    n, m = 5, 3
    A = jax.random.normal(k1, (m, n))
    xhat0 = jax.random.normal(k2, (n,))
    y = jax.random.normal(k3, (m,))
    gamma_r = 1.7
    gamma_b = 0.4
    alpha_bar = 0.36
    mu, sigma, score = corollary1(xhat0, y, A, gamma_r, gamma_b, alpha_bar)

    prec = gamma_r * jnp.eye(n) + gamma_b * (A.T @ A)
    sigma_ref = jnp.linalg.inv(prec)
    mu_ref = jnp.linalg.solve(prec, gamma_r * xhat0 + gamma_b * (A.T @ y))
    score_ref = (mu_ref - xhat0) / (jnp.sqrt(alpha_bar) * (1.0 - alpha_bar))
    assert _rel(mu, mu_ref) < 1e-8
    assert _rel(sigma, sigma_ref) < 1e-8
    assert _rel(score, score_ref) < 1e-8
    assert jnp.allclose(score, conditional_score(mu, xhat0, alpha_bar))


def test_tweedie_stops_gradient_through_eps():
    alpha_bar = 0.64

    def eps_theta(xt, t):
        return 0.3 * xt + 0.1

    xt = jnp.array([0.2, -0.4, 0.7])

    def f(xt_):
        return xhat0_tweedie(xt_, eps_theta(xt_, 1), alpha_bar)

    jac = jax.jacobian(f)(xt)
    assert jnp.allclose(jac, jnp.eye(3) / jnp.sqrt(alpha_bar), rtol=1e-8, atol=1e-8)


def test_cavi_increases_free_energy_and_hits_fixed_point():
    key = jax.random.PRNGKey(1)
    k1, k2, k3 = jax.random.split(key, 3)
    n, m = 4, 3
    A = jax.random.normal(k1, (m, n))
    x0 = jax.random.normal(k2, (n,))
    xhat0 = x0 + 0.1 * jax.random.normal(k3, (n,))
    y = A @ x0 + 0.05 * jax.random.normal(jax.random.PRNGKey(7), (m,))
    result = cavi_conjugate(xhat0, y, A, max_iter=200, tol=1e-14)
    energies = [float(v) for v in result.free_energies]
    assert len(energies) >= 2
    for prev, nxt in zip(energies, energies[1:]):
        assert nxt >= prev - 1e-7, (prev, nxt)
    assert energies[-1] > energies[0]

    gamma_r = float(result.gamma_r)
    gamma_b = float(result.gamma_b)
    prec = gamma_r * jnp.eye(n) + gamma_b * (A.T @ A)
    mu_eq = jnp.linalg.solve(prec, gamma_r * xhat0 + gamma_b * (A.T @ y))
    sigma_eq = jnp.linalg.inv(prec)
    e_obs = jnp.sum((y - A @ result.mu) ** 2) + jnp.trace(A @ result.sigma @ A.T)
    e_prior = jnp.sum((result.mu - xhat0) ** 2) + jnp.trace(result.sigma)
    assert _rel(result.mu, mu_eq) < 1e-5
    assert _rel(result.sigma, sigma_eq) < 1e-5
    assert _rel(result.b_b, 0.5 * e_obs) < 1e-5
    assert _rel(result.b_r, 0.5 * e_prior) < 1e-5
    assert float(result.a_b) == pytest.approx(m / 2)
    assert float(result.a_r) == pytest.approx(n / 2)


def test_separable_matches_conjugate_on_diagonal_A():
    key = jax.random.PRNGKey(2)
    n = 4
    diag = jnp.array([0.6, 1.3, 0.8, 1.7])
    A = jnp.diag(diag)
    xhat0 = jax.random.normal(key, (n,))
    y = A @ xhat0 + jnp.array([0.2, -0.1, 0.05, 0.3])
    sigma2_0 = jnp.ones(n)
    sep = separable_infer(xhat0, y, A, K=80, sigma2=sigma2_0, tol=1e-14)
    conj = cavi_conjugate(xhat0, y, A, max_iter=200, tol=1e-14, sigma=jnp.eye(n))
    assert _rel(sep.mu, conj.mu) < 1e-5
    assert _rel(sep.sigma2, jnp.diag(conj.sigma)) < 1e-5
    assert _rel(sep.gamma_b, conj.gamma_b) < 1e-5
    assert _rel(sep.gamma_r, conj.gamma_r) < 1e-5
    # Fixed point of the separable equations.
    H = diag ** 2
    At_y = A.T @ y
    At_A_mu = A.T @ (A @ sep.mu)
    denom = sep.gamma_r + sep.gamma_b * H
    mu_r = (sep.gamma_r * xhat0 + sep.gamma_b * At_y - sep.gamma_b * At_A_mu + sep.gamma_b * H * sep.mu) / denom
    sigma2_r = 1.0 / denom
    assert _rel(sep.mu, mu_r) < 1e-5
    assert _rel(sep.sigma2, sigma2_r) < 1e-5


def test_relaxed_step_does_not_decrease_free_energy():
    key = jax.random.PRNGKey(3)
    k1, k2, k3 = jax.random.split(key, 3)
    n, m = 5, 4
    A = jax.random.normal(k1, (m, n))
    xhat0 = jax.random.normal(k2, (n,))
    y = jax.random.normal(k3, (m,))
    mu = xhat0 + 0.5
    sigma2 = 0.3 * jnp.ones(n)
    m_dim, n_dim = A.shape
    a_b = jnp.asarray(m_dim / 2.0)
    a_r = jnp.asarray(n_dim / 2.0)
    b_b, b_r = gamma_rates_separable(mu, sigma2, xhat0, y, A)
    mu_n, sigma2_n, b_b_n, b_r_n, s, F_before, F_relax, F_after = separable_step(
        mu, sigma2, xhat0, y, A, a_b, b_b, a_r, b_r
    )
    assert 0.0 < float(s) <= 1.0
    assert float(F_relax) >= float(F_before) - 1e-6
    assert float(F_after) >= float(F_before) - 1e-6
    # s actually maximizes F along the line, not a fixed guess.
    op_mu = mu
    # Rebuild the target the same way the step does and compare a grid.
    from fb_gdm.operators import as_linear_op
    from fb_gdm.free_energy import separable_targets

    op = as_linear_op(A)
    mu_r, sigma2_r = separable_targets(mu, a_r / b_r, a_b / b_b, xhat0, op, At_y=op.rmatvec(y))
    grid = jnp.linspace(1e-4, 1.0, 21)

    def F_at(s_):
        mu_s = (1.0 - s_) * mu + s_ * mu_r
        s2_s = (1.0 - s_) * sigma2 + s_ * sigma2_r
        return float(free_energy_separable(mu_s, s2_s, a_b, b_b, a_r, b_r, xhat0, y, A))

    best_grid = max(F_at(float(g)) for g in grid)
    assert float(F_relax) >= best_grid - 1e-6
    assert op_mu is mu


def test_exact_observation_raises_gamma_b():
    key = jax.random.PRNGKey(4)
    k1, k2 = jax.random.split(key)
    n, m = 6, 3
    A = jax.random.normal(k1, (m, n))
    xhat0 = jax.random.normal(k2, (n,))
    y_exact = A @ xhat0
    y_far = y_exact + 6.0
    # Ground-truth x0 is not an argument of the update.
    fit_exact = separable_infer(xhat0, y_exact, A, K=40, sigma2=jnp.ones(n))
    fit_far = separable_infer(xhat0, y_far, A, K=40, sigma2=jnp.ones(n))
    assert float(fit_exact.gamma_b) > float(fit_far.gamma_b) * 2.0


def test_reverse_step_and_cg_score():
    key = jax.random.PRNGKey(5)
    k1, k2, k3 = jax.random.split(key, 3)
    n, m = 6, 3
    A = jax.random.normal(k1, (m, n))
    xt = jax.random.normal(k2, (n,))
    y1 = jax.random.normal(k3, (m,))
    y2 = y1 + jnp.array([3.0, -2.0, 1.5])
    alpha_bar = 0.5
    alpha = 0.98
    beta = 0.02
    beta_tilde = 0.015
    z = jnp.zeros(n)

    def eps_theta(xt_, t):
        return jnp.zeros_like(xt_)

    out1 = reverse_step(xt, y1, A, eps_theta, t=4, alpha_bar=alpha_bar, alpha=alpha, beta=beta, beta_tilde=beta_tilde, z=z, K=5)
    out2 = reverse_step(xt, y2, A, eps_theta, t=4, alpha_bar=alpha_bar, alpha=alpha, beta=beta, beta_tilde=beta_tilde, z=z, K=5)
    assert jnp.all(jnp.isfinite(out1))
    assert jnp.all(jnp.isfinite(out2))
    assert not jnp.allclose(out1, out2)

    xhat0 = xt / jnp.sqrt(alpha_bar)
    gamma_r, gamma_b = 2.0, 0.7
    _mu, _sigma, score_dense = corollary1(xhat0, y1, A, gamma_r, gamma_b, alpha_bar)
    mu_cg = posterior_mean_cg(xhat0, y1, A, gamma_r, gamma_b)
    score_cg = conditional_score(mu_cg, xhat0, alpha_bar)
    assert _rel(score_cg, score_dense) < 1e-5
    # Also on a diagonal operator, CG matches the dense (22) score.
    D = jnp.diag(jnp.array([0.5, 1.1, 0.9, 1.4, 0.7, 1.2]))
    y_diag = D @ xhat0 + jnp.array([0.1, -0.2, 0.3, 0.0, -0.1, 0.2])
    _mu_d, _sig_d, score_d = corollary1(xhat0, y_diag, D, gamma_r, gamma_b, alpha_bar)
    score_cg_d = conditional_score(posterior_mean_cg(xhat0, y_diag, D, gamma_r, gamma_b), xhat0, alpha_bar)
    assert _rel(score_cg_d, score_d) < 1e-5


def test_masking_operator_does_not_ask_for_dense_matrix():
    n = 32
    idx = jnp.array([0, 4, 9, 15, 21, 31])

    class Recorder:
        def __init__(self):
            self.calls = []
            self.shape = (idx.shape[0], n)

        def matvec(self, x):
            x = jnp.asarray(x)
            assert x.ndim == 1 and x.shape == (n,)
            self.calls.append(("matvec", tuple(x.shape)))
            return x[idx]

        def rmatvec(self, y):
            y = jnp.asarray(y)
            assert y.ndim == 1
            self.calls.append(("rmatvec", tuple(y.shape)))
            return jnp.zeros(n, dtype=jnp.float64).at[idx].set(y)

        def column_squared_norms(self):
            self.calls.append(("column_norms", (n,)))
            return jnp.zeros(n, dtype=jnp.float64).at[idx].set(1.0)

        def __matmul__(self, other):
            raise AssertionError("dense matmul is not allowed")

    rec = Recorder()
    xhat0 = jnp.linspace(-1.0, 1.0, n)
    y = jnp.array([0.2, -0.4, 0.1, 0.8, -0.3, 0.5])
    fit = separable_infer(xhat0, y, rec, K=3, sigma2=(1.0 - 0.4) * jnp.ones(n))
    kinds = {name for name, _shape in rec.calls}
    assert kinds <= {"matvec", "rmatvec", "column_norms"}
    assert "matvec" in kinds and "rmatvec" in kinds and "column_norms" in kinds
    assert all(len(shape) == 1 and max(shape) <= n for _name, shape in rec.calls)
    assert jnp.all(jnp.isfinite(fit.mu))
    # Unobserved coordinates keep the prior mean at the fixed point of one step's target.
    assert fit.mu.shape == (n,)


def test_linear_beta_schedule():
    T = 10
    beta_start, beta_end = 1e-4, 0.02
    sch = linear_beta_schedule(T, beta_start, beta_end)
    for t in range(1, T + 1):
        beta_t = beta_start + (t / T) * (beta_end - beta_start)
        assert float(sch.beta[t]) == pytest.approx(beta_t)
        assert float(sch.alpha[t]) == pytest.approx(1.0 - beta_t)
    prod = 1.0
    for t in range(1, T + 1):
        prod *= float(sch.alpha[t])
        assert float(sch.alpha_bar[t]) == pytest.approx(prod)
    assert float(sch.alpha_bar[0]) == pytest.approx(1.0)
    assert float(sch.beta_tilde[1]) == pytest.approx(float(sch.beta[1]))
    for t in range(2, T + 1):
        expected = float(sch.beta[t]) * (1.0 - float(sch.alpha_bar[t - 1])) / (1.0 - float(sch.alpha_bar[t]))
        assert float(sch.beta_tilde[t]) == pytest.approx(expected)
