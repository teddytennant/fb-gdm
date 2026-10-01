"""Matvec-only linear maps. Separable updates never form an n×n matrix."""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

Array = jnp.ndarray


class LinearOperator:
    """Linear map A: R^n → R^m given by matvecs and column squared norms.

    ``column_squared_norms`` is H = diag(Aᵀ A), the squared Euclidean norms
    of the columns. Callers that only have matvecs must supply H themselves.
    """

    def __init__(
        self,
        matvec: Callable[[Array], Array],
        rmatvec: Callable[[Array], Array],
        column_squared_norms: Array,
        shape: tuple[int, int],
    ) -> None:
        self._matvec = matvec
        self._rmatvec = rmatvec
        self._column_squared_norms = jnp.asarray(column_squared_norms, dtype=jnp.float64)
        self.shape = (int(shape[0]), int(shape[1]))

    def matvec(self, x: Array) -> Array:
        return jnp.asarray(self._matvec(x), dtype=jnp.float64)

    def rmatvec(self, y: Array) -> Array:
        return jnp.asarray(self._rmatvec(y), dtype=jnp.float64)

    def column_squared_norms(self) -> Array:
        return self._column_squared_norms

    def __matmul__(self, x: Array) -> Array:
        return self.matvec(x)


def as_linear_op(A) -> LinearOperator:
    """Wrap a dense matrix, or pass a duck-typed operator through."""
    if isinstance(A, LinearOperator):
        return A
    if hasattr(A, "matvec") and hasattr(A, "rmatvec"):
        shape = tuple(A.shape)
        if hasattr(A, "column_squared_norms"):
            norms = A.column_squared_norms
            H = norms() if callable(norms) else norms
        elif hasattr(A, "column_norms"):
            norms = A.column_norms
            H = norms() if callable(norms) else norms
        else:
            raise TypeError("operator must expose column_squared_norms or column_norms")
        return LinearOperator(A.matvec, A.rmatvec, H, shape)
    matrix = jnp.asarray(A, dtype=jnp.float64)
    if matrix.ndim != 2:
        raise ValueError(f"A must be an (m, n) matrix, got shape {matrix.shape}")
    m, n = matrix.shape
    H = jnp.sum(matrix * matrix, axis=0)

    def matvec(x: Array) -> Array:
        return matrix @ jnp.asarray(x, dtype=jnp.float64)

    def rmatvec(y: Array) -> Array:
        return matrix.T @ jnp.asarray(y, dtype=jnp.float64)

    return LinearOperator(matvec, rmatvec, H, (m, n))


def row_selection(indices, n: int) -> LinearOperator:
    """Masking operator: A x = x[indices]. Column norms are 0 or 1."""
    idx = jnp.asarray(indices, dtype=jnp.int32)
    n = int(n)
    m = int(idx.shape[0])
    H = jnp.zeros(n, dtype=jnp.float64).at[idx].set(1.0)

    def matvec(x: Array) -> Array:
        return jnp.asarray(x, dtype=jnp.float64)[idx]

    def rmatvec(y: Array) -> Array:
        return jnp.zeros(n, dtype=jnp.float64).at[idx].set(jnp.asarray(y, dtype=jnp.float64))

    return LinearOperator(matvec, rmatvec, H, (m, n))
