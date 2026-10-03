"""LU decomposition with partial pivoting.

Factors a square matrix A into:

    P A = L U

where P is a permutation matrix, L is unit lower-triangular, and U is
upper-triangular. Once the factorization is computed, solving A x = b
for any number of right-hand sides costs only O(n^2) each, instead of
O(n^3) for a fresh Gaussian elimination.

Use cases
---------
- Many systems with the same A, different b.
- Computing the inverse A^{-1} column by column.
- Computing the determinant of A (product of U's diagonal, times
  (-1)^(number of swaps)).

Complexity
----------
- Factorization: O(n^3).
- Solve per right-hand side: O(n^2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from numerical_lab.core import (
    InvalidInputError,
    Result,
    SingularMatrixError,
    Timer,
)

__all__ = ["LUFactorization", "lu_factor", "lu_solve"]


@dataclass
class LUFactorization:
    """Container for an LU factorization P A = L U.

    Attributes
    ----------
    L : np.ndarray, shape (n, n)
        Unit lower-triangular matrix.
    U : np.ndarray, shape (n, n)
        Upper-triangular matrix.
    P : np.ndarray, shape (n, n)
        Permutation matrix such that P A = L U.
    swaps : list of (int, int)
        Row swaps performed during pivoting (for diagnostics).
    n : int
        Size of the matrix.
    """

    L: np.ndarray
    U: np.ndarray
    P: np.ndarray
    swaps: list[tuple[int, int]]
    n: int

    def determinant(self) -> float:
        """Compute det(A) from U's diagonal and the parity of swaps."""
        det = float(np.prod(np.diag(self.U)))
        # Each swap flips the sign
        return det * ((-1) ** len(self.swaps))


def lu_factor(
    A: npt.ArrayLike,
    *,
    tol: float = 1e-12,
) -> LUFactorization:
    """Factor ``A`` into ``P A = L U`` with partial pivoting.

    Parameters
    ----------
    A : array_like, shape (n, n)
        Square matrix to factor.
    tol : float, default 1e-12
        Threshold below which a pivot is considered zero.

    Returns
    -------
    LUFactorization
        The L, U, P factors and pivoting info.

    Raises
    ------
    InvalidInputError
        If A is not square or tol is not positive.
    SingularMatrixError
        If a pivot is smaller than tol in magnitude.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[2.0, 1.0], [1.0, 3.0]])
    >>> fact = lu_factor(A)
    >>> np.allclose(fact.P @ A, fact.L @ fact.U)
    True
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise InvalidInputError(f"A must be square, got shape {A.shape}.")
    if tol <= 0:
        raise InvalidInputError(f"tol must be positive, got {tol}.")

    n = A.shape[0]
    U = A.copy()
    L = np.eye(n)
    P = np.eye(n)
    swaps: list[tuple[int, int]] = []

    for k in range(n - 1):
        # Find pivot
        pivot_row = k + int(np.argmax(np.abs(U[k:, k])))
        if abs(U[pivot_row, k]) < tol:
            raise SingularMatrixError(f"Matrix is singular (pivot {k} ≈ 0 at row {pivot_row}).")

        # Swap rows in U, L, P
        if pivot_row != k:
            U[[k, pivot_row]] = U[[pivot_row, k]]
            # Swap already-computed part of L (columns 0..k-1)
            L[[k, pivot_row], :k] = L[[pivot_row, k], :k]
            P[[k, pivot_row]] = P[[pivot_row, k]]
            swaps.append((k, pivot_row))

        # Eliminate below the pivot
        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] -= factor * U[k, k:]

    if abs(U[n - 1, n - 1]) < tol:
        raise SingularMatrixError("Matrix is singular (last pivot ≈ 0).")

    return LUFactorization(L=L, U=U, P=P, swaps=swaps, n=n)


def lu_solve(
    fact: LUFactorization,
    b: npt.ArrayLike,
) -> Result:
    """Solve ``A x = b`` given a precomputed LU factorization.

    Parameters
    ----------
    fact : LUFactorization
        Result of :func:`lu_factor`.
    b : array_like, shape (n,)
        Right-hand side vector.

    Returns
    -------
    Result
        With ``solution`` = x and metadata.

    Raises
    ------
    InvalidInputError
        If b has the wrong shape.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[2.0, 1.0], [1.0, 3.0]])
    >>> b = np.array([3.0, 4.0])
    >>> fact = lu_factor(A)
    >>> res = lu_solve(fact, b)
    >>> np.allclose(res.solution, [1.0, 1.0])
    True
    """
    b = np.asarray(b, dtype=float).copy()
    n = fact.n
    if b.shape != (n,):
        raise InvalidInputError(f"b must have shape ({n},), got {b.shape}.")

    with Timer() as timer:
        # Apply permutation to b
        b_perm = fact.P @ b

        # Forward substitution: L y = b_perm
        y = np.zeros(n)
        for i in range(n):
            y[i] = b_perm[i] - fact.L[i, :i] @ y[:i]

        # Back substitution: U x = y
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = (y[i] - fact.U[i, i + 1 :] @ x[i + 1 :]) / fact.U[i, i]

    return Result(
        solution=x,
        converged=True,
        iterations=n,
        history=[],
        n_eval=2 * n * n,  # rough estimate
        elapsed=timer.elapsed,
        info={"n": n, "det": fact.determinant()},
    )
