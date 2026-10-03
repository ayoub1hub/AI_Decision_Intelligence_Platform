"""Iterative solvers for linear systems: Jacobi and Gauss-Seidel.

Both methods decompose A = D + L + U (diagonal, strict lower, strict
upper) and iterate:

    Jacobi      : x^{k+1} = D^{-1} (b - (L + U) x^k)
    Gauss-Seidel: x^{k+1} = (D + L)^{-1} (b - U x^k)

Convergence is guaranteed when A is strictly diagonally dominant:

    |a_ii| > sum_{j != i} |a_ij|    for all i

Gauss-Seidel typically converges about twice as fast as Jacobi on the
same problem, at the cost of not being parallelizable.

Complexity
----------
- Per iteration: O(n^2) for dense A (or O(nnz) for sparse A).
- Number of iterations: depends on the spectral radius of the iteration
  matrix; typically O(n) for well-conditioned diagonally dominant A.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from numerical_lab.core import (
    ConvergenceError,
    InvalidInputError,
    Result,
    Timer,
)

__all__ = ["gauss_seidel", "jacobi"]


def _validate_inputs(
    A: npt.ArrayLike,
    b: npt.ArrayLike,
    x0: npt.ArrayLike | None,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shared validation for Jacobi / Gauss-Seidel."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise InvalidInputError(f"A must be square, got shape {A.shape}.")
    n = A.shape[0]
    if b.shape != (n,):
        raise InvalidInputError(f"b must have shape ({n},), got {b.shape}.")
    if tol <= 0:
        raise InvalidInputError(f"tol must be positive, got {tol}.")
    if max_iter <= 0:
        raise InvalidInputError(f"max_iter must be positive, got {max_iter}.")

    diag = np.diag(A)
    if np.any(diag == 0.0):
        raise InvalidInputError("A has a zero on its diagonal; cannot proceed.")

    if x0 is None:
        x = np.zeros(n)
    else:
        x = np.asarray(x0, dtype=float).copy()
        if x.shape != (n,):
            raise InvalidInputError(f"x0 must have shape ({n},), got {x.shape}.")

    return A, b, x


def jacobi(
    A: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    x0: npt.ArrayLike | None = None,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> Result:
    """Solve ``A x = b`` by the Jacobi iteration.

    Parameters
    ----------
    A : array_like, shape (n, n)
        Square coefficient matrix, ideally strictly diagonally dominant.
    b : array_like, shape (n,)
        Right-hand side vector.
    x0 : array_like, shape (n,), optional
        Initial guess. Defaults to zeros.
    tol : float, default 1e-10
        Convergence tolerance on ``||x^{k+1} - x^k||_inf``.
    max_iter : int, default 1000
        Maximum number of iterations.

    Returns
    -------
    Result
        With ``solution`` = x, ``iterations`` = k, ``history`` = residuals.

    Raises
    ------
    InvalidInputError
        On malformed inputs.
    ConvergenceError
        If the method does not converge within ``max_iter``.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[4.0, 1.0], [1.0, 3.0]])
    >>> b = np.array([1.0, 2.0])
    >>> res = jacobi(A, b, tol=1e-10)
    >>> np.allclose(A @ res.solution, b, atol=1e-8)
    True
    """
    A, b, x = _validate_inputs(A, b, x0, tol, max_iter)
    n = A.shape[0]
    D = np.diag(A)
    R = A - np.diag(D)  # off-diagonal part

    with Timer() as timer:
        history: list[float] = []
        converged = False

        for k in range(1, max_iter + 1):
            # x_new_i = (b_i - sum_{j != i} A_ij x_j) / A_ii
            x_new = (b - R @ x) / D
            step = float(np.max(np.abs(x_new - x)))
            history.append(step)

            if step < tol:
                x = x_new
                converged = True
                break

            x = x_new

    if not converged:
        raise ConvergenceError(
            f"Jacobi did not converge within {max_iter} iterations "
            f"(last step = {history[-1]:.3e}, tol = {tol:.3e})."
        )

    return Result(
        solution=x,
        converged=True,
        iterations=k,
        history=history,
        n_eval=k * n * n,  # rough estimate: one mat-vec per iteration
        elapsed=timer.elapsed,
        info={"tol": tol, "method": "jacobi"},
    )


def gauss_seidel(
    A: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    x0: npt.ArrayLike | None = None,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> Result:
    """Solve ``A x = b`` by the Gauss-Seidel iteration.

    Same as :func:`jacobi`, but uses already-updated values of x within
    the same iteration. Usually converges ~2x faster.

    Parameters
    ----------
    A, b, x0, tol, max_iter
        See :func:`jacobi`.

    Returns
    -------
    Result
        Same structure as :func:`jacobi`, but typically with fewer
        iterations.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[4.0, 1.0], [1.0, 3.0]])
    >>> b = np.array([1.0, 2.0])
    >>> res = gauss_seidel(A, b, tol=1e-10)
    >>> np.allclose(A @ res.solution, b, atol=1e-8)
    True
    """
    A, b, x = _validate_inputs(A, b, x0, tol, max_iter)
    n = A.shape[0]
    D = np.diag(A)

    with Timer() as timer:
        history: list[float] = []
        converged = False

        for k in range(1, max_iter + 1):
            x_old = x.copy()
            for i in range(n):
                # Use already-updated values for j < i, old values for j > i
                sigma = A[i, :i] @ x[:i] + A[i, i + 1 :] @ x[i + 1 :]
                x[i] = (b[i] - sigma) / D[i]

            step = float(np.max(np.abs(x - x_old)))
            history.append(step)

            if step < tol:
                converged = True
                break

    if not converged:
        raise ConvergenceError(
            f"Gauss-Seidel did not converge within {max_iter} iterations "
            f"(last step = {history[-1]:.3e}, tol = {tol:.3e})."
        )

    return Result(
        solution=x,
        converged=True,
        iterations=k,
        history=history,
        n_eval=k * n * n,
        elapsed=timer.elapsed,
        info={"tol": tol, "method": "gauss_seidel"},
    )
