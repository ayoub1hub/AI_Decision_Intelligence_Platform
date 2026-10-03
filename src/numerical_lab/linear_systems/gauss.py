"""Gaussian elimination with partial pivoting.

Solves the linear system A x = b by transforming A into an upper
triangular matrix U (forward elimination), then solving U x = c by
back-substitution.

Partial pivoting
----------------
At each step k, we search for the row i >= k with the largest
|A[i, k]| and swap rows k and i. This avoids division by zero and
improves numerical stability.

Complexity
----------
- Time: O(n^3) for the elimination, O(n^2) for back-substitution.
- Space: O(n^2) (in-place modification of A and b).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from numerical_lab.core import (
    InvalidInputError,
    Result,
    SingularMatrixError,
    Timer,
)

__all__ = ["gauss_solve"]


def gauss_solve(
    A: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    tol: float = 1e-12,
) -> Result:
    """Solve ``A x = b`` by Gaussian elimination with partial pivoting.

    Parameters
    ----------
    A : array_like, shape (n, n)
        Square coefficient matrix.
    b : array_like, shape (n,)
        Right-hand side vector.
    tol : float, default 1e-12
        Threshold below which a pivot is considered zero → singular matrix.

    Returns
    -------
    Result
        With ``solution`` = x (ndarray of shape (n,)), ``iterations`` = n
        (number of elimination steps), ``n_eval`` = number of arithmetic
        ops (rough estimate), and ``info`` containing pivot swaps.

    Raises
    ------
    InvalidInputError
        If A is not square, or if A and b have incompatible shapes.
    SingularMatrixError
        If a pivot is smaller than ``tol`` in magnitude.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([[2.0, 1.0], [1.0, 3.0]])
    >>> b = np.array([3.0, 4.0])
    >>> res = gauss_solve(A, b)
    >>> np.allclose(res.solution, [1.0, 1.0])
    True
    """
    # --- Convert & validate ---------------------------------------------
    A = np.asarray(A, dtype=float).copy()
    b = np.asarray(b, dtype=float).copy()

    if A.ndim != 2:
        raise InvalidInputError(f"A must be 2D, got shape {A.shape}.")
    n, m = A.shape
    if n != m:
        raise InvalidInputError(f"A must be square, got shape {A.shape}.")
    if b.shape != (n,):
        raise InvalidInputError(f"b must have shape ({n},), got {b.shape}.")
    if tol <= 0:
        raise InvalidInputError(f"tol must be positive, got {tol}.")

    with Timer() as timer:
        swaps: list[tuple[int, int]] = []
        op_count = 0

        # --- Forward elimination with partial pivoting ------------------
        for k in range(n - 1):
            # Find pivot: row with largest |A[i, k]| for i >= k
            pivot_row = k + int(np.argmax(np.abs(A[k:, k])))
            if abs(A[pivot_row, k]) < tol:
                raise SingularMatrixError(f"Matrix is singular (pivot {k} ≈ 0 at row {pivot_row}).")

            # Swap rows if needed
            if pivot_row != k:
                A[[k, pivot_row]] = A[[pivot_row, k]]
                b[[k, pivot_row]] = b[[pivot_row, k]]
                swaps.append((k, pivot_row))

            # Eliminate below
            for i in range(k + 1, n):
                factor = A[i, k] / A[k, k]
                A[i, k:] -= factor * A[k, k:]
                b[i] -= factor * b[k]
                op_count += 2 * (n - k)

        # Final pivot check
        if abs(A[n - 1, n - 1]) < tol:
            raise SingularMatrixError("Matrix is singular (last pivot ≈ 0).")

        # --- Back substitution ------------------------------------------
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = (b[i] - A[i, i + 1 :] @ x[i + 1 :]) / A[i, i]
            op_count += n - i

    return Result(
        solution=x,
        converged=True,  # direct method → always "converges"
        iterations=n,
        history=[],  # no iterative history for direct methods
        n_eval=op_count,
        elapsed=timer.elapsed,
        info={"swaps": swaps, "n": n},
    )
