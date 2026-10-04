"""Secant method for root finding.

The secant method replaces the derivative in Newton's method with a
finite-difference approximation using the two most recent iterates:

    x_{k+1} = x_k - f(x_k) * (x_k - x_{k-1}) / (f(x_k) - f(x_{k-1}))

It requires two initial points but no derivative, and converges with
order φ = (1 + sqrt(5)) / 2 ≈ 1.618 (the golden ratio).

Convergence
-----------
Near a simple root r with f'(r) != 0, and for x_0, x_1 sufficiently
close to r:

    |x_{k+1} - r| ≈ C * |x_k - r|^φ

Cost
----
- 1 function evaluation per iteration (after the two initial ones).
- Faster than bisection, slightly slower than Newton.
"""

from __future__ import annotations

from collections.abc import Callable

from numerical_lab.core import (
    ConvergenceError,
    InvalidInputError,
    Result,
    Timer,
    has_converged,
)

__all__ = ["secant"]


def secant(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    *,
    tol: float = 1e-10,
    max_iter: int = 100,
    mode: str = "absolute",
) -> Result:
    """Find a root of ``f`` by the secant method.

    Parameters
    ----------
    f : callable
        Function of one real variable.
    x0, x1 : float
        Two initial guesses. Must satisfy ``x0 != x1``.
    tol : float, default 1e-10
        Convergence tolerance on the step ``|x_{k+1} - x_k|``.
    max_iter : int, default 100
        Maximum number of iterations.
    mode : {"absolute", "relative", "residual"}, default "absolute"
        Convergence criterion. See :func:`numerical_lab.core.has_converged`.

    Returns
    -------
    Result
        With ``solution`` = final iterate, ``history`` = ``|x_{k+1}-x_k|``
        at each iteration, ``n_eval`` = number of ``f`` calls.

    Raises
    ------
    InvalidInputError
        If ``x0 == x1``, ``tol <= 0``, ``max_iter <= 0``, or if the
        secant slope vanishes during the iteration.
    ConvergenceError
        If the method does not converge within ``max_iter`` iterations.

    Notes
    -----
    The secant method converges with order φ ≈ 1.618, the golden ratio,
    strictly between bisection (order 1) and Newton (order 2). It is
    often the method of choice when ``f'`` is unavailable or expensive.

    Examples
    --------
    >>> import math
    >>> res = secant(lambda x: x**2 - 2, 1.0, 2.0)
    >>> abs(res.solution - math.sqrt(2)) < 1e-9
    True
    >>> res.converged
    True
    >>> 1.5 < res.order() < 1.75
    True
    """
    # --- Input validation ------------------------------------------------
    if x0 == x1:
        raise InvalidInputError(f"x0 and x1 must differ, got x0 = x1 = {x0}.")
    if tol <= 0:
        raise InvalidInputError(f"tol must be positive, got {tol}.")
    if max_iter <= 0:
        raise InvalidInputError(f"max_iter must be positive, got {max_iter}.")

    with Timer() as timer:
        x_prev = float(x0)
        x_curr = float(x1)
        f_prev = f(x_prev)
        f_curr = f(x_curr)
        n_eval = 2

        history: list[float] = []
        converged = False

        for k in range(1, max_iter + 1):
            denom = f_curr - f_prev
            if denom == 0.0:
                raise InvalidInputError(
                    f"Secant slope is zero between x={x_prev} and x={x_curr}; cannot proceed."
                )

            step = f_curr * (x_curr - x_prev) / denom
            x_new = x_curr - step
            history.append(abs(x_new - x_curr))

            if has_converged(abs(x_new - x_curr), tol, mode=mode, x=x_new):
                x_curr = x_new
                converged = True
                break

            # Shift for next iteration
            x_prev, f_prev = x_curr, f_curr
            x_curr = x_new
            f_curr = f(x_curr)
            n_eval += 1

    if not converged:
        raise ConvergenceError(
            f"Secant did not converge within {max_iter} iterations "
            f"(last step = {history[-1]:.3e}, tol = {tol:.3e})."
        )

    return Result(
        solution=x_curr,
        converged=converged,
        iterations=k,
        history=history,
        n_eval=n_eval,
        elapsed=timer.elapsed,
        info={"tol": tol, "mode": mode, "x0": x0, "x1": x1},
    )
