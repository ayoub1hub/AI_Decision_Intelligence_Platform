from __future__ import annotations

from collections.abc import Callable

from numerical_lab.core import (
    InvalidInputError,
    Result,
    Timer,
    has_converged,
)

__all__ = ["dichotomy"]


def dichotomy(
    f: Callable[[float], float],
    a: float,
    b: float,
    *,
    tol: float = 1e-10,
    max_iter: int = 200,
    mode: str = "absolute",
) -> Result:
    """Find a root of ``f`` on ``[a, b]`` by bisection.

    Parameters
    ----------
    f : callable
        Continuous function of one real variable.
    a, b : float
        Interval bounds. Must satisfy ``a < b`` and ``f(a) * f(b) < 0``.
    tol : float, default 1e-10
        Convergence tolerance on the interval half-width.
    max_iter : int, default 200
        Maximum number of iterations.
    mode : {"absolute", "relative", "residual"}, default "absolute"
        Convergence criterion. See :func:`numerical_lab.core.has_converged`.

    Returns
    -------
    Result
        With ``solution`` = midpoint of final interval, ``history`` =
        half-width at each iteration, ``n_eval`` = number of ``f`` calls.

    Raises
    ------
    InvalidInputError
        If ``a >= b``, or if ``f(a)`` and ``f(b)`` do not bracket a root.

    Examples
    --------
    >>> import math
    >>> res = dichotomy(lambda x: x**2 - 2, 0.0, 2.0)
    >>> abs(res.solution - math.sqrt(2)) < 1e-9
    True
    >>> res.converged
    True
    >>> res.order()  # ~1.0 for bisection
    1.0
    """
    # --- Input validation ------------------------------------------------
    if not (a < b):
        raise InvalidInputError(f"Require a < b, got a={a}, b={b}.")
    if tol <= 0:
        raise InvalidInputError(f"tol must be positive, got {tol}.")
    if max_iter <= 0:
        raise InvalidInputError(f"max_iter must be positive, got {max_iter}.")

    with Timer() as timer:
        fa = f(a)
        fb = f(b)
        n_eval = 2

        # Handle roots exactly at endpoints (fast exit)
        if fa == 0.0:
            return Result(
                solution=a,
                converged=True,
                iterations=0,
                history=[0.0],
                n_eval=n_eval,
                elapsed=timer.elapsed,
                info={"reason": "f(a) == 0"},
            )
        if fb == 0.0:
            return Result(
                solution=b,
                converged=True,
                iterations=0,
                history=[0.0],
                n_eval=n_eval,
                elapsed=timer.elapsed,
                info={"reason": "f(b) == 0"},
            )

        # Bracketing check
        if fa * fb > 0:
            raise InvalidInputError(
                f"f(a) and f(b) must have opposite signs; got f({a})={fa}, f({b})={fb}."
            )

        # --- Iteration loop ---------------------------------------------
        history: list[float] = []
        c = 0.5 * (a + b)
        converged = False

        for k in range(1, max_iter + 1):
            c = 0.5 * (a + b)
            fc = f(c)
            n_eval += 1

            half_width = 0.5 * (b - a)
            history.append(half_width)

            if has_converged(half_width, tol, mode=mode, x=c):
                converged = True
                break

            # Keep the sub-interval with the sign change
            if fa * fc < 0:
                b, fb = c, fc
            else:
                a, fa = c, fc

    return Result(
        solution=c,
        converged=converged,
        iterations=k,
        history=history,
        n_eval=n_eval,
        elapsed=timer.elapsed,
        info={"final_interval": (a, b), "tol": tol, "mode": mode},
    )
