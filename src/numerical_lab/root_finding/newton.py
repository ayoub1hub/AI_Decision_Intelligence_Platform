"""Newton-Raphson method for root finding.

Newton's method uses the tangent line at the current iterate to
produce the next one:

    x_{k+1} = x_k - f(x_k) / f'(x_k)

It converges quadratically (order 2) near a simple root, but requires
the derivative and a good initial guess.

Convergence
-----------
If f is twice differentiable near the root r, f(r) = 0, f'(r) != 0,
and x_0 is sufficiently close to r, then:

    |x_{k+1} - r| ≈ C * |x_k - r|²

so the number of correct digits roughly doubles each iteration.

Cost
----
- 2 function evaluations per iteration (f and f').
- ~log2(log2(1/tol)) iterations for quadratic convergence.
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

__all__ = ["newton"]


def newton(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    *,
    tol: float = 1e-10,
    max_iter: int = 100,
    mode: str = "absolute",
) -> Result:
    """Find a root of ``f`` by the Newton-Raphson method.

    Parameters
    ----------
    f : callable
        Function of one real variable, assumed differentiable.
    df : callable
        Derivative of ``f``.
    x0 : float
        Initial guess.
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
        at each iteration, ``n_eval`` = total number of ``f`` and ``df``
        calls (``f`` and ``df`` count separately).

    Raises
    ------
    InvalidInputError
        If ``tol <= 0``, ``max_iter <= 0``, or if ``f'(x_k) == 0``
        during the iteration (division by zero).
    ConvergenceError
        If the method does not converge within ``max_iter`` iterations.

    Notes
    -----
    Unlike :func:`dichotomy`, Newton's method does **not** require a
    bracketing interval, but it can diverge or oscillate if ``x0`` is
    far from the root or if ``f'`` vanishes nearby.

    Examples
    --------
    >>> import math
    >>> res = newton(lambda x: x**2 - 2, lambda x: 2*x, 1.0)
    >>> abs(res.solution - math.sqrt(2)) < 1e-9
    True
    >>> res.converged
    True
    >>> res.order() > 1.5
    True
    """
    # --- Input validation ------------------------------------------------
    if tol <= 0:
        raise InvalidInputError(f"tol must be positive, got {tol}.")
    if max_iter <= 0:
        raise InvalidInputError(f"max_iter must be positive, got {max_iter}.")

    with Timer() as timer:
        x = float(x0)
        history: list[float] = []
        n_eval = 0
        converged = False

        for k in range(1, max_iter + 1):
            fx = f(x)
            dfx = df(x)
            n_eval += 2

            if dfx == 0.0:
                raise InvalidInputError(
                    f"Derivative is zero at x={x}; Newton's method cannot proceed."
                )

            step = fx / dfx
            x_new = x - step
            history.append(abs(x_new - x))

            if has_converged(abs(x_new - x), tol, mode=mode, x=x_new):
                x = x_new
                converged = True
                break

            x = x_new

    if not converged:
        raise ConvergenceError(
            f"Newton did not converge within {max_iter} iterations "
            f"(last step = {history[-1]:.3e}, tol = {tol:.3e})."
        )

    return Result(
        solution=x,
        converged=converged,
        iterations=k,
        history=history,
        n_eval=n_eval,
        elapsed=timer.elapsed,
        info={"tol": tol, "mode": mode, "x0": x0},
    )
