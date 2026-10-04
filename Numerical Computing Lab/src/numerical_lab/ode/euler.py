"""Explicit Euler method for ODEs.

Solves the initial value problem:

    dy/dt = f(t, y),    y(t_0) = y_0

using the explicit (forward) Euler scheme:

    y_{n+1} = y_n + h * f(t_n, y_n)

This is the simplest ODE integrator. It has order 1 (global error
scales as O(h)) and is only conditionally stable, so it requires small
step sizes for stiff problems.

For production, prefer RK4 (order 4) or adaptive methods.

References
----------
- Hairer, Nørsett, Wanner, "Solving Ordinary Differential Equations I"
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from numerical_lab.core import (
    InvalidInputError,
    Result,
    Timer,
)

__all__ = ["euler"]


def euler(
    f: Callable[[float, npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    t_span: tuple[float, float],
    y0: npt.ArrayLike,
    *,
    h: float,
) -> Result:
    """Integrate ``dy/dt = f(t, y)`` with explicit Euler.

    Parameters
    ----------
    f : callable
        Right-hand side. Signature ``f(t, y) -> dy/dt``. Must return an
        array with the same shape as ``y``.
    t_span : (float, float)
        Integration interval ``[t0, tf]``. Must satisfy ``t0 < tf``.
    y0 : array_like
        Initial condition. Scalar or 1D array.
    h : float
        Fixed step size. Must be positive. The number of steps is
        ``ceil((tf - t0) / h)``.

    Returns
    -------
    Result
        ``solution`` = final state ``y(tf)`` (ndarray),
        ``history`` = list of error estimates is empty (no adaptive
        control), ``info`` contains ``t`` (array) and ``y`` (2D array,
        shape ``(n_steps+1, dim)``) for the full trajectory.

    Raises
    ------
    InvalidInputError
        If ``t_span`` is malformed, ``h <= 0``, or ``f`` returns the
        wrong shape.

    Examples
    --------
    >>> import numpy as np
    >>> f = lambda t, y: y
    >>> res = euler(f, (0.0, 1.0), [1.0], h=0.01)
    >>> np.allclose(res.solution, [np.e], rtol=1e-1)
    True
    """
    # --- Validate -------------------------------------------------------
    if not (isinstance(t_span, tuple) and len(t_span) == 2):
        raise InvalidInputError(f"t_span must be a (t0, tf) tuple, got {t_span!r}.")
    t0, tf = float(t_span[0]), float(t_span[1])
    if not (t0 < tf):
        raise InvalidInputError(f"Require t0 < tf, got t0={t0}, tf={tf}.")
    if h <= 0:
        raise InvalidInputError(f"h must be positive, got {h}.")

    y0_arr = np.atleast_1d(np.asarray(y0, dtype=float))
    if y0_arr.ndim != 1:
        raise InvalidInputError(f"y0 must be scalar or 1D, got shape {y0_arr.shape}.")

    # --- Setup grid -----------------------------------------------------
    n_steps = int(np.ceil((tf - t0) / h))
    # Adjust h so that (tf - t0) is an exact multiple of h
    h = (tf - t0) / n_steps
    t = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros((n_steps + 1, y0_arr.size))
    y[0] = y0_arr

    with Timer() as timer:
        # --- Time stepping ---------------------------------------------
        n_eval = 0
        for n in range(n_steps):
            dy = np.atleast_1d(np.asarray(f(t[n], y[n]), dtype=float))
            if dy.shape != y[n].shape:
                raise InvalidInputError(f"f returned shape {dy.shape}, expected {y[n].shape}.")
            y[n + 1] = y[n] + h * dy
            n_eval += 1

    return Result(
        solution=y[-1],
        converged=True,
        iterations=n_steps,
        history=[],  # no error estimate for fixed-step methods
        n_eval=n_eval,
        elapsed=timer.elapsed,
        info={
            "t": t,
            "y": y,
            "h": h,
            "n_steps": n_steps,
            "method": "euler",
        },
    )
