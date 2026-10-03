"""Runge-Kutta methods for ODEs.

Implements:

- RK4  : classic 4th-order Runge-Kutta, fixed step.
- RK45 : adaptive Dormand-Prince method, embedded 4(5) pair, with
         automatic step size control.

Both solve the initial value problem:

    dy/dt = f(t, y),    y(t_0) = y_0

with f(t, y) returning an array with the same shape as y.

References
----------
- Dormand, J. R.; Prince, P. J. (1980). "A family of embedded
  Runge-Kutta formulae". J. Comput. Appl. Math.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from numerical_lab.core import (
    ConvergenceError,
    InvalidInputError,
    Result,
    Timer,
)

__all__ = ["rk4", "rk45"]


# ======================================================================
# RK4 — fixed step
# ======================================================================
def rk4(
    f: Callable[[float, npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    t_span: tuple[float, float],
    y0: npt.ArrayLike,
    *,
    h: float,
) -> Result:
    """Integrate ``dy/dt = f(t, y)`` with classic RK4.

    Parameters
    ----------
    f : callable
        Right-hand side, ``f(t, y) -> dy/dt``.
    t_span : (float, float)
        Integration interval ``[t0, tf]``, with ``t0 < tf``.
    y0 : array_like
        Initial condition (scalar or 1D).
    h : float
        Step size, positive.

    Returns
    -------
    Result
        ``solution`` = final y, ``info`` contains ``t`` and ``y`` arrays.

    Raises
    ------
    InvalidInputError
        On malformed inputs.

    Examples
    --------
    >>> import numpy as np
    >>> f = lambda t, y: y
    >>> res = rk4(f, (0.0, 1.0), [1.0], h=0.1)
    >>> np.allclose(res.solution, [np.e], rtol=1e-5)
    True
    """
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

    n_steps = int(np.ceil((tf - t0) / h))
    h = (tf - t0) / n_steps
    t = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros((n_steps + 1, y0_arr.size))
    y[0] = y0_arr

    with Timer() as timer:
        n_eval = 0
        for n in range(n_steps):
            tn, yn = t[n], y[n]
            k1 = np.atleast_1d(np.asarray(f(tn, yn), dtype=float))
            k2 = np.atleast_1d(np.asarray(f(tn + h / 2, yn + h / 2 * k1), dtype=float))
            k3 = np.atleast_1d(np.asarray(f(tn + h / 2, yn + h / 2 * k2), dtype=float))
            k4 = np.atleast_1d(np.asarray(f(tn + h, yn + h * k3), dtype=float))
            y[n + 1] = yn + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            n_eval += 4

    return Result(
        solution=y[-1],
        converged=True,
        iterations=n_steps,
        history=[],
        n_eval=n_eval,
        elapsed=timer.elapsed,
        info={"t": t, "y": y, "h": h, "n_steps": n_steps, "method": "rk4"},
    )


# ======================================================================
# RK45 — adaptive step (Dormand-Prince)
# ======================================================================
# Dormand-Prince coefficients (order 4(5) embedded pair)
_DP_C = np.array([0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0])

_DP_A = np.array(
    [
        [],
        [1 / 5],
        [3 / 40, 9 / 40],
        [44 / 45, -56 / 15, 32 / 9],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
        [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
    ],
    dtype=object,
)

# 5th-order solution weights (also used as the "advancing" weights)
_DP_B5 = np.array([35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0])

# 4th-order solution weights (for error estimate)
_DP_B4 = np.array([5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40])


def _dp_step(f, t, y, h):
    """One Dormand-Prince step. Returns y5, y4, and the 7 k values."""
    k = np.zeros((7, y.size))
    for i in range(7):
        yi = y.copy()
        for j in range(i):
            if _DP_A[i][j] != 0:
                yi = yi + h * _DP_A[i][j] * k[j]
        k[i] = np.atleast_1d(np.asarray(f(t + _DP_C[i] * h, yi), dtype=float))
    y5 = y + h * sum(_DP_B5[i] * k[i] for i in range(7) if _DP_B5[i] != 0)
    y4 = y + h * sum(_DP_B4[i] * k[i] for i in range(7) if _DP_B4[i] != 0)
    return y5, y4, k


def rk45(
    f: Callable[[float, npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    t_span: tuple[float, float],
    y0: npt.ArrayLike,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    h0: float | None = None,
    h_max: float | None = None,
    max_steps: int = 10000,
) -> Result:
    """Integrate ``dy/dt = f(t, y)`` with adaptive Dormand-Prince RK45.

    Parameters
    ----------
    f : callable
        Right-hand side, ``f(t, y) -> dy/dt``.
    t_span : (float, float)
        Integration interval, ``t0 < tf``.
    y0 : array_like
        Initial condition.
    rtol : float, default 1e-6
        Relative tolerance per step.
    atol : float, default 1e-9
        Absolute tolerance per step.
    h0 : float, optional
        Initial step. Defaults to ``(tf - t0) / 100``.
    h_max : float, optional
        Maximum step size. Defaults to ``tf - t0``.
    max_steps : int, default 10000
        Safety bound on the number of steps.

    Returns
    -------
    Result
        ``solution`` = final y, ``history`` = step sizes, ``info``
        contains the full trajectory ``t`` and ``y``.

    Raises
    ------
    InvalidInputError
        On malformed inputs.
    ConvergenceError
        If ``max_steps`` is reached before ``tf``.

    Examples
    --------
    >>> import numpy as np
    >>> f = lambda t, y: y
    >>> res = rk45(f, (0.0, 1.0), [1.0], rtol=1e-8)
    >>> np.allclose(res.solution, [np.e], rtol=1e-6)
    True
    """
    if not (isinstance(t_span, tuple) and len(t_span) == 2):
        raise InvalidInputError(f"t_span must be a (t0, tf) tuple, got {t_span!r}.")
    t0, tf = float(t_span[0]), float(t_span[1])
    if not (t0 < tf):
        raise InvalidInputError(f"Require t0 < tf, got t0={t0}, tf={tf}.")
    if rtol <= 0 or atol <= 0:
        raise InvalidInputError(f"rtol and atol must be positive, got {rtol}, {atol}.")

    y0_arr = np.atleast_1d(np.asarray(y0, dtype=float))
    if y0_arr.ndim != 1:
        raise InvalidInputError(f"y0 must be scalar or 1D, got shape {y0_arr.shape}.")

    T = tf - t0
    h = h0 if h0 is not None else T / 100
    h_max_eff = h_max if h_max is not None else T

    t_list = [t0]
    y_list = [y0_arr.copy()]
    steps = [h]

    with Timer() as timer:
        t = t0
        y = y0_arr.copy()
        n_eval = 0
        step_count = 0

        while t < tf and step_count < max_steps:
            step_count += 1

            # Don't overshoot the final time
            h = min(h, tf - t)

            y5, y4, _ = _dp_step(f, t, y, h)
            n_eval += 7

            # Error estimate (scaled)
            scale = atol + rtol * np.maximum(np.abs(y), np.abs(y5))
            err = float(np.max(np.abs(y5 - y4) / scale))

            # Accept or reject
            if err <= 1.0 or h <= 1e-15 * max(abs(t), 1.0):
                t = t + h
                y = y5
                t_list.append(t)
                y_list.append(y.copy())

            # Adapt step size (PI controller simplified)
            if err == 0.0:
                factor = 5.0
            else:
                factor = 0.9 * err ** (-0.2)  # -1/5 for order-5 method
            factor = float(np.clip(factor, 0.2, 5.0))
            h = min(h * factor, h_max_eff)
            steps.append(h)

        if t < tf:
            raise ConvergenceError(f"rk45 reached max_steps={max_steps} before tf={tf} (t={t}).")

    return Result(
        solution=y,
        converged=True,
        iterations=step_count,
        history=steps,
        n_eval=n_eval,
        elapsed=timer.elapsed,
        info={
            "t": np.array(t_list),
            "y": np.array(y_list),
            "rtol": rtol,
            "atol": atol,
            "method": "rk45",
        },
    )
