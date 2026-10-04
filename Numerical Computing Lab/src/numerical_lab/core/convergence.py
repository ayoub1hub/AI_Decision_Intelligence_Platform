"""Convergence criteria and error estimators."""

from __future__ import annotations

import numpy as np


def has_converged(
    error: float,
    tol: float,
    *,
    mode: str = "absolute",
    x: float | None = None,
) -> bool:
    """Check a convergence criterion.

    Parameters
    ----------
    error : float
        Current error estimate (e.g. |x_{k+1} - x_k| or |f(x_k)|).
    tol : float
        Tolerance.
    mode : {"absolute", "relative", "residual"}
        - absolute : |x_{k+1} - x_k| < tol
        - relative : |x_{k+1} - x_k| / max(|x|, eps) < tol
        - residual : |f(x)| < tol
    x : float, optional
        Current iterate, required for ``relative`` mode.
    """
    if mode == "absolute":
        return error < tol
    if mode == "relative":
        if x is None:
            raise ValueError("`x` is required for relative convergence mode.")
        return error / max(abs(x), np.finfo(float).eps) < tol
    if mode == "residual":
        return error < tol
    raise ValueError(f"Unknown convergence mode: {mode!r}")


def empirical_order(errors: list[float]) -> float:
    """Standalone empirical order estimator (same logic as Result.order)."""
    e = [x for x in errors if x > 0]
    if len(e) < 3:
        return float("nan")
    orders = []
    for k in range(1, len(e) - 1):
        num = np.log(e[k + 1] / e[k])
        den = np.log(e[k] / e[k - 1])
        if abs(den) > 1e-15:
            orders.append(num / den)
    return float(np.mean(orders[-3:])) if orders else float("nan")
