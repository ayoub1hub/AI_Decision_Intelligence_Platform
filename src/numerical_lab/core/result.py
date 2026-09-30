"""Unified result object returned by every solver."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np


@dataclass
class Result:
    """Standard output for all numerical methods.

    Attributes
    ----------
    solution : float or np.ndarray
        The computed solution (scalar for root finding, vector for linear
        systems or ODEs).
    converged : bool
        Whether the method reached the tolerance criterion.
    iterations : int
        Number of iterations performed.
    history : list[float]
        Error estimate (or residual) at each iteration. Empty for direct
        methods (e.g. Gauss elimination).
    n_eval : int
        Number of function evaluations (useful to compare cost).
    elapsed : float
        Wall-clock time in seconds.
    info : dict
        Method-specific metadata (e.g. spectral radius, condition number).
    """

    solution: Any
    converged: bool
    iterations: int = 0
    history: list[float] = field(default_factory=list)
    n_eval: int = 0
    elapsed: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def final_error(self) -> float:
        """Last recorded error, or NaN if no history."""
        return self.history[-1] if self.history else float("nan")

    def order(self) -> float:
        """Empirical convergence order from the history of errors.

        Uses the standard estimator:
            p ≈ log(e_{k+1} / e_k) / log(e_k / e_{k-1})
        averaged over the last few valid triplets.
        """
        e = [x for x in self.history if x > 0]
        if len(e) < 3:
            return float("nan")
        orders = []
        for k in range(1, len(e) - 1):
            num = np.log(e[k + 1] / e[k])
            den = np.log(e[k] / e[k - 1])
            if abs(den) > 1e-15:
                orders.append(num / den)
        return float(np.mean(orders[-3:])) if orders else float("nan")

    def __repr__(self) -> str:
        status = "converged" if self.converged else "NOT converged"
        return (
            f"Result({status}, iter={self.iterations}, "
            f"err={self.final_error:.3e}, time={self.elapsed:.3e}s)"
        )


class Timer:
    """Context manager to measure elapsed time."""

    def __enter__(self) -> Self:
        self._start = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self._start