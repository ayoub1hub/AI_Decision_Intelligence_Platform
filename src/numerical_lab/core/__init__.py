"""Core building blocks shared by all numerical methods."""

from numerical_lab.core.convergence import empirical_order, has_converged
from numerical_lab.core.errors import (
    ConvergenceError,
    InvalidInputError,
    NumericalLabError,
    SingularMatrixError,
)
from numerical_lab.core.result import Result, Timer

__all__ = [
    "ConvergenceError",
    "InvalidInputError",
    "NumericalLabError",
    "Result",
    "SingularMatrixError",
    "Timer",
    "empirical_order",
    "has_converged",
]
