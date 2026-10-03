"""Custom exceptions for the numerical_lab package."""


class NumericalLabError(Exception):
    """Base exception for all numerical_lab errors."""


class ConvergenceError(NumericalLabError):
    """Raised when a method fails to converge within max_iter."""


class InvalidInputError(NumericalLabError):
    """Raised when inputs violate the method's assumptions."""


class SingularMatrixError(NumericalLabError):
    """Raised when a linear system is singular or near-singular."""
