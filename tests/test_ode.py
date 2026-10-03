"""Tests for ODE solvers."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from numerical_lab.core import InvalidInputError
from numerical_lab.ode import euler


# ======================================================================
# Basic correctness
# ======================================================================
class TestEulerBasic:
    def test_exponential_growth(self) -> None:
        """y' = y, y(0) = 1 → y(t) = exp(t)."""
        f = lambda t, y: y
        res = euler(f, (0.0, 1.0), [1.0], h=0.001)
        assert np.allclose(res.solution, [np.e], rtol=1e-2)

    def test_linear_decay(self) -> None:
        """y' = -y, y(0) = 1 → y(t) = exp(-t)."""
        f = lambda t, y: -y
        res = euler(f, (0.0, 2.0), [1.0], h=0.001)
        assert np.allclose(res.solution, [np.exp(-2.0)], rtol=1e-2)

    def test_first_order_convergence(self) -> None:
        """Euler is order 1: error ~ C * h."""
        f = lambda t, y: y
        t_final = 1.0
        exact = np.e

        errors = []
        hs = [0.1, 0.05, 0.025, 0.0125]
        for h in hs:
            res = euler(f, (0.0, t_final), [1.0], h=h)
            errors.append(abs(res.solution[0] - exact))

        # Order should be ~1: error halves when h halves
        ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
        for r in ratios:
            assert 1.7 < r < 2.3  # ~2 for order 1

    def test_harmonic_oscillator_short(self) -> None:
        """y'' + y = 0 as a system: y1' = y2, y2' = -y1."""
        f = lambda t, y: np.array([y[1], -y[0]])
        res = euler(f, (0.0, 1.0), [1.0, 0.0], h=0.001)
        # Exact: y1 = cos(t), y2 = -sin(t)
        assert np.allclose(res.solution, [np.cos(1.0), -np.sin(1.0)], atol=1e-3)

    def test_matches_scipy(self) -> None:
        """Compare against scipy.integrate.solve_ivp."""
        f = lambda t, y: y
        res = euler(f, (0.0, 1.0), [1.0], h=0.001)
        ref = solve_ivp(f, (0.0, 1.0), [1.0], method="RK45", rtol=1e-10)
        assert np.allclose(res.solution, ref.y[:, -1], rtol=1e-2)

    def test_trajectory_in_info(self) -> None:
        """info['t'] and info['y'] should contain the full path."""
        f = lambda t, y: y
        res = euler(f, (0.0, 1.0), [1.0], h=0.1)
        t = res.info["t"]
        y = res.info["y"]
        assert len(t) == len(y)
        assert t[0] == 0.0
        assert t[-1] == 1.0
        assert np.isclose(y[0, 0], 1.0)


# ======================================================================
# Vector systems
# ======================================================================
class TestEulerVector:
    def test_2d_system(self) -> None:
        """Lotka-Volterra style system."""

        # dx/dt = x - x*y, dy/dt = -y + x*y
        def f(t, y):
            return np.array([y[0] - y[0] * y[1], -y[1] + y[0] * y[1]])

        res = euler(f, (0.0, 1.0), [1.0, 1.0], h=0.001)
        assert res.solution.shape == (2,)
        assert np.all(np.isfinite(res.solution))


# ======================================================================
# Error handling
# ======================================================================
class TestEulerErrors:
    def test_invalid_t_span(self) -> None:
        f = lambda t, y: y
        with pytest.raises(InvalidInputError, match="Require t0 < tf"):
            euler(f, (1.0, 0.0), [1.0], h=0.1)

    def test_negative_h(self) -> None:
        f = lambda t, y: y
        with pytest.raises(InvalidInputError, match="h must be positive"):
            euler(f, (0.0, 1.0), [1.0], h=-0.1)

    def test_wrong_f_shape(self) -> None:
        """f returns a wrong-shaped array → InvalidInputError."""
        f = lambda t, y: np.array([1.0, 2.0])  # y has shape (1,)
        with pytest.raises(InvalidInputError, match="f returned shape"):
            euler(f, (0.0, 1.0), [1.0], h=0.1)
