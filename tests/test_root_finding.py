"""Tests for root-finding algorithms."""

from __future__ import annotations

import math

import pytest
from scipy.optimize import bisect as scipy_bisect

from numerical_lab.core import InvalidInputError
from numerical_lab.root_finding import dichotomy


# ======================================================================
# Basic correctness
# ======================================================================
class TestDichotomyBasic:
    def test_sqrt2(self) -> None:
        """Classic: f(x) = x² - 2 on [0, 2] → √2."""
        res = dichotomy(lambda x: x**2 - 2, 0.0, 2.0)
        assert res.converged
        assert abs(res.solution - math.sqrt(2)) < 1e-9
        assert res.iterations > 0
        assert res.n_eval == res.iterations + 2  # +2 for f(a), f(b)

    def test_matches_scipy(self) -> None:
        """Compare against scipy.optimize.bisect for several functions."""
        cases = [
            (lambda x: x**3 - x - 2, 1.0, 2.0),
            (lambda x: math.cos(x) - x, 0.0, 1.0),
            (lambda x: math.exp(x) - 3, 0.0, 2.0),
            (lambda x: x**5 - 5 * x + 1, 0.0, 1.0),
        ]
        for f, a, b in cases:
            ours = dichotomy(f, a, b, tol=1e-12)
            ref = scipy_bisect(f, a, b, xtol=1e-12)
            assert abs(ours.solution - ref) < 1e-9

    def test_convergence_order_is_one(self) -> None:
        """Bisection is linear: empirical order ≈ 1.0."""
        res = dichotomy(lambda x: x**2 - 2, 0.0, 2.0, tol=1e-14)
        assert 0.85 < res.order() < 1.15

    def test_result_dataclass_fields(self) -> None:
        res = dichotomy(lambda x: x**2 - 2, 0.0, 2.0)
        assert res.history, "history should not be empty"
        assert res.elapsed >= 0.0
        assert isinstance(res.info, dict)


# ======================================================================
# Edge cases — roots at endpoints
# ======================================================================
class TestDichotomyEndpoints:
    def test_root_at_left_endpoint(self) -> None:
        res = dichotomy(lambda x: x, 0.0, 1.0)
        assert res.converged
        assert res.solution == 0.0
        assert res.iterations == 0
        assert res.info["reason"] == "f(a) == 0"

    def test_root_at_right_endpoint(self) -> None:
        res = dichotomy(lambda x: x - 1, 0.0, 1.0)
        assert res.converged
        assert res.solution == 1.0
        assert res.iterations == 0
        assert res.info["reason"] == "f(b) == 0"


# ======================================================================
# Error handling
# ======================================================================
class TestDichotomyErrors:
    def test_invalid_interval(self) -> None:
        with pytest.raises(InvalidInputError, match="Require a < b"):
            dichotomy(lambda x: x, 2.0, 1.0)

    def test_no_sign_change(self) -> None:
        with pytest.raises(InvalidInputError, match="opposite signs"):
            dichotomy(lambda x: x**2 + 1, -1.0, 1.0)  # always positive

    def test_negative_tolerance(self) -> None:
        with pytest.raises(InvalidInputError, match="tol must be positive"):
            dichotomy(lambda x: x**2 - 2, 0.0, 2.0, tol=-1.0)

    def test_zero_max_iter(self) -> None:
        with pytest.raises(InvalidInputError, match="max_iter must be positive"):
            dichotomy(lambda x: x**2 - 2, 0.0, 2.0, max_iter=0)


# ======================================================================
# Non-convergence
# ======================================================================
class TestDichotomyNonConvergence:
    def test_max_iter_reached(self) -> None:
        """Very tight tolerance, few iterations → should NOT converge."""
        res = dichotomy(lambda x: x**2 - 2, 0.0, 2.0, tol=1e-15, max_iter=5)
        assert not res.converged
        assert res.iterations == 5
        assert len(res.history) == 5


# ======================================================================
# Newton's method
# ======================================================================
from numerical_lab.core import ConvergenceError
from numerical_lab.root_finding import newton


class TestNewtonBasic:
    def test_sqrt2(self) -> None:
        """f(x) = x² - 2, f'(x) = 2x, x0 = 1 → √2."""
        res = newton(lambda x: x**2 - 2, lambda x: 2 * x, 1.0)
        assert res.converged
        assert abs(res.solution - math.sqrt(2)) < 1e-12

    def test_matches_scipy(self) -> None:
        """Compare against scipy.optimize.newton."""
        from scipy.optimize import newton as scipy_newton

        cases = [
            (lambda x: x**3 - x - 2, lambda x: 3 * x**2 - 1, 1.5),
            (lambda x: math.cos(x) - x, lambda x: -math.sin(x) - 1, 1.0),
            (lambda x: math.exp(x) - 3, lambda x: math.exp(x), 1.0),
        ]
        for f, df, x0 in cases:
            ours = newton(f, df, x0, tol=1e-13)
            ref = scipy_newton(f, x0, fprime=df, tol=1e-13)
            assert abs(ours.solution - ref) < 1e-9

    def test_convergence_order_is_two(self) -> None:
        """Newton is quadratic: empirical order ≈ 2.0."""
        res = newton(lambda x: x**2 - 2, lambda x: 2 * x, 1.0, tol=1e-15)
        # On tolère une marge : near-machine-precision, l'ordre peut chuter
        assert res.order() > 1.5

    def test_fewer_iterations_than_dichotomy(self) -> None:
        """Newton doit converger en beaucoup moins d'itérations que Dichotomy."""
        from numerical_lab.root_finding import dichotomy

        f = lambda x: x**2 - 2
        df = lambda x: 2 * x

        res_dich = dichotomy(f, 0.0, 2.0, tol=1e-10)
        res_newt = newton(f, df, 1.0, tol=1e-10)

        # Newton doit être ~5x plus rapide en itérations
        assert res_newt.iterations < res_dich.iterations / 3


class TestNewtonErrors:
    def test_negative_tolerance(self) -> None:
        with pytest.raises(InvalidInputError, match="tol must be positive"):
            newton(lambda x: x**2 - 2, lambda x: 2 * x, 1.0, tol=-1.0)

    def test_zero_derivative(self) -> None:
        """f'(x) = 0 → Newton ne peut pas continuer."""
        with pytest.raises(InvalidInputError, match="Derivative is zero"):
            newton(lambda x: x**2 + 1, lambda x: 2 * x, 0.0)

    def test_non_convergence(self) -> None:
        """Trop peu d'itérations → ConvergenceError."""
        with pytest.raises(ConvergenceError, match="did not converge"):
            newton(lambda x: x**2 - 2, lambda x: 2 * x, 1.0, max_iter=1, tol=1e-15)


# ======================================================================
# Secant method
# ======================================================================
from numerical_lab.root_finding import secant


class TestSecantBasic:
    def test_sqrt2(self) -> None:
        """f(x) = x² - 2, x0 = 1, x1 = 2 → √2."""
        res = secant(lambda x: x**2 - 2, 1.0, 2.0)
        assert res.converged
        assert abs(res.solution - math.sqrt(2)) < 1e-12

    def test_matches_scipy(self) -> None:
        """Compare against scipy.optimize.newton (secant variant)."""
        from scipy.optimize import newton as scipy_newton

        cases = [
            (lambda x: x**3 - x - 2, 1.0, 2.0),
            (lambda x: math.cos(x) - x, 0.5, 1.5),
            (lambda x: math.exp(x) - 3, 0.5, 2.0),
        ]
        for f, x0, x1 in cases:
            ours = secant(f, x0, x1, tol=1e-13)
            # scipy_newton sans fprime utilise la secante
            ref = scipy_newton(f, x1, x1=x0, tol=1e-13)
            assert abs(ours.solution - ref) < 1e-9

    def test_convergence_order_is_golden_ratio(self) -> None:
        """Secant converges with order φ ≈ 1.618."""
        res = secant(lambda x: x**2 - 2, 1.0, 2.0, tol=1e-12)
        # Large marge : le bruit machine peut abaisser l'ordre mesuré
        assert res.order() > 1.3

    def test_faster_than_dichotomy(self) -> None:
        """Secant doit être bien plus rapide que Dichotomy."""
        from numerical_lab.root_finding import dichotomy

        f = lambda x: x**2 - 2

        res_dich = dichotomy(f, 0.0, 2.0, tol=1e-10)
        res_sec = secant(f, 1.0, 2.0, tol=1e-10)

        assert res_sec.iterations < res_dich.iterations / 2

    def test_n_eval_is_optimal(self) -> None:
        """1 eval/itération (hors 2 initiales) — c'est l'avantage clé."""
        res = secant(lambda x: x**2 - 2, 1.0, 2.0)
        # n_eval = 2 initiales + (iterations - 1) pendant la boucle
        assert res.n_eval <= res.iterations + 2


class TestSecantErrors:
    def test_identical_initial_points(self) -> None:
        with pytest.raises(InvalidInputError, match="must differ"):
            secant(lambda x: x**2 - 2, 1.0, 1.0)

    def test_negative_tolerance(self) -> None:
        with pytest.raises(InvalidInputError, match="tol must be positive"):
            secant(lambda x: x**2 - 2, 1.0, 2.0, tol=-1.0)

    def test_zero_slope(self) -> None:
        """f(x0) == f(x1) → pente nulle."""
        # f = constante → f(x0) == f(x1) == 0
        with pytest.raises(InvalidInputError, match="slope is zero"):
            secant(lambda x: 0.0, 1.0, 2.0)

    def test_non_convergence(self) -> None:
        with pytest.raises(ConvergenceError, match="did not converge"):
            secant(lambda x: x**2 - 2, 1.0, 2.0, max_iter=1, tol=1e-15)
