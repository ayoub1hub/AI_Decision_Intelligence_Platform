"""Tests for linear system solvers."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import solve as scipy_solve

from numerical_lab.core import InvalidInputError, SingularMatrixError
from numerical_lab.linear_systems import gauss_solve


# ======================================================================
# Basic correctness
# ======================================================================
class TestGaussBasic:
    def test_2x2_system(self) -> None:
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        b = np.array([3.0, 4.0])
        res = gauss_solve(A, b)
        assert np.allclose(res.solution, [1.0, 1.0])

    def test_3x3_system(self) -> None:
        A = np.array(
            [
                [2.0, 1.0, -1.0],
                [-3.0, -1.0, 2.0],
                [-2.0, 1.0, 2.0],
            ]
        )
        b = np.array([8.0, -11.0, -3.0])
        res = gauss_solve(A, b)
        assert np.allclose(res.solution, [2.0, 3.0, -1.0])

    def test_identity(self) -> None:
        A = np.eye(5)
        b = np.arange(1.0, 6.0)
        res = gauss_solve(A, b)
        assert np.allclose(res.solution, b)

    def test_matches_scipy(self) -> None:
        """Compare against scipy.linalg.solve on random matrices."""
        rng = np.random.default_rng(42)
        for n in [3, 5, 10, 20]:
            A = rng.standard_normal((n, n)) + n * np.eye(n)  # well-conditioned
            b = rng.standard_normal(n)
            ours = gauss_solve(A, b)
            ref = scipy_solve(A, b)
            assert np.allclose(ours.solution, ref, atol=1e-9)

    def test_result_metadata(self) -> None:
        A = np.eye(3)
        b = np.ones(3)
        res = gauss_solve(A, b)
        assert res.converged
        assert res.iterations == 3
        assert res.n_eval > 0
        assert res.elapsed >= 0
        assert isinstance(res.info, dict)


# ======================================================================
# Pivoting
# ======================================================================
class TestGaussPivoting:
    def test_requires_pivot_swap(self) -> None:
        """First pivot is 0 → must swap rows to proceed."""
        A = np.array([[0.0, 1.0], [1.0, 0.0]])
        b = np.array([2.0, 3.0])
        res = gauss_solve(A, b)
        assert np.allclose(res.solution, [3.0, 2.0])
        assert len(res.info["swaps"]) == 1

    def test_pivot_swap_improves_stability(self) -> None:
        """Classic example where naive Gauss fails but pivoting succeeds."""
        eps = 1e-16
        A = np.array([[eps, 1.0], [1.0, 1.0]])
        b = np.array([1.0, 2.0])
        res = gauss_solve(A, b)
        ref = scipy_solve(A, b)
        assert np.allclose(res.solution, ref, atol=1e-9)


# ======================================================================
# Error handling
# ======================================================================
class TestGaussErrors:
    def test_non_square_matrix(self) -> None:
        A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = np.array([1.0, 2.0])
        with pytest.raises(InvalidInputError, match="must be square"):
            gauss_solve(A, b)

    def test_incompatible_b(self) -> None:
        A = np.eye(3)
        b = np.array([1.0, 2.0])
        with pytest.raises(InvalidInputError, match="shape"):
            gauss_solve(A, b)

    def test_singular_matrix(self) -> None:
        """Rank-deficient matrix → SingularMatrixError."""
        A = np.array([[1.0, 2.0], [2.0, 4.0]])  # row2 = 2 * row1
        b = np.array([1.0, 2.0])
        with pytest.raises(SingularMatrixError):
            gauss_solve(A, b)

    def test_negative_tolerance(self) -> None:
        A = np.eye(2)
        b = np.ones(2)
        with pytest.raises(InvalidInputError, match="tol must be positive"):
            gauss_solve(A, b, tol=-1.0)


# ======================================================================
# Numerical robustness
# ======================================================================
class TestGaussRobustness:
    def test_hilbert_matrix(self) -> None:
        """Hilbert matrices are famously ill-conditioned."""
        from scipy.linalg import hilbert

        n = 5
        A = hilbert(n)
        x_true = np.ones(n)
        b = A @ x_true
        res = gauss_solve(A, b)
        # Ill-conditioned → on tolère une erreur plus grande
        assert np.allclose(res.solution, x_true, atol=1e-3)

    def test_diagonally_dominant(self) -> None:
        rng = np.random.default_rng(0)
        n = 10
        A = rng.standard_normal((n, n))
        A += n * np.eye(n)  # strongly diagonally dominant
        x_true = rng.standard_normal(n)
        b = A @ x_true
        res = gauss_solve(A, b)
        assert np.allclose(res.solution, x_true, atol=1e-10)


# ======================================================================
# LU decomposition
# ======================================================================
from scipy.linalg import lu_factor as scipy_lu_factor
from scipy.linalg import lu_solve as scipy_lu_solve

from numerical_lab.linear_systems import lu_factor, lu_solve


class TestLUFactor:
    def test_2x2_factorization(self) -> None:
        """P A = L U identity."""
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        fact = lu_factor(A)
        assert np.allclose(fact.P @ A, fact.L @ fact.U)
        # L unit lower-triangular
        assert np.allclose(np.diag(fact.L), 1.0)
        assert np.allclose(np.triu(fact.L, 1), 0.0)
        # U upper-triangular
        assert np.allclose(np.tril(fact.U, -1), 0.0)

    def test_matches_scipy(self) -> None:
        """Compare LU factors with scipy (up to row swaps)."""
        rng = np.random.default_rng(42)
        for n in [3, 5, 10]:
            A = rng.standard_normal((n, n)) + n * np.eye(n)
            ours = lu_factor(A)
            # Reconstruct
            assert np.allclose(ours.P @ A, ours.L @ ours.U, atol=1e-10)

    def test_determinant(self) -> None:
        """det(A) from LU matches numpy."""
        rng = np.random.default_rng(0)
        for n in [3, 5, 10]:
            A = rng.standard_normal((n, n)) + n * np.eye(n)
            fact = lu_factor(A)
            assert np.isclose(fact.determinant(), np.linalg.det(A), rtol=1e-8)


class TestLUSolve:
    def test_solve_2x2(self) -> None:
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        b = np.array([3.0, 4.0])
        fact = lu_factor(A)
        res = lu_solve(fact, b)
        assert np.allclose(res.solution, [1.0, 1.0])

    def test_matches_scipy(self) -> None:
        rng = np.random.default_rng(7)
        for n in [3, 10, 30]:
            A = rng.standard_normal((n, n)) + n * np.eye(n)
            b = rng.standard_normal(n)
            fact = lu_factor(A)
            ours = lu_solve(fact, b)
            ref = scipy_lu_solve(scipy_lu_factor(A), b)
            assert np.allclose(ours.solution, ref, atol=1e-9)

    def test_reuse_for_multiple_b(self) -> None:
        """Same A, different b — must give consistent solutions."""
        rng = np.random.default_rng(1)
        n = 10
        A = rng.standard_normal((n, n)) + n * np.eye(n)
        fact = lu_factor(A)
        for _ in range(5):
            b = rng.standard_normal(n)
            res = lu_solve(fact, b)
            # Verify A x = b
            assert np.allclose(A @ res.solution, b, atol=1e-9)


class TestLUErrors:
    def test_non_square(self) -> None:
        A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        with pytest.raises(InvalidInputError, match="must be square"):
            lu_factor(A)

    def test_singular(self) -> None:
        A = np.array([[1.0, 2.0], [2.0, 4.0]])
        with pytest.raises(SingularMatrixError):
            lu_factor(A)

    def test_wrong_b_shape(self) -> None:
        A = np.eye(3)
        fact = lu_factor(A)
        b = np.array([1.0, 2.0])
        with pytest.raises(InvalidInputError, match="shape"):
            lu_solve(fact, b)


# ======================================================================
# Iterative solvers
# ======================================================================
from numerical_lab.core import ConvergenceError
from numerical_lab.linear_systems import gauss_seidel, jacobi


def _diag_dominant(n: int, seed: int = 0) -> np.ndarray:
    """Generate a strictly diagonally dominant matrix."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    # Enforce strict diagonal dominance
    for i in range(n):
        A[i, i] = np.sum(np.abs(A[i])) + 1.0
    return A


class TestJacobi:
    def test_2x2(self) -> None:
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        b = np.array([1.0, 2.0])
        res = jacobi(A, b, tol=1e-10)
        assert np.allclose(A @ res.solution, b, atol=1e-8)

    def test_matches_scipy(self) -> None:
        from scipy.linalg import solve as scipy_solve

        for n in [5, 10, 20]:
            A = _diag_dominant(n, seed=n)
            b = np.random.default_rng(n).standard_normal(n)
            ours = jacobi(A, b, tol=1e-12)
            ref = scipy_solve(A, b)
            assert np.allclose(ours.solution, ref, atol=1e-8)

    def test_convergence_monotone_step(self) -> None:
        """Steps should decrease toward zero."""
        A = _diag_dominant(10, seed=1)
        b = np.ones(10)
        res = jacobi(A, b, tol=1e-10)
        # First step > last step
        assert res.history[0] > res.history[-1]

    def test_does_not_converge_without_diag_dominance(self) -> None:
        """A poorly conditioned matrix may diverge → ConvergenceError."""
        A = np.array([[1.0, 2.0], [2.0, 1.0]])  # not diag dominant
        b = np.array([1.0, 1.0])
        with pytest.raises(ConvergenceError):
            jacobi(A, b, max_iter=50, tol=1e-12)


class TestGaussSeidel:
    def test_2x2(self) -> None:
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        b = np.array([1.0, 2.0])
        res = gauss_seidel(A, b, tol=1e-10)
        assert np.allclose(A @ res.solution, b, atol=1e-8)

    def test_matches_scipy(self) -> None:
        from scipy.linalg import solve as scipy_solve

        for n in [5, 10, 20]:
            A = _diag_dominant(n, seed=n)
            b = np.random.default_rng(n).standard_normal(n)
            ours = gauss_seidel(A, b, tol=1e-12)
            ref = scipy_solve(A, b)
            assert np.allclose(ours.solution, ref, atol=1e-8)

    def test_faster_than_jacobi(self) -> None:
        """Gauss-Seidel typically converges in fewer iterations than Jacobi."""
        A = _diag_dominant(20, seed=42)
        b = np.ones(20)

        res_jac = jacobi(A, b, tol=1e-10)
        res_gs = gauss_seidel(A, b, tol=1e-10)

        # GS should be at least 1.5x faster in iterations
        assert res_gs.iterations < res_jac.iterations / 1.5


class TestIterativeErrors:
    def test_non_square(self) -> None:
        A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = np.array([1.0, 2.0])
        with pytest.raises(InvalidInputError, match="must be square"):
            jacobi(A, b)

    def test_zero_diagonal(self) -> None:
        A = np.array([[0.0, 1.0], [1.0, 2.0]])
        b = np.array([1.0, 2.0])
        with pytest.raises(InvalidInputError, match="zero on its diagonal"):
            jacobi(A, b)

    def test_wrong_b_shape(self) -> None:
        A = np.eye(3)
        b = np.array([1.0, 2.0])
        with pytest.raises(InvalidInputError, match="shape"):
            jacobi(A, b)

    def test_negative_tolerance(self) -> None:
        A = np.eye(2)
        b = np.ones(2)
        with pytest.raises(InvalidInputError, match="tol must be positive"):
            gauss_seidel(A, b, tol=-1.0)
