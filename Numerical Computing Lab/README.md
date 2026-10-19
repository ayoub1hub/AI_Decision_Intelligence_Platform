# Numerical Computing Lab

A from-scratch implementation of numerical methods in Python, with **no SciPy in the implementation** (SciPy used only as a reference for validation).

## Philosophy

Every method is:

- **Implemented from scratch** — understand the algorithm, not just call it
- **Tested against SciPy** — numerical correctness verified
- **Benchmarked** — convergence order measured empirically
- **Documented** — docstrings with theory, complexity, and examples

## Modules

### Root Finding

| Method | Order | Iterations (for `1e-10`) | Derivative needed? |
|---|---|---|---|
| **Dichotomy** | 1.0 | ~35 | No |
| **Newton** | 2.0 | ~6 | Yes |
| **Secant** | ~1.618 | ~8 | No |

### Linear Systems

| Method | Type | Complexity | Use case |
|---|---|---|---|
| **Gauss elimination** | Direct | O(n³) | Single solve |
| **LU decomposition** | Direct | O(n³) + O(n²) per solve | Multiple `b` |
| **Jacobi** | Iterative | O(n²)/iter | Sparse matrices |
| **Gauss-Seidel** | Iterative | O(n²)/iter | Sparse, faster than Jacobi |

### ODE Solvers

| Method | Order | Fixed/Adaptive | Use case |
|---|---|---|---|
| **Euler** | 1 | Fixed | Pedagogical |
| **RK4** | 4 | Fixed | Standard production |
| **RK45** | 4(5) | Adaptive | Production with error control |

## Quick Start

```bash
pip install -e ".[dev]"
pytest
```

## Examples

### Root finding

```python
from numerical_lab.root_finding import newton

res = newton(
    f=lambda x: x**2 - 2,
    df=lambda x: 2*x,
    x0=1.0,
    tol=1e-10,
)
print(res.solution)      # 1.4142135623...
print(res.iterations)    # 6
print(res.order())       # ~2.0 (quadratic convergence)
```

### Linear systems

```python
import numpy as np
from numerical_lab.linear_systems import lu_factor, lu_solve

A = np.array([[4.0, 1.0], [1.0, 3.0]])
fact = lu_factor(A)

# Multiple right-hand sides, factor once
for b in [np.array([1.0, 2.0]), np.array([3.0, 4.0])]:
    res = lu_solve(fact, b)
    print(res.solution)
```

### ODE

```python
import numpy as np
from numerical_lab.ode import rk45

# y' = y, y(0) = 1 → y(1) = e
res = rk45(
    f=lambda t, y: y,
    t_span=(0.0, 1.0),
    y0=[1.0],
    rtol=1e-8,
)
print(res.solution)   # [2.718281828...]
```

## Structure

```
src/numerical_lab/
├── core/                    — Result, Timer, has_converged
├── root_finding/            — Dichotomy, Newton, Secant
├── linear_systems/          — Gauss, LU, Jacobi, Gauss-Seidel
└── ode/                     — Euler, RK4, RK45
```

## Tests

**83 tests** covering:

- Numerical correctness (vs SciPy)
- Edge cases (singular matrices, roots at endpoints)
- Convergence order (empirical measurement)
- Error handling (invalid inputs)
- Boundary conditions (non-convergence, singular matrices)

```bash
pytest -v          # all tests
pytest --cov       # with coverage
```

## Benchmarks

All methods are **benchmarked against SciPy** in `tests/`:

- **Dichotomy** vs `scipy.optimize.bisect`
- **Newton** vs `scipy.optimize.newton`
- **LU** vs `scipy.linalg.lu_factor` / `lu_solve`
- **RK4/RK45** vs `scipy.integrate.solve_ivp`

Convergence orders are measured empirically via `Result.order()`:

```python
res = newton(f, df, x0=1.0)
print(res.order())   # ~2.0 (quadratic for Newton)
```

## Key Design Decisions

- **Unified `Result` object** — every solver returns `Result(solution, iterations, history, ...)`
- **Configurable convergence** — absolute, relative, or residual mode
- **1 eval per iteration** — Secant reuses previous `f(x)`, Newton uses `f` and `df` separately
- **Pivot partial** — Gauss and LU use partial pivoting for numerical stability
- **Adaptive step** — RK45 uses Dormand-Prince embedded pair with error control

## License

MIT