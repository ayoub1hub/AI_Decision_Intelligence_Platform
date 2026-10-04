# mon_script.py
import numpy as np
from numerical_lab.root_finding import newton
from numerical_lab.linear_systems import lu_factor, lu_solve
from numerical_lab.ode import rk45

# 1. Trouver une racine
res = newton(lambda x: x**3 - x - 2, lambda x: 3*x**2 - 1, 1.5)
print(f"Racine trouvée : {res.solution} en {res.iterations} itérations")

# 2. Résoudre un système linéaire (plusieurs b réutilisent la factorisation)
A = np.array([[4.0, 1.0], [1.0, 3.0]])
fact = lu_factor(A)
for b in [np.array([1.0, 2.0]), np.array([3.0, 4.0])]:
    res = lu_solve(fact, b)
    print(f"Solution : {res.solution}")

# 3. Intégrer une ODE
f = lambda t, y: y
res = rk45(f, (0.0, 1.0), [1.0], rtol=1e-8)
print(f"y(1) = {res.solution[0]:.10f} (exact : {np.e:.10f})")