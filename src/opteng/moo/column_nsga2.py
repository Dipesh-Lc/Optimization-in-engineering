from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

# constants related to material and physical properties
@dataclass(frozen=True)
class ColumnParams:
    rho: float = 7850.0
    l: float = 5.0
    E: float = 200e9
    M: float = 1000.0
    g: float = 9.81
    sigma_max: float = 250e6


class ColumnProblem(Problem):
    def __init__(self, p: ColumnParams = ColumnParams()):
        self.p = p
        super().__init__(n_var=2, n_obj=2, n_ieq_constr=2, xl=np.array([0.04, 0.04]), xu=np.array([0.5, 0.5]))

    def _evaluate(self, X, out, *args, **kwargs):
        p = self.p
        x1 = X[:, 0]
        x2 = X[:, 1]

        mass = p.rho * p.l * x1 * x2

        term = (p.E * x1 * x2**3) / (4 * p.l**3 * (p.M + (33/140) * p.rho * p.l * x1 * x2))
        freq = np.sqrt(term) / (2 * np.pi)

        # Objectives: minimize mass, maximize frequency -> minimize -frequency
        out["F"] = np.column_stack([mass, -freq])

        # Constraints (<= 0 in pymoo): stress and buckling margins should be <=0 when violated
        stress = (p.M * p.g) / (x1 * x2)
        g1 = stress - p.sigma_max  # <=0 is feasible
        buckling = (np.pi**2 * p.E * x2**2) / (48 * p.l**2)
        g2 = (p.M * p.g) / (x1 * x2) - buckling  # <=0 is feasible
        out["G"] = np.column_stack([g1, g2])


def solve_column_nsga2(
    params: ColumnParams = ColumnParams(),
    pop_size: int = 100,
    n_gen: int = 200,
    seed: int = 1,
) -> Dict[str, Any]:
    """Run NSGA-II and return results."""
    problem = ColumnProblem(params)
    algorithm = NSGA2(pop_size=pop_size)
    res = minimize(problem, algorithm, ("n_gen", n_gen), seed=seed, verbose=False)
    return {"X": res.X, "F": res.F, "raw": res}


def plot_pareto(F: np.ndarray) -> None:
    """Plot Pareto front (mass vs frequency)."""
    mass = F[:, 0]
    freq = -F[:, 1]  # back to +frequency
    plt.figure()
    plt.scatter(mass, freq)
    plt.xlabel("Mass (kg)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Pareto Front: Mass vs Frequency")


def main() -> None:
    out = solve_column_nsga2()
    X = out["X"]
    F = out["F"]

    print(f"Found {F.shape[0]} Pareto solutions.")
    print("\nOptimal designs (x1, x2) and objectives (mass, frequency):")
    # F[:,1] is -frequency in this module, so convert back:
    freq = -F[:, 1]
    table = np.column_stack([X, F[:, 0], freq])
    # nice header
    print("   x1        x2        mass(kg)    freq(Hz)")
    for row in table:
        print(f"{row[0]:.6f}  {row[1]:.6f}  {row[2]:.4f}   {row[3]:.4f}")

    plot_pareto(F)
    plt.show()
    plt.pause(0.001)  # allow GUI to render


if __name__ == "__main__":
    main()
