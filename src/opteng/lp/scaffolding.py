"""Scaffolding support system (Linear Programming).

Ref: Thesis 'Optimization in Engineering Problems' (Dipesh Lamichhane, 2025).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
from scipy.optimize import linprog


@dataclass(frozen=True)
class ScaffoldingParams:
    """Problem constants (maxmum tension that the ropes can withstand)."""
    W1: float = 150
    W2: float = 150
    W3: float = 100
    W4: float = 100


def solve_scaffolding(params: ScaffoldingParams = ScaffoldingParams()) -> Dict[str, Any]:
    """Solve the LP and return a simple result dict.

    Returns keys: success, objective_value, x (decision variables), raw (SciPy result).
    """
    # Objective: maximize x1+x2+x3+x4 -> minimize negative
    c = [-1, -1, -1, -1]

    # Inequality constraints matrix (A_ub * x <= b_ub)
    A_ub = [
        [1/2, 0,   3/8,  3/20],    # TA <= W1
        [1/2, 0,   1/8,  1/20],    # TB <= W1
        [0,   1/2, 2/5,  17/50],   # TC <= W2
        [0,   1/2, 1/10, 23/50],   # TD <= W2
        [0,   0,   1/2,  1/5],     # TE <= W3
        [0,   0,   1/2,  3/10],    # TF <= W3
        [0,   0,   0,    1/2],     # TG & TH <= W4
    ]

    #right-hand side values for inequality constraints
    b_ub = [params.W1, params.W1, params.W2, params.W2, params.W3, params.W3, params.W4]

    # Define variable bounds (x1, x2, x3, x4 must be at least 50)
    bounds = [(50, None), (50, None), (50, None), (50, None)]

    # Solve the linear programming problem
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    obj = None
    x = None
    if res.success:
        x = res.x
        obj = -res.fun  # convert back to maximization

    return {"success": bool(res.success), "objective_value": obj, "x": x, "raw": res}


def main() -> None:
    out = solve_scaffolding()
    if not out["success"]:
        raise SystemExit("Optimization failed.")
    x1, x2, x3, x4 = out["x"]
    print("Optimal solution:")
    print(f"x1={x1:.2f}, x2={x2:.2f}, x3={x3:.2f}, x4={x4:.2f}")
    print(f"Max objective (x1+x2+x3+x4) = {out['objective_value']:.6f}")
    
    # Calculate actual constraint values using the optimal solution
    TA = (1/2)*x1 + (3/8)*x3 + (3/20)*x4
    TB = (1/2)*x1 + (1/8)*x3 + (1/20)*x4
    TC = (1/2)*x2 + (2/5)*x3 + (17/50)*x4
    TD = (1/2)*x2 + (1/10)*x3 + (23/50)*x4
    TE = (1/2)*x3 + (1/5)*x4
    TF = (1/2)*x3 + (3/10)*x4
    TG = (1/2)*x4
    TH = (1/2)*x4    

    # Display constraint values at the optimal solution
    print("\nConstraint Values at Optimal Solution:")
    print(f"TA = {TA:.2f} (<= W1 = {ScaffoldingParams.W1})")
    print(f"TB = {TB:.2f} (<= W1 = {ScaffoldingParams.W1})")
    print(f"TC = {TC:.2f} (<= W2 = {ScaffoldingParams.W2})")
    print(f"TD = {TD:.2f} (<= W2 = {ScaffoldingParams.W2})")
    print(f"TE = {TE:.2f} (<= W3 = {ScaffoldingParams.W3})")
    print(f"TF = {TF:.2f} (<= W3 = {ScaffoldingParams.W3})")
    print(f"TG = {TG:.2f} (<= W4 = {ScaffoldingParams.W4})")
    print(f"TH = {TH:.2f} (<= W4 = {ScaffoldingParams.W4})")       

if __name__ == "__main__":
    main()
