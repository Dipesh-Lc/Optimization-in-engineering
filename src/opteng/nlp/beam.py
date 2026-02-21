from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class BeamBounds:
    x1: Tuple[float, float] = (0.04, 0.12)
    x2: Tuple[float, float] = (0.06, 0.20)


def objective(x: np.ndarray) -> float:
    x1, x2 = x
    return float(x1 * x2)


def constraint1(x: np.ndarray) -> float:
    x1, x2 = x
    return float(x1 * x2**2 - (9 / 2200))


def constraint2(x: np.ndarray) -> float:
    x1, x2 = x
    return float(x1 * x2**3 - (725 / (1656 * 10**4)))


def constraint3(x: np.ndarray) -> float:
    x1, x2 = x
    return float(x2 - x1)  # x2 >= x1


def solve_beam(
    x0: Tuple[float, float] = (0.04, 0.06),
    bounds: BeamBounds = BeamBounds(),
    method: str = "SLSQP",
) -> Dict[str, Any]:
    """Solve the beam design NLP.

    Returns keys: success, x, objective_value, raw.
    """
    bnds = [bounds.x1, bounds.x2]
    cons = [
        {"type": "ineq", "fun": constraint1},
        {"type": "ineq", "fun": constraint2},
        {"type": "ineq", "fun": constraint3},
    ]
    res = minimize(objective, np.array(x0, dtype=float), method=method, bounds=bnds, constraints=cons)
    obj = float(res.fun) if res.success else None
    return {"success": bool(res.success), "x": res.x if res.success else None, "objective_value": obj, "raw": res}


def main() -> None:
    out = solve_beam()
    if not out["success"]:
        raise SystemExit("Optimization failed.")
    x1, x2 = out["x"]
    print("Optimal solution:")
    print(f"x1={1000*x1:.4f} mm, x2={1000*x2:.4f} mm")
    print(f"Minimum W = {(76.5*10**3)*out['objective_value']:.4f} N")


if __name__ == "__main__":
    main()
