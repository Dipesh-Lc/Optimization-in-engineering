from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class TrussParams:
    b: float = 0.5 # Half base in meters
    h: float = 1.0 # Height in meters
    W: float = 25000.0 # Load in Newtons
    E: float = 200e9 # Young's modulus (Pa)
    sigma_y: float = 310e6 # Yield stress (Pa)
    rho: float = 77000.0  # weight density (N/m^3)


@dataclass(frozen=True)
class TrussBounds:
    d: Tuple[float, float] = (0.02, 0.05)
    t: Tuple[float, float] = (0.005, 0.05)


def objective(x: np.ndarray, p: TrussParams = TrussParams()) -> float:
    d, t = x
    return float(np.sqrt(p.b**2 + p.h**2) * np.pi * 2 * p.rho * d * t)


def g1(x: np.ndarray, p: TrussParams = TrussParams()) -> float:
    d, t = x
    return float(d * t - (p.W * np.sqrt(p.b**2 + p.h**2)) / (np.pi * p.h * p.sigma_y))


def g2(x: np.ndarray, p: TrussParams = TrussParams()) -> float:
    d, t = x
    return float((d**3 * t + t**3 * d) - (8 * p.W * (p.b**2 + p.h**2)**1.5) / (np.pi**3 * p.E * p.h))


def g3(x: np.ndarray) -> float:
    d, t = x
    return float(d - t)  # t <= d


def solve_truss(
    x0: Tuple[float, float] = (0.02, 0.02),
    params: TrussParams = TrussParams(),
    bounds: TrussBounds = TrussBounds(),
    method: str = "SLSQP",
) -> Dict[str, Any]:
    bnds = [bounds.d, bounds.t]
    cons = [
        {"type": "ineq", "fun": lambda xx: g1(xx, params)},
        {"type": "ineq", "fun": lambda xx: g2(xx, params)},
        {"type": "ineq", "fun": g3},
    ]
    res = minimize(lambda xx: objective(xx, params), np.array(x0, dtype=float), method=method, bounds=bnds, constraints=cons)
    obj = float(res.fun) if res.success else None
    return {"success": bool(res.success), "x": res.x if res.success else None, "objective_value": obj, "raw": res}


def main() -> None:
    out = solve_truss()
    if not out["success"]:
        raise SystemExit("Optimization failed.")
    d, t = out["x"]
    print("Optimal solution:")
    print(f"d={1000 *d:.4f} mm, t={1000 *t:.4f} mm")
    print(f"Objective (Minimum Weight) = {out['objective_value']:.4f} N")


if __name__ == "__main__":
    main()
