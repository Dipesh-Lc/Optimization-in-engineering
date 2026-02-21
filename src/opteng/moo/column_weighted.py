from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
from scipy.optimize import minimize, OptimizeResult


@dataclass(frozen=True)
class ColumnParams:
    rho: float = 7850.0
    l: float = 5.0
    E: float = 200e9
    M: float = 1000.0
    g: float = 9.81
    sigma_max: float = 250e6
    bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.04, 0.5), (0.04, 0.5))


def _stress(x, p: ColumnParams) -> float:
    x1, x2 = x
    return (p.M * p.g) / (x1 * x2)


def _buckling_capacity(x, p: ColumnParams) -> float:
    x1, x2 = x
    return (np.pi**2 * p.E * x2**2) / (48 * p.l**2)


def _term(x, p: ColumnParams) -> float:
    """Shared term used in the frequency expression."""
    x1, x2 = x
    return (p.E * x1 * x2**3) / (4 * p.l**3 * (p.M + (33 / 140) * p.rho * p.l * x1 * x2))


def objective_mass(x, p: ColumnParams) -> float:
    x1, x2 = x
    return float(p.rho * p.l * x1 * x2)


def objective_frequency_raw(x, p: ColumnParams) -> float:
    """Matches the original script's 'frequency' inside optimization: sqrt(term) (no / 2π)."""
    return float(np.sqrt(_term(x, p)))


def frequency_hz(x, p: ColumnParams) -> float:
    """Frequency reported in Hz (matches original printing): sqrt(term)/(2π)."""
    return float(np.sqrt(_term(x, p)) / (2 * np.pi))


def constraints(p: ColumnParams):
    # SciPy 'ineq' means fun(x) >= 0 is feasible
    return [
        {"type": "ineq", "fun": lambda x: p.sigma_max - _stress(x, p)},
        {"type": "ineq", "fun": lambda x: _buckling_capacity(x, p) - _stress(x, p)},
    ]


def _safe_x(res: OptimizeResult, fallback: Tuple[float, float]) -> np.ndarray:
    """Return res.x if present; else fallback."""
    if res is not None and getattr(res, "x", None) is not None:
        return np.array(res.x, dtype=float)
    return np.array(fallback, dtype=float)


def solve_weighted(
    alpha: float = 0.01,
    x0: Tuple[float, float] = (0.1, 0.1),
    params: ColumnParams = ColumnParams(),
    method: str = "SLSQP",
) -> Dict[str, Any]:
    """
    Solve:
    1) min mass
    2) max frequency (via min -sqrt(term))
    3) weighted normalized objective:
       (alpha/mass1)*mass - ((1-alpha)/frequency2)*sqrt(term)
    """
    cons = constraints(params)
    x0_arr = np.array(x0, dtype=float)

    # 1) Min mass
    res_mass = minimize(
        lambda x: objective_mass(x, params),
        x0_arr,
        method=method,
        bounds=params.bounds,
        constraints=cons,
    )
    x_mass = _safe_x(res_mass, x0)
    mass1 = objective_mass(x_mass, params)  # normalize

    # 2) Max frequency 
    res_freq = minimize(
        lambda x: -objective_frequency_raw(x, params),
        x0_arr,
        method=method,
        bounds=params.bounds,
        constraints=cons,
    )
    x_freq = _safe_x(res_freq, x0)
    frequency2 = frequency_hz(x_freq, params)  # normalize denominator uses Hz 

    # Guard against divide-by-zero 
    if not np.isfinite(mass1) or mass1 <= 0 or not np.isfinite(frequency2) or frequency2 <= 0:
        return {
            "success": False,
            "raw": {"mass": res_mass, "freq": res_freq},
            "message": "Normalization values invalid (mass1 or frequency2).",
        }

    # 3) Weighted objective 
    def obj(x):
        m = objective_mass(x, params)
        f_raw = objective_frequency_raw(x, params)  # sqrt(term), no /2π
        return (alpha / mass1) * m - ((1 - alpha) / frequency2) * f_raw

    res_weighted = minimize(
        obj,
        x0_arr,
        method=method,
        bounds=params.bounds,
        constraints=cons,
    )
    x_w = _safe_x(res_weighted, x0)

    return {
        "alpha": alpha,
        "x": x_w,
        "mass": objective_mass(x_w, params),
        "frequency": frequency_hz(x_w, params),
        "normalizers": {"mass1": mass1, "frequency2": frequency2},
        "raw": {"mass": res_mass, "freq": res_freq, "weighted": res_weighted},
        "solver_status": {
            "mass_success": bool(getattr(res_mass, "success", False)),
            "mass_message": str(getattr(res_mass, "message", "")),
            "freq_success": bool(getattr(res_freq, "success", False)),
            "freq_message": str(getattr(res_freq, "message", "")),
            "weighted_success": bool(getattr(res_weighted, "success", False)),
            "weighted_message": str(getattr(res_weighted, "message", "")),
        },
    }


def main() -> None:
    out = solve_weighted(alpha=0.01)
    x1, x2 = out["x"]
    print(f"\nNormalized Weighted Solution When alpha = {out['alpha']}")
    print(f"x1 = {x1:.6f} m")
    print(f"x2 = {x2:.6f} m")
    print(f"Mass = {out['mass']:.2f} kg")
    print(f"Frequency = {out['frequency']:.2f} Hz")



if __name__ == "__main__":
    main()