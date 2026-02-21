# Optimization in Engineering Problems
This project develops a modular Python framework for constrained structural optimization.  
It implements linear, nonlinear, and multi-objective optimization models derived from mechanics-based formulations and analyzes solver behavior in boundary-dominated systems.

---

## 🔍 Overview

Four engineering optimization case studies are included:

| Type | Problem | Method |
|------|---------|--------|
| LP | Scaffolding support system | SciPy HiGHS |
| NLP | Beam design optimization | SLSQP |
| NLP | Two-bar truss optimization | SLSQP |
| MOO | Water-tank column design | Weighted Sum & NSGA-II |

The focus is on:

- Constraint-based structural modeling  
- Solver implementation using SciPy  
- Multi-objective trade-off analysis  
- Reproducible computational workflow  

---

## 🛠 Technical Stack

- Python 3.12  
- NumPy  
- SciPy (HiGHS, SLSQP)  
- pymoo (NSGA-II)  
- Matplotlib  
- MATLAB
---

## 📐 Problems Implemented

### Linear Programming – Scaffolding
Maximization of load capacity subject to tension constraints.

### Nonlinear Programming – Beam & Truss
Weight minimization under stress, buckling, and geometric constraints.

### Multi-Objective Optimization – Column Design
Simultaneous:
- Minimization of mass  
- Maximization of natural frequency  

Subject to stress and buckling limits.

Two approaches are implemented:
- Normalized weighted-sum scalarization (SLSQP)
- NSGA-II (pymoo)

![Pareto Front](docs/figures/moo_column/pareto.png)

---

## 📊 Key Observations

- Optimal solutions lie on active constraint boundaries.
- Scaling influences convergence behavior in SLSQP.
- Weighted-sum scalarization captures convex Pareto regions efficiently.
- NSGA-II approximates the full Pareto front more robustly.

Details about problems, formulations and figures are available in:
- `docs/problems/`
- `docs/figures/`

## Repo layout
- `src/opteng` # Optimization modules (LP, NLP, MOO)
- `examples`   # Runnable scripts
- `docs`       # Problems, Formulations and figures
- `validation` # Matlab solution for validation
- `pyproject.toml`
- `README.md`

MATLAB implementations are included in `validation/` for independent cross-platform verification.

## Installation

Clone and install in editable mode:

```bash
git clone https://github.com/Dipesh-Lc/Optimization-in-engineering.git
cd Optimization-in-engineering
pip install -e .
```

## Quick start

```bash
python examples/run_beam.py
python examples/run_truss.py
python examples/run_scaffolding.py
python examples/run_column_weighted.py
python examples/run_column_nsga2.py
```
