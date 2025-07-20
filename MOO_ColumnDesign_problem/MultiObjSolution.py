#Import Libraries 
import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import matplotlib.pyplot as plt

# Define the optimization problem by defining a custom class inheriting from Problem
class ColumnProblem(Problem):
# Initialize constants related to material and physical properties    
    def __init__(self):
        self.rho = 7850       # Density (kg/m³)
        self.l = 5            # Column length (m)
        self.E = 200e9        # Young's modulus (Pa)
        self.M = 1000         # Mass of tank (kg)
        self.g = 9.81         # Gravity (m/s²)
        self.sigma_max = 250e6  # Permissible stress (Pa)
        
# Call the parent constructor to set up the problem dimensions and bounds
        super().__init__(
            n_var=2,          # Variables: x1, x2
            n_obj=2,          # Objectives: min mass, max freq
            n_constr=2,       # Constraints: g1, g2
            xl=[0.04, 0.04],  # Lower bounds 
            xu=[0.5, 0.5]     # Upper bounds 
        )

    def _evaluate(self, X, out, *args, **kwargs):
        x1, x2 = X[:, 0], X[:, 1]
        
        # Objectives
        f1 = self.rho * self.l * x1 * x2  # Minimize mass
        term = (self.E * x1 * x2**3) / (4 * self.l**3 * (self.M + (33/140) * self.rho * self.l * x1 * x2))
        f2 = -np.sqrt(term)/(2*np.pi)  # Maximize frequency 
        
        # Constraints (g <= 0)
        g1 = (self.M * self.g) / (x1 * x2) - self.sigma_max  # Stress constraint
        g2 = (self.M * self.g) / (x1 * x2) - (np.pi**2 * self.E * x2**2) / (48 * self.l**2)  # Buckling
        
        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2])

# Solve the problem
problem = ColumnProblem()
algorithm = NSGA2(pop_size=100)
res = minimize(
    problem,
    algorithm,
    ('n_gen', 100),  # Terminate after 100 generations
    seed=1,
    verbose=False
)

# Plot Pareto front (mass vs. natural frequency)
plt.scatter(res.F[:, 0], -res.F[:, 1], c="blue")
plt.xlabel("Mass (kg)")
plt.ylabel("Natural Frequency (Hz)")
plt.title("Pareto Front")
plt.show()

# Extract solutions (x1, x2) from `res.X`
print("Optimal designs (x1, x2):")
print(res.X, res.F)