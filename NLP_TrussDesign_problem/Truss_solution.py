import numpy as np
from scipy.optimize import minimize

# Constants (Known Data)
b = 0.5  # Half base in meters
h = 1  # Height in meters
W = 25000  # Load in Newtons
E = 200e9  # Young's modulus (Pa)
sigma_y = 310e6  # Yield stress (Pa)
rho = 77000  # Weight density (N/m³)

# Define the objective function (to be minimized)
def objective(x):
    d, t = x
    return np.sqrt(b**2 + h**2) * np.pi * 2 * rho * d * t

# Define the inequality constraints (must be >= 0)
def constraint_g1(x):
    d, t = x
    return d * t - (W * np.sqrt(b**2 + h**2)) / (np.pi * h * sigma_y)

def constraint_g2(x):
    d, t = x
    return (d**3 * t + t**3 * d) - (8 * W * (b**2 + h**2)**1.5) / (np.pi**3 * E * h)

def constraint_g3(x):
    d, t = x
    return d - t  # Ensuring t <= d (d - t >= 0)

# Initial guess
x0 = [0.02, 0.02]

# Bounds for d and t
bounds = [(0.02, 0.05), (0.005, 0.05)]  # (min, max) values

# Define constraints in dictionary format
constraints = [
    {'type': 'ineq', 'fun': constraint_g1},
    {'type': 'ineq', 'fun': constraint_g2},
    {'type': 'ineq', 'fun': constraint_g3}
]

# Solve the optimization problem using SLSQP
result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

# Display the results
if result.success:
    print("Optimal Solution:")
    print(f"d = {1000 * result.x[0]:.4f} mm")
    print(f"t = {1000 * result.x[1]:.4f} mm")
    print(f"Minimum Weight = {result.fun:.4f} N")
else:
    print("Optimization failed:", result.message)
