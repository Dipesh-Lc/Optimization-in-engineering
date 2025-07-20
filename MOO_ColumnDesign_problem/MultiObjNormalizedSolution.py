import numpy as np
from scipy.optimize import minimize

# Constants
rho = 7850       # Density of the material (kg/m^3)
l = 5            # Column length (m)
E = 200e9        # Young's modulus (Pa)
M = 1000        # Mass of the tank (kg)
g = 9.81         # Acceleration due to gravity (m/s^2)
sigma_max = 250e6  # Maximum permissible stress (Pa)
alpha = 0.01     # Weighting factor for mass and frequency optimization

# Objective function to minimize mass
def objective1(x):
    x1, x2 = x
    mass = rho * l * x1 * x2  # Compute mass of the column
    return mass

# Objective function to maximize frequency (negative for minimization approach)
def objective2(x):
    x1, x2 = x
    term = (E * x1 * x2**3) / (4 * l**3 * (M + (33/140) * rho * l * x1 * x2))
    frequency = np.sqrt(term) 
    return -frequency  # Negative since we minimize by default

# Define the inequality constraints (must be >= 0)
def constraint1(x):
    x1, x2 = x
    return sigma_max - (M * g) / (x1 * x2)

def constraint2(x):
    x1, x2 = x
    return (np.pi**2 * E * x2**2) / (48 * l**2) - (M * g) / (x1 * x2)

# Bounds for design variables (x1 and x2)
bounds = [(0.04, 0.5), (0.04, 0.5)]

# Constraints in dictionary format
constraints = [
    {'type': 'ineq', 'fun': constraint1},
    {'type': 'ineq', 'fun': constraint2}
]

# Initial guess for optimization
x0 = [0.1, 0.1]

# Solve for minimum mass
result1 = minimize(objective1, x0, method='SLSQP', bounds=bounds, constraints=constraints)
print("\nMin Mass Solution:")
print(f"x1 = {result1.x[0]:.6f} m")
print(f"x2 = {result1.x[1]:.6f} m")
mass1 = rho * l * result1.x[0] * result1.x[1]
print(f"Mass = {mass1:.2f} kg")
term1 = (E * result1.x[0] * result1.x[1]**3) / (4 * l**3 * (M + (33/140) * rho * l * result1.x[0] * result1.x[1]))
frequency1 = np.sqrt(term1) / (2 * np.pi)
print(f"Frequency = {frequency1:.2f} Hz")

# Solve for maximum frequency
result2 = minimize(objective2, x0, method='SLSQP', bounds=bounds, constraints=constraints)
print("\nMax Frequency Solution:")
print(f"x1 = {result2.x[0]:.6f} m")
print(f"x2 = {result2.x[1]:.6f} m")
mass2 = rho * l * result2.x[0] * result2.x[1]
term2 = (E * result2.x[0] * result2.x[1]**3) / (4 * l**3 * (M + (33/140) * rho * l * result2.x[0] * result2.x[1]))
frequency2 = np.sqrt(term2) / (2 * np.pi)
print(f"Mass = {mass2:.2f} kg")
print(f"Frequency = {frequency2:.2f} Hz")

# Multi-objective optimization balancing mass and frequency
def objective(x):
    x1, x2 = x
    mass = rho * l * x1 * x2
    term = (E * x1 * x2**3) / (4 * l**3 * (M + (33/140) * rho * l * x1 * x2))
    frequency = np.sqrt(term)
    return (alpha / mass1) * mass - ((1 - alpha) / frequency2) * frequency

# Solve for weighted multi-objective optimization
result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

# Display final results
print(f"\nNormalized Weighted Solution When alpha = {alpha}")
print(f"x1 = {result.x[0]:.6f} m")
print(f"x2 = {result.x[1]:.6f} m")
mass = rho * l * result.x[0] * result.x[1]
term = (E * result.x[0] * result.x[1]**3) / (4 * l**3 * (M + (33/140) * rho * l * result.x[0] * result.x[1]))
frequency = np.sqrt(term) / (2 * np.pi)
print(f"Mass = {mass:.2f} kg")
print(f"Frequency = {frequency:.2f} Hz")