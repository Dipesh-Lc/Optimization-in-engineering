import numpy as np
from scipy.optimize import minimize

# Define the objective function (to be minimized)
def objective(vars):
    x1, x2 = vars
    return x1 * x2  

# Define the inequality constraints (must be >= 0)
def constraint1(vars):
    x1, x2 = vars
    return x1 * x2**2 - (9 / 2200)  # Constraint 1

def constraint2(vars):
    x1, x2 = vars
    return x1 * x2**3 - (725 / (1656 * 10**4))  # Constraint 2

def constraint3(vars):
    x1, x2 = vars
    return x2 - x1  # Ensures x2 >= x1

# Define variable bounds for x1 and x2
bounds = [(0.04, 0.12), (0.06, 0.20)]  # (min, max) values

# Initial guess for the optimization
initial_guess = [0.04, 0.06]

# Define the constraints in dictionary format
constraints = [
    {'type': 'ineq', 'fun': constraint1},  # constraint1 >= 0
    {'type': 'ineq', 'fun': constraint2},  # constraint2 >= 0
    {'type': 'ineq', 'fun': constraint3}   # constraint3 >= 0
]

# Solve the optimization problem using SLSQP
result = minimize(fun=objective, x0=initial_guess, method='SLSQP', bounds=bounds,
    constraints=constraints)

# Display the results
if result.success:
    print("Optimization was successful!")
    print(f"Optimal values: x1 = {1000*result.x[0]:.4f} mm, x2 = {1000*result.x[1]:.4f} mm")
    print(f"Minimum W = {(76.5*10**3)*result.fun:.4f} N")  # Compute the minimum W value
else:
    print("Optimization failed:", result.message)
