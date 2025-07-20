from scipy.optimize import linprog

# Define the maxmum tension that the ropes can withstand
W1 = 150  
W2 = 150  
W3 = 100
W4 = 100

# Objective function coefficients (minimizing -x1 - x2 - x3 - x4)
c = [-1, -1, -1, -1]

# Inequality constraints matrix (A_ub * x <= b_ub)
A_ub = [
    [1/2,0,3/8,3/20],    # TA <= W1
    [1/2,0,1/8,1/20],    # TB <= W1
    [0, 1/2, 2/5, 17/50], # TC <= W2 
    [0, 1/2, 1/10, 23/50],# TD <= W2 
    [0, 0, 1/2, 1/5],     # TE <= W3
    [0, 0, 1/2, 3/10],    # TF <= W3
    [0, 0, 0, 1/2]        # TG & TH<= W4 (same expression)
]
# Compute right-hand side values for inequality constraints
b_ub = [
    W1,  # TA
    W1,  # TB
    W2,  # TC 
    W2,  # TD
    W3,  # TE
    W3,  # TF
    W4   # TG & TH (same expression)
]

# Define variable bounds (x1, x2, x3, x4 must be at least 50)
x_bounds = [(50, None), (50, None), (50, None), (50, None)]

# Solve the linear programming problem
result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=x_bounds, method='highs')

# Check if an optimal solution was found
if result.success:
    x1, x2, x3, x4 = result.x
    total = x1 + x2 + x3 + x4

    # Display optimal solution
    print("Optimal solution found:")
    print(f"x1 = {x1:.2f}")
    print(f"x2 = {x2:.2f}")
    print(f"x3 = {x3:.2f}")
    print(f"x4 = {x4:.2f}")
    print(f"Total (x1 + x2 + x3 +x4) = {total:.2f}")

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
    print(f"TA = {TA:.2f} (<= W1 = {W1})")
    print(f"TB = {TB:.2f} (<= W1 = {W1})")
    print(f"TC = {TC:.2f} (<= W2 = {W2})")
    print(f"TD = {TD:.2f} (<= W2 = {W2})")
    print(f"TE = {TE:.2f} (<= W3 = {W3})")
    print(f"TF = {TF:.2f} (<= W3 = {W3})")
    print(f"TG = {TG:.2f} (<= W4 = {W4})")
    print(f"TH = {TH:.2f} (<= W4 = {W4})")
else:
    # Display error message if no solution was found
    print("No optimal solution found. Message:", result.message)
