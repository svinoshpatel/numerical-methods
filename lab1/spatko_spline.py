import numpy as np
import matplotlib.pyplot as plt

# Main parameters
a_bound = 3.0
b_bound = 8.0
N = 28  # number of intervals
points_per_interval = 30

# Function to interpolate
def f(x):
    return np.sin(x**2) * np.exp(x/2) + np.log(x + 3)

# Thomas algorithm for tridiagonal systems
def thomas_algorithm(a, b, c, d):
    n = len(d)
    
    # Forward elimination
    for i in range(1, n):
        w = a[i] / b[i-1]
        b[i] = b[i] - w * c[i-1]
        d[i] = d[i] - w * d[i-1]
    
    # Back substitution
    x = [0] * n
    x[n-1] = d[n-1] / b[n-1]
    for i in range(n-2, -1, -1):
        x[i] = (d[i] - c[i] * x[i+1]) / b[i]
    
    return x

# Natural cubic spline solver
def cubic_spline(x, y):
    n = len(x) - 1  # number of intervals
    h = []
    for i in range(n):
        h.append(x[i+1] - x[i])
    
    # Set up system for second derivatives
    # For natural spline: c[0] = 0, c[n] = 0
    # We solve for c[1], c[2], ..., c[n-1]
    if n == 1:
        c = [0, 0]
        residual_inf = 0.0
    else:
        m = n - 1  # number of unknowns
        
        # Create tridiagonal system
        lower = [0] * m  # lower diagonal (a)
        main = [0] * m   # main diagonal (b)
        main = [0] * m   # main diagonal (b)
        upper = [0] * m  # upper diagonal (c)
        rhs = [0] * m    # right hand side (d)
        
        for i in range(m):
            # i corresponds to equation for c[i+1]
            if i > 0:
                lower[i] = h[i]
            main[i] = 2 * (h[i] + h[i+1])
            if i < m - 1:
                upper[i] = h[i+1]
            
            rhs[i] = 3 * ((y[i+2] - y[i+1]) / h[i+1] - (y[i+1] - y[i]) / h[i])
        
        # Solve using Thomas algorithm
        c_inner = thomas_algorithm(lower, main, upper, rhs)
        
        # Add boundary conditions
        c = [0]
        for val in c_inner:
            c.append(val)
        c.append(0)
        
        # Calculate residual for verification
        residual = []
        for i in range(m):
            r = main[i] * c_inner[i]
            if i > 0:
                r += lower[i] * c_inner[i-1]
            if i < m - 1:
                r += upper[i] * c_inner[i+1]
            r -= rhs[i]
            residual.append(abs(r))
        residual_inf = max(residual) if residual else 0.0
    
    # Calculate coefficients for each piece
    a = []
    b_coef = []
    d = []
    
    for i in range(n):
        a.append(y[i])
        b_coef.append((y[i+1] - y[i]) / h[i] - h[i] * (2*c[i] + c[i+1]) / 3)
        d.append((c[i+1] - c[i]) / (3 * h[i]))
    
    return a, b_coef, c, d, h, residual_inf

# Evaluate spline at given points
def evaluate_spline(x_nodes, a, b_coef, c, d, h, x_eval):
    result = []
    
    for x_val in x_eval:
        # Find which interval x_val belongs to
        i = 0
        for j in range(len(x_nodes)-1):
            if x_val >= x_nodes[j] and x_val <= x_nodes[j+1]:
                i = j
                break
        
        # Calculate local parameter t
        t = x_val - x_nodes[i]
        
        # Evaluate cubic polynomial
        value = a[i] + b_coef[i]*t + c[i]*t*t + d[i]*t*t*t
        result.append(value)
    
    return np.array(result)

# Create interpolation nodes
x_nodes = []
for i in range(N + 1):
    x_nodes.append(a_bound + i * (b_bound - a_bound) / N)
x_nodes = np.array(x_nodes)

# Calculate function values at nodes
y_nodes = f(x_nodes)

# Get spline coefficients
a, b_coef, c, d, h, residual_inf = cubic_spline(x_nodes, y_nodes)

# Create dense grid for evaluation
x_dense = []
for i in range(N):
    for j in range(points_per_interval):
        x_val = x_nodes[i] + j * h[i] / points_per_interval
        x_dense.append(x_val)
x_dense.append(x_nodes[-1])  # Add last point

# Evaluate spline and original function
spline_values = evaluate_spline(x_nodes, a, b_coef, c, d, h, x_dense)
function_values = []
for x_val in x_dense:
    function_values.append(f(x_val))

# Calculate error
error = []
for i in range(len(spline_values)):
    error.append(abs(spline_values[i] - function_values[i]))

max_error = max(error)

# Calculate L2 norm approximation using trapezoidal rule
dx = []
for i in range(len(x_dense) -1):
    dx.append(x_dense[i+1] - x_dense[i])

error_squared = []
for i in range(len(error) - 1):
    first_error_squared = error[i] * error[i]
    second_error_squared = error[i+1] * error[i+1]
    error_squared.append(first_error_squared + second_error_squared)

sum_products = 0
for i in range(len(dx)):
    sum_products += dx[i] * error_squared[i]

L2_norm = np.sqrt(0.5 * sum_products)

# Print results
print("Інтервал: [{a_bound}, {b_bound}], кількість інтервалів N: {N}, вузлів: {N+1}")
print("Кількість точок густої сітки: {len(x_dense)}, max|похибка| = {max_error:.6e}, L2 ≈ {l2_norm:.6e}")
print("Нев'язка тридіагональної СЛАР (норма ∞): {residual_inf:.3e}")

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(x_dense, function_values, 'b-', label='f(x)', linewidth=2)
plt.plot(x_dense, spline_values, 'r--', label='Spline S(x)', linewidth=2)
plt.plot(x_nodes, y_nodes, 'ko', markersize=4, label='Nodes')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Function and Cubic Spline Interpolation')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(x_dense, error, 'g-', linewidth=2)
plt.xlabel('x')
plt.ylabel('|S(x) - f(x)|')
plt.title('Absolute Error')
plt.grid(True)
plt.show()

