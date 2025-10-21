import numpy as np
import matplotlib.pyplot as plt

A_BOUND = 0.0
B_BOUND = np.pi
N = 5  # number of intervals (nodes - 1)
POINTS_PER_INTERVAL = 20

def f(x):
    return np.sin(x) + 0.5 * np.cos(2 * x)

def compute_divided_differences(x_nodes, f_nodes):
    n = len(x_nodes)
    table = []
    
    # Initialize first column with function values
    for i in range(n):
        table.append([f_nodes[i]])
    
    # Fill remaining columns
    for j in range(1, n):
        for i in range(n - j):
            diff = (table[i + 1][j - 1] - table[i][j - 1]) / (x_nodes[i + j] - x_nodes[i])
            table[i].append(diff)
    
    return table

def w_k(x, x_nodes, k):
    if k < 0:
        return 1.0
    
    product = 1.0
    for i in range(k + 1):
        product *= (x - x_nodes[i])
    
    return product

def newton_polynomial(x, x_nodes, divided_diff_table):
    n = len(x_nodes) - 1
    result = divided_diff_table[0][0]
    
    for k in range(1, n + 1):
        result += w_k(x, x_nodes, k - 1) * divided_diff_table[0][k]
    
    return result

def interpolation_error(x, x_nodes, divided_diff_table, f_func):
    return abs(f_func(x) - newton_polynomial(x, x_nodes, divided_diff_table))

# Main interpolation solver
def newton_interpolation(x0, xn, n, f_func):
    h = (xn - x0) / n
    
    # Generate interpolation nodes
    x_nodes = []
    for i in range(n + 1):
        x_nodes.append(x0 + i * h)
    x_nodes = np.array(x_nodes)
    
    # Calculate function values at nodes
    f_nodes = []
    for x_val in x_nodes:
        f_nodes.append(f_func(x_val))
    f_nodes = np.array(f_nodes)
    
    # Compute divided differences
    divided_diff_table = compute_divided_differences(x_nodes, f_nodes)
    
    return x_nodes, f_nodes, divided_diff_table

def evaluate_on_dense_grid(x_nodes, divided_diff_table, f_func, points_per_interval):
    n = len(x_nodes) - 1
    x_dense = []
    
    # Generate dense grid
    for i in range(n):
        h_i = x_nodes[i + 1] - x_nodes[i]
        for j in range(points_per_interval):
            x_val = x_nodes[i] + j * h_i / points_per_interval
            x_dense.append(x_val)
    x_dense.append(x_nodes[-1])
    
    # Evaluate functions
    f_values = []
    n_values = []
    error_values = []
    w_n_values = []
    
    for x_val in x_dense:
        f_val = f_func(x_val)
        n_val = newton_polynomial(x_val, x_nodes, divided_diff_table)
        
        f_values.append(f_val)
        n_values.append(n_val)
        error_values.append(abs(f_val - n_val))
        w_n_values.append(w_k(x_val, x_nodes, n))
    
    return x_dense, f_values, n_values, error_values, w_n_values

def calculate_error_norms(error_values, x_dense):
    max_error = max(error_values)
    
    # L2 norm using trapezoidal rule
    dx = []
    for i in range(len(x_dense) - 1):
        dx.append(x_dense[i + 1] - x_dense[i])
    
    error_squared = []
    for i in range(len(error_values) - 1):
        sum_squared = error_values[i]**2 + error_values[i + 1]**2
        error_squared.append(sum_squared)
    
    sum_products = 0
    for i in range(len(dx)):
        sum_products += dx[i] * error_squared[i]
    
    L2_norm = np.sqrt(0.5 * sum_products)
    
    return max_error, L2_norm

# Create interpolation nodes
x_nodes, f_nodes, divided_diff_table = newton_interpolation(A_BOUND, B_BOUND, N, f)

# Evaluate on dense grid
x_dense, f_values, n_values, error_values, w_n_values = evaluate_on_dense_grid(
    x_nodes, divided_diff_table, f, POINTS_PER_INTERVAL
)

# Calculate errors
max_error, L2_norm = calculate_error_norms(error_values, x_dense)

# Print results
print(f"Інтервал: [{A_BOUND}, {B_BOUND}], кількість інтервалів N: {N}, вузлів: {N+1}")
print(f"Кількість точок густої сітки: {len(x_dense)}, max|похибка| = {max_error:.6e}, L2 ≈ {L2_norm:.6e}")

# Display divided differences
print("\nРозділені різниці (перші 6 порядків):")
for k in range(min(6, len(x_nodes))):
    print(f"f[x_0,...,x_{k}] = {divided_diff_table[0][k]:.6f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(x_dense, f_values, 'b-', label='f(x)', linewidth=2)
plt.plot(x_dense, n_values, 'r--', label='N_n(x)', linewidth=2)
plt.plot(x_nodes, f_nodes, 'ko', markersize=6, label='Вузли')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Функція та інтерполяція Ньютона')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(x_dense, error_values, 'g-', linewidth=2)
plt.xlabel('x')
plt.ylabel('|N_n(x) - f(x)|')
plt.title('Абсолютна похибка')
plt.grid(True)
plt.yscale('log')
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(x_dense, w_n_values, 'orange', linewidth=2)
plt.plot(x_nodes, np.zeros_like(x_nodes), 'ro', markersize=6, label='Вузли (корені)')
plt.xlabel('x')
plt.ylabel('w_n(x)')
plt.title('Функція w_n(x)')
plt.legend()
plt.grid(True)
plt.show()

