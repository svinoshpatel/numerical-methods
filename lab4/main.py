import numpy as np
import matplotlib.pyplot as plt

def transcendent_function(x):
    return np.exp(x) * np.sin(x**2) + np.log(x+2) + np.arctan(x)

def tabulate_function(x0, xn, n):
    h = (xn - x0) / n
    x_vals = np.array([x0 + i * h for i in range(n + 1)])
    f_vals = transcendent_function(x_vals)
    return x_vals, f_vals

def form_matrix_B(x_vals, m):
    n = len(x_vals)
    B = np.zeros((n, m + 1))
    for i in range(n):
        for j in range(m + 1):
            B[i, j] = x_vals[i] ** j
    return B

def form_matrix_C(B):
    return B.T @ B

def form_vector_d(B, f_vals):
    return B.T @ f_vals

def solve_least_squares_gauss(C, d):
    n = len(d)
    
    aug = []
    for i in range(n):
        row = []
        # Copy row from C
        for j in range(n):
            row.append(C[i][j])
        row.append(d[i])
        aug.append(row)
    
    # Forward elimination with column pivoting
    for k in range(n):
        max_val = abs(aug[k][k])
        max_idx = k
        for i in range(k + 1, n):
            if abs(aug[i][k]) > max_val:
                max_val = abs(aug[i][k])
                max_idx = i
        
        # Swap rows if needed
        if max_idx != k:
            aug[k], aug[max_idx] = aug[max_idx], aug[k]
        
        for i in range(k + 1, n):
            if aug[k][k] != 0:
                factor = aug[i][k] / aug[k][k]
                for j in range(k, n + 1):
                    aug[i][j] -= factor * aug[k][j]
    
    # Back substitution
    A = [0.0] * n
    for i in range(n - 1, -1, -1):
        sum_val = 0.0
        for j in range(i + 1, n):
            sum_val += aug[i][j] * A[j]
        A[i] = (aug[i][-1] - sum_val) / aug[i][i]
    
    return A

def polynomial_approximation(x, A):
    result = 0
    for j in range(len(A)):
        result += A[j] * (x ** j)
    return result

def compute_approximation_error(x_vals, f_vals, A):
    phi_vals = np.array([polynomial_approximation(x, A) for x in x_vals])
    epsilon = np.abs(f_vals - phi_vals)
    dispersion = np.sum((f_vals - phi_vals) ** 2)
    return epsilon, dispersion, phi_vals

# Task 3: Find optimal polynomial degree
def find_optimal_degree(x_vals, f_vals, m_values):
    results = {}
    
    for m in m_values:
        # Form matrices
        B = form_matrix_B(x_vals, m)
        C = form_matrix_C(B)
        d = form_vector_d(B, f_vals)
        
        # Solve for coefficients
        A = solve_least_squares_gauss(C, d)
        
        # Compute error and dispersion
        epsilon, dispersion, phi_vals = compute_approximation_error(x_vals, f_vals, A)
        
        results[m] = {
            'coefficients': A,
            'epsilon': epsilon,
            'dispersion': dispersion,
            'phi_vals': phi_vals
        }
        
        print(f"m = {m}: Dispersion = {dispersion:.6e}")
    
    return results

def plot_error_functions(x0, xn, n, results, m_values):
    h1 = (xn - x0) / (20 * n)
    x_fine = np.arange(x0, xn + h1/2, h1)
    f_fine = transcendent_function(x_fine)
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Dispersion vs degree
    plt.subplot(2, 2, 1)
    dispersions = [results[m]['dispersion'] for m in m_values]
    plt.plot(m_values, dispersions, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Polynomial degree m')
    plt.ylabel('Dispersion')
    plt.title('Dispersion vs Polynomial Degree')
    plt.grid(True)
    plt.yscale('log')
    
    # Find optimal degree (minimum dispersion)
    optimal_m = m_values[np.argmin(dispersions)]
    plt.axvline(x=optimal_m, color='r', linestyle='--', label=f'Optimal m = {optimal_m}')
    plt.legend()
    
    # Plot 2-4: Error functions for selected degrees
    selected_m = [1, 5, optimal_m] if optimal_m not in [1, 5] else [1, 5, 10]
    
    for idx, m in enumerate(selected_m[:3]):
        plt.subplot(2, 2, idx + 2)
        
        # Compute approximation on fine grid
        A = results[m]['coefficients']
        phi_fine = np.array([polynomial_approximation(x, A) for x in x_fine])
        error_fine = np.abs(f_fine - phi_fine)
        
        plt.plot(x_fine, error_fine, 'r-', linewidth=2)
        plt.xlabel('x')
        plt.ylabel('|f(x) - φ(x)|')
        plt.title(f'Approximation Error for m = {m}')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nOptimal polynomial degree: m = {optimal_m} (minimum dispersion)")
    
    return optimal_m

# Main program
def main():
    # Parameters
    x0 = 0      # Left bound
    xn = 5      # Right bound
    n = 30      # Number of intervals (approximately 30 as per task)
    
    print("=" * 70)
    print("LEAST SQUARES POLYNOMIAL APPROXIMATION OF TRANSCENDENT FUNCTION")
    print("=" * 70)
    
    # Task 1: Tabulate the function
    print(f"\nTask 1: Tabulating f(x) on [{x0}, {xn}] with n = {n} nodes")
    x_vals, f_vals = tabulate_function(x0, xn, n)
    print(f"Created {len(x_vals)} data points with step h = {(xn-x0)/n:.4f}")
    print(f"Sample points (first 5):")
    for i in range(min(5, len(x_vals))):
        print(f"  x[{i}] = {x_vals[i]:.4f}, f(x[{i}]) = {f_vals[i]:.6f}")
    
    # Task 2 & 3: Find approximations for m = 1,...,10
    print(f"\nTask 2-3: Computing polynomial approximations for degrees m = 1,...,10")
    print("-" * 70)
    m_values = list(range(1, 11))
    results = find_optimal_degree(x_vals, f_vals, m_values)
    
    # Task 4: Plot error functions
    print(f"\nTask 4: Plotting error functions with finer step h1 = (xn-x0)/(20*n)")
    optimal_m = plot_error_functions(x0, xn, n, results, m_values)
    
    # Display optimal polynomial coefficients
    print(f"\nOptimal polynomial (m = {optimal_m}) coefficients:")
    A_opt = results[optimal_m]['coefficients']
    for j, coef in enumerate(A_opt):
        print(f"  A[{j}] = {coef:.10f}")
    
    print("\n" + "=" * 70)
    print("PROGRAM COMPLETED SUCCESSFULLY")
    print("=" * 70)

# Run the program
if __name__ == "__main__":
    main()

