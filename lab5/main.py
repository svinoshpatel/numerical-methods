import numpy as np
import matplotlib.pyplot as plt

# Define the analytical function
def f(x):
    return np.sin(x) + x**2

def f_derivative_analytical(x):
    return np.cos(x) + 2*x

# Task 1: Calculate explicit analytical derivative at x0
def task1(x0):
    return f_derivative_analytical(x0)

# Task 2: First-order derivative approximation
def y0_prime_h(x0, h):
    return (f(x0 + h) - f(x0 - h)) / (2 * h)

# Task 2-3: Find optimal step size
def find_optimal_h(x0, h_min=1e-20, h_max=1e-5):
    h_values = np.logspace(np.log10(h_min), np.log10(h_max), 1000)
    y_exact = f_derivative_analytical(x0)
    
    errors = []
    for h in h_values:
        y_approx = y0_prime_h(x0, h)
        error = abs(y_approx - y_exact)
        errors.append(error)
    
    optimal_idx = np.argmin(errors)
    optimal_h = h_values[optimal_idx]
    min_error = errors[optimal_idx]
    
    return optimal_h, min_error, h_values, errors

# Task 4: Calculate derivative using two step sizes
def task4(x0, h):
    y_h = (f(x0 + h) - f(x0 - h)) / (2 * h)
    y_2h = (f(x0 + 2*h) - f(x0 - 2*h)) / (4 * h)
    return y_h, y_2h

# Task 5: Calculate error at step h
def task5(x0, h):
    y_approx = y0_prime_h(x0, h)
    y_exact = f_derivative_analytical(x0)
    R1 = abs(y_approx - y_exact)
    return R1

# Task 6: Richardson extrapolation (Runge-Romberg method)
def task6(x0, h):
    y_h = y0_prime_h(x0, h)
    y_2h = y0_prime_h(x0, 2*h)
    
    # Richardson extrapolation formula
    y_R = y_h + (y_h - y_2h) / 3
    
    # Error estimation
    y_exact = f_derivative_analytical(x0)
    R2 = abs(y_R - y_exact)
    
    return y_R, R2

# Task 7: Aitken's method with three step sizes
def task7(x0, h):
    # Calculate derivatives at h, 2h, 4h
    y_h = (f(x0 + h) - f(x0 - h)) / (2 * h)
    y_2h = (f(x0 + 2*h) - f(x0 - 2*h)) / (4 * h)
    y_4h = (f(x0 + 4*h) - f(x0 - 4*h)) / (8 * h)
    
    # Aitken's formula
    numerator = (y_2h)**2 - y_4h * y_h
    denominator = 2 * y_2h - (y_4h + y_h)
    
    if abs(denominator) > 1e-15:
        y_E = numerator / denominator
    else:
        y_E = y_h
    
    # Order of accuracy
    if abs(y_2h - y_h) > 1e-15:
        p = (1 / np.log(2)) * np.log(abs((y_4h - y_2h) / (y_2h - y_h)))
    else:
        p = float('nan')
    
    # Error estimation
    y_exact = f_derivative_analytical(x0)
    R3 = abs(y_E - y_exact)
    
    return y_E, p, R3

print("=" * 70)
print("NUMERICAL DERIVATIVE APPROXIMATION")
print("=" * 70)

# Define point x0
x0 = 1.0

# Task 1
print(f"\nTask 1: Analytical derivative at x₀ = {x0}")
y_exact = task1(x0)
print(f"y'(x₀) = {y_exact:.10f}")

# Task 2-3: Find optimal h
print(f"\nTask 2-3: Finding optimal step size h")
optimal_h, min_error, h_values, errors = find_optimal_h(x0)
print(f"Optimal h = {optimal_h:.2e}")
print(f"Minimum error R₀ = {min_error:.2e}")

# Use h = 1e-5 as specified
h = 1e-5
print(f"\nUsing h = {h:.2e} for remaining calculations")

# Task 4
print(f"\nTask 4: Derivatives at step h and 2h")
y_h, y_2h = task4(x0, h)
print(f"y'₀(h) = {y_h:.10f}")
print(f"y'₀(2h) = {y_2h:.10f}")

# Task 5
print(f"\nTask 5: Error at step h")
R1 = task5(x0, h)
print(f"R₁ = |y'₀(h) - y'(x₀)| = {R1:.2e}")

# Task 6: Richardson extrapolation
print(f"\nTask 6: Richardson extrapolation (Runge-Romberg)")
y_R, R2 = task6(x0, h)
print(f"y'_R = {y_R:.10f}")
print(f"R₂ = |y'_R - y'(x₀)| = {R2:.2e}")
print(f"Error improvement: {R1/R2:.2f}x better")

# Task 7: Aitken's method
print(f"\nTask 7: Aitken's method")
y_E, p, R3 = task7(x0, h)
print(f"y'_E = {y_E:.10f}")
print(f"Order of accuracy p = {p:.4f}")
print(f"R₃ = |y'_E - y'(x₀)| = {R3:.2e}")

# Summary comparison
print("\n" + "=" * 70)
print("SUMMARY - ERROR COMPARISON")
print("=" * 70)
print(f"Exact derivative:              y'(x₀) = {y_exact:.10f}")
print(f"Finite difference (h):         y'₀(h) = {y_h:.10f}, Error = {R1:.2e}")
print(f"Richardson extrapolation:      y'_R   = {y_R:.10f}, Error = {R2:.2e}")
print(f"Aitken's method:               y'_E   = {y_E:.10f}, Error = {R3:.2e}")

# Plot error vs step size
plt.figure(figsize=(10, 6))
plt.loglog(h_values, errors, 'b-', linewidth=2)
plt.axvline(optimal_h, color='r', linestyle='--', label=f'Optimal h = {optimal_h:.2e}')
plt.xlabel('Step size h', fontsize=12)
plt.ylabel('Absolute error |y\'₀(h) - y\'(x₀)|', fontsize=12)
plt.title(f'Numerical Derivative Error vs Step Size at x₀ = {x0}', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('derivative_error_analysis.png', dpi=300)
print("\nPlot saved as 'derivative_error_analysis.png'")
plt.show()

