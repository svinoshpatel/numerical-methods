import numpy as np
import matplotlib.pyplot as plt

# Main parameters
X0 = 0
XN = 5
STEP = 0.01
N_VALUES = [5, 10, 20]

def factorial(k):
    if k == 0 or k == 1:
        return 1
    result = 1
    for i in range(2, k + 1):
        result *= i
    return result

def binomial(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def finite_difference(f_vals, k):
    result = 0
    for j in range(k + 1):
        result += ((-1)**(k - j)) * binomial(k, j) * f_vals[j]
    return result

def factorial_poly(t, k):
    if k == 0:
        return 1
    result = 1
    for i in range(k):
        result *= (t - i)
    return result

def f_approx(t, delta_f_vals):
    result = 0
    for k in range(len(delta_f_vals)):
        result += (delta_f_vals[k] / factorial(k)) * factorial_poly(t, k)
    return result

def transcendent_function(x):
    return np.exp(-x/4) * np.sinh(x/2) + np.cos(x**2)

def compute_tabulation(x0, xn, n, step):
    h = (xn - x0) / n
    xs = np.linspace(x0, xn, n + 1)
    f_vals = transcendent_function(xs)
    
    delta_f_vals = []
    for k in range(n + 1):
        delta_f_vals.append(finite_difference(f_vals, k))
    
    ts = np.arange(x0, xn + step, step)
    f_real = transcendent_function(ts)
    
    f_aprx_vals = []
    for t in ts:
        f_aprx_vals.append(f_approx((t - x0) / h, delta_f_vals))
    f_aprx_vals = np.array(f_aprx_vals)
    
    eps_vals = np.abs(f_real - f_aprx_vals)
    
    return xs, ts, f_real, f_aprx_vals, eps_vals

def plot_results(x0, xn, n_values, step):
    fig, axes = plt.subplots(len(n_values), 3, figsize=(15, 12))
    
    for i in range(len(n_values)):
        n = n_values[i]
        xs, ts, f_real, f_aprx_vals, eps_vals = compute_tabulation(x0, xn, n, step)
        
        axes[i, 0].plot(xs, transcendent_function(xs), 'ro-', label='Function nodes')
        axes[i, 0].set_title(f'n={n}: Function nodes')
        axes[i, 0].legend()
        
        axes[i, 1].plot(ts, f_real, 'b-', label='Real f(t)')
        axes[i, 1].plot(ts, f_aprx_vals, 'g--', label='Approx f_aprx(t)')
        axes[i, 1].set_title(f'n={n}: Real and Approximate')
        axes[i, 1].legend()
        
        axes[i, 2].plot(ts, eps_vals, 'm-', label='Approximation error')
        axes[i, 2].set_title(f'n={n}: Approximation error')
        axes[i, 2].legend()
    
    plt.tight_layout()
    plt.show()

plot_results(X0, XN, N_VALUES, STEP)

