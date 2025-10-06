import numpy as np
import matplotlib.pyplot as plt
import argparse

def factorial(k):
    if k == 0 or k == 1:
        return 1
    f = 1
    for i in range(2, k + 1):
        f *= i
    return f

def permutations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def finite_difference(f_vals, k):
    n = len(f_vals)
    diff = 0
    for j in range(k + 1):
        diff += ((-1)**(k - j)) * permutations(k, j) * f_vals[j]
    return diff

def factorial_poly(t, k):
    if k == 0:
        return 1
    result = 1
    for i in range(k):
        result *= (t - i)
    return result

def f_approx(t, delta_f_vals):
    n = len(delta_f_vals)
    s = 0
    for k in range(n):
        s += (delta_f_vals[k] / factorial(k)) * factorial_poly(t, k)
    return s

def transcendent_function(x):
    return np.exp(x) * np.sin(x**2) + np.log(x+2) + np.arctan(x)

def compute_tabulation(x0, xn, n, step=0.01):
    h = (xn - x0) / n
    xs = np.array([x0 + i*h for i in range(n + 1)])
    f_vals = transcendent_function(xs)
    delta_f_vals = [finite_difference(f_vals, k) for k in range(n + 1)]
    ts = np.arange(x0, xn + step, step)
    f_real = transcendent_function(ts)
    f_aprx_vals = np.array([f_approx(t, delta_f_vals) for t in ts])
    eps_vals = np.abs(f_real - f_aprx_vals)
    return xs, ts, f_real, f_aprx_vals, eps_vals

def plot_results(x0, xn, n_values):
    plt.figure(figsize=(15, 12))
    for i, n in enumerate(n_values, 1):
        xs, ts, f_real, f_aprx_vals, eps_vals = compute_tabulation(x0, xn, n)
        plt.subplot(len(n_values), 3, 3*(i-1) + 1)
        plt.plot(xs, transcendent_function(xs), 'ro-', label='Function nodes')
        plt.title(f'n={n}: Function nodes')
        plt.legend()
        plt.subplot(len(n_values), 3, 3*(i-1) + 2)
        plt.plot(ts, f_real, 'b-', label='Real f(t)')
        plt.plot(ts, f_aprx_vals, 'g--', label='Approx f_aprx(t)')
        plt.title(f'n={n}: Real and Approximate')
        plt.legend()
        plt.subplot(len(n_values), 3, 3*(i-1) + 3)
        plt.plot(ts, eps_vals, 'm-', label='Approximation error')
        plt.title(f'n={n}: Approximation error')
        plt.legend()
    plt.tight_layout()
    plt.show()

parser = argparse.ArgumentParser(description='Factorial polynomial approximation of a transcendent function')
parser.add_argument('--x0', type=float, default=0, help='Left bound of interval')
parser.add_argument('--xn', type=float, default=5, help='Right bound of interval')
parser.add_argument('--step', type=float, default=0.01, help='Step for fine tabulation')
args = parser.parse_args()
plot_results(args.x0, args.xn, [5, 10, 20])

