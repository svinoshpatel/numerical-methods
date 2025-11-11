import numpy as np

def f(x):
    return np.sin(x) + x**2

def y_analytical_derivative(x):
    return np.cos(x) + 2*x

def prime_approximation(f, x0, h):
    return (f(x0 + h) - f(x0 - h)) / (2 * h)

def prime_approximation_2h(f, x0, h):
    return (f(x0 + 2*h) - f(x0 - 2*h)) / (4 * h)

def prime_approximation_4h(f, x0, h):
    return (f(x0 + 4*h) - f(x0 - 4*h)) / (8 * h)

def runge_romberg(y0_h, y0_2h):
    return y0_h + (y0_h - y0_2h) / 3

def eitken(y0_h, y0_2h, y0_4h):
    numerator = y0_2h**2 - y0_4h * y0_h
    denominator = 2*y0_2h - (y0_4h + y0_h)
    return numerator / denominator

def order_of_accuracy(y0_h, y0_2h, y0_4h):
    numerator = y0_4h - y0_2h
    denominator = y0_2h - y0_h
    if numerator * denominator > 0 and denominator != 0:
        return (1 / np.log(2)) * np.log(abs(numerator / denominator))
    else:
        return np.nan

x0 = 0.56
exact = y_analytical_derivative(x0)

h_values = np.logspace(-20, 3, 100)
errors = []

for h in h_values:
    approx = prime_approximation(f, x0, h)
    error = abs(approx - exact)
    errors.append(error)

min_error_index = np.argmin(errors)
h_optimal = h_values[min_error_index]
R0 = errors[min_error_index]

print(f"Optimal h0: {h_optimal:.2e}")
print(f"Minimum error R0: {R0:.2e}")
print(f"Numerical derivative at h0: {prime_approximation(f, x0, h_optimal)}")
print(f"Exact derivative: {exact}")



h = 1e-3
y0_h = prime_approximation(f, x0, h)
y0_2h = prime_approximation_2h(f, x0, h)

R1 = abs(y0_h - exact)

print()
print(f"Numerical derivative for h: {y0_h}")
print(f"Numerical derivative for 2h: {y0_2h}")
print(f"Exact derivative: {exact}")
print(f"Error for h (R1): {R1:.2e}")



y_R = runge_romberg(y0_h, y0_2h)
R2 = (y_R - exact)

print()
print(f"Runge-Romberg corrected value y_R: {y_R}")
print(f"Error after correction R2: {R2:.2e}")



y0_4h = prime_approximation_4h(f, x0, h)
y_E = eitken(y0_h, y0_2h, y0_4h)
p = order_of_accuracy(y0_h, y0_2h, y0_4h)
R3 = abs(y_E - exact)

print()
print(f"Eitken refined derivative: {y_E}")
print(f"Order of accuracy p: {p:.2}")
print (f"Error R3: {R3:.2e}")
