import numpy as np
import matplotlib.pyplot as plt

# 1. Analytical integral (for example function)
from scipy.integrate import quad

def f(x):
    return np.exp(x) * np.sin(x**2) + np.log(x+2) + np.arctan(x)

a = 0
b = np.pi

I0, _ = quad(f, a, b)

# 2. Simpson's method implementation
def simpson(f, a, b, N):
    if N % 2 == 1:
        N += 1
    h = (b - a) / N
    x = np.linspace(a, b, N+1)
    y = f(x)
    S = y[0] + y[-1] + 4 * np.sum(y[1:N:2]) + 2 * np.sum(y[2:N-1:2])
    return S * h / 3

# 3. Dependence of accuracy on N (Plotting error graph)
N_vals = np.arange(10, 1001, 10)
errors = [abs(simpson(f, a, b, N) - I0) for N in N_vals]
plt.plot(N_vals, errors)
plt.yscale('log')
plt.xlabel('N')
plt.ylabel('Error')
plt.title('Error vs Number of intervals N (Simpson)')
plt.grid(True)
plt.show()

# Find minimum N for given tolerance
eps = 1e-8
N_req = next(N for N, err in zip(N_vals, errors) if err < eps)
epsopt = abs(simpson(f, a, b, N_req) - I0)
print(f"Required N for eps={eps}: {N_req}, error: {epsopt}")

# 4. Error for rounded N0 (nearest multiple of 8)
N0 = (N_req // 8) * 8
if N0 == 0: N0 = 8
eps0 = abs(simpson(f, a, b, N0) - I0)
print(f"N0 (nearest multiple of 8): {N0}, error: {eps0}")

# 5. Runge–Romberg refinement
N1 = N0
N2 = 2 * N0
I_N1 = simpson(f, a, b, N1)
I_N2 = simpson(f, a, b, N2)
I_RR = I_N2 + (I_N2 - I_N1) / (2**4 - 1)
eps_RR = abs(I_RR - I0)
print(f"Runge–Romberg result: {I_RR}, error: {eps_RR}")

# 6. Eitken acceleration refinement
def eitken_acceleration(I_N1, I_N2, I_N4):
    p = np.log(np.abs(I_N4 - I_N2) / np.abs(I_N2 - I_N1)) / np.log(2)
    I_Eitken = (I_N4 * I_N1 - I_N2**2) / (I_N4 + I_N1 - 2*I_N2)
    return I_Eitken, p

N4 = 4*N0
I_N4 = simpson(f, a, b, N4)
I_Eitken, p = eitken_acceleration(I_N1, I_N2, I_N4)
eps_Eitken = abs(I_Eitken - I0)
print(f"Eitken result: {I_Eitken}, error: {eps_Eitken}, extrapolated order: {p}")

# 7. Adaptive Simpson's method (optional refinement)
def adaptive_simpson(f, a, b, tol=1e-12):
    def helper(f, a, b, fa, fb, fm, I, tol):
        m = (a + b) / 2
        lm = (a + m) / 2
        rm = (m + b) / 2
        fla, flm, flr = f(a), f(lm), f(m)
        frm = f(rm)
        frb, frb = f(b), f(rm)
        left = (m - a) / 6 * (fa + 4*f(lm) + fm)
        right = (b - m) / 6 * (fm + 4*f(rm) + fb)
        if abs(left + right - I) < 15 * tol:
            return left + right
        else:
            return helper(f, a, m, fa, fm, f(lm), left, tol/2) + \
                   helper(f, m, b, fm, fb, f(rm), right, tol/2)
    fa, fb = f(a), f(b)
    fm = f((a + b) / 2)
    I = (b - a) / 6 * (fa + 4*fm + fb)
    return helper(f, a, b, fa, fb, fm, I, tol)

adaptive_result = adaptive_simpson(f, a, b, eps)
print(f"Adaptive Simpson result: {adaptive_result}, error: {abs(adaptive_result - I0)}")

