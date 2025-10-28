import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.exp(-x/4) * np.sinh(x/2) + np.cos(x**2)

def tabulate_function(x0, xn, n):
    h = (xn - x0) / n
    x = np.array([x0 + i * h for i in range(n + 1)])
    y = f(x)
    return x, y

def form_normal_equations(x, y, m):
    n = len(x)
    B = np.zeros((m + 1, m + 1))
    C = np.zeros(m + 1)
    for k in range(m + 1):
        for j in range(m + 1):
            B[k, j] = np.sum(x**(k + j))
        C[k] = np.sum(y * x**k)
    return B, C

def gauss_elimination(B, C):
    m = len(C)
    # Copy the arrays into lists for easier manipulation
    A = B.copy().tolist()
    b = C.copy().tolist()
    # Forward elimination
    for k in range(m):
        # Pivot: find the max element in column k
        maxrow = k
        for i in range(k + 1, m):
            if abs(A[i][k]) > abs(A[maxrow][k]):
                maxrow = i
        # Swap rows if needed
        if maxrow != k:
            A[k], A[maxrow] = A[maxrow], A[k]
            b[k], b[maxrow] = b[maxrow], b[k]
        # Eliminate lower rows
        for i in range(k + 1, m):
            if A[k][k] == 0:
                continue
            factor = A[i][k] / A[k][k]
            for j in range(k, m):
                A[i][j] -= factor * A[k][j]
            b[i] -= factor * b[k]
    # Back substitution
    a = [0.0 for _ in range(m)]
    for i in range(m - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, m):
            s += A[i][j] * a[j]
        if A[i][i] == 0:
            a[i] = 0
        else:
            a[i] = (b[i] - s) / A[i][i]
    return np.array(a)

def poly_approx(a, x):
    return sum([a[j] * x**j for j in range(len(a))])

def dispersion(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

x0 = 0
xn = np.pi
n = 30

x_tab, y_tab = tabulate_function(x0, xn, n)

dispersions = []
poly_coeffs = []
degrees = list(range(1, 11))

for m in degrees:
    B, C = form_normal_equations(x_tab, y_tab, m)
    a = gauss_elimination(B, C)
    y_approx = poly_approx(a, x_tab)
    disp = dispersion(y_tab, y_approx)
    dispersions.append(disp)
    poly_coeffs.append(a)

min_disp_index = int(np.argmin(dispersions))
best_m = degrees[min_disp_index]
best_coeffs = poly_coeffs[min_disp_index]

print(f"Lowest dispersion: {dispersions[min_disp_index]:.6e} for degree m={best_m}")

h1 = (xn - x0) / (20 * n)
x_fine = np.arange(x0, xn + h1 / 2, h1)
plt.figure(figsize=(10, 6))
plt.plot(x_fine, f(x_fine), label="f(x)", color='black', linewidth=2)

for idx in range(len(degrees)):
    m = degrees[idx]
    a = poly_coeffs[idx]
    y_fine_approx = poly_approx(a, x_fine)
    plt.plot(x_fine, y_fine_approx, label=f'Poly deg {m}, disp={dispersions[idx]:.2e}')

plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Least Squares Polynomial Approximations (Gauss method)")
plt.grid(True)
plt.show()

