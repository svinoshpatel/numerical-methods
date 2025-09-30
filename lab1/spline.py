import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

# Function we want to interpolate
def f(x: np.ndarray) -> np.ndarray:
    return np.exp(x) * np.sin(x**2) + np.log(x+2) + np.arctan(x)

# Natural cubic spline coefficient solver
def natural_cubic_spline(x: np.ndarray, y: np.ndarray):
    N = x.size - 1             # number of intervals
    h = np.diff(x)             # step sizes between nodes
    m = N - 1                  # number of unknown c's inside

    if m > 0:
        # Prepare diagonals of tridiagonal matrix
        l = np.zeros(m); d = np.zeros(m); u = np.zeros(m)
        d[:] = 2.0 * (h[:m] + h[1:m+1])         # main diagonal
        if m > 1:
            l[1:] = h[1:m]                      # lower diagonal
            u[:-1] = h[1:m]                     # upper diagonal

        # Right-hand side vector g
        g = 3.0 * ((y[2:m+2] - y[1:m+1]) / h[1:m+1] - (y[1:m+1] - y[:m]) / h[:m])

        # Thomas algorithm
        cp = np.zeros(m); dp = np.zeros(m)
        cp[0] = u[0] / d[0] if m > 1 else 0.0
        dp[0] = g[0] / d[0]

        # Forward elimination
        for i in range(1, m):
            denom = d[i] - l[i] * cp[i-1]
            cp[i] = (u[i] / denom) if i < m-1 else 0.0
            dp[i] = (g[i] - l[i] * dp[i-1]) / denom

        # Backward substitution
        c_internal = np.zeros(m)
        c_internal[-1] = dp[-1]
        for i in range(m-2, -1, -1):
            c_internal[i] = dp[i] - cp[i] * c_internal[i+1]

        # Full c array (with boundary natural spline conditions = 0)
        c = np.zeros(N + 1)
        c[1:N] = c_internal
    else:
        c = np.zeros(N + 1)

    # Coefficients on each interval
    a_i = y[:-1].copy()
    b_i = (y[1:] - y[:-1]) / h - (h / 3.0) * (2.0 * c[:-1] + c[1:])
    d_i = (c[1:] - c[:-1]) / (3.0 * h)
    return a_i, b_i, c, d_i, h

# Evaluate spline on dense grid
def evaluate_spline(x_nodes, a_i, b_i, c, d_i, h, points_per_interval=20):
    N = x_nodes.size - 1
    xx_list, Sf_list = [], []
    for i in range(N):
        t = np.linspace(0.0, h[i], points_per_interval, endpoint=False)   # local t
        xx_list.append(x_nodes[i] + t)                                   # grid points
        Sf_list.append(a_i[i] + b_i[i] * t + c[i] * t**2 + d_i[i] * t**3) # cubic formula
    # Last node explicitly
    xx_list.append(np.array([x_nodes[-1]]))
    Sf_list.append(np.array([a_i[-1] + (b_i[-1]*h[-1] + c[-1]*h[-1]**2 + d_i[-1]*h[-1]**3)]))
    return np.concatenate(xx_list), np.concatenate(Sf_list)

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Natural cubic spline lab (Python)")
parser.add_argument("-a", type=float, default=0.0, help="Left bound")
parser.add_argument("-b", type=float, default=1.0, help="Right bound")
parser.add_argument("-N", type=int, default=25, help="Number of intervals (20..30 recommended)")
parser.add_argument("-p", type=int, default=20, help="Points per interval for dense grid")
parser.add_argument("--no-plots", action="store_true", help="Do not show plots")
args = parser.parse_args()

# Unpack args
a, b, N, p = args.a, args.b, args.N, args.p

# Construct interpolation nodes
x = np.linspace(a, b, N + 1)   # equally spaced nodes
y = f(x)                       # function values

# Get spline coefficients
a_i, b_i, c, d_i, h = natural_cubic_spline(x, y)

# Evaluate spline on dense grid
xx, Sf = evaluate_spline(x, a_i, b_i, c, d_i, h, points_per_interval=p)

# Compute reference f and absolute error
ff = f(xx)
err = np.abs(Sf - ff)

# Save input table (nodes, steps)
input_table = pd.DataFrame({
    "i": np.arange(N + 1, dtype=int),
    "x_i": x, "y_i": y,
    "h_i": np.concatenate(([np.nan], h))
})
input_table.to_csv("input.txt", index=False, sep="\t", float_format="%.12g")

# Save output table (grid values, spline values, error)
output_table = pd.DataFrame({
    "x": xx, "f(x)": ff, "S(x)": Sf, "|S-f|": err
})
output_table.to_csv("output.txt", index=False, sep="\t", float_format="%.12g")

# Residual check for tridiagonal system
if N > 1:
    m = N - 1
    A = np.zeros(m); A[1:] = h[1:m]
    B = 2.0 * (h[:m] + h[1:m+1])
    C = np.zeros(m); C[:-1] = h[1:m]
    D = 3.0 * ((y[2: m + 2] - y[1: m + 1]) / h[1: m + 1] - (y[1: m + 1] - y[:m]) / h[:m])
    r = A * c[:m] + B * c[1:N] + C * c[2:N+1] - D
    residual_inf = float(np.max(np.abs(r)))
else:
    residual_inf = 0.0

# Print diagnostics
print(f"Інтервал: [{a}, {b}], кількість інтервалів N: {N}, вузлів: {N+1}")
print(f"Кількість точок густої сітки: {xx.size}, max|похибка| = {np.max(err):.6e}, L2 ≈ {np.sqrt(np.trapezoid(err**2, xx)):.6e}")
print(f"Нев’язка тридіагональної СЛАР (норма ∞): {residual_inf:.3e}")

# Optionally show plots
if not args.no_plots:
    plt.figure()
    plt.plot(xx, ff, label="f(x)")
    plt.plot(xx, Sf, label="S(x)", linestyle="--")
    plt.title("f(x) та кубічний сплайн S(x)")
    plt.xlabel("x"); plt.ylabel("значення")
    plt.legend(); plt.grid(True); plt.show()

    plt.figure()
    plt.plot(xx, err, label="|S(x) - f(x)|")
    plt.title("Абсолютна похибка інтерполяції")
    plt.xlabel("x"); plt.ylabel("абсолютна похибка")
    plt.legend(); plt.grid(True); plt.show()
