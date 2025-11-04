import random

def generateRandomMatrix(a : float, b : float, n : int):
    matrix_A : list[list[float]] = [[]]
    for i in range(1, n+1):
        row : list[float] = [0.0]
        for j in range(1, n+1):
            row_number : float = a + b * random.random()
            row.append(row_number)
        matrix_A.append(row)
    return matrix_A

def norm_vector(x : list[float]) -> float:
    max_val = 0.0
    for i in range(1, n+1):
        if abs(x[i]) > max_val:
            max_val = abs(x[i])
    return max_val

def LU_find (
    L : list[list[float]],
    U : list[list[float]],
    A : list[list[float]],
    n : int
):
    for k in range(1, n+1):
        for i in range(k, n+1):
            sum_ : float = 0.0
            for j in range(1, k):
                sum_ += L[i][j] * U[j][k]
            L[i][k] = A[i][k] - sum_

        for i in range(k+1, n+1):
            sum_ = 0.0
            for j in range(1, k):
                sum_ += L[k][j] * U[j][i]
            U[k][i] = (A[k][i] - sum_) / L[k][k]

def LU_solve(
    L : list[list[float]],
    U : list[list[float]],
    B : list[float],
    X : list[float],
    n : int
):
    Z : list[float] = [0] * (n+1)
    Z[1] = B[1] / L[1][1]

    for k in range(2, n+1):
        sum_ : float = 0.0
        for j in range(1, k):
            sum_ += L[k][j] * Z[j]
        Z[k] = (B[k] - sum_) / L[k][k]

    X[n] = Z[n]
    for k in range(n-1, 0, -1):
        sum_ = 0.0
        for j in range(k+1, n+1):
            sum_ += U[k][j] * X[j]
        X[k] = Z[k] - sum_

def new_B(A : list[list[float]], X : list[float], B0 : list[float], n : int):
    for i in range(1, n+1):
        sum_ : float = 0.0
        for j in range(1, n+1):
            sum_ += A[i][j] * X[j]
        B0[i] = sum_


a : float = 1.0
b : float = 9.0
n : int = 100
x_0 : float = 0.51
eps : float = 1e-12
kmax : int = int(1e6)

B : list[float] = [0.0] * (n+1)
B0 : list[float] = [0.0] * (n+1)
X : list[float] = [0.0] * (n+1)
R : list[float] = [0.0] * (n+1)
dX : list[float] = [0.0] * (n+1)

A : list[list[float]] = generateRandomMatrix(a, b, n)
L : list[list[float]] = [[0.0] * (n+1) for _ in range(n+1)]
U : list[list[float]] = [[0.0] * (n+1) for _ in range(n+1)]

for i in range(1, n+1):
    sum_ : float = 0.0
    for j in range(1, n+1):
        sum_ += A[i][j]
    B[i] = sum_ * x_0

for i in range(1, n+1):
    for j in range(1, n+1):
        if i == j:
            U[i][j] = 1.0

LU_find(L, U, A, n)
LU_solve(L, U, B, X, n)

for i in range(1, n+1):
    dX[i] = X[i] - x_0
r_eps : float = norm_vector(dX)
print(f"Start solution = {r_eps:.12e}")

k : int = 0
while True:
    new_B(A, X, B0, n)
    for i in range(1, n+1):
        R[i] = B[i] - B0[i]
    LU_solve(L, U, R, dX, n)
    for i in range(1, n+1):
        X[i] += dX[i]

    if (norm_vector(dX) < eps and norm_vector(R) < eps):
        break

    if k >= kmax:
        print("Reached max iteration limit.")
        break

    k += 1


print(f"Number of iterations = {k}")
print(f"‖dX‖ = {norm_vector(dX):.3e}")
print(f"‖R‖  = {norm_vector(R):.3e}")

print("\nFirst 5 elements of X:")
for i in range(1, min(6, n+1)):
    print(f"X[{i}] = {X[i]:.6f}")
