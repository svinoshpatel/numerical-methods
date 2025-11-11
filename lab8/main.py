import random

def generate_random_matrix(a : float, b : float, n : int, scale : int):
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        off_diag_sum = 0.0
        for j in range(n):
            val = a + b * random.random()
            if i != j:
                A[i][j] = val
                off_diag_sum += val
        A[i][i] = off_diag_sum * scale
    return A

def norm_vector_diff(Y : list[float], X : list[float], n : int):
    max_val : float = 0.0
    for i in range(n):
        if abs(Y[i] - X[i]) > max_val:
            max_val = abs(Y[i] - X[i])
    return max_val

def norm_matrix(X : list[list[float]], n : int):
    max_val : float = 0.0
    for i in range(n):
        sum_ : float = 0.0
        for j in range(n):
            sum_ += abs(X[i][j])
        if sum_ > max_val:
             max_val = sum_
    return max_val

def gauss_seidel(
    A : list[list[float]],
    B : list[float],
    X0 : list[float],
    X1 : list[float],
    n : int,
):
    eps : float = 1e-14
    kmax : float = 1e5
    k : int = 1

    while True:
        for i in range(n):
            X0[i] = X1[i]
        for i in range(n):
            sum_ = 0.0
            for j in range(n):
                if j < i:
                    sum_ += A[i][j] * X1[j]
                elif j > i:
                    sum_ += A[i][j] * X0[j]
            X1[i] = (B[i] - sum_) / A[i][i]
        k += 1

        if norm_vector_diff(X1, X0, n) < eps or k > kmax:
            break
        
    if k >= kmax:
        print("Gauss-Seidel: Solution not found!")
    else:
        print(f"Gauss-Seidel: Solution found in {k} iterations")

def yacobi(
    A : list[list[float]],
    B : list[float],
    X0 : list[float],
    X1 : list[float],
    n : int
):
    eps : float = 1e-14
    kmax : float = 1e5
    k : int = 1

    while True:
        for i in range(n):
            X0[i] = X1[i]
        for i in range(n):
            sum_ : float = 0.0
            for j in range(n):
                if j != i:
                    sum_ += A[i][j] * X0[j]
            X1[i] = (B[i] - sum_) / A[i][i]
        k += 1
        
        if norm_vector_diff(X1, X0, n) < eps or k > kmax:
            break

    if k >= kmax:
        print("Jacobi: Solution not found!")
    else:
        print(f"Jacobi: Solution found in {k} iterations")

def simple_iteration(
    A : list[list[float]],
    B : list[float],
    X0 : list[float],
    X1 : list[float],
    tau : float,
    n : int
):
    eps : float = 1e-14
    kmax : float = 1e5
    k : int = 1

    while True:
        for i in range(n):
            X0[i] = X1[i]
        for i in range(n):
            sum_ : float = 0.0
            for j in range(n):
                sum_ += X0[j] * A[i][j]
            X1[i] = X0[i] - tau * sum_ + tau * B[i]
        k += 1

        if norm_vector_diff(X1, X0, n) < eps or k > kmax:
            break

    if k >= kmax:
        print("Simple Iteration: Solution not found!")
    else:
        print(f"Simple Iteration: Solution found in {k} iterations")

a : float = 1.0
b : float = 9.0
x : float = 2.51
n : int = 100

A = generate_random_matrix(a, b, n, 10)

B : list[float] = [0.0] * n
for i in range(n):
    sum_ : float = 0.0
    for j in range(n):
        sum_ += A[i][j]
    B[i] = sum_ * x

tau : float = 1 / norm_matrix(A, n)

X0 : list[float] = [0.0] * n
X1 : list[float] = [0.0] * n

simple_iteration(A, B, X0, X1, tau, n)

for i in range(n):
    X1[i] = 0.0

yacobi(A, B, X0, X1, n)

for i in range(n):
    X1[i] = 0.0

gauss_seidel(A, B, X0, X1, n)
