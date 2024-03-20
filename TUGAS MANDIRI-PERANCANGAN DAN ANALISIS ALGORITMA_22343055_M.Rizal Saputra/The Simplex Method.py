import numpy as np

def simplex(c, A, b):
    m, n = A.shape
    
    # Step 1: Inisialisasi
    tableau = np.zeros((m + 1, n + m + 1))
    tableau[0, 1:n + 1] = -c
    tableau[1:, 0:n] = A
    tableau[np.arange(1, m + 1), np.arange(n + 1, n + m + 1)] = 1
    tableau[1:, -1] = b
    
    # Step 2: Iterasi hingga optimal
    while np.any(tableau[0, 1:] < 0):
        entering_var = np.argmin(tableau[0, 1:])
        
        ratios = tableau[1:, -1] / tableau[1:, entering_var + 1]
        leaving_var = np.argmin(ratios) + 1
        
        pivot = tableau[leaving_var, entering_var + 1]
        tableau[leaving_var, 1:] /= pivot
        for i in range(m + 1):
            if i != leaving_var:
                tableau[i, 1:] -= tableau[i, entering_var + 1] * tableau[leaving_var, 1:]
    
    # Step 3: Ekstrak solusi optimal
    optimal_solution = np.zeros(n)
    for i in range(m):
        row = np.where(tableau[i + 1, n + 1:] == 1)[0]
        if len(row) == 1:
            optimal_solution[row] = tableau[i + 1, 0]
    
    return optimal_solution

# Contoh penggunaan
c = np.array([2, 3, 0, 0])
A = np.array([[1, 4, 1, 0],
              [2, 3, 0, 1]])
b = np.array([12, 18])

solusi_optimal = simplex(c, A, b)
print("Solusi optimal:", solusi_optimal)
