
import numpy as np
from .LABO6 import diagRH, productoMatricial
from .LABO3 import traspuesta, norma

# Recordatorio: V es la matriz de autovectores de AᵀA, Σ es la matriz de valores singulares (raíz de los autovalores de AᵀA)
# AᵀA es simétrica definida positiva → sus autovalores son no negativos y sus autovectores forman una base ortonormal.

def svd_reducida(A, k="max", tol=1e-6):
    m, n = A.shape
    if m >= n:
        Ad = productoMatricial(traspuesta(A), A)    #Caso m >= n, calculo primero V
        modo = "V"
    else:
        Ad = productoMatricial(A, traspuesta(A))    #Caso m < n, calculo primero U
        modo = "U"
    
    #diagonalizo la que convenga
    S, D = diagRH(Ad, tol)
    
    
    sigmas = np.zeros(D.shape[0])   #calculo los VS
    for i in range(D.shape[0]):
        sigmas[i] = np.sqrt(D[i, i]) if D[i, i] > 0 else 0

    for i in range(len(sigmas)):    #filtro aquellos cuasi nulos
        if sigmas[i] < tol:
            sigmas = sigmas[:i]
            S = S[:, :i]
            break

    if k != "max":
        k = min(k, len(sigmas))
        sigmas = sigmas[:k]
        S = S[:, :k]

    
    if modo == "V":     #Si tengo V, calculo U
        hatV = S
        B = productoMatricial(A, hatV)
        hatU = np.zeros(B.shape, dtype=float)
        for i in range(B.shape[1]):
            if sigmas[i] > tol:
                hatU[:, i] = B[:, i] / sigmas[i]    #formula + normaliza
            else:
                hatU[:, i] = 0.0

    else:               #Si tengo U, calculo V
        hatU = S
        B = productoMatricial(traspuesta(A), hatU)
        hatV = np.zeros(B.shape, dtype=float)
        for i in range(B.shape[1]):
            if sigmas[i] > tol:
                hatV[:, i] = B[:, i] / sigmas[i]    #formula + normaliza
            else:
                hatV[:, i] = 0.0

    return hatU, sigmas, hatV

