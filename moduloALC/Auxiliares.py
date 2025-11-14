import numpy as np

#Funcion auxiliar que concatena dos matrices que tienen misma cantidad de filas
#es decir "pega" una al lado de la otra
def concatenaColumnas(A, B):
    
    if A.shape[0] != B.shape[0]:
        return None

    filas = A.shape[0]
    cols = A.shape[1] + B.shape[1]
    
    R = np.zeros((filas, cols))

    for i in range(A.shape[1]):     #Copia A
        R[:, i] = A[:, i]

    rango = A.shape[1]
    for j in range(B.shape[1]):     #Copia B
        R[:, rango + j] = B[:, j]

    return R
