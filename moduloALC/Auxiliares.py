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
    

#Funcion para decidir si dos matrices son iguales con determinada tolerancia
def SonMatricesIguales(A,B,tol=1e-6):   
    m,n = A.shape
    
    if A.shape != B.shape : return False
    
    M = A - B
    
    for i in range(m):
        for j in range(n):
            if abs(M[i][j]) > tol : return False
    
    return True
