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




#Esta funcion es necesaria para el punto 4 del TP, necesitamos eliminar la restriccion de m==n
def QR_con_GS_MatRectangular(A, tol=1e-12, nops=False):
    m, n = A.shape
    
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    operaciones = 0  

    #Primera columna
    Q[:,0] = A[:,0]/LABO3.norma(A[:,0],2)
    R[0,0] = LABO3.norma(A[:,0],2)
    operaciones += 4*m

    for j in range(1,n):
        aJ = A[:, j].copy()
        
        for k in range(j):
            R[k, j] = LABO5.producto_interno(Q[:, k], A[:, j])
            operaciones += 2*m
            aJ = aJ - R[k, j] * Q[:, k]
            operaciones += 2*m

        R[j, j] = LABO3.norma(aJ, 2)
        operaciones += 2*m

        if abs(R[j, j]) < tol:
            R[j, j] = 0
            Q[:, j] = 0
        else:
            Q[:, j] = aJ / R[j, j]
            operaciones += m

    if nops:
        return Q, R, operaciones
    else:
        return Q, R
