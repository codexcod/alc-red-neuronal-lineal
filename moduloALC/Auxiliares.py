import numpy as np
import moduloALC.LABO3 as LABO3
import moduloALC.LABO5 as LABO5

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




#La manera de resolver AX=B para X, donde A,B Y X son matrices se puede ver con multiplicacion por bloques: (Ax1 | Ax2 ... | Axn) = (b1 | b2 ... | bn),
#entonces solo queda resolver individualmente Axi=bi donde xi y bi son vectores columnas, y ya tenemos una funcion para eso, si A es triangular

def res_tri_mat(U, B):
    n, m = B.shape
    
    X = np.zeros((n, m))

    for j in range(m):
        X[:, j] = LABO4.res_tri(U, B[:, j])

    return X





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


#Algoritmo para armar la matriz de confusion, con accuracy

W = #METODO CORRESPONDIENTE

print("Matriz W:\n", W)

#Ahora quiero ver que tanto se parece W @ Xv a Yv

Yv_pred = W @ Xv

#Separo por clases

pred = np.argmax(Yv_pred, axis=0)
true = np.argmax(Yv, axis=0)

print("True labels:", true)
print("Pred labels:", pred)

accuracy = np.mean(pred == true)
print("Accuracy:", accuracy)


def confusion_matrix(true, pred, num_classes):
    M = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(true, pred):
        M[t, p] += 1
    return M

cm = confusion_matrix(true, pred, num_classes=2)
print("Confusion matrix:\n", cm)

