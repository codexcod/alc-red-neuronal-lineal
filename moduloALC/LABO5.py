import numpy as np
from .LABO3 import norma

def producto_interno(a, b):
    
    if a.shape != b.shape : return None
    
    res = []
    
    for i in range(a.shape[0]):
        res.append(a[i]*b[i])
    
    return sum(res)

#Recordar estrucutra QR:
#Q es la base ortonormal generada a partir vectores
#R tiene las coordenas de en la base ortonormal de los vectores originales

def QR_con_GS(A, tol=1e-12, nops=False):
    m, n = A.shape
    if m != n: return None
    
    Q = np.zeros((n, n))
    R = np.zeros((n, n))
    
    operaciones = 0  
    
    Q[:,0] = A[:,0]/norma(A[:,0],2)
    R[0,0] = norma(A[:,0],2)
    operaciones += 4*n  #calculo 2 normas
    
    for j in range(1,n):
        aJ = A[:, j].copy()   #tomamos la col j de A
        
        for k in range(j):
            R[k, j] = producto_interno(Q[:, k], A[:, j])
            operaciones += 2*n          #2n operaciones multiplicaciones y sumas
            
            aJ = aJ - R[k, j] * Q[:, k]     #restamos la proyeccion
            operaciones += 2*n          #resta y prod interno
        
        R[j, j] = norma(aJ, 2)
        operaciones += 2*n  #norma
        
        
        if abs(R[j, j]) < tol:  #R[i,i] tiene las normas de los vectores, si son muy cercanas a 0, vamos a casi dividir por 0, las filtro por tolerancia
            R[j, j] = 0         
            Q[:, j] = 0         #reemplazqamos por 0 tmb en Q, ya que el vector es ld
        else:
            Q[:, j] = aJ / R[j, j]      #si no esta muy cerca de 0, lo normalizamos
            operaciones += n    #division
    
    if nops:
        return Q, R, operaciones
    
    else:
        return Q, R


def QR_con_HH(A, tol=1e-12):

    m, n = A.shape
    
    if m < n:  return None
    
    R = A.copy()
    Q = np.eye(m)

    for k in range(n):
        
        # Vector x desde la fila k
        x = R[k:, k]
        e1 = np.zeros(x.shape)
        e1[0] = 1

        esc = -np.sign(x[0]) * norma(x,2)
        
        if esc == 0:
            continue

        v = x - esc * e1
        v = v / norma(v,2)

        #reflecion en R (solo parte inferior)
        for j in range(k, n):
            # columna j de la submatriz
            col = R[k:, j]
            # producto interno v^T col
            proj = 2 * producto_interno(v, col)
            # col = col - 2*v*(v^T col)
            R[k:, j] = col - proj * v

        #reflexion en Q
        
        for j in range(m):
            fila = Q[j, k:]
            proj = 2 * producto_interno(fila, v)
            Q[j, k:] = fila - proj * v

    # Filtramos elementos pequenios
    R[np.abs(R) < tol] = 0

    return Q, R


def calculaQR(A, metodo='RH', tol=1e-12):
    
    m, n = A.shape
    if m != n: return None
    
    if metodo != 'RH' and metodo != 'GS' : return None
    
    if metodo == 'GS':
        return QR_con_GS(A,tol,False)
    
    else:
        return QR_con_HH(A,tol)
    
    