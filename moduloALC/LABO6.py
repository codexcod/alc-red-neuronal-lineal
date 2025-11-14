import numpy as np
from .LABO3 import calcularAx, norma, traspuesta, normaExacta
from .LABO5 import producto_interno

#NECESITAMOS multiplicar A por B 
def productoMatricial(A,B):
    n,m = A.shape
    m2, l = B.shape 
    
    if m!=m2 : return None
    
    R = np.zeros((n,l))
    
    for i in range(n):
        for j in range(l):
            for k in range(m):
                R[i, j] += A[i, k] * B[k, j]
    return R

def transformacionF(A, v):
    
    n, m = A.shape
    if n != m : return None
    if v.shape[0] != n : return None
    
    w1 = calcularAx(A,v)
    
    if norma(w1,2) > 0:
        w = w1/(norma(w1,2))
    
    else:
        w = np.zeros(n)
    
    return w
    
def transformacionF_kVeces(A,v,k):
    
    if k < 1 : return None
    
    for i in range(k):
        v = transformacionF(A,v)
        i+=1
    
    return v

#recordemos que la idea del metodo es calcular Ax repetidas veces ya que esto converge al autovalor dominante(el de mayor modulo)

def metpot2k(A, tol=1e-15, K = 1000):
    n, m = A.shape
    
    v = np.random.rand(n)
    v2 = transformacionF_kVeces(A,v,2)
    e = producto_interno(v2,v)      #esto mide el error, o sea que tanto se parece v2 a v
    k = 0
    
    while abs(e-1) > tol and k < K:     #como v y v2 tienen norma 1, si su prod int da cerca de 1 quiere decir que estan casi aliniados (x . y = ||x||||y|| cos(0) = cos(0) = 1) 
        v = v2
        v2 = transformacionF_kVeces(A,v,2)
        e = producto_interno(v2,v)
        
        k +=1
        
    autoValor = producto_interno(v2, calcularAx(A,v2))      #formula de Reyl... para hallar autovalor a partir de un autovector 
    
    return v2, autoValor, k


#AHORA HOUSE HOLDER 

#Se asume que, A es de R^(nxn), simetrica

def producto_deVectores(a, b):
    
    n = len(a)
    m = len(b)
    M = np.zeros((n, m))
    
    for i in range(n):
        for j in range(m):
            M[i, j] = a[i] * b[j]
    
    return M


def diagRH(A, tol=1e-15,K=1000):
    n,m = A.shape 
    if n!= m : return None
    
    v1 = metpot2k(A,tol,K)[0]
    lambda1 = metpot2k(A,tol,K)[1]
    e1 = np.zeros(n)
    e1[0] = 1 
    
    Hv1 = np.eye(n) - 2 * ((producto_deVectores(e1-v1,e1-v1))/ (norma(e1-v1,2)**2)) 
    
    if n==2:
        S = Hv1
        D = productoMatricial(Hv1, productoMatricial(A,traspuesta(Hv1)))
        return S,D
    
    else:
        B = productoMatricial(Hv1, productoMatricial(A,traspuesta(Hv1)))
        A2 = B[1:n,1:n]
        S0, D0 = diagRH(A2,tol,K)
        
        D = np.zeros((n,n))
        D[0,0] = lambda1
        D[1:, 1:] = D0      #submatriz inferior izq
        
        Saux = np.zeros((n,n))
        Saux[0,0] = 1
        Saux[1:,1:] = S0
        S = productoMatricial(Hv1,Saux)
        
        return S, D

#pasaron los tests