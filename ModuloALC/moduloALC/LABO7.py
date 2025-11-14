import numpy as np
from .LABO3 import traspuesta
from .LABO6 import productoMatricial, diagRH

def generaVectorSuma1(n):
    v=np.random.rand(n)
    k = sum(v)
    res = v/k
    
    return res

def transiciones_al_azar_continuas(n):
    M = np.zeros((n,n))
    
    for j in range(n):
        v = generaVectorSuma1(n)
        M[:,j] = v
    
    return M

def transiciones_al_azar_uniformes(n, thres):
    
    M = np.zeros((n,n))
    
    for j in range(n):
        v = generaVectorSuma1(n)
        
        posNoNulas = 0
        for i in range(n):
            
            if v[i] >= thres:
                v[i] = 0
            else: posNoNulas+=1
        
        if posNoNulas == 0:
            v[:] = 1/n
        else:
            for k in range(n):
                if v[k] != 0:
                    v[k] = 1/posNoNulas
    
        M[:,j] = v
    
    return M


#Aca A es de mxn
#recordar que, si un autovector tiene al 0 como autovalor asociado, forma parte del nucleo de a. Pues es sol no trivial de Ax = 0
#dado que At @ A es siempre simetrica, es diagonalizable con RH. At @ A  es de nxn

def nucleo(A, tol=1e-15):

    Ad = productoMatricial(traspuesta(A), A)
    
    S, D = diagRH(Ad)
    
    n = A.shape[1]
    resList = []

    for i in range(n):
        if abs(D[i, i]) <= tol:
            resList.append(S[:, i])
    
    if len(resList) == 0: return np.array([])
    
    return traspuesta(np.array(resList))

def crea_rala(listado, m_filas, n_columnas, tol=1e-15):
    
    if len(listado) != 3 : return {}, (m_filas, n_columnas)
    
    I, J, V = listado
    ralaDict = {}
    
    for i, j, v in zip(I, J, V):    #aca tomamos lo elementos de las listas en paralelo
        if abs(v) >= tol:
            
            ralaDict[(i,j)] = v
    
    return ralaDict, (m_filas, n_columnas)

# def multiplica_rala_por_vector(A,v): 