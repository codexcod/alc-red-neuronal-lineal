import numpy as np

#FUNCIONES AUXILIARES
def sumatoria(x:np.array):
    res = 0

    for elem in x:
        res+= elem
    return res

def absValue(e : int) -> int:
    if e < 0:
        return -e
    else:
        return e

def sumatoriaAbs(x:np.array):
    res = 0

    for elem in x:
        res+= absValue(elem)
    return res
    

def calcularAx(A, x):
    c = A.shape[1]
    res = []

    for fila in A:
        num = 0
        for j in range(0,c):
            num += fila[j] * x[j]
        res.append(num)

    return np.array(res)
    
def dimCasera(m : np.array):
    i = 0
    j = 0
    for elem in m:
        i += 1
    for elem in m[0]:
        j += 1

    return (i,j)

def traspuesta(m):

    c = dimCasera(m)[1]
    j = 0
    res = []
    
    while (j < c):
        newFila = []
        
        for elem in m:
            newFila.append(float(elem[j]))
            
        res.append(newFila)
        j += 1
    
    return np.array(res)

#FUNCIONES DEL MODULO

def norma (x,p):
    if (p == "inf"):
        actelem = absValue(x[0]) 
        for elem in x:
            if absValue(elem) > actelem: actelem = absValue(elem)
        return actelem
    else:        
        res = []
        for elem in x:
            res.append(float(elem**p))

        res = sumatoria(res)
        
    return res**(1/p)

def normaliza(X, p):
    
    if X.ndim == 1:
        n = norma(X, p)
        return X * (1 / n)
    
    res = []
    for elem in X:
        res.append(np.array(elem) * (1 / norma(elem, p)))
        
    return res

def normaExacta(A,p=[1, 'inf']):
    
    if p != 'inf' and p != 1 : return None
    
    normaInf = 0
    for elem in A:
        if sumatoriaAbs(elem) > normaInf:
            normaInf  = sumatoriaAbs(elem)

    norma1 = 0
    trasA = traspuesta(A)
    for elem in trasA:
        if sumatoriaAbs(elem) > norma1:
            norma1  = sumatoriaAbs(elem)

    return  [float(norma1),float(normaInf)]
    

def normaMatMC2(A, q , p , Np):
    
    vectores = np.random.rand(Np, dimCasera(A)[1])
    trueVectors = normaliza(vectores,p)
    maximo = 0

    for elem in trueVectors:
        resultAx = calcularAx(A,elem)
        if maximo < norma(resultAx, q):
            maximo = norma(resultAx, q)
            bestVector = resultAx

    return maximo

def normaMatMC(A, q , p , Np):
    
    vectores = np.random.rand(Np, dimCasera(A)[1])
    trueVectors = normaliza(vectores,p)
    maximo = 0

    for elem in trueVectors:
        resultAx = calcularAx(A,elem)
        if maximo < norma(resultAx, q):
            maximo = norma(resultAx, q)
            bestVector = resultAx

    return maximo, np.array(bestVector)
 

def condMC(A, p):
    invA = np.linalg.inv(A)

    nA = normaMatMC2(A, p , p , 1000)
    ninvA = normaMatMC2(invA, p, p ,1000)

    return float(nA) * float(ninvA)

#def condExacto(A, p):
    
