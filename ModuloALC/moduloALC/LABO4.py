import numpy as np

def calculaLU(A):
    if A is None : return None, None, 0
    
    cant_op = 0
    m, n = A.shape
    Ac = A.copy()
    
    if m != n:
        print("Matriz no cuadrada")
        return None, None, 0

    for k in range(n-1):
        pivote = Ac[k, k]
        if pivote == 0:
            print("Pivote nulo")
            return None, None, 0
        
        for i in range(k+1, n):
            m_ik = Ac[i, k] / pivote
            Ac[i, k] = m_ik
            cant_op += 1
            
            for h in range(k+1, n):
                Ac[i, h] = Ac[i, h] - m_ik * Ac[k, h]
                cant_op += 2

    L = obtenerL(Ac)
    U = obtenerU(Ac)

    return L, U, cant_op


def obtenerL(Ac):
    
    n = Ac.shape[0]
    L = np.zeros(Ac.shape)
    
    for i in range(n):
        for j in range(n):
            if i > j:
                L[i, j] = Ac[i, j]
            elif i == j:
                L[i, j] = 1
    return L


def obtenerU(Ac):
    
    n = Ac.shape[0]
    U = np.zeros(Ac.shape)
    
    for i in range(n):
        for j in range(n):
            if i <= j:
                U[i, j] = Ac[i, j]
                
    return U

M = np.array([[1, 1, 1],
              [2, 7, 3],
              [3, 17, 7]])


# xi = 1/A(i,i) (b[i] - suma)
# suma prod interno de el vextor x con los elementos de la fila que vienen antes de i

def res_tri(L, b, inferior=True):
    
    if inferior == False:
        return res_tri_sup(L,b)
    
    n = L.shape[0]
    x = np.zeros(n)

    for i in range(n):
        suma = 0
        
        for j in range(i):      # hasta i-1 para que no haga el producto con (i,i)
            suma += L[i, j] * x[j]
            
        x[i] = (b[i] - suma) / L[i, i]

    return x

def res_tri_sup(U, b):
    
    n = U.shape[0]
    x = np.zeros(n)
    
    for i in reversed(range(n)):
        suma = 0
        
        for j in range(i+1, n):   # elementos después de la diagonal, sin incluir
            suma += U[i, j] * x[j]
            
        x[i] = (b[i] - suma) / U[i, i]
    
    return x


def traspuesta(m : np.array) -> np.array:

    c = m.shape[1]
    j = 0
    res = []
    
    while (j < c):
        newFila = []
        
        for elem in m:
            newFila.append(float(elem[j]))
            
        res.append(newFila)
        j += 1
    
    return np.array(res)


#Si A = LU => Axi =ei  <=> L(Uxi) = ei,
#Hay q resolver Lyi = ei y luego Uxi = yi, ahora A^-1 es (x1, ... ,xn)

def tiene_fila_o_columna_nula(A):
    
    A = np.array(A)  
    filas_nulas = [i for i in range(A.shape[0]) if np.all(A[i, :] == 0)]
    columnas_nulas = [j for j in range(A.shape[1]) if np.all(A[:, j] == 0)]
    
    tiene_nula = bool(filas_nulas or columnas_nulas)
    return tiene_nula

def inversa(A):
    n = A.shape[0]
    I = np.eye(n)
    L = calculaLU(A)[0]
    U = calculaLU(A)[1]
    
    if L is None or U is None : return None
    
    if tiene_fila_o_columna_nula(L) : return None
    if tiene_fila_o_columna_nula(U) : return None
    
    if n != A.shape[1]:
        print("La matriz no es cuadrada")
        return None
    
    #Lyi = ei, I es cuadrada simetrica
    solY = []
    for elem in I:                  
        solY.append(res_tri(L,elem))
    
    solY = np.array(solY)   #Notar que estan en forma de filas las sol, no importa por como vamos a iterar
    
    #Uxi = yi
    
    solX = []
    for elem in solY:
        solX.append(res_tri_sup(U,elem))
    
    solX = np.array(solX)
    
    return traspuesta(solX)

#En la descomposici´on LDV la matriz L es la misma que en
#la factorizaci´on LU , y la matrices D y V resultan de aplicar la descomposici´on
#LU de la matriz U t. Gracias a que la matriz U es triangular superior e in-
#versible, aplicar LU a U t resulta en una matriz diagonal D cuyos elementos son
#iguales a la diagonal de U y una matriz V triangular superior tal que V tD = U t
#o DV = U .

def calculaLDV(A):
    L = calculaLU(A)[0]   
    U = calculaLU(A)[1]
    
    L2 = calculaLU(traspuesta(U))[0]
    U2 = calculaLU(traspuesta(U))[1]
    
    D = traspuesta(U2)
    V = traspuesta(L2)
    
    return L, D, V


def esSimetrica(A, tol):
    n = A.shape[0]
    if n != A.shape[1] : return False
    
    At = traspuesta(A)
    
    for i in range(n):
        for j in range(n):
            if abs(A[i][j] - At[i][j]) > tol : return False
            
    return True
    
    
def esSDP(A, atol=1e-8):
    
    if not esSimetrica(A, atol) : return False
    
    n= A.shape[0]
    D = calculaLDV(A)[1]
    
    for i in range(n):
        if D[i][i] <= 0 : return False
    
    return True

    
    
    
    
    
    
    
        
    
