import numpy as np

# ------------------------------
# LABO1.py
# ------------------------------
def dimCasera(m : np.array):
    i = 0
    j = 0
    for elem in m:
        i += 1
    for elem in m[0]:
        j += 1

    return (i,j)

def absValue(e:float):
    if e < 0 :
        return -e
    else:
        return e


def error(x:float ,y:float):

    return absValue(x-y)


def errorRelativo(x:float ,y:float):
    if (x != 0):
        return absValue(x-y)/absValue(x)


def matricesIguales(A,B):

    if dimCasera(A) != dimCasera(B):
        return False

    for i in range(dimCasera(A)[0]):
        for j in range(dimCasera(A)[1]):
            if not sonIguales (A[i][j],B[i][j],atol=1e-08):
                return False

    return True

def matricesIgualesSinError(A,B):

    if dimCasera(A) != dimCasera(B):
        return False

    for i in range(dimCasera(A)[0]):
        for j in range(dimCasera(A)[1]):
            if A[i][j] != B[i][j]:
                return False

    return True

        
def sonIguales(x,y,atol=1e-08):
    
    return np.allclose(error(x,y),0,atol=atol)


# ------------------------------
# LABO2.py
# ------------------------------
def rota(tetha):
    R = np.array([[np.cos(tetha), -np.sin(tetha)],
                  [np.sin(tetha),  np.cos(tetha)]])
    return R                  




def escala(s):
    n = len(s)
    T = np.zeros((n,n))
    for i in range(n):
        T[i,i] = s[i]
    return T     



def rota_y_escala(tetha, s):
    RT = escala(s) @ rota(tetha)
    return RT




def afin(tetha, s, b): # s es de 1x2
    RT = rota_y_escala(tetha, s) #matriz de 2x2
    T = np.eye(3)
    T[:2, :2] = RT       # parte lineal (rotación + escala)
    T[:2, 2] = b         # vector de traslación
    
    return T




def trans_afin(v, tetha, s, b):
    r = np.zeros((3))
    r[:2]= v
    r[2] = 1

    T = afin(tetha, s, b) @ r
    return T[:-1]


# ------------------------------
# LABO3.py
# ------------------------------
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
    

# ------------------------------
# LABO4.py
# ------------------------------
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
        if abs(pivote) < 1e-12:
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

# M = np.array([[1, 1, 1],
#               [2, 7, 3],
#               [3, 17, 7]])


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
    
    L2 = calculaLU(traspuesta(U))[0] #V traspuesta
    U2 = calculaLU(traspuesta(U))[1] #D
    
    D = traspuesta(U2) #Diagonal (no cambia)
    V = traspuesta(L2) #Traspuesta de V traspuesta (queda V)
    
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

def calculaCholesky(A, atol=1e-8):
    if not esSDP(A, atol) : print("La matriz no es SDP"); return None

    D = calculaLDV(A)[1]
    L = calculaLDV(A)[0]

    D_sqrt = dividirDiagonal(D)

    L_chol = productoMatricial(L, D_sqrt) 

    return L_chol
   
def dividirDiagonal(D):
    n = D.shape[0]
    D_sqrt  = np.zeros((n,n))
    
    for i in range(n):
        D_sqrt[i,i] = np.sqrt(D[i,i])
        
    return D_sqrt   


# ------------------------------
# LABO5.py
# ------------------------------
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
    
    
# ------------------------------
# LABO6.py
# ------------------------------
#NECESITAMOS multiplicar A por B 
def productoMatricial(A,B):
    n,m = A.shape
    m2, l = B.shape 
    
    if m!=m2 : return None

    #R = np.zeros((n,l))
    
    #for i in range(n):
    #    for j in range(l):
    #        for k in range(m):
    #            R[i, j] += A[i, k] * B[k, j]
    #return R
    
    # Usa BLAS a través de NumPy para una multiplicación eficiente
    return A @ B

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
    
    # Power method una sola vez (autovector y autovalor dominante)
    v1, lambda1, _ = metpot2k(A, tol, K)
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


# ------------------------------
# LABO7.py
# ------------------------------
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


# ------------------------------
# LABO8.py
# ------------------------------
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


# ------------------------------
# Auxiliares.py
# ------------------------------
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
#entonces solo queda resolver individualmente Axi=bi donde xi y bi son vectores columnas, y ya tenemos una funcion para eso, si A es triangular superior
def res_tri_sup_mat(U, B):
    n, k = B.shape
    X = np.zeros((n, k))
    for i in reversed(range(n)):
        X[i, :] = (B[i, :] - U[i, i+1:] @ X[i+1:, :]) / U[i, i]
    return X

def res_tri_mat(U, B):
    n, m = B.shape
    
    X = np.zeros((n, m))

    for j in range(m):
        X[:, j] = res_tri(U, B[:, j])

    return X




#Esta funcion es necesaria para el punto 4 del TP, necesitamos eliminar la restriccion de m==n
def QR_con_GS_MatRectangular(A, tol=1e-12, nops=False):
    m, n = A.shape
    
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    operaciones = 0  

    #Primera columna
    Q[:,0] = A[:,0]/norma(A[:,0],2)
    R[0,0] = norma(A[:,0],2)
    operaciones += 4*m

    for j in range(1,n):
        aJ = A[:, j].copy()
        
        for k in range(j):
            R[k, j] = producto_interno(Q[:, k], A[:, j])
            operaciones += 2*m
            aJ = aJ - R[k, j] * Q[:, k]
            operaciones += 2*m

        R[j, j] = norma(aJ, 2)
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


# ------------------------------
# Funciones movidas desde mainScript.py (solo funciones, no código de ejecución)
# ------------------------------
# 1. Lectura de Datos 

#Dentro de cats_and_dogs, tenemos train y val, dentro de cada una una carpeta para "cats" y "dogs"
#cada una de estas, contiene un archivo .npy que es la matriz que queremos

def cargarDataset(carpeta):
    
    #Cargamos los embeddings de entrenamiento, np.load lee el archivo .npy
    #Cada archivo contiene una matriz de embeddings, donde cada columna es un embedding
    
    #---Parte de TRAIN---
    cats_train = np.load(carpeta + "/train/cats/efficientnet_b3_embeddings.npy")
    dogs_train = np.load(carpeta + "/train/dogs/efficientnet_b3_embeddings.npy")

    Xt = concatenaColumnas(cats_train, dogs_train)   #Unimos los embeddings

    #Para armar Yt necesitamos poner como columnas las etiquetas, esto lo hacemos con un producto matricial:
        #los vectores etiquetas son vectores columna de 2x1: [1,0] para gato o [0,1] para perro
        #multiplicamos el vector etiqueta por otro vector (1,1, ... ,1) de R^n, donde n es el numero de columnas de la matriz "cats_train" o "dogs_train"
        #esto devuelve una matriz de 2xn, que tiene como columnas las etiquetas correspondientes al embedding
        
    nc = cats_train.shape[1]
    nd = dogs_train.shape[1]

    Y_cats = productoMatricial(np.array([[1],[0]]) , np.ones((1, nc)))
    Y_dogs = productoMatricial(np.array([[0],[1]]) , np.ones((1, nd)))

    Yt = concatenaColumnas(Y_cats, Y_dogs)

    #---Parte de VALIDATION--- (misma idea)
    cats_val = np.load(carpeta + "/val/cats/efficientnet_b3_embeddings.npy")
    dogs_val = np.load(carpeta + "/val/dogs/efficientnet_b3_embeddings.npy")

    Xv = concatenaColumnas(cats_val, dogs_val)

    nv = cats_val.shape[1]
    mv = dogs_val.shape[1]

    Yv_cats = productoMatricial(np.array([[1],[0]]) , np.ones((1, nv)))
    Yv_dogs = productoMatricial(np.array([[0],[1]]) , np.ones((1, mv)))

    Yv = concatenaColumnas(Yv_cats, Yv_dogs)

    return Xt, Yt, Xv, Yv


# 2. Ecuaciones normales:

#Resumen:
    #Para resolver el problema minW ||Y - WX||^2 , donde X es la matriz de embeddings y Y la matriz de etiquetas
    #usamos la pseudoinversa de X, que se define como X+ = (XtX)^(-1) Xt si n > p o X+ = Xt (XXt)^(-1) si n < p
    #si n = p, X+ = X^(-1)

    #donde n = dim embeddings, p = num ejemplos

#Luego la solucion W es W = Y X+
#Vamos a aprovechar la factorizacion de Cholesky con L triangular para resolver varios sistemas
#No seria conveniente calcular la inversa directamente ya que tomaria mucho tiempo y puede ser inestable numericamente 

def pinvEcuacionesNormales(X, _, Y):        #Recibe X, L y Y, devuelve W solucion del problema minW ||Y - WX||^2
    n = X.shape[0]
    p = X.shape[1]
        
    if n > p: #caso n > p
        #Queremos resolver el sistema (XtX) U = Xt usando la factorizacion de Cholesky XtX = LLt
        #nos queda el sistema LLt U = Xt. Llamo Z = Lt U
        #primero resolvemos L Z = Xt con sustitucion adelante (L es triangular inferior)
        #Cada columna de Z es resultado de resolver L zi = xi (xi es columna de Xt)

        L = calculaCholesky(productoMatricial(traspuesta(X), X))


        Z = res_tri_mat(L,traspuesta(X))   

        #luego resolvemos Lt U = Z con sustitucion atras (Lt es triangular superior) donde U es la pseudoinversa de X

        U = res_tri_sup_mat(traspuesta(L), Z)

        #Finalmente calculamos W = Y U donde U es la pseudoinversa 
        W = productoMatricial(Y, U)   # WX = Y  -> W = Y @ X+
        
        return W
    
    if n < p:   #caso n < p 
        #Queremos resolver el sistema V(XXt) = Xt usando la factorizacion de Cholesky XXt = LLt
        #nos queda el sistema V LLt = X. Si transponemos queda LLt Vt = Xt. Llamo Z = Lt Vt 
        #primero resolvemos L Z = X con sustitucion adelante (L es triangular inferior)
        #Cada columna de Z es resultado de resolver L zi = xi (xi es columna de Xt) 

        L = calculaCholesky(productoMatricial(X , traspuesta(X))) 
        Z = res_tri_mat(L, X)

        #luego resolvemos Lt Vt = Z con sustitucion atras (Lt es triangular superior) donde Vt es la pseudoinversa de X transpuesta
        
        Vt = res_tri_sup_mat(traspuesta(L), Z)
        U = traspuesta(Vt)   #pseudoinversa de X

        #Finalmente calculamos W = Y U donde U es la pseudoinversa
        W = productoMatricial(Y, U)    

        return W
    
    if n == p:   #caso n = p:
        X_inv = inversa(X)
        W = productoMatricial(Y, X_inv)
        return W
    

# 3. Descomposicion en Valores singulares  

#Resumen: 
    #La SVD nos deja descomponer cualquier matriz X como X = USVt , donde U y V son ortogonales, osea su inversa es la traspuesta
    #por lo tanto son inversibles, la idea aca es calcular la pseudoinversa de la matriz X, que viene a ser X+ = V S+ Ut
    #donde Vt^(-1) = V , U^(-1) = Ut y S+ es la matriz Sigma pero en su diagonal el inverso multiplicativo de los valores singulares

#Para este ejercicio seguimos el algoritmo 2:
#Requiere: Dada X ∈ Rn×p con n < p, rango(X) = n (rango completo) y Y ∈ Rm×p
#Asegura: Solucion del problema mınW ∥Y − W X∥2

def pinvSVD(U, S, V, Y):
    
    #V = traspuesta(V)  #hay que trasponer la V
    
    #Construimos  S+
    n = S.shape[0]
    S_inv = np.zeros((n,n))
    for i in range(n):
        S_inv[i,i] = 1.0 / S[i]

    #Calculamos la pseudoinversa X+
    X_pinv = productoMatricial(productoMatricial(V , S_inv), traspuesta(U))    #X+ = V S+ Ut

    #Calculamos el W
    W = productoMatricial(Y, X_pinv)    # WX = Y  -> W = Y @ X+
    return W

#Esto es lo que habria que correr para el W del tp pero tarda mucho:
    #U, S, V = svd_reducida(Xt)
    #res = pinvSVD(U,S,V,Yt)
    #print(res)



# 4. Descomposicion QR

#Queremos ahora calcular los pesos W a traves de la factorizacion QR, obteniendo la misma en un caso con GramSmidth y en otro con HouseHolder
#para hallar W seguimos el algoritmo 3

#En el requerimiento del algoritmo: "Dada X ∈ Rn×p y Y ∈ Rm×p"
#y como sabemos que X ∈ Rn×p con n < p, rango(X) = n (rango completo), esto quiere decir como en los anteriores casos que X tiene muchas columnas y menos filas
#pero estas ultimas son linealmente independientes, luego esto garantiza que X @ X.T es inversible, podemos entonces escribir X+ = X.T @ (X @ X.T)^-1

#Las siguientes funciones calculan W usando la factorizacion 

#Aca la Q y R vendrian de QR_con_GS_MatRectangular( traspuesta(X) )

def pinvGramSchmidt(Q, R, Y):
    
    #Aca tenemos que X+ = (QR) @ (R.T @ Q.T) @ (QR)−1 = Q @ (R.T)−1, entonces si multiplicamos a derecha por R.T
    #nos queda X+ @ R.T = Q, si renombramos X+ = V , buscamos V dado por V @ R.T = Q, pero necesitamos "acomodar V", 
    #si trasponemos, (V @ R.T).T = Q.T, luego lo que necesitamos resolver es R @ V.T = Q.T
    #Esto es un sistema matricial triangular superior

    Qt = traspuesta(Q)
    Vt = res_tri_sup_mat(R, Qt)     #La explicacion de esta funcion esta en "Auxiliares.py"

    #Trasponemos para recuperar V
    V = traspuesta(Vt)

    #Luego:
    W = productoMatricial(Y , V)

    return W
    
    
#Ahora Q, R viene de QR_con_HH( traspuesta(X) )
#Aca hay un problema pues QR_con_HH para matrices rectangulares devuelve una R que no es cuadrada, por lo que para trabajar con ella tenemos que
#reducirla y tomar la parte cuadrada util, por lo tanto tambien recortar Q

def pinvHouseHolder(Q, R, Y):
    
    #dimensiones
    mQ, _ = Q.shape
    mR, nR = R.shape  

    #La parte superior cuadrada util de R es
    R_util = R[:nR, :nR]  

    #Ahora necesitamos Q_util que son las primeras nR columnas de Q
    Q_util = Q[:, :nR]      

    Qt = traspuesta(Q_util) 

    #Luego como antes resolvemos el sistema triangular superior R_util @ Vt = Qt
    Vt = res_tri_sup_mat(R_util, Qt)

    V = traspuesta(Vt)

    W = productoMatricial(Y, V)

    return W


#5. Pseudo-Inversa de Moore-Penrose

#Resumen:
    #El objetivo aca es: dadas dos matrices X y pX (una pseudo-Inversa de X) , decidir si cumplen las condiciones de Moore-Penrose, 
    #con una determinada tolerancia. Para que esto suceda, se deben cumplir las siguientes propiedades:
        #I)   X @ pX @ X = X
        #II)  pX @ X @ pX = pX
        #III) (X @ pX)^t = X @ pX
        #IV)  (pX @ X)^t = pX @ X

#Para ello, realizamos una funcion que devuelve un valor booleano, la misma verifica si todas las propiedades se cumplen y, en caso afirmativo retorna True.

def esPseudoInversa(X, pX, tol=1e-7):
    
    m, n = X.shape   #Si X es de tamaño m × n

    #La pseudo-inversa de MP debe ser de tamanio n × m
    if (n, m) != pX.shape:
        return False

    #Propiedad I)
    X_aprox = productoMatricial(        #calculamos el producto que corresponde
        productoMatricial(X,pX) , X
    )

    if not SonMatricesIguales(X_aprox,X,tol):   #Esta funcion compara si las matices son "iguales" con determinada tolerancia
        return False

    #Propiedad II)  misma idea que con la propiedad I
    pX_aprox = productoMatricial(
        productoMatricial(pX,X) , pX
    )
    
    if not SonMatricesIguales(pX_aprox,pX,tol):
        return False
    
    #Propiedad III) y IV), aca basta con chequear si las pseudo-Identidades de nxn y mxm, son simetricas, siempre con la tolerancia pedida
    pseudo_I = productoMatricial(X,pX)
    
    if not esSimetrica(pseudo_I,tol):
        return False
    
    pseudo_I_2 = productoMatricial(pX,X)
    
    if not esSimetrica(pseudo_I_2,tol):
        return False
    
    return True


