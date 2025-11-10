import math
import numpy as np
import scipy.linalg

# ──────────────────────────────────────────────────────────────────────────────
#  Funciones básicas adicionales (solo uso mínimo de numpy en estas funciones)
#  - Permitido: conversión a arrays, tamaño, máximos/mínimos
#  - Implementación con listas y bucles donde es posible
# ──────────────────────────────────────────────────────────────────────────────

def error_relativo(x, y):
    """Recibe dos numeros x e y, y calcula el error relativo de aproximar x usando y en float64."""
    x64 = float(x)
    y64 = float(y)
    denom = abs(x64)
    if denom == 0.0:
        return abs(y64)  # si x=0, error relativo se interpreta como |y|
    return abs(x64 - y64) / denom


def matricesIguales(A, B):
    """Devuelve True si ambas matrices son iguales (misma forma y valores ~ iguales)."""
    A_np = np.array(A)
    B_np = np.array(B)
    if A_np.shape != B_np.shape:
        return False
    return np.allclose(A_np, B_np)


# ──────────────────────────────────────────────────────────────────────────────
#  Transformaciones lineales y afines en R^2 (homogéneas 3x3 cuando aplique)
# ──────────────────────────────────────────────────────────────────────────────

def rota(theta):
    """Retorna una matriz 2x2 que rota un vector un ángulo theta (en radianes)."""
    c = math.cos(theta)
    s = math.sin(theta)
    return [[c, -s],
            [s,  c]]


def escala(s):
    """Recibe una tira de números s y retorna matriz diagonal n x n que escala la componente i por s[i]."""
    if hasattr(s, "tolist"):
        s = s.tolist()
    n = len(s)
    M = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        M[i][i] = float(s[i])
    return M


def _matmul_2x2(A, B):
    return [
        [A[0][0]*B[0][0] + A[0][1]*B[1][0], A[0][0]*B[0][1] + A[0][1]*B[1][1]],
        [A[1][0]*B[0][0] + A[1][1]*B[1][0], A[1][0]*B[0][1] + A[1][1]*B[1][1]],
    ]


def rota_y_escala(theta, s):
    """Rota en ángulo theta y luego escala por factores s (longitud 2). Retorna 2x2."""
    if hasattr(s, "tolist"):
        s = s.tolist()
    if isinstance(s, (int, float)):
        s = [s, s]
    R = rota(theta)
    S = escala(s)  # 2x2 diagonal
    # Primero rota, luego escala: matriz equivalente es S @ R
    return _matmul_2x2(S, R)


def afin(theta, s, b):
    """
    Retorna matriz 3x3 homogénea:
      - Rota por theta
      - Escala por s (R^2)
      - Traslada por b (R^2)
    """
    if hasattr(s, "tolist"):
        s = s.tolist()
    if hasattr(b, "tolist"):
        b = b.tolist()
    if isinstance(s, (int, float)):
        s = [s, s]
    L = rota_y_escala(theta, s)  # 2x2
    # Matriz afín homogénea
    return [
        [L[0][0], L[0][1], float(b[0])],
        [L[1][0], L[1][1], float(b[1])],
        [0.0,     0.0,     1.0],
    ]


def trans_afin(v, theta, s, b):
    """Aplica la transformación afín a v (en R2): w = S(R v) + b."""
    if hasattr(v, "tolist"):
        v = v.tolist()
    L = rota_y_escala(theta, s)
    w0 = L[0][0]*v[0] + L[0][1]*v[1]
    w1 = L[1][0]*v[0] + L[1][1]*v[1]
    if hasattr(b, "tolist"):
        b = b.tolist()
    return [w0 + b[0], w1 + b[1]]


# ──────────────────────────────────────────────────────────────────────────────
#  Normas y número de condición
# ──────────────────────────────────────────────────────────────────────────────

def _abs(x):
    return -x if x < 0 else x


def norma(x, p):
    """Devuelve la norma p del vector x. p puede ser un escalar > 0 o 'inf'."""
    if hasattr(x, "tolist"):
        x = x.tolist()
    if p == 'inf':
        m = 0.0
        for xi in x:
            axi = _abs(float(xi))
            if axi > m:
                m = axi
        return m
    if p <= 0:
        raise ValueError("p debe ser > 0 o 'inf'")
    acc = 0.0
    for xi in x:
        acc += _abs(float(xi))**p
    return acc**(1.0/p)


def normaliza(X, p):
    """Normaliza cada vector de X con la norma p. X es lista de vectores no vacíos."""
    Y = []
    for v in X:
        nv = norma(v, p)
        if nv == 0.0:
            Y.append([0.0 for _ in range(len(v))])
        else:
            Y.append([float(vi)/nv for vi in (v.tolist() if hasattr(v, "tolist") else v)])
    return Y


def _matvec(A, x):
    if hasattr(A, "tolist"):
        A = A.tolist()
    if hasattr(x, "tolist"):
        x = x.tolist()
    m = len(A)
    n = len(A[0]) if m > 0 else 0
    y = [0.0 for _ in range(m)]
    for i in range(m):
        s = 0.0
        row = A[i]
        for j in range(n):
            s += float(row[j]) * float(x[j])
        y[i] = s
    return y


def _uniform_random_vector(n, *, seed_state):
    # LCG simple para determinismo sin numpy
    a = 1664525
    c = 1013904223
    m = 2**32
    state = seed_state
    def rnd():
        nonlocal state
        state = (a * state + c) % m
        return state / m
    return [2.0 * rnd() - 1.0 for _ in range(n)], state


def normaMatMC(A, q, p, Np):
    """Aproxima ||A||_{q,p} por Monte Carlo; devuelve (valor, x) con ||x||_p=1."""
    if hasattr(A, "tolist"):
        A = A.tolist()
    m = len(A)
    n = len(A[0]) if m > 0 else 0
    best_val = -1.0
    best_x = [0.0 for _ in range(n)]
    state = 123456789
    for _ in range(int(Np)):
        x, state = _uniform_random_vector(n, seed_state=state)
        # normalizar en norma p
        npv = norma(x, p)
        if npv == 0.0:
            continue
        x = [xi / npv for xi in x]
        y = _matvec(A, x)
        val = norma(y, q)
        if val > best_val:
            best_val = val
            best_x = x[:]
    return best_val, best_x


def normaExacta(A, p=[1, 'inf']):
    """Devuelve normas exactas 1 e infinito.
    - p == [1,'inf'] (default): retorna [||A||_1, ||A||_inf]
    - p == 1: retorna ||A||_1
    - p == 'inf': retorna ||A||_inf
    - otro p: retorna None
    """
    if hasattr(A, "tolist"):
        A = A.tolist()
    m = len(A)
    n = len(A[0]) if m > 0 else 0
    # ||A||_1: máximo de sumas por columnas
    col_sums = [0.0 for _ in range(n)]
    for j in range(n):
        s = 0.0
        for i in range(m):
            s += _abs(float(A[i][j]))
        col_sums[j] = s
    norm1 = max(col_sums) if col_sums else 0.0
    # ||A||_inf: máximo de sumas por filas
    row_sums = [0.0 for _ in range(m)]
    for i in range(m):
        s = 0.0
        for j in range(n):
            s += _abs(float(A[i][j]))
        row_sums[i] = s
    norminf = max(row_sums) if row_sums else 0.0
    if p == [1, 'inf']:
        return [norm1, norminf]
    if p == 1:
        return norm1
    if p == 'inf':
        return norminf
    return None


def condMC(A, p):
    """Número de condición κ_p(A) ≈ ||A||_p ||A^{-1}||_p por Monte Carlo."""
    if hasattr(A, "tolist"):
        A = A.tolist()
    if p not in (1, 2, 'inf'):
        p = 2
    valA, _ = normaMatMC(A, p, p, Np=200)
    Ain = inversa(A)
    valInv, _ = normaMatMC(Ain, p, p, Np=200)
    return valA * valInv


def condExacto(A, p):
    """κ_p(A) exacto para p in {1, 'inf'}; para otros p, usa Monte Carlo."""
    if p == 1 or p == 'inf':
        nA1, nAinf = normaExacta(A)
        Ain = inversa(A)
        nAinv1, nAinfinf = normaExacta(Ain)
        if p == 1:
            return nA1 * nAinv1
        else:
            return nAinf * nAinfinf
    # fallback MC
    return condMC(A, p)


# Nota: calculaLU ya está implementada en este módulo (más abajo). No se redefine.

def res_tri(L, b, inferior=True):
    """
    Resuelve L x = b, donde L es triangular.
    - inferior=True: L triangular inferior con 1s (o no) en diagonal
    - inferior=False: L triangular superior
    """
    # Convertimos a listas
    if hasattr(L, "tolist"):
        L = L.tolist()
    if hasattr(b, "tolist"):
        b = b.tolist()
    n = len(L)
    x = [0.0 for _ in range(n)]
    if inferior:
        for i in range(n):
            s = float(b[i])
            for j in range(i):
                s -= float(L[i][j]) * x[j]
            di = float(L[i][i])
            x[i] = s / di if di != 0.0 else float('inf')
    else:
        for i in range(n - 1, -1, -1):
            s = float(b[i])
            for j in range(i + 1, n):
                s -= float(L[i][j]) * x[j]
            di = float(L[i][i])
            x[i] = s / di if di != 0.0 else float('inf')
    return x


def inversa(A):
    """Calcula la inversa de A empleando la factorización LU y resolución triangular."""
    # Convertimos A a numpy array solo para reutilizar la calculaLU existente
    A_np = np.array(A, dtype=float)
    L, U, _ = calculaLU(A_np)
    if L is None:
        return None
    n = A_np.shape[0]
    Inv = [[0.0 for _ in range(n)] for _ in range(n)]
    # Resolución por columnas con e_i
    for i in range(n):
        e = [0.0 for _ in range(n)]
        e[i] = 1.0
        y = res_tri(L, e, inferior=True)
        x = res_tri(U, y, inferior=False)
        for r in range(n):
            Inv[r][i] = x[r]
    return Inv


def calculaLDV(A):
    """
    Calcula L D V tal que A = L D V.
    Para generalidad: usa LU sin pivoteo y separa D = diag(diag(U)), V = D^{-1} U.
    """
    A_np = np.array(A, dtype=float)
    L_np, U_np, _ = calculaLU(A_np)
    if L_np is None:
        return None
    n = L_np.shape[0]
    # Construimos D y V en listas
    D = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        D[i][i] = float(U_np[i, i])
        if D[i][i] == 0.0:
            return None
    V = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            V[i][j] = float(U_np[i, j]) / D[i][i]
    # L en listas
    L = L_np.tolist()
    return L, D, V


def _ldlt(A, atol=1e-12):
    """Descomposición LDL^T para matriz simétrica. Retorna (L, D) o None si falla."""
    if hasattr(A, "tolist"):
        A = A.tolist()
    n = len(A)
    # Copia
    L = [[0.0 for _ in range(n)] for _ in range(n)]
    D = [0.0 for _ in range(n)]
    for i in range(n):
        L[i][i] = 1.0
    for k in range(n):
        s = 0.0
        for j in range(k):
            s += (L[k][j]**2) * D[j]
        Dk = float(A[k][k]) - s
        if abs(Dk) < atol:
            return None
        D[k] = Dk
        for i in range(k + 1, n):
            s2 = 0.0
            for j in range(k):
                s2 += L[i][j] * L[k][j] * D[j]
            L[i][k] = (float(A[i][k]) - s2) / Dk
    # Convert D list to diagonal matrix
    Dmat = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        Dmat[i][i] = D[i]
    return L, Dmat


def esSDP(A, atol=1e-8):
    """Checkea si A es simétrica definida positiva usando LDL^T (LDV con V=L^T)."""
    if hasattr(A, "tolist"):
        A = A.tolist()
    n = len(A)
    # Chequeo de simetría
    for i in range(n):
        for j in range(n):
            if abs(float(A[i][j]) - float(A[j][i])) > atol:
                return False
    # LDL^T
    ldlt = _ldlt(A, atol=1e-12)
    if ldlt is None:
        return False
    _, D = ldlt
    # D positiva
    for i in range(n):
        if D[i][i] <= atol:
            return False
    return True

# ──────────────────────────────────────────────────────────────────────────────
#  Factorización QR (Gram-Schmidt y Householder) sin usar @ / np.matmul
# ──────────────────────────────────────────────────────────────────────────────

def _dot_vec(x, y):
    s = 0.0
    for xi, yi in zip(x, y):
        s += float(xi) * float(yi)
    return s

def _norm2(x):
    return math.sqrt(_dot_vec(x, x))

def QR_con_GS(A, tol=1e-12, retorna_nops=False):
    """
    A una matriz de n x n
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna_nops permite (opcionalmente) retornar el numero de operaciones realizado
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones).
    Si la matriz A no es de n x n, debe retornar None
    """
    if not isinstance(A, np.ndarray) or A.ndim != 2:
        return None
    m, n = A.shape
    if m != n:
        return None
    # Copia de trabajo
    V = A.astype(float).copy()
    Q = np.zeros((n, n), dtype=float)
    R = np.zeros((n, n), dtype=float)
    nops = 0
    for k in range(n):
        # Vector v_k (columna k de V)
        v = V[:, k].copy()
        # Ortogonalización modificada
        for j in range(k):
            # R[j, k] = q_j^T v
            rjk = _dot_vec(Q[:, j], v)
            R[j, k] = rjk if abs(rjk) >= tol else 0.0
            nops += 2 * n  # aproximación (mults+sums) del dot
            # v = v - rjk * q_j
            for i in range(n):
                v[i] = v[i] - rjk * Q[i, j]
            nops += 2 * n
        # R[k, k] = ||v||
        rkk = _norm2(v)
        R[k, k] = rkk if rkk >= tol else 0.0
        nops += 2 * n  # norma ~ dot
        if R[k, k] == 0.0:
            # Vector casi nulo: columna de Q queda cero
            Q[:, k] = 0.0
        else:
            inv = 1.0 / R[k, k]
            for i in range(n):
                Q[i, k] = v[i] * inv
            nops += 2 * n
    if retorna_nops:
        return Q, R, nops
    return Q, R

def QR_con_HH(A, tol=1e-12):
    """
    A una matriz de m x n (m>=n)
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna matrices Q y R calculadas con reflexiones de Householder
    Si la matriz A no cumple m>=n, debe retornar None
    """
    if not isinstance(A, np.ndarray) or A.ndim != 2:
        return None
    m, n = A.shape
    if m < n:
        return None
    R = A.astype(float).copy()
    Q = np.eye(m, dtype=float)
    for k in range(n):
        # Construir reflector para anular debajo de R[k,k] en columna k
        # x = R[k:, k]
        x = R[k:, k].copy()
        # norma de x
        normx = _norm2(x)
        if normx < tol:
            continue
        # Elegir alpha para evitar cancelación
        alpha = -math.copysign(normx, x[0]) if x[0] != 0 else -normx
        # v = x - alpha * e1
        v = x.copy()
        v[0] = v[0] - alpha
        vnorm = _norm2(v)
        if vnorm < tol:
            continue
        # Normalizar v
        for i in range(len(v)):
            v[i] = v[i] / vnorm
        # Aplicar H = I - 2 v v^T a R[k:, k:]
        # w = v^T R_sub (vector fila)
        rows = m - k
        cols = n - k
        w = [0.0 for _ in range(cols)]
        for j in range(cols):
            s = 0.0
            for i in range(rows):
                s += v[i] * R[k + i, k + j]
            w[j] = s
        # R_sub = R_sub - 2 v w
        for i in range(rows):
            for j in range(cols):
                R[k + i, k + j] = R[k + i, k + j] - 2.0 * v[i] * w[j]
        # Forzar ceros por tolerancia debajo de la diagonal
        for i in range(k + 1, m):
            if abs(R[i, k]) < tol:
                R[i, k] = 0.0
        # Actualizar Q = Q * H_k (aplicar a columnas k: en Q)
        # t = Q[:, k:] v  (vector columna tamaño m)
        t = [0.0 for _ in range(m)]
        for i in range(m):
            s = 0.0
            for r in range(rows):
                s += Q[i, k + r] * v[r]
            t[i] = s
        # Q[:, k:] = Q[:, k:] - 2 t v^T
        for i in range(m):
            for r in range(rows):
                Q[i, k + r] = Q[i, k + r] - 2.0 * t[i] * v[r]
    return Q, R

def calculaQR(A, metodo='RH', tol=1e-12):
    """
    A una matriz de n x n
    tol la tolerancia con la que se filtran elementos nulos en R
    metodo = ['RH','GS'] usa reflectores de Householder (RH) o Gram Schmidt (GS) para realizar la factorizacion
    retorna matrices Q y R calculadas con el método elegido
    Si el metodo no esta entre las opciones, retorna None
    """
    if not isinstance(A, np.ndarray) or A.ndim != 2:
        return None
    m, n = A.shape
    if m != n:
        return None
    if metodo == 'RH':
        return QR_con_HH(A, tol=tol)
    if metodo == 'GS':
        Q, R = QR_con_GS(A, tol=tol, retorna_nops=False)
        return Q, R
    return None

# ──────────────────────────────────────────────────────────────────────────────
#  LU, inversas y utilidades relacionadas
# ──────────────────────────────────────────────────────────────────────────────

def calculaLU(A):
    # Calcula LU sin pivoteo y retorna (L, U, nops). Si falla, (None, None, 0).
    if A is None:
        return None, None, 0
    Ac = np.array(A, dtype=float, copy=True)
    if Ac.ndim != 2 or Ac.shape[0] != Ac.shape[1]:
        return None, None, 0
    n = Ac.shape[0]
    nops = 0
    for j in range(n):
        if Ac[j, j] == 0.0:
            return None, None, 0
        for i in range(j + 1, n):
            multiplicador = Ac[i, j] / Ac[j, j]
            nops += 1  # división
            Ac[i, j] = multiplicador
            for k in range(j + 1, n):
                Ac[i, k] = Ac[i, k] - multiplicador * Ac[j, k]
                nops += 2  # multiplicación + resta
    L = np.tril(Ac, -1) + np.eye(n)
    U = np.triu(Ac)
    return L, U, nops

# ──────────────────────────────────────────────────────────────────────────────
#  Método de la potencia y variantes
# ──────────────────────────────────────────────────────────────────────────────

def metpot1(M, *, tol=1e-8, maxrep=1000, seed: int | None = 42):
    rng = np.random.default_rng(seed)
    v = rng.uniform(-1, 1, M.shape[0])
    v /= np.sqrt((v * v).sum())
    λ_prev = v @ (M @ v)
    for _ in range(maxrep):
        v = M @ v
        v /= np.sqrt((v * v).sum())
        λ = v @ (M @ v)
        if abs(λ - λ_prev) / (abs(λ_prev) + 1e-15) < tol:
            return v, float(λ), True
        λ_prev = λ
    return v, float(λ_prev), False


def metpot2k(A, tol=1e-15, K=1000):
    """
    A: matriz de n x n
    tol: tolerancia en la diferencia entre pasos consecutivos del autovector
    K: número máximo de iteraciones
    Retorna: (v, lambda, k) donde v es el autovector dominante (||v||=1),
             lambda el autovalor dominante (en magnitud) y k las iteraciones realizadas.
    No usa multiplicación matricial de numpy.
    """
    if not isinstance(A, np.ndarray) or A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A debe ser cuadrada")
    n = A.shape[0]
    # Inicialización determinista
    rng = np.random.default_rng(42)
    v = rng.uniform(-1.0, 1.0, size=n)
    nv = np.linalg.norm(v)
    if nv == 0.0:
        v = np.zeros(n); v[0] = 1.0
    else:
        v = v / nv
    k_done = 0
    for k in range(int(K)):
        # w = A v  (sin @): por filas con dot 1D
        w = np.empty(n, dtype=float)
        for i in range(n):
            w[i] = float(np.dot(A[i, :], v))
        nw = float(np.linalg.norm(w))
        if nw == 0.0:
            # v fue enviado a un eigenvector nulo, re-inicializar
            v = rng.uniform(-1.0, 1.0, size=n)
            nv = float(np.linalg.norm(v))
            if nv == 0.0:
                continue
            v = v / nv
            continue
        v_next = w / nw
        # criterio de parada: ||v_next - v||_2 <= tol
        diff = float(np.linalg.norm(v_next - v))
        v = v_next
        k_done = k + 1
        if diff <= tol:
            break
    # autovalor por cociente de Rayleigh: v^T (A v)
    Av = np.empty(n, dtype=float)
    for i in range(n):
        Av[i] = float(np.dot(A[i, :], v))
    λ = float(np.dot(v, Av))
    return v, λ, k_done


def inversa_LU(A):
    L, U, _ = calculaLU(A)
    if L is None:
        raise ValueError("No se pudo factorizar LU (pivote nulo o matriz no cuadrada).")
    n = A.shape[0]
    B = np.zeros_like(A, dtype=float)
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        y = scipy.linalg.solve_triangular(L, e, lower=True)
        B[:, i] = scipy.linalg.solve_triangular(U, y)
    return B


def metpotI(M, μ, *, tol=1e-8, maxrep=1000, seed: int | None = 42):
    B = inversa_LU(M + μ * np.eye(M.shape[0]))
    v_shift, λ_B, _ = metpot1(B, tol=tol, maxrep=maxrep, seed=seed)
    λ_min = 1.0 / λ_B - μ
    return v_shift, λ_min, True







# ───────────────────────────────────────────────────────────────
# L06 - Diagonalización por deflación usando metpot2k
# ───────────────────────────────────────────────────────────────

def diagRH(A, tol=1e-15, K=1000):
    """
    A: matriz SIMÉTRICA de n x n
    tol: tolerancia del criterio de metpot2k
    K: máximo de iteraciones para metpot2k
    Retorna S (autovectores columnas) y D (diagonal de autovalores) tal que A = S D S.T
    No usa multiplicación matricial de numpy.
    Si A no es simétrica, retorna None.
    """
    if not isinstance(A, np.ndarray) or A.ndim != 2 or A.shape[0] != A.shape[1]:
        return None
    if not np.allclose(A, A.T, atol=1e-12):
        return None
    n = A.shape[0]
    B = A.astype(float).copy()
    S = np.zeros((n, n), dtype=float)
    Ddiag = np.zeros(n, dtype=float)
    for k in range(n):
        v, lam, _ = metpot2k(B, tol=tol, K=K)
        # asegurar normalización
        nv = math.sqrt(_dot_vec(v.tolist(), v.tolist()))
        if nv == 0.0:
            # fallback si algo degenerado ocurre
            e = np.zeros(n); e[k] = 1.0
            v = e
            lam = 0.0
        else:
            v = (v / nv)
        S[:, k] = v
        Ddiag[k] = lam
        # Deflación: B = B - lam * v v^T  (sin @)
        for i in range(n):
            vi = float(v[i])
            for j in range(n):
                B[i, j] = B[i, j] - lam * vi * float(v[j])
        # Limpieza numérica de simetría
        for i in range(n):
            for j in range(i+1, n):
                mij = 0.5 * (B[i, j] + B[j, i])
                B[i, j] = mij
                B[j, i] = mij
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        D[i, i] = Ddiag[i]
    return S, D

# ───────────────────────────────────────────────────────────────
# L07 - Transiciones estocásticas y matrices ralas
# ───────────────────────────────────────────────────────────────

def transiciones_al_azar_continuas(n: int) -> np.ndarray:
    """
    Retorna T (n x n) con entradas ~ U(0,1) y normalizada por columnas.
    Sin columnas nulas.
    """
    T = np.random.random((n, n))
    for j in range(n):
        s = float(T[:, j].sum())
        if s <= 0.0:
            T[:, j] = 1.0 / n
        else:
            T[:, j] /= s
    return T


def transiciones_al_azar_uniformes(n: int, thres: float) -> np.ndarray:
    """
    Retorna T (n x n) con columnas cuyos elementos no nulos son iguales y
    suman 1 en cada columna. Cada entrada es no nula con prob. thres.
    Si una columna queda sin no nulos, se fuerza un 1 en una fila aleatoria.
    """
    T = np.zeros((n, n), dtype=float)
    rng = np.random.default_rng()
    for j in range(n):
        mask = rng.random(n) <= thres
        k = int(mask.sum())
        if k == 0:
            i = int(rng.integers(0, n))
            T[i, j] = 1.0
        else:
            val = 1.0 / k
            T[mask, j] = val
    return T


def nucleo(A: np.ndarray, tol: float = 1e-15) -> np.ndarray:
    """
    Núcleo de A calculando autovectores de A^T A con autovalor <= tol.
    Retorna matriz n x k con autovectores (o vector vacío si k=0).
    No usa @ para A^T A.
    """
    if not isinstance(A, np.ndarray) or A.ndim != 2:
        return np.zeros((0,))
    m, n = A.shape
    # G = A^T A
    G = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(m):
                s += float(A[k, i]) * float(A[k, j])
            G[i, j] = s
    S_est, D_est = diagRH(G, tol=tol, K=10000)
    if S_est is None:
        return np.zeros((0,))
    vals = np.diag(D_est)
    k = sum(1 for lam in vals if abs(lam) <= tol)
    if k == 0:
        return np.zeros((0,))
    # Refinar base del núcleo buscando los k autovectores mínimos con metpotI
    Gw = G.copy()
    Vs = []
    for _ in range(k):
        v_small, lam_min, _ = metpotI(Gw, μ=1e-6, tol=1e-12, maxrep=5000)
        # normalizar
        v_small = v_small / (np.linalg.norm(v_small) + 1e-30)
        Vs.append(v_small)
        # deflación: Gw = Gw - lam * v v^T
        lam = float(np.dot(v_small, np.dot(Gw, v_small)))
        outer = np.outer(v_small, v_small)
        Gw = Gw - lam * outer
        # forzar simetría
        Gw = 0.5 * (Gw + Gw.T)
    S_zero = np.column_stack(Vs)
    return S_zero


def crea_rala(listado, m_filas: int, n_columnas: int, tol: float = 1e-15):
    """
    Crea representación rala:
    - dict {(i,j): aij} con |aij|>=tol
    - dims (m_filas, n_columnas)
    """
    A_dict = {}
    if not listado:
        return A_dict, (m_filas, n_columnas)
    if len(listado) != 3:
        return A_dict, (m_filas, n_columnas)
    I, J, V = listado
    for i, j, v in zip(I, J, V):
        if abs(float(v)) >= tol:
            A_dict[(int(i), int(j))] = float(v)
    return A_dict, (m_filas, n_columnas)


def multiplica_rala_vector(A_rala, v: np.ndarray) -> np.ndarray:
    """
    Multiplica matriz rala (dict, dims) por vector v.
    """
    if isinstance(A_rala, tuple):
        A_dict, dims = A_rala
        m, n = dims
    else:
        A_dict = A_rala
        m = len(set(i for (i, _) in A_dict.keys())) if A_dict else v.shape[0]
        n = v.shape[0]
    w = np.zeros(m, dtype=float)
    for (i, j), aij in A_dict.items():
        w[i] += aij * float(v[j])
    return w


# ───────────────────────────────────────────────────────────────
# L08 - SVD reducida (sin usar @ / np.matmul)
# ───────────────────────────────────────────────────────────────

def _AtA(A: np.ndarray) -> np.ndarray:
    """Construye G = A^T A sin usar @/matmul."""
    m, n = A.shape
    G = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(m):
                s += float(A[k, i]) * float(A[k, j])
            G[i, j] = s
    return G


def svd_reducida(A: np.ndarray, k="max", tol: float = 1e-15):
    """
    A: matriz m x n
    k: cantidad de valores singulares a retener o 'max' para todos los > tol
    tol: umbral para considerar singular value ~ 0
    Retorna: (U_hat: m x k, Sig_hat: vector k, V_hat: n x k)
    """
    if not isinstance(A, np.ndarray) or A.ndim != 2:
        raise ValueError("A debe ser una matriz 2D de numpy")
    U, S, VT = np.linalg.svd(A)
    if k == "max":
        r = int(np.sum(np.abs(S) > tol))
    else:
        r = int(k)
    r = max(0, min(r, S.shape[0]))
    U_hat = U[:, :r]
    Sig_hat = S[:r]
    V_hat = VT.T[:, :r]
    return U_hat, Sig_hat, V_hat


