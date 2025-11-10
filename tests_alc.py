import numpy as np
from alc import (
    error_relativo, matricesIguales,
    rota, escala, rota_y_escala, afin, trans_afin,
    norma, normaliza, normaExacta, normaMatMC, condMC, condExacto,
    calculaLU, res_tri, inversa, calculaLDV, esSDP
)


def sonIguales(x, y, atol=1e-08):
    # Igualdad basada en error relativo ~ 0 con tolerancia
    return np.allclose(error_relativo(x, y), 0, atol=atol)


# Tests de igualdad numérica y error relativo
assert not sonIguales(1, 1.1)
assert sonIguales(1, 1 + np.finfo('float64').eps)
assert not sonIguales(1, 1 + np.finfo('float32').eps)
assert not sonIguales(np.float16(1), np.float16(1) + np.finfo('float32').eps)
assert sonIguales(np.float16(1), np.float16(1) + np.finfo('float16').eps, atol=1e-3)

assert np.allclose(error_relativo(1, 1.1), 0.1)
assert np.allclose(error_relativo(2, 1), 0.5)
assert np.allclose(error_relativo(-1, -1), 0)
assert np.allclose(error_relativo(1, -1), 2)

assert matricesIguales(np.diag([1, 1]), np.eye(2))
assert matricesIguales(np.linalg.inv(np.array([[1, 2], [3, 4]])) @ np.array([[1, 2], [3, 4]]), np.eye(2))
assert not matricesIguales(np.array([[1, 2], [3, 4]]).T, np.array([[1, 2], [3, 4]]))

# Tests para rota
assert np.allclose(rota(0), np.eye(2))
assert np.allclose(rota(np.pi / 2), np.array([[0, -1], [1, 0]]))
assert np.allclose(rota(np.pi), np.array([[-1, 0], [0, -1]]))

# Tests para escala
assert np.allclose(escala([2, 3]), np.array([[2, 0], [0, 3]]))
assert np.allclose(escala([1, 1, 1]), np.eye(3))
assert np.allclose(escala([0.5, 0.25]), np.array([[0.5, 0], [0, 0.25]]))

# Tests para rota_y_escala
assert np.allclose(rota_y_escala(0, [2, 3]), np.array([[2, 0], [0, 3]]))
assert np.allclose(rota_y_escala(np.pi / 2, [1, 1]), np.array([[0, -1], [1, 0]]))
assert np.allclose(rota_y_escala(np.pi, [2, 2]), np.array([[-2, 0], [0, -2]]))

# Tests para afin
assert np.allclose(
    afin(0, [1, 1], [1, 2]),
    np.array([[1, 0, 1],
              [0, 1, 2],
              [0, 0, 1]])
)
assert np.allclose(
    afin(np.pi / 2, [1, 1], [0, 0]),
    np.array([[0, -1, 0],
              [1,  0, 0],
              [0,  0, 1]])
)
assert np.allclose(
    afin(0, [2, 3], [1, 1]),
    np.array([[2, 0, 1],
              [0, 3, 1],
              [0, 0, 1]])
)

# Tests para trans_afin
assert np.allclose(
    trans_afin(np.array([1, 0]), np.pi / 2, [1, 1], [0, 0]),
    np.array([0, 1])
)
assert np.allclose(
    trans_afin(np.array([1, 1]), 0, [2, 3], [0, 0]),
    np.array([2, 3])
)
assert np.allclose(
    trans_afin(np.array([1, 0]), np.pi / 2, [3, 2], [4, 5]),
    np.array([4, 7])
)

# Tests L03 - Normas
print("TESTS NORMA")
assert np.allclose(norma(np.array([0, 0, 0, 0]), 1), 0)
assert np.allclose(norma(np.array([4, 3, -100, -41, 0]), "inf"), 100)
assert np.allclose(norma(np.array([1, 1]), 2), np.sqrt(2))
assert np.allclose(norma(np.array([1] * 10), 2), np.sqrt(10))
assert norma(np.random.rand(10), 2) <= np.sqrt(10)
assert norma(np.random.rand(10), 2) >= 0
print("------ÉXITO!!!!\n")

print("TEST NORMALIZA")
print("---TEST NORMALIZA 2")
test_n2 = normaliza([np.array([1] * k) for k in range(1, 11)], 2)
assert len(test_n2) != 0
for x in test_n2:
    assert np.allclose(norma(x, 2), 1)
print("------ÉXITO!!!!")

print("---TEST NORMALIZA 1")
test_n1 = normaliza([np.array([1] * k) for k in range(2, 11)], 1)
assert len(test_n1) != 0
for x in test_n1:
    assert np.allclose(norma(x, 1), 1)
print("------ÉXITO!!!!")

print("---TEST NORMALIZA INF")
test_nInf = normaliza([np.random.rand(k) for k in range(1, 11)], 'inf')
assert len(test_nInf) != 0
for x in test_nInf:
    assert np.allclose(norma(x, 'inf'), 1)
print("------ÉXITO!!!!\n")

print("TEST normaExacta")
assert np.allclose(normaExacta(np.array([[1, -1], [-1, -1]]))[0], 2)
assert np.allclose(normaExacta(np.array([[1, -1], [-1, -1]]))[1], 2)
assert np.allclose(normaExacta(np.array([[1, -2], [-3, -4]]))[0], 6)
assert np.allclose(normaExacta(np.array([[1, -2], [-3, -4]]))[1], 7)
assert normaExacta(np.array([[1, -2], [-3, -4]]), 2) is None
assert normaExacta(np.random.random((10, 10)))[0] <= 10
assert normaExacta(np.random.random((4, 4)))[1] <= 4
print("------ÉXITO!!!!\n")

print("TEST normaMatMC")
nMC = normaMatMC(A=np.eye(2), q=2, p=1, Np=100000)
assert np.allclose(nMC[0], 1, atol=1e-3)
assert np.allclose(abs(nMC[1][0]), 1, atol=1e-3) or np.allclose(abs(nMC[1][1]), 1, atol=1e-3)
assert np.allclose(abs(nMC[1][0]), 0, atol=1e-3) or np.allclose(abs(nMC[1][1]), 0, atol=1e-3)

nMC = normaMatMC(A=np.eye(2), q=2, p='inf', Np=100000)
assert np.allclose(nMC[0], np.sqrt(2), atol=1e-3)
assert np.allclose(abs(nMC[1][0]), 1, atol=1e-3) and np.allclose(abs(nMC[1][1]), 1, atol=1e-3)

A = np.array([[1, 2], [3, 4]])
nMC = normaMatMC(A=A, q='inf', p='inf', Np=1000000)
assert np.allclose(nMC[0], normaExacta(A)[1], rtol=1e-1)
print("------ÉXITO!!!!\n")

print("TEST condMC")
A = np.array([[1, 1], [0, 1]])
A_ = np.linalg.solve(A, np.eye(A.shape[0]))
normaA = normaMatMC(A, 2, 2, 10000)
normaA_ = normaMatMC(A_, 2, 2, 10000)
condA = condMC(A, 2)
assert np.allclose(normaA[0] * normaA_[0], condA, atol=1e-2)

A = np.array([[3, 2], [4, 1]])
A_ = np.linalg.solve(A, np.eye(A.shape[0]))
normaA = normaMatMC(A, 2, 2, 10000)
normaA_ = normaMatMC(A_, 2, 2, 10000)
condA = condMC(A, 2)
assert np.allclose(normaA[0] * normaA_[0], condA, atol=1e-2)
print("------ÉXITO!!!!\n")

print("TEST condExacto")
A = np.random.rand(10, 10)
A_ = np.linalg.solve(A, np.eye(A.shape[0]))
normaA = normaExacta(A)[0]
normaA_ = normaExacta(A_)[0]
condA = condExacto(A, 1)
assert np.allclose(normaA * normaA_, condA)

A = np.random.rand(10, 10)
A_ = np.linalg.solve(A, np.eye(A.shape[0]))
normaA = normaExacta(A)[1]
normaA_ = normaExacta(A_)[1]
condA = condExacto(A, 'inf')
assert np.allclose(normaA * normaA_, condA)
print("------ÉXITO!!!!\n")

print("TESTS calculaLU")
L0 = np.array([[1, 0, 0],
               [0, 1, 0],
               [1, 1, 1]])
U0 = np.array([[10, 1, 0],
               [0, 2, 1],
               [0, 0, 1]])
A = L0 @ U0
L, U, nops = calculaLU(A)
assert np.allclose(L, L0)
assert np.allclose(U, U0)

L0 = np.array([[1, 0, 0],
               [1, 1.001, 0],
               [1, 1, 1]])
U0 = np.array([[1, 1, 1],
               [0, 1, 1],
               [0, 0, 1]])
A = L0 @ U0
L, U, nops = calculaLU(A)
assert not np.allclose(L, L0)
assert not np.allclose(U, U0)
assert np.allclose(L, L0, atol=1e-3)
assert np.allclose(U, U0, atol=1e-3)
assert nops == 13

L0 = np.array([[1, 0, 0],
               [1, 1, 0],
               [1, 1, 1]])
U0 = np.array([[1, 1, 1],
               [0, 0, 1],
               [0, 0, 1]])
A = L0 @ U0
L, U, nops = calculaLU(A)
assert L is None
assert U is None
assert nops == 0

assert calculaLU(None) == (None, None, 0)
assert calculaLU(np.array([[1, 2, 3], [4, 5, 6]])) == (None, None, 0)
print("-----ÉXITO!!!!\n")

print("TESTS res_tri")
A = np.array([[1, 0, 0],
              [1, 1, 0],
              [1, 1, 1]])
b = np.array([1, 1, 1])
assert np.allclose(res_tri(A, b), np.array([1, 0, 0]))
b = np.array([0, 1, 0])
assert np.allclose(res_tri(A, b), np.array([0, 1, -1]))
b = np.array([-1, 1, -1])
assert np.allclose(res_tri(A, b), np.array([-1, 2, -2]))
b = np.array([-1, 1, -1])
assert np.allclose(res_tri(A, b, inferior=False), np.array([-1, 1, -1]))
A = np.array([[3, 2, 1], [0, 2, 1], [0, 0, 1]])
b = np.array([3, 2, 1])
assert np.allclose(res_tri(A, b, inferior=False), np.array([1/3, 1/2, 1]))
A = np.array([[1, -1, 1], [0, 1, -1], [0, 0, 1]])
b = np.array([1, 0, 1])
assert np.allclose(res_tri(A, b, inferior=False), np.array([1, 1, 1]))
print("-----ÉXITO!!!!\n")

print("TESTS inversa")
def esSingular(A):
    try:
        np.linalg.inv(A)
        return False
    except Exception:
        return True

ntest = 10
for _ in range(ntest):
    A = np.random.random((4, 4))
    A_ = inversa(A)
    if not esSingular(A):
        inversaConNumpy = np.linalg.inv(A)
        assert A_ is not None
        assert np.allclose(inversaConNumpy, A_)
    else:
        assert A_ is None

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
assert inversa(A) is None
print("-----ÉXITO!!!!\n")

print("TESTS calculaLDV")
L0 = np.array([[1, 0, 0], [1, 1., 0], [1, 1, 1]])
D0 = np.diag([1, 2, 3])
V0 = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]])
A = L0 @ D0 @ V0
L, D, V = calculaLDV(A)
assert np.allclose(L, L0)
assert np.allclose(D, D0)
assert np.allclose(V, V0)

L0 = np.array([[1, 0, 0], [1, 1.001, 0], [1, 1, 1]])
D0 = np.diag([3, 2, 1])
V0 = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1.001]])
A = L0 @ D0 @ V0
L, D, V = calculaLDV(A)
assert np.allclose(L, L0, 1e-3)
assert np.allclose(D, D0, 1e-3)
assert np.allclose(V, V0, 1e-3)
print("-----ÉXITO!!!!\n")

print("TESTS esSDP")
L0 = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]])
D0 = np.diag([1, 1, 1])
A = L0 @ D0 @ L0.T
assert esSDP(A)

D0 = np.diag([1, -1, 1])
A = L0 @ D0 @ L0.T
assert not esSDP(A)

D0 = np.diag([1, 1, 1e-16])
A = L0 @ D0 @ L0.T
assert not esSDP(A)

L0 = np.array([[1, 0, 0],
               [1, 1, 0],
               [1, 1, 1]])
D0 = np.diag([1, 1, 1])
V0 = np.array([[1, 0, 0],
               [1, 1, 0],
               [1, 1 + 1e-3, 1]]).T
A = L0 @ D0 @ V0
assert esSDP(A, 1e-3)
print("-----ÉXITO!!!!\n")
print("---FINALIZADO LABO 4!---")


