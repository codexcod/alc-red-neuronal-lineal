import unittest
import numpy as np
from alc import cholesky, esSDP, matricesIguales, solve_cholesky

class TestCholesky(unittest.TestCase):
    def test_identidad(self):
        A = np.eye(5)
        self.assertTrue(esSDP(A))
        L = cholesky(A)
        self.assertIsNotNone(L)
        self.assertTrue(matricesIguales(L @ L.T, A))
        # L debe ser triangular inferior
        self.assertTrue(np.allclose(np.triu(L, 1), 0.0))

    def test_random_spd(self):
        rng = np.random.default_rng(0)
        B = rng.normal(size=(8, 8))
        A = B.T @ B + 1e-6 * np.eye(8)  # asegurar SPD
        self.assertTrue(esSDP(A))
        L = cholesky(A)
        self.assertIsNotNone(L)
        self.assertTrue(matricesIguales(L @ L.T, A))
        self.assertTrue(np.all(np.diag(L) > 0.0))

    def test_no_spd(self):
        # Matriz simétrica no definida positiva
        A = np.array([[0.0, 1.0],
                      [1.0, 0.0]])
        self.assertFalse(esSDP(A))
        with self.assertRaises(ValueError):
            _ = cholesky(A)

    def test_resolver_sistema(self):
        rng = np.random.default_rng(1)
        M = rng.normal(size=(6, 6))
        A = M.T @ M + 1e-6 * np.eye(6)   # SPD
        b = rng.normal(size=6)
        x_chol = solve_cholesky(A, b)
        self.assertIsNotNone(x_chol)
        # Comparar con solución directa
        x_np = np.linalg.solve(A, b)
        self.assertTrue(np.allclose(x_chol, x_np, atol=1e-8))
        # Residuo pequeño
        r = A @ x_chol - b
        self.assertLess(np.linalg.norm(r), 1e-8)

    # Pruebas simples adicionales (sin múltiples RHS)
    def test_simple_cases(self):
        rng = np.random.default_rng(7)
        # SPD 2x2 básica
        A = np.array([[4.0, 2.0],[2.0, 3.0]])
        L = cholesky(A)
        self.assertIsNotNone(L)
        self.assertTrue(np.allclose(L @ L.T, A, atol=1e-12))
        # Resolver Ax=b con vector
        b = np.array([1.0, -1.0])
        x = solve_cholesky(A, b)
        self.assertIsNotNone(x)
        self.assertTrue(np.allclose(A @ x, b, atol=1e-10))

if __name__ == "__main__":
    unittest.main()


