# Biblioteca ALC (Álgebra Lineal Computacional)

Colección de funciones de álgebra lineal y análisis de matrices, implementadas en `alc.py`. Incluye utilidades para normas y condición, factorizaciones LU/QR, métodos de potencia, matrices de transición (Markov), manejo de matrices ralas y SVD reducida. La suite de tests (`tests_*.py`) valida cada módulo.

## Requisitos
- Python 3.10+ recomendado
- Dependencias: `numpy`, `scipy`

Instalación rápida:
```bash
python3 -m pip install --user numpy scipy
```

## Ejecución de tests
```bash
cd tp-neuronas
python3 tests_alc.py
python3 tests_qr.py
python3 tests_metpot.py
python3 tests_markov.py
python3 tests_svd.py
```

## Contenido principal (resumen)

- Básicas:
  - `error_relativo(x,y)`: error relativo de aproximar x con y.
  - `matricesIguales(A,B)`: compara por forma y `allclose`.

- Transformaciones (R²):
  - `rota(theta)`, `escala(s)`, `rota_y_escala(theta,s)`.
  - `afin(theta,s,b)`, `trans_afin(v,theta,s,b)`.

- Normas y condición:
  - `norma(x,p)`, `normaliza(X,p)`.
  - `normaMatMC(A,q,p,Np)`: norma inducida por Monte Carlo.
  - `normaExacta(A,p)`: 1 e infinito (exactas).
  - `condMC(A,p)`, `condExacto(A,p)`.

- LU e inversa:
  - `calculaLU(A)`: LU sin pivoteo (L, U, nops).
  - `res_tri(L,b,inferior=True)`: resolución triangular.
  - `inversa(A)`: por LU.
  - `calculaLDV(A)`: factoriza L D V (a partir de LU).
  - `esSDP(A)`: chequeo simétrica definida positiva (LDLᵗ).

- Cholesky y ecuaciones normales:
  - `cholesky(A, tol=1e-12)`: descomposición A = L Lᵗ para matrices SPD; retorna L (triangular inferior con diagonal positiva). Valida simetría/SPD y no usa `numpy.linalg.cholesky`.
  - `solve_cholesky(A, b, tol=1e-12)`: resuelve Ax = b usando A = L Lᵗ (forward/backward con `res_tri`). Requiere A SPD y `b` vector 1D.
  - `pinvEcuacionesNormales(X, Y, tol=1e-12)`: calcula W que minimiza ||Y − W X||_F² vía ecuaciones normales con Cholesky. Maneja casos delgados/ancho/cuadrado; entradas `X ∈ ℝ^{n×p}`, `Y ∈ ℝ^{m×p}`, salida `W ∈ ℝ^{m×n}`.

- QR:
  - `QR_con_GS(A)`: Gram–Schmidt modificado.
  - `QR_con_HH(A)`: Householder.
  - `calculaQR(A,metodo)`: orquestador (`'GS'|'RH'`).

- Métodos de potencia:
  - `metpot2k(A,tol,K)`: potencia para autovalor dominante.
  - `inversa_LU(A)`, `metpotI(M,μ,...)`: potencia con desplazamiento/inversa.

- Markov y ralas:
  - `transiciones_al_azar_continuas(n)`: T aleatoria, columnas normalizadas.
  - `transiciones_al_azar_uniformes(n,thres)`: columnas con valores uniformes no nulos.
  - `nucleo(A,tol)`: base del núcleo vía diag. de AᵗA (con refinamiento).
  - `crea_rala(listado,m,n,tol)`: dict ralo y dimensiones.
  - `multiplica_rala_vector(A_rala,v)`: producto ralo–vector.

- SVD:
  - `svd_reducida(A,k='max',tol)`: retorna `U_hat`, `Sig_hat`, `V_hat` reducidas.

## Estructura de archivos
- `alc.py`: implementación de la biblioteca.
- `tests_alc.py`, `tests_qr.py`, `tests_metpot.py`, `tests_markov.py`, `tests_svd.py`: tests por módulo.
- `.gitignore`, `README.md`.

## Notas
- Se evitó el uso de `@/np.matmul` donde el enunciado lo prohíbe.
- Algunas rutinas usan `numpy/scipy` para utilidades numéricas y resolución triangular.


