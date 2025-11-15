import numpy as np
import moduloALC as alc


# 1. Lectura de Datos 

#Dentro de cats_and_dogs, tenemos train y val, dentro de cada una una carpeta para "cats" y "dogs"
#cada una de estas, contiene un archivo .npy que es la matriz que queremos

ruta = ""  #Aca va el path donde tengas guardado "cats_and_dogs"

def cargarDataset(carpeta):
    
    #Cargamos los embeddings de entrenamiento, np.load lee el archivo .npy
    #Cada archivo contiene una matriz de embeddings, donde cada columna es un embedding
    
    #---Parte de TRAIN---
    cats_train = np.load(carpeta + "/train/cats/efficientnet_b3_embeddings.npy")
    dogs_train = np.load(carpeta + "/train/dogs/efficientnet_b3_embeddings.npy")

    Xt = alc.concatenaColumnas(cats_train, dogs_train)   #Unimos los embeddings

    #Para armar Yt necesitamos poner como columnas las etiquetas, esto lo hacemos con un producto matricial:
        #los vectores etiquetas son vectores columna de 2x1: [1,0] para gato o [0,1] para perro
        #multiplicamos el vector etiqueta por otro vector (1,1, ... ,1) de R^n, donde n es el numero de columnas de la matriz "cats_train" o "dogs_train"
        #esto devuelve una matriz de 2xn, que tiene como columnas las etiquetas correspondientes al embedding
        
    nc = cats_train.shape[1]
    nd = dogs_train.shape[1]

    Y_cats = alc.productoMatricial(np.array([[1],[0]]) , np.ones((1, nc)))
    Y_dogs = alc.productoMatricial(np.array([[0],[1]]) , np.ones((1, nd)))

    Yt = alc.concatenaColumnas(Y_cats, Y_dogs)

    #---Parte de VALIDATION--- (misma idea)
    cats_val = np.load(carpeta + "/val/cats/efficientnet_b3_embeddings.npy")
    dogs_val = np.load(carpeta + "/val/dogs/efficientnet_b3_embeddings.npy")

    Xv = alc.concatenaColumnas(cats_val, dogs_val)

    nv = cats_val.shape[1]
    mv = dogs_val.shape[1]

    Yv_cats = alc.productoMatricial(np.array([[1],[0]]) , np.ones((1, nv)))
    Yv_dogs = alc.productoMatricial(np.array([[0],[1]]) , np.ones((1, mv)))

    Yv = alc.concatenaColumnas(Yv_cats, Yv_dogs)

    return Xt, Yt, Xv, Yv


Xt, Yt, Xv, Yv = cargarDataset(ruta)

#Train
print("Xt", Xt.shape)   # -> Xt (1536, 2000) 2000 imagenes representadas por vectores de dim 1536
print("Yt", Yt.shape)   # -> Yt (2, 2000)    2000 etiquetas correspondientes

#Validation
print("Xv", Xv.shape)   # -> Xv (1536, 1000)
print("Yv", Yv.shape)   # -> Yv (2, 1000)

#Por enunciado: "El set completo contiene 3.000 imagenes de entrenamiento y 2.000 imagenes de test o validacion."
#Como los archivos tienen el mismo tamaño para perro y gato, asumimos que esta bien que como hay 2000 en train, son 1000 gatos y 1000 perros
#luego en validacion quedan los otros 1000, que son 500/500 



# ---------------------------------------------------------------------------------------


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

        L = alc.calculaCholesky(alc.productoMatricial(alc.traspuesta(X), X))


        Z = resolver_sistema_matricial(L,alc.traspuesta(X))   

        #luego resolvemos Lt U = Z con sustitucion atras (Lt es triangular superior) donde U es la pseudoinversa de X

        U = resolver_sistema_matricial(alc.traspuesta(L),Z)

        #Finalmente calculamos W = Y U donde U es la pseudoinversa 
        W = alc.productoMatricial(Y, U)   # WX = Y  -> W = Y @ X+
        
        return W
    
    if n < p:   #caso n < p 
        #Queremos resolver el sistema V(XXt) = Xt usando la factorizacion de Cholesky XXt = LLt
        #nos queda el sistema V LLt = Xt. Si transponemos queda LLt Vt = Xt. Llamo Z = Lt Vt 
        #primero resolvemos L Z = Xt con sustitucion adelante (L es triangular inferior)
        #Cada columna de Z es resultado de resolver L zi = xi (xi es columna de Xt) 

        L = alc.calculaCholesky(alc.productoMatricial(X , alc.traspuesta(X))) 
        Z = resolver_sistema_matricial(L, alc.traspuesta(X))

        #luego resolvemos Lt Vt = Z con sustitucion atras (Lt es triangular superior) donde Vt es la pseudoinversa de X transpuesta
        
        Vt = resolver_sistema_matricial(alc.traspuesta(L), Z)
        U = alc.traspuesta(Vt)   #pseudoinversa de X

        #Finalmente calculamos W = Y U donde U es la pseudoinversa
        W = alc.productoMatricial(Y, U)    

        return W
    
    if n == p:   #caso n = p:
        X_inv = alc.inversa(X)
        W = alc.productoMatricial(Y, X_inv)
        return W
    


def resolver_sistema_matricial(L, B):           #Crea una matriz Z solucion del sistema columna a columna, resolviendo L zi = bi en cada paso (L debe ser triangular)
    Z = np.zeros((L.shape[0], B.shape[1]))
    for i in range(B.shape[1]):
        Z[:, i] = alc.res_tri(L, B[:, i])
    return Z


#--------------------------------------------------------------------------------


# 3. Descomposicion en Valores singulares  

#Resumen: 
    #La SVD nos deja descomponer cualquier matriz X como X = USVt , donde U y V son ortogonales, osea su inversa es la traspuesta
    #por lo tanto son inversibles, la idea aca es calcular la pseudoinversa de la matriz X, que viene a ser X+ = V S+ Ut
    #donde Vt^(-1) = V , U^(-1) = Ut y S+ es la matriz Sigma pero en su diagonal el inverso multiplicativo de los valores singulares

#Para este ejercicio seguimos el algoritmo 2:
#Requiere: Dada X ∈ Rn×p con n < p, rango(X) = n (rango completo) y Y ∈ Rm×p
#Asegura: Solucion del problema mınW ∥Y − W X∥2

def pinvSVD(U, S, V, Y):
    
    #V = alc.traspuesta(V)  #hay que trasponer la V
    
    #Construimos  S+
    n = S.shape[0]
    S_inv = np.zeros((n,n))
    for i in range(n):
        S_inv[i,i] = 1.0 / S[i]

    #Calculamos la pseudoinversa X+
    X_pinv = alc.productoMatricial(alc.productoMatricial(V , S_inv), alc.traspuesta(U))    #X+ = V S+ Ut

    #Calculamos el W
    W = alc.productoMatricial(Y, X_pinv)    # WX = Y  -> W = Y @ X+
    return W

#Esto es lo que habria que correr para el W del tp pero tarda mucho:
    #U, S, V = alc.svd_reducida(Xt)
    #res = pinvSVD(U,S,V,Yt)
    #print(res)



#TEST DE pinvSVD
X = np.array([[1, 2, 0, 1, 3],
              [0, 1, 1, 2, 1],
              [1, 0, 2, 1, 0]])

# Matriz Y (m=2, p=5)
Y = np.array([[1, 0, 1, 0, 2],
              [0, 1, 0, 1, 1]])

# SVD reducida
U, S, Vt = alc.svd_reducida(X)

W = pinvSVD(U, S, Vt, Y)

# print("W =", W)

Y_approx = W @ X    # si quisiesemos recuperar Y con WX, quiero ver que tanto se parece a la Y original
# print("Y_approx =\n", Y_approx)
# print("Error =\n", Y_approx - Y)



#--------------------------------------------------------------------------------


# 4. POR HACER


#--------------------------------------------------------------------------------


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
    X_aprox = alc.productoMatricial(        #calculamos el producto que corresponde
        alc.productoMatricial(X,pX) , X
    )

    if not alc.SonMatricesIguales(X_aprox,X,tol):   #Esta funcion compara si las matices son "iguales" con determinada tolerancia
        return False

    #Propiedad II)  misma idea que con la propiedad I
    pX_aprox = alc.productoMatricial(
        alc.productoMatricial(pX,X) , pX
    )
    
    if not alc.SonMatricesIguales(pX_aprox,pX,tol):
        return False
    
    #Propiedad III) y IV), aca basta con chequear si las pseudo-Identidades de nxn y mxm, son simetricas, siempre con la tolerancia pedida
    pseudo_I = alc.productoMatricial(X,pX)
    
    if not alc.esSimetrica(pseudo_I,tol):
        return False
    
    pseudo_I_2 = alc.productoMatricial(pX,X)
    
    if not alc.esSimetrica(pseudo_I_2,tol):
        return False
    
    return True


