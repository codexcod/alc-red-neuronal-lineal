import numpy as np
import moduloALC as alc

# 1. Lectura de Datos 

#Dentro de cats_and_dogs, tenemos train y val, dentro de cada una una carpeta para "cats" y "dogs"
#cada una de estas, contiene un archivo .npy que es la matriz que queremos

ruta = ""  #Aca va el path donde tengas guardado "cats_and_dogs"

def cargarDataset(carpeta):
    
    #Cargamos los embeedings de entrenamiento, np.load lee el archivo .npy
    #Cada archivo contiene una matriz de embeedings, donde cada columna es un embedding
    
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
#Como los archivos tienen el mismo tamanio para perro y gato, asumimos que esta bien que como hay 2000 en train, son 1000 gatos y 1000 perros
#luego en validacion quedan los otros 1000, que son 500/500 

# 2.

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

print("W =", W)

Y_approx = W @ X    # si quisiesemos recuperar Y con WX, quiero ver que tanto se parece a la Y original
print("Y_approx =\n", Y_approx)
print("Error =\n", Y_approx - Y)

