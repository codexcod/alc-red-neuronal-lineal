import os
import numpy as np
from alc import cargarDataset, pinvEcuacionesNormales

def main():
    # Ruta al split de entrenamiento
    repo_root = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(repo_root, "template-alumnos", "dataset", "cats_and_dogs", "train")

    Xt, Yt, Xv, Yv = cargarDataset(train_dir)

    if Xt is None or Yt is None:
        print("No se pudo cargar el split de entrenamiento desde:", train_dir)
        return

    print("Xt shape:", None if Xt is None else Xt.shape)
    print("Yt shape:", None if Yt is None else Yt.shape)
    # Chequeos básicos
    if Yt is not None:
        assert Yt.shape[0] == 2, "Yt debe tener 2 filas (one-hot)"
        n_cats = int(np.sum(Yt[0, :] == 1))
        n_dogs = int(np.sum(Yt[1, :] == 1))
        print("Cantidad de embeddings gatos (train):", n_cats)
        print("Cantidad de embeddings perros (train):", n_dogs)
        print("Total columnas en Xt:", Xt.shape[1])
        assert Xt.shape[1] == Yt.shape[1], "Xt y Yt deben tener igual cantidad de columnas"
        print("One-hot ok y dimensiones consistentes.")

    # Imprimir una vista simple de las primeras columnas
    print("Primeras 5 etiquetas (columnas):")
    print(Yt[:, :5])

    # Probar pinvEcuacionesNormales: entrenar W en train y evaluar residuo/accuracy en train
    print("\nEntrenando W con pinvEcuacionesNormales...")
    W = pinvEcuacionesNormales(Xt, Yt, tol=1e-12)
    print("W shape:", W.shape)
    # Residuo en train
    Y_pred = W @ Xt
    residuo = np.linalg.norm(Yt - Y_pred, ord="fro")
    print("||Yt - W Xt||_F:", residuo)
    # Accuracy en train (argmax por columna)
    y_true = np.argmax(Yt, axis=0)
    y_hat = np.argmax(Y_pred, axis=0)
    acc = np.mean(y_true == y_hat)
    print("Accuracy (train):", f"{acc*100:.2f}%")

if __name__ == "__main__":
    main()


