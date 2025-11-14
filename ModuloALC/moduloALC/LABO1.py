import numpy as np

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

