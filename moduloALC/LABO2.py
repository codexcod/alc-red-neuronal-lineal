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