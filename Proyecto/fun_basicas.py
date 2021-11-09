import numpy as np

def sigmoide(Z):
    # Calculo de la funcion sigmoide 
    sigmoide = 1 / (1 + np.exp(-Z))
    return sigmoide


def one_hot(y, et):
    # Transforma y en one_hot con et numero de etiquetas
    m = len(y)

    y = (y - 1)
    y_onehot = np.zeros((m, et))
    
    for i in range(m):
        y_onehot[i][int(y[i])] = 1

    return y_onehot
