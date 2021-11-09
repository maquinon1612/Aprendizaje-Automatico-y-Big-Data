import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as skl
from scipy.io import loadmat
from sklearn.metrics import accuracy_score


def evaluar_probar(Ex,Ey,Px,Py,C,Sigma):
    # Recibe un conjunto de entrenamiento y otro de prueba. Entrena una SVM con los ejemplos del conjunto de entrenamiento
    #   y los parámetros C y Sigma , y devuelve el porcentaje de aciertos en el entrenamiento y en la prueba.
    score = 0

    svm = skl.SVC(kernel = 'rbf' , C = C, gamma = 1/(2*Sigma**2))
    
    svm.fit(Ex,Ey)
    predP = svm.predict(Px)
    aciertosP = sum((predP == Py)*1) / Px.shape[0] * 100

    predE = svm.predict(Ex)
    aciertosE = sum((predE == Ey)*1) / Ex.shape[0] * 100

    return aciertosE , aciertosP

def evalua_Lambdas(Ex, Ey, Vx, Vy, Px, Py, params):
    # Recibe un conjunto de entrenamiento, otro de validación, otro de prueba y una lista de parametros.
    # Emplea el conjunto de entrenamiento y el de validación para encontrar los dos valores de la lista params que mejor porcentaje de aciertos
    #   dan, y después muestra el comportamiento de ese mejor clasificador para datos nuevos empleando el conjunto de pruebas.
    scores = np.zeros((len(params), len(params)))
    res = list()

    for v in params:
        for sigma in params:

            svm = skl.SVC(kernel = 'rbf', C = v, gamma = 1/(2*sigma**2))
            svm.fit(Ex, Ey)
            xpred = svm.predict(Vx)
            acertados = sum(xpred == Vy) / xpred.shape[0] * 100
            res.append(acertados)
            scores[params.index(v), params.index(sigma)] = acertados
        
    print(scores)
    ind = np.where(scores == scores.max())

    indp1 = ind[0][0]
    indp2 = ind[1][0]


    svm = skl.SVC(kernel = 'rbf', C = params[indp1], gamma = 1/(2*params[indp2]**2))
    svm.fit(Ex, Ey)
    print(accuracy_score(Py, svm.predict(Px)))

    print("C = {} , sigma = {}".format(params[indp1],params[indp2]))
    
    return params[indp1], params[indp2]


def evalua_Kernels(Ex, Ey, Vx, Vy, Px, Py, C, sigma):
    # Recibe un conjunto de entrenamiento, uno de validacion, otro de prueba y dos parametros C y Sigma
    # Emplea el conjunto de entrenamiento para entrenar una SVM con cada Kernel disponible, después evalua el comportamiento de los Kernels
    #   con el conjunto de validacion y obtiene el que mas porcentaje de aciertos tiene, y finalmente comprueba el comportamiento de la mejor SVM
    #   con el conjunto de pruebas.
    
    kernels= ['rbf', 'poly', 'sigmoid']
    scores = np.zeros((len(kernels)))
    res = list()

    for k in kernels:
        svm = skl.SVC(kernel = k, C = C, gamma = 1/(2*sigma**2))
        svm.fit(Ex, Ey)
        xpred = svm.predict(Vx)
        acertados = sum(xpred == Vy) / xpred.shape[0] * 100
        print(k, "acertados : ", acertados)
        res.append(acertados)
        scores[kernels.index(k)] = acertados
        
    print(scores)
    ind = np.where(scores == scores.max())

    svm = skl.SVC(kernel = kernels[ind[0][0]] , C = C , gamma = 1/(2*sigma**2))
    svm.fit(Ex, Ey)
    print(kernels[ind[0][0]], accuracy_score(Py, svm.predict(Px)))
    
    return kernels[ind[0][0]], accuracy_score(Py, svm.predict(Px))


