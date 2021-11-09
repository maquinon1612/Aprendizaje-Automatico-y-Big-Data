import loader as ld
import pandas as pd
import fun_basicas as fun
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

def prepara_datos(X, y, et):
    # Transforma el parametro Y en un array que en la posicion i-esima contiene Y2[i] = 0 si Y[i] != et
    #   y Y2[i] = 1 si Y[i] == et
    Y2 = (y == et) * 1
    ## Aquí hay que hacer ravel de Y2 para pasar de (5000,1) a (5000,1)
    ## y conseguir que funcione como en la practica anterior
    Y2 = np.ravel(Y2)
    return (X, Y2)

def coste(Theta, X, Y):
    # Calculo del coste en regresion logistica
    G = fun.sigmoide(np.dot(X, Theta))
    sum1 = np.dot(Y, np.log(G))
    sum2 = np.dot((1-Y), np.log(1 - G + 1e-6))
    return (-1 / X.shape[0]) * (sum1 + sum2)

def gradiente(Theta, X, Y):
    # Calculo del gradiente de regresion logistica
    m = X.shape[0]
    G = fun.sigmoide( np.matmul(X,Theta) )
    gradiente  = (1 / len(Y)) * np.matmul(X.T, G - Y)
    return gradiente


def coste_reg(Theta, X, Y, Lambda):
    # Coste regularizado en regresion logistica
    # Utiliza la funcion coste anterior
    c = coste(Theta, X, Y)
    m = X.shape[0]
    e = 0

    for t in range(1, len(Theta)):
        e += Theta[t] ** 2

    return c + (Lambda / (2 * m)) * e

def gradiente_reg(Theta,X,Y,Lambda):
    # Gradiente regularizado de la regresion logistica
    # Emplea la funcion gradiente anterior
    m = X.shape[0]
    gr = gradiente(Theta,X,Y)
    theta2 = (Lambda/m)*Theta
    return (gr + theta2)


def optimiza_reg(X, Y, Lambda, et):
    # Recibe los ejemplos de entrenamiento X , las etiquetas asociadas a esos ejemplos en Y
    #   , un parametro de regularizacion Lambda y el numero de etiquetas diferentes en et.
    # Calcula los parametros Theta optimos entrenando el modelo de regresion logistica con los datos de X
    #   y el parametro de regularizacion Lambda
    X, Y = prepara_datos(X, Y, et)
    c, gr = preparaFunciones(Lambda)

    T = np.zeros(X.shape[1])

    result = opt.fmin_tnc(func=c, x0=T, fprime=gr, args=(X, Y))
    c_f = coste(result[0], X, Y)
    print("coste:", c_f)
    return result[0]


def oneVsAll(X, y, num_etiquetas, reg):
    # Genera 'num_etiquetas' clasificadores. Cada uno distingue a los elementos de una clase del resto
    params = []

    # Por cada tipo de etiqueta se devuelve la Theta optima que reconoce la misma y se añade a un array de Thetas
    for et in range(0, num_etiquetas):
        p = optimiza_reg(X, y, reg, et)
        params.append(p)
    return np.array(params)


def evalua():
    X, Y, _ = ld.cargarDatos()
    X = X.to_numpy()
    Y = Y.to_numpy()

    L = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

    for Lamda in L:
        Theta = oneVsAll(X, Y, 4, Lamda)
        asig = []
        for i in range(X.shape[0]):
            l = np.dot(Theta, X[i])
            m = max(l)
            i = np.where(l == m)
            asig.append(i[0][0])

        y2 = np.ravel(Y)
        t = (asig == y2) * 1
        perc = (sum(t) / X.shape[0]) * 100
        print("Porcentaje de aciertos con lambda = {} : ".format(Lamda), perc, "%")


def evaluar_validacion(L, EX, EY, VX, VY):
    # Recibe en L un parametro de regularizacion
    #           EX los ejemplos de entrenamiento
    #           EY las etiquetas de los ejemplos de entrenamiento
    #           VX los ejemplos de validacion
    #           VY las etiquetas de los ejemplos de validacion
    # Entrena los clasificadores de los ejemplos de entrenamiento y después
    #    comprueba la precisión del conjunto de clasificadores usando el conjunto
    #    de validacion
    Theta = oneVsAll(EX, EY, 4, L)
    asig = []
    for i in range(VX.shape[0]):
        l = np.dot(Theta, VX[i])
        m = max(l)
        i = np.where(l == m)
        asig.append(i[0][0])
    
    y2 = np.ravel(VY)
    t = (asig == y2) * 1
    perc = (sum(t) / VX.shape[0]) * 100

    return perc, asig


def preparaFunciones(Lambda):
    # Genera funciones que calculan el coste y el gradiente regularizados de la regresion
    # logistica dado un parametro de regularizacion
    c = lambda Theta, X, Y: coste_reg(Theta, X, Y, Lambda)
    gr = lambda Theta, X, Y: gradiente_reg(Theta, X, Y, Lambda)

    return (c, gr)
