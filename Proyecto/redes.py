import loader as ld
import fun_basicas as fun
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from scipy.optimize import minimize


def coste(theta1, theta2, X, Y, num_etiquetas):  # Y preparada

    A1, A2, h = forward_prop(X, theta1, theta2)

    sum1 = Y * np.log(h)
    sum2 = (1 - Y) * np.log(1 - h + 1e-6)

    return (-1 / X.shape[0]) * np.sum(sum1 + sum2)


def coste_reg(theta1, theta2, X, Y, num_etiquetas, Lambda):
    c = coste(theta1, theta2, X, Y, num_etiquetas)
    m = X.shape[0]

    e = sum(sum(theta1[:, 1:] ** 2)) + sum(sum(theta2[:, 1:] ** 2))

    return c + (Lambda / (2 * m)) * e


def forward_prop(X, theta1, theta2):
    n = X.shape[0]

    # Se añade una fila de unos a la matriz inicial
    X = np.hstack([np.ones([n, 1]), X])

    # La capa oculta utiliza la primera matriz de pesos para crear sus neuronas y le añade una fila de unos
    Oculta = fun.sigmoide(np.dot(X, theta1.T))
    Oculta = np.hstack([np.ones([n, 1]), Oculta])

    # El resultado se calcula pasando por la segunda matriz de pesos todas las neuronas de la capa oculta
    Resultado = fun.sigmoide(np.dot(Oculta, theta2.T))

    return X, Oculta, Resultado


def gradiente(theta1, theta2, X, y):
    # Creamos los Delta con la forma de theta pero inicializados a cero
    Delta1 = np.zeros(np.shape(theta1))
    Delta2 = np.zeros(np.shape(theta2))

    m = len(y)

    # Se realiza la propagación hacia delante
    A1, A2, h = forward_prop(X, theta1, theta2)

    # Se realiza la propagación hacia atras para cada
    # elemento para comprobar el fallo
    for k in range(m):
        a1k = A1[k, :]
        a2k = A2[k, :]
        a3k = h[k, :]
        yk = y[k, :]

        d3 = a3k - yk
        g_prima = (a2k * (1 - a2k))
        d2 = np.dot(theta2.T, d3) * g_prima

        Delta1 = Delta1 + np.dot(d2[1:, np.newaxis], a1k[np.newaxis, :])
        Delta2 = Delta2 + np.dot(d3[:, np.newaxis], a2k[np.newaxis, :])

    # Se devuelven los Deltas que corresponden al gradiente
    return Delta1 / m, Delta2 / m


def gradiente_reg(theta1, theta2, X, y, Lambda):
    m = len(y)
    Delta1, Delta2 = gradiente(theta1, theta2, X, y)

    # A cada elemento del gradiente (menos la primera columna) se le añade el termino de regularización Lambda
    # multiplicado por cada elemento de las matriz theta 1 y theta2
    Delta1[:, 1:] = Delta1[:, 1:] + (Lambda / m) * theta1[:, 1:]
    Delta2[:, 1:] = Delta2[:, 1:] + (Lambda / m) * theta2[:, 1:]

    return Delta1, Delta2


def backprop(params_rn, num_entradas, num_ocultas, num_etiquetas, X, y, reg):
    # backprop devuelve una tupla (coste, gradiente) con el coste y el gradiente de
    # una red neuronal de tres capas , con num_entradas , num_ocultas nodos en la capa
    # oculta y num_etiquetas nodos en la capa de salida. Si m es el numero de ejemplos
    # de entrenamiento, la dimensión de ’X’ es (m, num_entradas) y la de ’y’ es
    # (m, num_etiquetas)

    theta1 = np.reshape(params_rn[:num_ocultas * (num_entradas + 1)], (num_ocultas, (num_entradas + 1)))
    theta2 = np.reshape(params_rn[num_ocultas * (num_entradas + 1):], (num_etiquetas, (num_ocultas + 1)))

    m = len(y)

    D1, D2 = gradiente_reg(theta1, theta2, X, y, reg)

    coste = coste_reg(theta1, theta2, X, y, num_etiquetas, reg)

    gradiente = np.concatenate((np.ravel(D1), np.ravel(D2)))

    return coste, gradiente


def prueba_neurona(X, y, theta1, theta2):
    """función que devuelve el porcentaje de acierto de una red neuronal utilizando unas matrices de pesos dadas"""
    n = len(y)

    y = np.ravel(y)

    _, _, result = forward_prop(X, theta1, theta2)

    result = np.argmax(result, axis=1)

    return (sum((result + 1)%4 == y) / n * 100)


def validacion_redes(random_state, num_labels, iteraciones, hiddens, lambdas, colores = ['r', 'b' , 'g', 'm']):
    
    Ex, Ey, Vx, Vy, Px, Py = ld.carga_Numpy(random_state)
    
    y_onehot = fun.one_hot(Ey, 4)

    input_size = Ex.shape[1]

    INIT_EPSILON = 0.12

    for hidden_size in hiddens:

        theta1 = np.random.random((hidden_size,(input_size + 1)))*(2*INIT_EPSILON) - INIT_EPSILON
        theta2 = np.random.random((num_labels,(hidden_size + 1)))*(2*INIT_EPSILON) - INIT_EPSILON

        params = np.concatenate((np.ravel(theta1), np.ravel(theta2)))

        plt.figure()

        i = 0

        for reg in lambdas:
            percent = []
            for iters in iteraciones:
                fmin = minimize(fun=backprop, x0=params,
                        args=(input_size, hidden_size,
                        num_labels, Ex, y_onehot, reg),
                        method='TNC', jac=True,
                        options={'maxiter': iters})

                theta1 = np.reshape(fmin.x[:hidden_size*(input_size + 1)],(hidden_size,(input_size + 1)))
                theta2 = np.reshape(fmin.x[hidden_size * (input_size+1):],(num_labels,(hidden_size + 1)))

                p = prueba_neurona(Vx, Vy, theta1, theta2)
                print(p)
                percent.append(p)
            plt.plot(iteraciones, percent, c = colores[i] , label = ' lambda = {} '.format(reg))
            i = i+1

        plt.legend()
        plt.title("hidden sizes: {}".format(hidden_size))
        plt.show()
        

        
        
        
def prueba_redes(random_state, num_labels, iteraciones, hiddens, lambdas, colores = ['r', 'b' , 'g', 'm']):
    
    Ex, Ey, Vx, Vy, Px, Py = ld.carga_Numpy(random_state)
    
    y_onehot = fun.one_hot(Ey, 4)

    input_size = Ex.shape[1]

    INIT_EPSILON = 0.12

    for hidden_size in hiddens:

        theta1 = np.random.random((hidden_size,(input_size + 1)))*(2*INIT_EPSILON) - INIT_EPSILON
        theta2 = np.random.random((num_labels,(hidden_size + 1)))*(2*INIT_EPSILON) - INIT_EPSILON

        params = np.concatenate((np.ravel(theta1), np.ravel(theta2)))

        plt.figure()

        i = 0

        for reg in lambdas:
            percent1 = []
            percent2 = []
            for iters in iteraciones:
                fmin = minimize(fun=backprop, x0=params,
                        args=(input_size, hidden_size,
                        num_labels, Ex, y_onehot, reg),
                        method='TNC', jac=True,
                        options={'maxiter': iters})

                theta1 = np.reshape(fmin.x[:hidden_size*(input_size + 1)],(hidden_size,(input_size + 1)))
                theta2 = np.reshape(fmin.x[hidden_size * (input_size+1):],(num_labels,(hidden_size + 1)))

                p1 = prueba_neurona(Vx, Vy, theta1, theta2)
                print("validación = {}".format(p1))
                p2 = prueba_neurona(Px, Py, theta1, theta2)
                print("prueba = {}".format(p2))
                percent1.append(p1)
                percent2.append(p2)

            plt.plot(iteraciones, percent1, c = colores[i] , label = 'validación')
            plt.plot(iteraciones, percent2, c = colores[i + 1] , label = 'prueba')
            i = i+1

        plt.legend()
        plt.title("hidden sizes: {}".format(hidden_size))
        plt.show()
        
        
        
        




#### Para redes de dos capas ocultas(luego hare una para un numero de capas ocultas cualquiera)



def coste2(theta1, theta2, theta3, X, Y, num_etiquetas):  # Y preparada

    A1, A2, A3, h = forward_prop2(X, theta1, theta2, theta3)

    sum1 = Y * np.log(h)
    sum2 = (1 - Y) * np.log(1 - h + 1e-6)

    return (-1 / X.shape[0]) * np.sum(sum1 + sum2)


def coste_reg2(theta1, theta2, theta3, X, Y, num_etiquetas, Lambda):
    c = coste2(theta1, theta2, theta3, X, Y, num_etiquetas)
    m = X.shape[0]

    e = sum(sum(theta1[:, 1:] ** 2)) + sum(sum(theta2[:, 1:] ** 2)) + sum(sum(theta3[:, 1:] ** 2))

    return c + (Lambda / (2 * m)) * e


def forward_prop2(X, theta1, theta2, theta3):
    n = X.shape[0]

    # Se añade una fila de unos a la matriz inicial
    X = np.hstack([np.ones([n, 1]), X])

    # Las capas ocultas utilizan la primera y segunda matrices de pesos para crear sus neuronas y les añaden una fila de unos
    Oculta1 = fun.sigmoide(np.dot(X, theta1.T))
    Oculta1 = np.hstack([np.ones([n, 1]), Oculta1])
    
    Oculta2 = fun.sigmoide(np.dot(Oculta1, theta2.T))
    Oculta2 = np.hstack([np.ones([n, 1]), Oculta2])

    # El resultado se calcula pasando por la segunda matriz de pesos todas las neuronas de la capa oculta
    Resultado = fun.sigmoide(np.dot(Oculta2, theta3.T))

    return X, Oculta1, Oculta2, Resultado




def gradiente2(theta1, theta2, theta3, X, y):
    # Creamos los Delta con la forma de theta pero inicializados a cero
    Delta1 = np.zeros(np.shape(theta1))
    Delta2 = np.zeros(np.shape(theta2))
    Delta3 = np.zeros(np.shape(theta3))

    m = len(y)

    # Se realiza la propagación hacia delante
    A1, A2, A3, h = forward_prop2(X, theta1, theta2, theta3)

    # Se realiza la propagación hacia atras para cada
    # elemento para comprobar el fallo
    for k in range(m):
        a1k = A1[k, :]
        a2k = A2[k, :]
        a3k = A3[k, :]
        a4k = h[k, :]
        yk = y[k, :]

        d4 = a4k - yk
        g_prima = (a3k * (1 - a3k))
        d3 = np.dot(theta3.T, d4) * g_prima
        
        g_prima = (a2k * (1 - a2k))
        d2 = np.dot(theta2.T, d3[1:]) * g_prima

        Delta1 = Delta1 + np.dot(d2[1:, np.newaxis], a1k[np.newaxis, :])
        Delta2 = Delta2 + np.dot(d3[1:, np.newaxis], a2k[np.newaxis, :])
        Delta3 = Delta3 + np.dot(d4[:, np.newaxis], a3k[np.newaxis, :])

    # Se devuelven los Deltas que corresponden al gradiente
    return Delta1 / m, Delta2 / m, Delta3 / m




def gradiente_reg2(theta1, theta2, theta3, X, y, Lambda):
    m = len(y)
    Delta1, Delta2, Delta3 = gradiente2(theta1, theta2, theta3, X, y)

    # A cada elemento del gradiente (menos la primera columna) se le añade el termino de regularización Lambda
    # multiplicado por cada elemento de las matriz theta 1 y theta2
    Delta1[:, 1:] = Delta1[:, 1:] + (Lambda / m) * theta1[:, 1:]
    Delta2[:, 1:] = Delta2[:, 1:] + (Lambda / m) * theta2[:, 1:]
    Delta3[:, 1:] = Delta3[:, 1:] + (Lambda / m) * theta3[:, 1:]

    return Delta1, Delta2, Delta3


def backprop2(params_rn, num_entradas, num_ocultas1, num_ocultas2, num_etiquetas, X, y, reg):
    # backprop pero con dos capas ocultas
    
    pos = (num_ocultas1 * (num_entradas + 1)) + (num_ocultas2 * (num_ocultas1 + 1))
    
    theta1 = np.reshape(params_rn[:num_ocultas1 * (num_entradas + 1)], (num_ocultas1, (num_entradas + 1)))
    theta2 = np.reshape(params_rn[num_ocultas1 * (num_entradas + 1): pos ], (num_ocultas2, (num_ocultas1 + 1)))
    theta3 = np.reshape(params_rn[pos :], (num_etiquetas, (num_ocultas2 + 1)))

    m = len(y)

    D1, D2, D3 = gradiente_reg2(theta1, theta2, theta3, X, y, reg)

    coste = coste_reg2(theta1, theta2, theta3, X, y, num_etiquetas, reg)

    gradiente = np.concatenate((np.ravel(D1), np.ravel(D2), np.ravel(D3)))

    return coste, gradiente



def prueba_neurona2(X, y, theta1, theta2, theta3):
    """función que devuelve el porcentaje de acierto de una red neuronal utilizando unas matrices de pesos dadas"""
    n = len(y)

    y = np.ravel(y)

    _, _, _, result = forward_prop2(X, theta1, theta2, theta3)

    result = np.argmax(result, axis=1)

    return (sum((result + 1)%4 == y) / n * 100)




def validacion_redes2(random_state, num_labels, iteraciones, hiddens1, hiddens2, lambdas, colores = ['r', 'b' , 'g', 'm']):
    
    Ex, Ey, Vx, Vy, Px, Py = ld.carga_Numpy(random_state)
    
    y_onehot = fun.one_hot(Ey, 4)

    input_size = Ex.shape[1]
    
    INIT_EPSILON = 0.12
    
    for hidden_size1 in hiddens1:
    
        for hidden_size2 in hiddens2:

            if hidden_size1 >= hidden_size2:

                theta1 = np.random.random((hidden_size1,(input_size + 1)))*(2*INIT_EPSILON) - INIT_EPSILON
                theta2 = np.random.random((hidden_size2,(hidden_size1 + 1)))*(2*INIT_EPSILON) - INIT_EPSILON
                theta3 = np.random.random((num_labels,(hidden_size2 + 1)))*(2*INIT_EPSILON) - INIT_EPSILON

                params = np.concatenate((np.ravel(theta1), np.ravel(theta2), np.ravel(theta3)))

                plt.figure()

                i = 0

                for reg in lambdas:

                    percent = []

                    for iters in iteraciones:

                        fmin = minimize(fun=backprop2, x0=params,
                                args=(input_size, hidden_size1, hidden_size2,
                                num_labels, Ex, y_onehot, reg),
                                method='TNC', jac=True,
                                options={'maxiter': iters})

                        pos = (hidden_size1 * (input_size + 1)) + (hidden_size2 * (hidden_size1 + 1))

                        theta1 = np.reshape(fmin.x[:hidden_size1 * (input_size + 1)], (hidden_size1, (input_size + 1)))
                        theta2 = np.reshape(fmin.x[hidden_size1 * (input_size + 1): pos ], (hidden_size2, (hidden_size1 + 1)))
                        theta3 = np.reshape(fmin.x[pos :], (num_labels, (hidden_size2 + 1)))

                        p = prueba_neurona2(Vx, Vy, theta1, theta2, theta3)
                        print(p)
                        percent.append(p)

                    plt.plot(iteraciones, percent, c = colores[i] , label = ' lambda = {} '.format(reg))
                    i = i+1
                plt.title("hidden sizes: {}, {}".format(hidden_size1, hidden_size2))
                plt.legend()
                plt.show()
                
           
        
                
                
def prueba_redes2(random_state, num_labels, iteraciones, hiddens1, hiddens2, lambdas, colores = ['r', 'b' , 'g', 'm']):
    
    Ex, Ey, Vx, Vy, Px, Py = ld.carga_Numpy(random_state)
    
    y_onehot = fun.one_hot(Ey, 4)

    input_size = Ex.shape[1]
    
    INIT_EPSILON = 0.12
    
    for hidden_size1 in hiddens1:
    
        for hidden_size2 in hiddens2:

            if hidden_size1 >= hidden_size2:

                theta1 = np.random.random((hidden_size1,(input_size + 1)))*(2*INIT_EPSILON) - INIT_EPSILON
                theta2 = np.random.random((hidden_size2,(hidden_size1 + 1)))*(2*INIT_EPSILON) - INIT_EPSILON
                theta3 = np.random.random((num_labels,(hidden_size2 + 1)))*(2*INIT_EPSILON) - INIT_EPSILON

                params = np.concatenate((np.ravel(theta1), np.ravel(theta2), np.ravel(theta3)))

                plt.figure()

                i = 0

                for reg in lambdas:

                    percent1 = []
                    percent2 = []

                    for iters in iteraciones:

                        fmin = minimize(fun=backprop2, x0=params,
                                args=(input_size, hidden_size1, hidden_size2,
                                num_labels, Ex, y_onehot, reg),
                                method='TNC', jac=True,
                                options={'maxiter': iters})

                        pos = (hidden_size1 * (input_size + 1)) + (hidden_size2 * (hidden_size1 + 1))

                        theta1 = np.reshape(fmin.x[:hidden_size1 * (input_size + 1)], (hidden_size1, (input_size + 1)))
                        theta2 = np.reshape(fmin.x[hidden_size1 * (input_size + 1): pos ], (hidden_size2, (hidden_size1 + 1)))
                        theta3 = np.reshape(fmin.x[pos :], (num_labels, (hidden_size2 + 1)))

                        p1 = prueba_neurona2(Vx, Vy, theta1, theta2, theta3)
                        print("validación = {}".format(p1))
                        p2 = prueba_neurona2(Px, Py, theta1, theta2, theta3)
                        print("prueba = {}".format(p2))
                        percent1.append(p1)
                        percent2.append(p2)

                    plt.plot(iteraciones, percent1, c = colores[i] , label = 'validación')
                    plt.plot(iteraciones, percent2, c = colores[i + 1] , label = 'prueba')
                    i = i+1
                plt.title("hidden sizes: {}, {}".format(hidden_size1, hidden_size2))
                plt.legend()
                plt.show()