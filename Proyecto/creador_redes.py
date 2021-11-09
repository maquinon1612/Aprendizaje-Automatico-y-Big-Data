import numpy as np
import fun_basicas as fun
import loader as ld
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import confusion_matrix

# Esta clase representa una red neuronal que se puede parametrizar
#    Nodos_capa : una lista de el numero de nodos de cada capa oculta de la red
#    Entradas   : numero de entradas de la red
#    Salidas    : numero de salidas de la red
class red_neuronal:
    
    nodos_capa = []
    entradas = 0
    salidas = 0

    
    def __init__ (self, nodos_capa, entradas, salidas):
        self.entradas = entradas
        self.salidas = salidas
        self.nodos_capa = nodos_capa
        
        
        

    def forward_prop(self, entrada,matrices_pesos):
        """ Algoritmo de propagacion hacia delante, devuelve la lista de activaciones de cada capa
            El ultimo elemento de la lista devuelta es el resultado de la red para las entradas proporcionadas """
        activaciones = []

        X = entrada
        n = X.shape[0]
        X = np.hstack([np.ones([n, 1]), X])
        activaciones.append(X)

        for i in range(len(matrices_pesos) - 1):

            A = fun.sigmoide(np.dot(X, matrices_pesos[i].T ))
            A = np.hstack([np.ones([n,1]), A])

            X = A

            activaciones.append(A)
            
        A = fun.sigmoide(np.dot(X, matrices_pesos[i + 1].T))
        activaciones.append(A)

        return activaciones

    
    
    
    def coste_reg(self,X,y,reg,matrices_pesos):
        """ Calculo del coste con regularizacion para redes neuronales utilizando las matrices de pesos que se le pasan """
        activaciones = self.forward_prop(X,matrices_pesos)
        h = activaciones[-1]
        
        s1 = y * np.log(h)
        s2 = (1 - y) * np.log( 1 - h + 1e-6)

        c = (-1 / X.shape[0]) * np.sum(s1 + s2)
        e = sum([sum(sum(matrices_pesos[i][:,1:] ** 2)) for i in range(len(matrices_pesos))])

        return c + (reg / (2*X.shape[0])) * e

    
    
    
    def gradiente_reg(self, X, y, reg,matrices_pesos):
        """ Calculo de gradiente regularizado para redes neuronales haciendo uso de las matrices de pesos dadas """ 
        Deltas = [np.zeros(np.shape(matrices_pesos[i])) for i in range(len(matrices_pesos))]
        activaciones = self.forward_prop(X,matrices_pesos)

        # Bucle para el calculo de de matrices Delta para el calculo de gradiente
        
        for k in range(len(y)):
            activ = [activaciones[i][k, :] for i in range(len(activaciones))]

            ultimo = y[k, :]
            d = []
            
            j = (len(activaciones) - 1)
            
            # las dos ultimas deltas tienen un caculo diferente al resto y por tanto se calculan fuera del bucle
            
            daux = activ[j] - ultimo
            g_aux = activ[j-1] * ( 1 - activ[j-1] )
            Deltas[j - 1] = Deltas[j - 1] + np.dot(daux[:,np.newaxis], activ[j-1][np.newaxis, :])
            daux = np.dot(matrices_pesos[j-1].T, daux) * g_aux
            Deltas[j - 2] = Deltas[j - 2] + np.dot(daux[1:,np.newaxis], activ[j-2][np.newaxis, :])

            # bucle que calcula los valores de las matrices delta que no son las dos ultimas
            
            for j in range(len(activ)-3, 1, -1):
                g_aux = activ[j] * ( 1 - activ[j])
                daux = np.dot(matrices_pesos[j].T, daux[1:]) * g_aux
                Deltas[j - 1] = Deltas[j - 1] + np.dot(daux[1:,np.newaxis], activ[j-1][np.newaxis, :])
        
        # Parte de regularización de las matrices Delta
        for i in range(len(Deltas)):
            Deltas[i] = Deltas[i] / len(y)
            Deltas[i][:, 1:] = Deltas[i][:, 1:] + (reg/len(y)) * matrices_pesos[i][:, 1:]

        return Deltas
    
    
    
    def init_aleatorio(self, init):
    
        """ Función para inicializar matrices de pesos aleatoriamente """
        
        matrices_pesos = []
        
        M = np.random.random((self.nodos_capa[0], self.entradas + 1))  * (2 * init) - init
        matrices_pesos.append(M)
        
        i = -1
        
        for i in range(len(self.nodos_capa) - 1):
            M = np.random.random((self.nodos_capa[i+1], (1 + self.nodos_capa[i])))  * (2 * init) - init
            matrices_pesos.append(M)
            
        M = np.random.random((self.salidas, (1 + self.nodos_capa[i + 1])))  * (2 * init) - init
        matrices_pesos.append(M)
        
        pesos_ravel = np.concatenate(tuple(map(np.ravel, matrices_pesos)))
        
        return pesos_ravel
    
    
    
    def desenlazado(self, params_rn):
        """ Función que crea una lista con las matrices formadas con sus correctas dimensiones """ 
        matrices_pesos = []
        
        # matriz desde la entrada hasta la primera capa oculta
        matrices_pesos.append(np.reshape(params_rn[:self.nodos_capa[0] * (self.entradas + 1)],
                                        (self.nodos_capa[0], (self.entradas + 1))))
        
        
        # las matrices entre capas ocultas
        ini = self.nodos_capa[0] * (self.entradas + 1)
        fin = ini
        
        for i in range(len(self.nodos_capa) - 1):
            fin = fin + (self.nodos_capa[i + 1] * (self.nodos_capa[i] + 1))
            matrices_pesos.append(np.reshape(params_rn[ini : fin], (self.nodos_capa[i + 1], (self.nodos_capa[i] + 1))))
            ini = fin
            
            
        # la matriz desde la ultima oculta hasta la salida    
        matrices_pesos.append(np.reshape(params_rn[ini :], (self.salidas, (self.nodos_capa[len(self.nodos_capa) - 1] + 1))))
        
        return matrices_pesos
    
    
    

    def backprop(self, params_rn, X, y, reg):
        """ Devuelve una funcion que calcula el coste y el gradiente de la red neuronal 
            Se usa para el optimizar los parametros de la red en la funcion minimize """ 
        matrices_pesos = self.desenlazado(params_rn)
        
        deltas = self.gradiente_reg(X,y,reg, matrices_pesos)
        coste = self.coste_reg(X,y,reg, matrices_pesos)

        gr = tuple(map(np.ravel, deltas))

        gradiente = np.concatenate(gr)

        return coste, gradiente

    
    
    
    def prueba_neurona(self, X, y, matrices_pesos):
        """ Función que devuelve el porcentaje de acierto de una red neuronal utilizando unas matrices de pesos dadas """
        n = len(y)

        y = np.ravel(y)

        forward = self.forward_prop(X, matrices_pesos)
        
        result = forward[len(forward)-1]

        result = np.argmax(result, axis=1)
        
        # Escrive en pantalla el número de elementos que relaciono con cada clase en orden
        
        for i in range(self.salidas):
            print(np.where(result == i)[0].shape)

        return (sum((result + 1)%4 == y) / n * 100)
    
    
    
    def confusion(self, X, y, matrices_pesos):
        """ Función que devuelve el porcentaje de acierto de una red neuronal utilizando unas matrices de pesos dadas 
            como una matriz de confusión """
        n = len(y)

        y = np.ravel(y)

        forward = self.forward_prop(X, matrices_pesos)
        
        result = forward[len(forward)-1]

        result = np.argmax(result, axis=1)
        
        df_cm = pd.DataFrame(confusion_matrix(y, (result+1)%4), index = [i for i in "0123"],
                  columns = [i for i in "0123"])
        
        plt.figure(figsize = (7,5))
        
        return sn.heatmap(df_cm, annot=True)
    
    
    
    
    def entrenar(self, X, y, Vx, Vy, Px, Py, reg, iters, init):
        
        """ Función que para entrenar una la red neuronal creada con unos datos de entrenamiento 'X' e 'y' una serie de iteraciones
            'iters', con un parametro de regularización 'reg' y con un parametro de inicialización aleatoria de matrices 'init'. 
            Devuelve el porcentaje de aciertos en el conjunto de validación y de prueba """
        
        pesos_ravel = self.init_aleatorio(init)
        
        # calculo del minimo

        fmin = minimize(fun = self.backprop , x0 = pesos_ravel , args = (X,y,reg),
                        method = 'TNC' , jac = True , options = {'maxiter' : iters})
        
        matrices_pesos = self.desenlazado(fmin.x)

        p1 = self.prueba_neurona(Vx, Vy, matrices_pesos)
        print("validación = {}".format(p1))
        p2 = self.prueba_neurona(Px, Py, matrices_pesos)
        print("prueba = {}".format(p2))

        return p1, p2
    
    
    def matriz_confusion(self, X, y, Px, Py,reg, iters, init):
        """ Entrena la red de neuronal y devuelve la matriz de confusión generada con los datos de prueba """
        
        pesos_ravel = self.init_aleatorio(init)
        
        # calculo del minimo

        fmin = minimize(fun = self.backprop , x0 = pesos_ravel , args = (X,y,reg),
                        method = 'TNC' , jac = True , options = {'maxiter' : iters})
        
        matrices_pesos = self.desenlazado(fmin.x)

        return self.confusion(Px, Py, matrices_pesos)




def pruebas(redes, Lambdas, salidas, random_state, INIT_EPSILON, iteraciones):
        """Función para entrenar, validar y probar una serie de formatos de red 'redes' con unos valores
           de regularización 'Lambda' y un número de 'iteraciones' """

        # cargamos los datos de los dataset
        Ex, Ey, Vx, Vy, Px, Py = ld.carga_Numpy(random_state)
        y_onehot = fun.one_hot(Ey, salidas)

        # Los normalizamos
        Ex2, scaler = ld.normalizar(Ex)
        Vx2 = scaler.transform(Vx)
        Px2 = scaler.transform(Px)


        # Por cada formato de red se crea la red deseada
        for red in redes:

            r = red_neuronal(red, Ex2.shape[1], salidas)

            # Por cada valor de LAmbda y cada valore de iteración se ejecuta la red con ese Lambda
            # esas iteraciones y se guarda el porcentaje en el conjunto de validación y prueba
            for L in Lambdas:

                validaciones = []
                pruebas = []

                for i in range(len(iteraciones)):
                    p1, p2 = r.entrenar(Ex2, y_onehot, Vx2, Vy, Px2, Py, L, iteraciones[i], INIT_EPSILON)
                    validaciones.append(p1)
                    pruebas.append(p2)

                # Se dibujan las graficas de aciertos por cada red y cada valor de Lambda

                plt.figure()
                plt.title("red: {}, Lambda = {}".format(red, L))
                plt.plot(np.array(iteraciones), np.array(validaciones), label = 'validaciones' , Color = 'red')
                plt.plot(np.array(iteraciones), np.array(pruebas), label = 'pruebas' , Color = 'blue')
                plt.legend()
                plt.show()