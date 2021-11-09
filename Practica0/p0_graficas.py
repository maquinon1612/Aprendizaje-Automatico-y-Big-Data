import sys
import numpy as np
import random
import time

import matplotlib.pyplot as plt

def main():
    args = sys.argv
    a, b = args[1], args[2]
    t = integra_mc(func, 1, 5)
    print(t)

def func (x) :
    return -(x**2)+ 5*x + 5

#version numpy
def integra_mc(func, a, b, num_puntos = 100):
    puntos = np.linspace(a,b,num_puntos)
    m = max(func(puntos))
    fxpuntos = func((np.random.randint(a, size = num_puntos)) + (b-a))
    ypuntos = np.random.randint(m, size=num_puntos) #----------------------------------cambio
    menores = ypuntos < fxpuntos
    por_debajo = sum(menores)
    return (por_debajo / num_puntos)*(b-a)*m


#-----------------------------------------funciones añadidas--------------------------------------------#
#para tomar tiempos y pintar la grafica
def main3():
    args = sys.argv
    a, b = args[1], args[2]
    t = toma_tiempos(func,2,5)
    #bucles, vectorizado, tamaños
    plt.figure()
    plt.scatter(t[2],t[0], c = 'red', label = 'bucle')
    plt.scatter(t[2],t[1], c = 'blue', label = 'vectorizado')
    plt.legend()
    plt.savefig('tiempos3.png')

def main1():
    args = sys.argv
    a, b = args[1], args[2]
    t = integra_mc(func, 1, 5)
    print(t)

def main2():
    from scipy import integrate
    x2 = lambda x: func (x)
    t = integrate.quad(x2, 1, 5)
    print(t)
    
#version iterativa
def integra_mc1(func, a, b, num_puntos = 10000):
    
    punto_f = func(a)
    por_debajo = 0
    
    for num in range(num_puntos):
        aux = a + (b-a)*num/num_puntos
        aux_f = func(aux)
        
        if aux_f > punto_f:
            punto_f = aux_f
            
    for num in range(num_puntos):
        fx = func((random.randrange(a)) + (b-a))
        y = random.randrange(int(punto_f))
        
        if y < fx:
            por_debajo = por_debajo + 1
        
    return (por_debajo / num_puntos)*(b-a)*punto_f  


def toma_tiempos(f, a, b):
    t_bucles = []
    t_vectorizado = []
    tamanios = np.linspace(1,1000000,1000, dtype = int)
    for i in tamanios:
        tv = tiempo_vectorizado(f,a,b,i)
        tb = tiempo_bucles(f,a,b,i)
        t_bucles += [tb]
        t_vectorizado += [tv]
    return (t_bucles,t_vectorizado,tamanios)
    
def tiempo_vectorizado(f,a,b,num_puntos):
    prev = time.process_time()
    res = integra_mc(f,a,b,num_puntos)
    fin = time.process_time()
    return (fin-prev)*1000

def tiempo_bucles(f,a,b,num_puntos):
    prev = time.process_time()
    res = integra_mc1(f,a,b,num_puntos)
    fin = time.process_time()
    return (fin-prev)*1000

main3()