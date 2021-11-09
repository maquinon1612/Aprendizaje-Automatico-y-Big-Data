#!/usr/bin/python

import sys
import numpy as np
import random
import matplotlib.pyplot as plt

def main():
    args = sys.argv
    a, b = args[1], args[2]
    t = integra_mc(func, 0, 5.5)
    print(t)

def func (x) :
    return -(x**2)+ 5*x + 5

#version numpy
def integra_mc(func, a, b, num_puntos = 100):
    puntos = np.linspace(a,b,num_puntos)
    m = max(func(puntos))
    fxpuntos = func(puntos)
    ypuntos = np.random.uniform(0, m, [num_puntos])
    menores = ypuntos < fxpuntos
    por_debajo = sum(menores)
    plt.plot(puntos, func(puntos), label="function")
    plt.plot(puntos,ypuntos,ls = 'None', marker = 'x', c = 'r', label="rand generated numbers")
    plt.legend(loc='upper left')
    ax = plt.gca() # gca stands for 'get current axis'
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))
    plt.show()
    return (por_debajo / num_puntos)*(b-a)*m


#-----------------------------------------funciones añadidas--------------------------------------------#
    
def main1():
    args = sys.argv
    a, b = args[1], args[2]
    t = integra_mc1(func, 0, 5.5)
    print(t)

def main2():
    from scipy import integrate
    x2 = lambda x: func (x)
    t = integrate.quad(x2, 0, 5.5)
    print(t)

#version iterativa
def integra_mc1(func, a, b, num_puntos = 1000000):
    
    punto_f = func(a)
    por_debajo = 0
    
    for num in range(num_puntos):
        aux = a + (b-a)*num/num_puntos
        aux_f = func(aux)
        
        if aux_f > punto_f:
            punto_f = aux_f
            
    for num in range(num_puntos):
        fx = func(random.randrange(a,int(b)))
        y = random.randrange(int(punto_f))
        
        if y < fx:
            por_debajo = por_debajo + 1
        
    return (por_debajo / num_puntos)*(b-a)*punto_f  

main() # 379 µs ± 12.9 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

main1() # 2 s ± 4.79 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

main2() # 10.7 µs ± 22.7 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
