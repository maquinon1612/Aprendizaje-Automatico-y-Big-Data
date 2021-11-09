#!/usr/bin/python

import sys
import numpy as np
import random

def main():
    args = sys.argv
    a, b = args[1], args[2]
    t = integra_mc(func, float(a), float(b))
    print(t)



def func (x) :
    return -(x**2)+ 5*x + 5


def integra_mc(func, a, b, num_puntos = 10000000):
    puntos = np.linspace(a,b,num_puntos)
    m = max(func(puntos))
    fxpuntos = func((np.random.rand(num_puntos) % a) + (b-a))
    ypuntos = np.random.rand(num_puntos) % m
    menores = ypuntos < fxpuntos
    por_debajo = sum(menores)
    return (por_debajo / num_puntos)*(b-a)*m

main()