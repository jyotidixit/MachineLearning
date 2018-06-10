# -*- coding: utf-8 -*-
"""
Created on Sun May 27 15:06:16 2018

@author: jyoti
"""

import numpy as np
import matplotlib.pyplot as plt

N = 100
D = 2

X = np.random.randn(N, D)
X[:50, :] = X[:50, :] - 2*np.ones((50, D)) #centered at -2
X[50:, :] = X[50:, :] + 2*np.ones((50, D)) #centered at +2

T = np.array([0]*50 + [1]*50) #setting forst 50 elements of array to 0 and next 50 to 1

ones = np.array([[1]*N]).T
Xb = np.concatenate((ones, X), axis = 1)
w = np.random.randn(D + 1)


def sigmoid(a):
    return 1/(1 + np.exp(-a))

Y = sigmoid(Xb.dot(w))

def crossEntropyErrorFunction(T, Y):
    E = 0
    for i in range(N):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E

learning_rate = 0.1
for i in range(100):
    if i % 10 == 0:
        print(crossEntropyErrorFunction(T, Y))
        
    w += learning_rate*Xb.T.dot(T - Y)
    Y = sigmoid(Xb.dot(w))
    
print("Final weight, w: ", w)