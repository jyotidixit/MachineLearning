# -*- coding: utf-8 -*-
"""
Created on Mon May 28 16:22:16 2018

@author: j.dixit
"""

import numpy as np
import matplotlib.pyplot as plt

N = 4
D = 2

X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
        ])

T = np.array([0, 1, 1, 0])

ones = np.array([[1]*N]).T
#plt.scatter(X[:, 0], X[:, 1], c=T)
#plt.show()

xy = np.matrix(X[:, 0]*X[:, 1]).T
Xb = np.array(np.concatenate((ones, xy, X), axis = 1))

w = np.random.rand(D + 2)

z = Xb.dot(w)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

Y = sigmoid(z)

def cross_entropy(T, Y):
    E = 0
    for i in range(N):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1-np.log(Y[i]))
    return E

learning_rate = 0.0001
error = []

for i in range(5000):
    e = cross_entropy(T, Y)
    error.append(e)
    if i % 100 == 0:
        print(e)
        
    w += learning_rate*(np.dot((T-Y).T, Xb) - 0.01*w)
    
    Y = sigmoid(Xb.dot(w))
    
plt.plot(error)
plt.title("Cross-entropy")
print("Final w: ", w)
print("Final classification rate", 1-np.abs(T-np.round(Y)).sum()/N)
