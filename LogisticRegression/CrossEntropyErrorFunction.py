# -*- coding: utf-8 -*-
"""
Created on Sun May 27 13:33:29 2018

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
Z = Xb.dot(w)

def sigmoid(a):
    return 1/(1 + np.exp(-a))

#def forward(X, w, b):
#    return sigmoid(X.dot(w) + b)
Y = sigmoid(Z)

def crossEntropyErrorFunction(T, Y):
    E = 0
    for i in range(N):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E

crossEntropyError = crossEntropyErrorFunction(T, Y)
print("With random/normally distributed weights: ",crossEntropyError)

w = np.array([0, 4, 4])
Z = Xb.dot(w)

Y = sigmoid(Z)

crossEntropyError = crossEntropyErrorFunction(T, Y)
print("With calculated weights/closed form solution: ",crossEntropyError)

plt.scatter(X[:, 0], X[:, 1], c = T, s = 100, alpha = 0.5)
plt.title("Two Gaussian clouds and the discriminating line")
x_axis = np.linspace(-6, 6, 100)
y_axis = -x_axis
plt.plot(x_axis, y_axis)
plt.show()