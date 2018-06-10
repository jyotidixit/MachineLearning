# -*- coding: utf-8 -*-
"""
Created on Tue May 29 22:07:08 2018

@author: jyoti
"""

import numpy as np              #importing the numpy package with alias np
import matplotlib.pyplot as plt #importing the matplotlib.pyplot as plt

N = 50                         
D = 50

X = (np.random.random((N, D))-0.5)*10       
w_dash =  np.array([1, 0.5, -0.5] + [0]*(D-3))
Y = X.dot(w_dash) + np.random.randn(N)*0.5

Y[-1]+=30   #setting last element of Y as Y + 30
Y[-2]+=30   #setting second last element of Y as Y + 30

plt.scatter(X, Y)
plt.title('Relationship between Y and X[:, 1]')
plt.xlabel('X[:, 1]')
plt.ylabel('Y')
plt.show()

X = np.vstack([np.ones(N), X]).T       #appending bias data points colummn to X

w_ml = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))     #finding weights for maximum likelihood estimation
Y_ml = np.dot(X, w_ml)    

plt.scatter(X[:,1], Y)
plt.plot(X[:,1],Y_ml, color='red')
plt.title('Graph of maximum likelihood method(Red line: predictions)')
plt.xlabel('X[:, 1]')
plt.ylabel('Y')
plt.show()

costs = []
w = np.random.randn(D)/np.sqrt(D)
L1_coeff = 5    
learning_rate = 0.001
for t in range(500):
    Yhat = X.dot(w)
    delta = Yhat - Y
    w = w - learning_rate*(X.T.dot(delta) + L1_coeff*np.sign(w))
    meanSquareError = delta.dot(delta)/N
    costs.append(meanSquareError)
  
w_map = w
Y_map = X.dot(w_map)
        
plt.scatter(X[:,1], Y)
plt.plot(X[:,1],Y_ml, color='red',label="maximum likelihood")
plt.plot(X[:,1],Y_map, color='green', label="map")
plt.title('Graph of MAP v/s ML method')
plt.legend()
plt.xlabel('X[:, 1]')
plt.ylabel('Y')
plt.show()