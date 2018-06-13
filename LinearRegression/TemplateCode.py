# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 17:55:24 2018

@author: jyoti
"""
from __future__ import division, print_function
from builtins import range

import numpy as np
import matplotlib.pyplot as plt

class LinearRegression(object):
    def __init__(self):
        pass
    
    def fit(self, X, Y, eta=10, epochs=2000):
        N, D = X.shape
        self.w = np.random.randn(D)
        #self.b = 0
        
        
        for i in range(epochs):
            Yhat = self.predict(X)
            delta = Yhat - Y #the error between predicted output and actual output
            self.w = self.w - eta*(X.T.dot(delta)) #performing gradient descent for w
        
        print("Final weights are ", self.w)
        #print("Final bias point is ", self.b)
        print("Final cost is ", self.costs)
        
        
        
    def predict(self, X):
        Y_cap = X.dot(self.w)
        return Y_cap
    
    def costs(self, X, Y):
        Yhat = self.predict(X)
        cost = (Yhat-Y).dot(Yhat-Y)
        return cost
            
def main():
    X = []
    Y = []
    
    for line in open("data_2d.csv"):
        x1, x2, y = line.split(",")
        X.append([float(x1), float(x2)])
        Y.append(float(y))
    X = np.array(X)
    Y = np.array(Y)
    
    
    model = LinearRegression()
    model.fit(X, Y)
    #prediction = model.predict()
   
    
    
    
if __name__ == '__main__':
    main()
   
    
            