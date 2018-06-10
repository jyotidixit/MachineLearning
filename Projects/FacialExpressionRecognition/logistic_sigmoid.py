<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 14:57:05 2018

@author: jyoti
"""

from __future__ import print_function, division
from builtins import range


import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from util import getBinaryData, sigmoid, sigmoid_cost, error_rate


class LogisticModel(object):
    def __init__(self):
        pass

    def fit(self, X, Y, learning_rate=1e-6, reg=0., epochs=120000, show_fig=False):
        X, Y = shuffle(X, Y)
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        X, Y = X[:-1000], Y[:-1000]

        N, D = X.shape
        self.W = np.random.randn(D) / np.sqrt(D)
        self.b = 0

        costs = []
        best_validation_error = 1
        for i in range(epochs):
                
                pY = self.forward(X)

                
                self.W -= learning_rate*(X.T.dot(pY - Y) + reg*self.W)
                self.b -= learning_rate*((pY - Y).sum() + reg*self.b)

                
                if i % 20 == 0:
                    pYvalid = self.forward(Xvalid)
                    c = sigmoid_cost(Yvalid, pYvalid)
                    costs.append(c)
                    e = error_rate(Yvalid, np.round(pYvalid))
                    print("i:", i, "cost:", c, "error:", e)
                    if e < best_validation_error:
                        best_validation_error = e
        print("best_validation_error:", best_validation_error)
        print("final weight matrix is ", self.W)
        print("final bias node is: ", self.b)

        if show_fig:
            plt.plot(costs)
            plt.show()


    def forward(self, X):
        return sigmoid(X.dot(self.W) + self.b)

    def predict(self, X):
        pY = self.forward(X)
        return np.round(pY)


    def score(self, X, Y):
        prediction = self.predict(X)
        return 1 - error_rate(Y, prediction)


def main():
    X, Y = getBinaryData()

    X0 = X[Y==0, :]
    X1 = X[Y==1, :]
    X1 = np.repeat(X1, 9, axis=0)
    X = np.vstack([X0, X1])
    Y = np.array([0]*len(X0) + [1]*len(X1))
    
    model = LogisticModel()
    model.fit(X, Y, show_fig=True)
    print("Model score is ", model.score(X, Y))
    

if __name__ == '__main__':
    main()
            
            
=======
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 14:57:05 2018

@author: jyoti
"""

from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from util import getBinaryData, sigmoid, sigmoid_cost, error_rate


class LogisticModel(object):
    def __init__(self):
        pass

    def fit(self, X, Y, learning_rate=1e-6, reg=0., epochs=120000, show_fig=False):
        X, Y = shuffle(X, Y)
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        X, Y = X[:-1000], Y[:-1000]

        N, D = X.shape
        self.W = np.random.randn(D) / np.sqrt(D)
        self.b = 0

        costs = []
        best_validation_error = 1
        for i in range(epochs):
                # forward propagation and cost calculation
                pY = self.forward(X)

                # gradient descent step
                self.W -= learning_rate*(X.T.dot(pY - Y) + reg*self.W)
                self.b -= learning_rate*((pY - Y).sum() + reg*self.b)

                
                if i % 20 == 0:
                    pYvalid = self.forward(Xvalid)
                    c = sigmoid_cost(Yvalid, pYvalid)
                    costs.append(c)
                    e = error_rate(Yvalid, np.round(pYvalid))
                    print("i:", i, "cost:", c, "error:", e)
                    if e < best_validation_error:
                        best_validation_error = e
        print("best_validation_error:", best_validation_error)
        print("final weight matrix is ", self.W)

        if show_fig:
            plt.plot(costs)
            plt.show()


    def forward(self, X):
        return sigmoid(X.dot(self.W) + self.b)

    def predict(self, X):
        pY = self.forward(X)
        return np.round(pY)


    def score(self, X, Y):
        prediction = self.predict(X)
        return 1 - error_rate(Y, prediction)


def main():
    X, Y = getBinaryData()

    X0 = X[Y==0, :]
    X1 = X[Y==1, :]
    X1 = np.repeat(X1, 9, axis=0)
    X = np.vstack([X0, X1])
    Y = np.array([0]*len(X0) + [1]*len(X1))
    
    model = LogisticModel()
    model.fit(X, Y, show_fig=True)
    print("Model score is ", model.score(X, Y))
    # scores = cross_val_score(model, X, Y, cv=5)
    # print "score mean:", np.mean(scores), "stdev:", np.std(scores)

if __name__ == '__main__':
    main()
            
            
>>>>>>> 4e75187c8d52f84c0923213e3de4fdc2f6d8b7ec
