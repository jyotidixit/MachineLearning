# -*- coding: utf-8 -*-
"""
Created on Sun May 27 15:21:54 2018

@author: jyoti
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 26 19:13:44 2018

@author: jyoti
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


def get_data():
    df = pd.read_csv("ecommerce_data.csv")
    data = df.as_matrix()
    X = data[:, :-1]
    Y = data[:, -1]
    X = np.array(X)
    Y = np.array(Y)
    X[:, 1] = (X[:, 1]-X[:, 1].mean())/X[:, 1].std()
    X[:, 2] = (X[:, 2]-X[:, 2].mean())/X[:, 2].std()
    N, D = X.shape
    
    X2 = np.zeros((N, D+3))
    X2[:, 0: D-2] = X[:, 0: D-2]
            
    for n in range(N):
        t = int(X[n, D-1])
        X2[n, t+(D-1)] = 1
    
    Z = np.zeros((N, 4))
    Z[np.arange(N), X[:, D-1].astype(np.int32)] = 1
    #X2[:, -4:] = Z
    assert(np.abs(X2[:, -4:]- Z).sum() < 10e-10)
    return X2, Y

def get_binary_data():
    X, Y = get_data()
    X2 = X[Y <= 1]
    Y2 = Y[Y <= 1]
    return X2, Y2

X, Y = get_binary_data()
X, Y = shuffle(X, Y)
X_train = X[:-100]
Y_train = Y[:-100]
X_test = X[-100:]
Y_test = Y[-100:]

D = X.shape[1]
N = X.shape[0]
w = np.random.randn(D)
b = 0

def sigmoid(a):
    return 1/(1 + np.exp(-a))


def forward(x, w, b):
    return sigmoid(x.dot(w) + b)

def classification_rate(Y, P):
    return np.mean(Y == P)

def crossEntropyErrorFunction(T, Y):
    return -np.mean(T*np.log(Y) + (1 - T)*np.log(1 - Y))

train_costs = []
test_costs = []
learning_rate = 0.001

for i in range(10000):
    pY_train = forward(X_train, w, b)
    pY_test = forward(X_test, w, b)
    
    ctrain = crossEntropyErrorFunction(Y_train, pY_train)
    ctest = crossEntropyErrorFunction(Y_test, pY_test)
    train_costs.append(ctrain)
    test_costs.append(ctest)
    
    w -= learning_rate*X_train.T.dot(pY_train - Y_train)
    b -= learning_rate*(pY_train - Y_train).sum()
    if i % 1000 == 0:
        print(i, ctrain, ctest)
        
print("Final training classification rate: ", classification_rate(Y_train, np.round(pY_train)))
print("Final test classification rate: ", classification_rate(Y_test, np.round(pY_test)))

legend1, = plt.plot(train_costs, label="train cost")
legend2, = plt.plot(test_costs, label="test cost")

plt.legend([legend1, legend2])
plt.show()