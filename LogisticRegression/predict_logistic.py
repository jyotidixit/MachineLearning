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
D = X.shape[1]
W = np.random.randn(D)
b = 0

def sigmoid(a):
    return 1/(1 + np.exp(-a))

def forward(x, w, b):
    return sigmoid(x.dot(w) + b)

P_Y_Given_X = forward(X, W, b)
predictions = np.round(P_Y_Given_X)

def classification_rate(Y, P):
    return np.mean(Y == P)

print("Score: ", classification_rate(Y, predictions))

