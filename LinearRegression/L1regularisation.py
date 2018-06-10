# -*- coding: utf-8 -*-
"""
Created on Tue May 29 21:54:38 2018

@author: jyoti
"""
from __future__ import print_function, division
from builtins import range

import numpy as np # importing numpy with alias np
import matplotlib.pyplot as plt # importing matplotlib.pyplot with alias plt

No_of_observations = 50  
No_of_Dimensions = 50

X_input = (np.random.random((No_of_observations, No_of_Dimensions))-0.5)*10 #Generating 50x50 matrix forX with random values centered round 0.5      
w_dash =  np.array([1, 0.5, -0.5] + [0]*(No_of_Dimensions-3)) # Making first 3 features significant by setting w for them as non-zero and others zero
Y_output = X_input.dot(w_dash) + np.random.randn(No_of_observations)*0.5 #Setting Y = X.w + some random noise

costs = [] #Setting empty list for costs
w = np.random.randn(No_of_Dimensions)/np.sqrt(No_of_Dimensions) #Setting w to random values
L1_coeff = 5    
learning_rate = 0.001

for t in range(500):
    Yhat = X_input.dot(w)
    delta = Yhat - Y_output #the error between predicted output and actual output
    w = w - learning_rate*(X_input.T.dot(delta) + L1_coeff*np.sign(w)) #performing gradient descent for w
    meanSquareError = delta.dot(delta)/No_of_observations #Finding mean square error
    costs.append(meanSquareError) #Appending mse for each iteration in costs list
    
plt.plot(costs)
plt.title("Plot of costs of L1 Regularization")
plt.ylabel("Costs")
plt.show()

print("final w:", w) #The final w output. As you can see, first 3 w's are significant , the rest are very small

# plot our w vs true w
plt.plot(w_dash, label='true w')
plt.plot(w, label='w_map')
plt.legend()
plt.show()