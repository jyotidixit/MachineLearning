
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 13:01:51 2018

@author: jyoti
"""

import numpy as np
import matplotlib.pyplot as plt

from util import getData

labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def main():
    X, Y = getData(balance_ones = False)
    
    while(True):
        for i in range(7):
            x, y = X[Y == i], Y[Y == i]
            N = len(y)
            j = np.random.choice(N)
            plt.imshow(x[j].reshape(48, 48), cmap = 'gray')
            plt.title(labels[y[j]])
            plt.show()
        prompt = input("Quit the program? Y/N\n")
        if prompt == 'Y':
            break
        
if __name__ == '__main__':
    main()
        

    