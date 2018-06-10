<<<<<<< HEAD
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
        
=======
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
        
>>>>>>> 4e75187c8d52f84c0923213e3de4fdc2f6d8b7ec
    