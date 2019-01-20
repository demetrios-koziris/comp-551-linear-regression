# -*- coding: utf-8 -*-
"""
Linear regression example using least squares from COMP551 lecture 2
@author: Demetrios Koziris
"""

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


"""
Least-Squares solution
"""
def least_squares(order=1, save_plots=False):
    #print('Least-Squares:\n')
    
    x = [0.86, 0.09, -0.85, 0.87, -0.44, -0.43, -1.1, 0.40, -0.96, 0.17]
    y = [2.49, 0.83, -0.25, 3.10, 0.87, 0.02, -0.12, 1.81, -0.83, 0.43]
    
    # input matrix of feature vectors
    X = np.array([np.power(x,p) for p in range(order+1)]).T
    # target vector
    Y = np.array(y).T

    Xt_X = X.T @ X
    # print(f'X^tX = \n{Xt_X}\n')
    Xt_Y = X.T @ Y
    # print(f'X^tY = \n{Xt_Y}\n')
    w =  inv(Xt_X) @ Xt_Y
    # print(f'W = (X^tX)^-1(X^tY) = \n{w}\n')
    
    def fit_function(x):
        return sum([weight*(x**p) for p,weight in enumerate(w)])
    
    def fit_function_label():
        terms = [f'{weight:.2f}x^{p}' for p,weight in enumerate(w)]
        return '$y = ' + ' + '.join(terms[::-1]) + '$'      
    
    plot_title = f'Least-Squares Linear Regression Order {order}'
    x_interval = np.linspace(-1.6, 1.6, 1000)
    plt.plot(x_interval, fit_function(x_interval), 'r', label=fit_function_label())
    plt.scatter(x, y, marker='x')
    plt.xlim(-1.6, 1.6);
    plt.ylim(-2, 5);
    plt.title(plot_title)
    plt.legend(loc=2)
    if (save_plots):
        plt.savefig(plot_title.lower().replace(' ','_'), bbox_inches="tight")
    plt.show()


for i in range(10):
    least_squares(i, True)
