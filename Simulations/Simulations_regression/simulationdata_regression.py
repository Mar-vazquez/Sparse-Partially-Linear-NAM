#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 10:07:58 2023

@author: marvazquezrabunal
"""

"""
SIMULATION DATA REGRESSION


Description:
    
Code to simulate the regression datasets.

"""


###----------------------------------------------------------------------------

### Call libraries

import numpy as np
import random
import copy
from sklearn.preprocessing import StandardScaler

###----------------------------------------------------------------------------
## Define some non-linear functions

def square(x):
    return x**2

def cube(x):
    return x**3

def log_abs(x):
    return np.log(abs(x) + 0.01)

def sin2(x):
    return np.sin(2*x)

def cos2(x):
    return np.cos(2*x)


###----------------------------------------------------------------------------
## Simulate the regression data

def simulation_regression(nfeat, prop, size):
    """Simulate the regression datasets.

    Parameters
    ----------
    nfeat: number of explanatory features in the data.
    prop: proportion of sparse/linear/non-linear features.
    size: sample size of the data.
        
    Returns
    -------
    The explanatory simulated features, the response and a list with the
    information about the simulated data.
    
    """
    
    # Simulate X uniformly
    x=np.random.uniform(-2, 2, size).reshape(size, 1)
    for i in range(nfeat-1):
        xi = np.random.uniform(-2, 2, size).reshape(size, 1)
        x = np.append(x, xi, axis = 1)
        
    # Scale and center X
    x = StandardScaler().fit_transform(x)
    
    # Define list of possible non-linearities and slopes
    non_lin_list=[sin2, cos2, np.exp, log_abs, square, cube]
    slope_options = [0.5, 0.6, 0.8, 1, 1.2, 1.5, 1.8, 2]
    
    # Correct possible missmatches in the number of features of each kind
    elem = [int(el*nfeat) for el in prop]
    while sum(elem) > nfeat:
        elem[np.argmax(elem)] -= 1
      
    while sum(elem) < nfeat:
        elem[np.argmin(elem)] += 1
    
    # Randomly choose some non-linearities and slopes
    nl_selected = random.choices(non_lin_list, k = int(elem[2]))
    slopes_lin = random.choices(slope_options, k = int(elem[1]))
    
    # Randomly select which variable has each behaviour and their sign
    ind = random.sample(range(nfeat), nfeat)
    new_x = copy.deepcopy(x)
    new_x[:,ind[0:int(elem[0])]] *=  0
    for i in range(int(elem[1])):
        new_x[:,ind[int(elem[0]) +i]] = round(slopes_lin[i], 2) * (new_x[:,ind[int(elem[0]) + i]])
    for i in range(int(elem[2])):
        new_x[:,ind[int(elem[0]) + int(elem[1])+i]] = nl_selected[i](new_x[:,ind[int(elem[0]) + int(elem[1]) + i]])
    sign = random.sample([1,-1]*nfeat, nfeat)
    for i in range(len(sign)):
        new_x[:,i] = new_x[:,i] * sign[i]
    
    # Obtain y and center it
    y = np.sum(new_x, axis = 1) + np.random.normal(0, 1, size)
    y = y - y.mean()
    return x, y, list([elem, ind, nl_selected, slopes_lin, sign, nfeat, prop, size])   
        


