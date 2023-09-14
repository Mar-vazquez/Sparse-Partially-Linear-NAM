#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 10:18:48 2023

@author: marvazquezrabunal
"""


"""
SIMULATION DATA SURVIVAL


Description:
    
Code to simulate the survival datasets.

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

def absolute_val(x):
    return abs(x)

def abs_log(x):
    return 4*abs(np.log(abs(x) + 0.01))

def exp_square(x):
    return np.exp(x**2)

def fourth(x):
    return x**4

def square_mod(x):
    return 0.5*x**2

###----------------------------------------------------------------------------
## Simulate the survival data

def simulation_survival(nfeat, prop, size):
    """Simulate the survival datasets.

    Parameters
    ----------
    nfeat: number of explanatory features in the data.
    prop: proportion of sparse/linear/non-linear/time features.
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
    non_lin_time_list = [np.exp, square_mod, absolute_val]
    slope_options = [1, 0.5, 1.5, 1.2]
    
    # Correct possible missmatches in the number of features of each kind
    elem = [int(el*nfeat) for el in prop]
    while sum(elem) > nfeat:
        elem[np.argmax(elem)] -= 1
      
    while sum(elem) < nfeat:
        elem[np.argmin(elem)] += 1
    
    # Randomly choose some non-linearities, time dependencies and slopes
    nl_selected = random.choices(non_lin_list, k = int(elem[2] + elem[3]))
    slopes_lin = random.choices(slope_options, k = int(elem[1] + elem[2] + elem[3]))
    
    # Randomly select which variable has each behaviour and their sign
    ind = random.sample(range(nfeat), nfeat)
    a_x = copy.deepcopy(x)
    a_nl = copy.deepcopy(x)
    b_x = copy.deepcopy(x)
    a_x[:,ind[0:int(elem[0])]] *=  0
    a_nl[:,ind[0:int(elem[0] + elem[1])]] *=  0
    for i in range(int(elem[1] + elem[2] + elem[3])):
        a_x[:,ind[int(elem[0]) +i]] = round(slopes_lin[i], 2) * (a_x[:,ind[int(elem[0]) + i]])
    for i in range(int(elem[2] + elem[3])):
        a_nl[:,ind[int(elem[0]) + int(elem[1])+i]] = nl_selected[i](a_nl[:,ind[int(elem[0]) + int(elem[1]) + i]])
    sign = random.sample([1,-1]*nfeat, nfeat)
    for i in range(len(sign)):
        a_x[:,i] = a_x[:,i] * sign[i]

    a_lin = np.sum(a_x, axis = 1)
    a_nl = np.sum(a_nl, axis = 1)
    a = a_nl + a_lin
    
    b_x[:,ind[0:int(elem[0] + elem[1] + elem[2])]] *=  0
    nl_time = random.choices(non_lin_time_list, k = int(elem[3]))
    for i in range(int(elem[3])):
        b_x[:, ind[int(elem[0]) + int(elem[1]) + int(elem[2]) + i]] = nl_time[i](b_x[:,ind[int(elem[0]) + int(elem[1]) + int(elem[2]) + i]])
    b = np.sum(b_x, axis = 1)
    
    # Obtain time
    v = np.random.exponential(size = size)
    if sum(abs(b)) != 0:
        h0 = 0.01
        t = 1/b * np.log(1 + v * b / (h0 * np.exp(a)))
    else:
        h0 = 0.5
        t = v/ (h0 * np.exp(a))
    
    # Obtain censoring
    t_perc = max([x for j,x in enumerate(t) if x <=  np.percentile(t, 90)])
    c_ind = [j for j,x in enumerate(t) if x >=  np.percentile(t, 90)]
    c = np.ones(size)
    for k in c_ind:
        t[k] = t_perc
        c[k] = 0
    y = np.hstack((t.reshape((size, 1)),c.reshape((size, 1))))
    x = np.hstack ((x, t.reshape((size, 1))))
    return x, y, list([elem, ind, nl_selected, slopes_lin, sign, nl_time, nfeat, prop, size])   


