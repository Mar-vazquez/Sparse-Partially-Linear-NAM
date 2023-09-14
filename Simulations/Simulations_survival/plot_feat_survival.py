#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 12:12:59 2023

@author: marvazquezrabunal
"""


"""
PLOT BEHAVIOUR OF INDIVIDUAL WEIGHTS SURVIVAL


Description:

Code to plot the behaviour learnt by each feature in the survival data from 
the simulation study.

"""

###----------------------------------------------------------------------------

### Call libraries
import matplotlib.pyplot as plt
from model_survival import model_survival_copy
import pickle
import numpy as np
from functions_survival import final_model
import matplotlib.cm as cm
import matplotlib
import copy


###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
### prop = 3430
    
# Read data
name = "results_file/n_2000/res_3430"
f=open( name, "rb")
b=pickle.load(f)
f.close()

X2 = b[2]
y2 = b[3]



# Fit model
final_model_fitted = final_model(b[1][3], X2, y2, b[7])

model_new = model_survival_copy(X2.shape[1] - 1)
model_new.load_state_dict(final_model_fitted.module_.state_dict())

x0 = np.sort(X2[:,0])
x1 = np.sort(X2[:,1])
x2 = np.sort(X2[:,2])
x3 = np.sort(X2[:,3])
x4 = np.sort(X2[:,4])
x5 = np.sort(X2[:,5])
x6 = np.sort(X2[:,6])
x7 = np.sort(X2[:,7])
x8 = np.sort(X2[:,8])
x9 = np.sort(X2[:,9])

# Define true functions
f0 = 1 * x0
f1 = 0.5*x1 + np.exp(x1)
f2 = 0 * x2
f3 = -1.2*x3
f4 = -1.5 * x4
f5 = 0*x5
f6 =  0*x6
f7 = -1.2*x7
f8 = 1*x8 + np.exp(x8)
f9 = -1.2 * x9 + np.exp(x9)

f= [f0 - f0.mean(), f1 - f1.mean(), f2 - f2.mean(), f3 - f3.mean(), f4 - f4.mean(), f5 - f5.mean(), f6 - f6.mean(), f7 - f7.mean(), f8 - f8.mean(), f9 - f9.mean()]


# Plot the results
font = {'family' : 'normal',
        'size'   : 27}

matplotlib.rc('font', **font)

colors = cm.rainbow(np.linspace(0, 1, 50))
plt.rcParams["figure.figsize"] = (50,19) 
X3 = copy.deepcopy(X2)
X3[:,-1] = -1.19
final, linear1, non_linear1, time1  = model_new(X3)
items = list(model_new.state_dict().items())
for i in range(X2.shape[1] -1):
    plt.subplot(2, 5, i+1)
    axis_x = X2[:,i]
    axis_y = linear1[:,i].detach().numpy() + non_linear1[:,i].detach().numpy() + time1[:,i].detach().numpy()
    axis_y = axis_y - axis_y.mean()
    plt.scatter(axis_x, axis_y, color = colors[5*i], label = "Prediction")
    axis_x = x1
    axis_y = f[i]
    y_lim = plt.ylim()
    x_lim = plt.xlim()
    plt.plot(np.sort(X3[:,i]), f[i], c = 'r', label = "Truth", linestyle='dashed',  linewidth=3.5)
    plt.title("Feature " + str(i+1))
    plt.legend()

# Save figure    
plt.savefig("results_file/feat_plots_3430.png", format="png", dpi = 200)


###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
### prop = 2521
    
# Read data
name = "results_file/n_2000/res_2521"
f=open( name, "rb")
b=pickle.load(f)
f.close()

X2 = b[2]
y2 = b[3]


# Fit model
final_model_fitted = final_model(b[1][2], X2, y2, b[7])

model_new = model_survival_copy(X2.shape[1] - 1)
model_new.load_state_dict(final_model_fitted.module_.state_dict())

x0 = np.sort(X2[:,0])
x1 = np.sort(X2[:,1])
x2 = np.sort(X2[:,2])
x3 = np.sort(X2[:,3])
x4 = np.sort(X2[:,4])
x5 = np.sort(X2[:,5])
x6 = np.sort(X2[:,6])
x7 = np.sort(X2[:,7])
x8 = np.sort(X2[:,8])
x9 = np.sort(X2[:,9])


# Define true functions
f0 = 1.2 * x0
f1 = -1.2*x1 + np.exp(x1)
f2 = -1*x2
f3 = 0.5*x3
f4 =  0*x4
f5 = 1.5*x5 + x5**3
f6 = 0*x6
f7 = -1.5*x7 + np.cos(2*x7) + 2 * 0.5 * x7**2 
f8 = 1.5*x8
f9 = 1 * x9

f= [f0 - f0.mean(), f1 - f1.mean(), f2 - f2.mean(), f3 - f3.mean(), f4 - f4.mean(), f5 - f5.mean(), f6 - f6.mean(), f7 - f7.mean(), f8 - f8.mean(), f9 - f9.mean()]


# Plot the results
font = {'family' : 'normal',
        'size'   : 27}

matplotlib.rc('font', **font)

colors = cm.rainbow(np.linspace(0, 1, 50))
plt.rcParams["figure.figsize"] = (50,20) 

    
time_seq = [1, 5, 10]

colors2= cm.rainbow(np.linspace(0, 2, 20))
X3 = copy.deepcopy(X2)
X3[:,-1] = (2 - y2[:,0].mean())/y2[:,0].std()
final, linear1, non_linear1, time1  = model_new(X3)
items = list(model_new.state_dict().items())
for i in range(X2.shape[1] -1):
    if i == 7:
        plt.subplot(2, 5, i+1)
        axis_x = X2[:,i]
        for t in time_seq:
            axis_x = X2[:,i]
            f7 = -1.5*x7 + np.cos(2*x7) + t * 0.5 * x7**2 
            f= [f0 - f0.mean(), f1 - f1.mean(), f2 - f2.mean(), f3 - f3.mean(), f4 - f4.mean(), f5 - f5.mean(), f6 - f6.mean(), f7 - f7.mean(), f8 - f8.mean(), f9 - f9.mean()]
            X3[:,-1] = (t - y2[:,0].mean())/y2[:,0].std()
            final, linear1, non_linear1, time1  = model_new(X3)
            axis_y = linear1[:,i].detach().numpy() + non_linear1[:,i].detach().numpy() + time1[:,i].detach().numpy()
            axis_y = axis_y - axis_y.mean()
            plt.scatter(axis_x, axis_y, color = colors2[t], label = "Prediction t = " + str(t))
            axis_x = x1
            axis_y = f[i]
            plt.plot(np.sort(X3[:,i]), f[i], c = colors2[t], label = "Truth t = " + str(t), linestyle='dashed',  linewidth=3.5)
            plt.title("Feature " + str(i+1))
            plt.legend()
    else:
        plt.subplot(2, 5, i+1)
        axis_x = X2[:,i]
        axis_y = linear1[:,i].detach().numpy() + non_linear1[:,i].detach().numpy() + time1[:,i].detach().numpy()
        axis_y = axis_y - axis_y.mean()
        plt.scatter(axis_x, axis_y, color = colors[5*i], label = "Prediction")
        axis_x = x1
        axis_y = f[i]
        y_lim = plt.ylim()
        x_lim = plt.xlim()
        plt.plot(np.sort(X3[:,i]), f[i], c = 'r', label = "Truth", linestyle='dashed',  linewidth=3.5)
        plt.title("Feature " + str(i+1))
        plt.legend()

# Save figure
plt.savefig("results_file/feat_plots_2521.png", format="png", dpi = 200)


    