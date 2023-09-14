#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 10:58:02 2023

@author: marvazquezrabunal
"""


"""
PLOT BEHAVIOUR OF INDIVIDUAL WEIGHTS REGRESSION


Description:

Code to plot the behaviour learnt by each feature in the regression data from 
the simulation study.

"""

###----------------------------------------------------------------------------

### Call libraries
import matplotlib.pyplot as plt
from model_regression import model_regression_copy
import pickle
import numpy as np
from functions_regression import final_model, SimulationDataset
import matplotlib.cm as cm
import matplotlib

###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
### prop = 262


# Read data
name = "results_file/n_2000/res_262"
f=open( name, "rb")
b=pickle.load(f)
f.close()

X = b[2]
y = b[3]
dataset = SimulationDataset(X, y, scale_data=True)
X2 = dataset.X.numpy().astype(np.float32)
y2 = dataset.y.numpy().astype(np.float32)


# Fit model
final_model_fitted = final_model(b[1][0], X2, y2, b[7])

model_new = model_regression_copy(X2.shape[1])
model_new.load_state_dict(final_model_fitted.module_.state_dict())
x1 = np.linspace(-1.7, 1.7, 100)


# Define true functions
f0 = np.cos(2*x1)
f1 = -0.5*x1
f2 = -1.2*x1
f3 = 0 * x1
f4 = x1**2
f5 = 0.6*x1
f6 = 1.5*x1
f7 = -2*x1
f8 = 2 * x1
f9 = 0 * x1

f= [f0 - f0.mean(), f1 - f1.mean(), f2 - f2.mean(), f3 - f3.mean(), f4 - f4.mean(), f5 - f5.mean(), f6 - f6.mean(), f7 - f7.mean(), f8 - f8.mean(), f9 - f9.mean()]


# Plot the results
font = {'family' : 'normal',
        'size'   : 27}

matplotlib.rc('font', **font)

colors = cm.rainbow(np.linspace(0, 1, 50))
plt.rcParams["figure.figsize"] = (50,19) 
final, non_linear1, linear1  = model_new(dataset.X)
items = list(model_new.state_dict().items())
for i in range(X2.shape[1]):
    plt.subplot(2, 5, i+1)
    axis_x = X2[:,i]
    axis_y = linear1[:,i].detach().numpy() + non_linear1[:,i].detach().numpy()
    axis_y = axis_y - axis_y.mean()
    plt.scatter(axis_x, axis_y, color = colors[5*i], label = "Prediction")
    axis_x = x1
    axis_y = f[i]
    y_lim = plt.ylim()
    x_lim = plt.xlim()
    plt.plot(x1, f[i], c = 'r', label = "Truth", linestyle='dashed',  linewidth=3.5)
    plt.title("Feature " + str(i+1))
    plt.legend()
    
#  Save figure 
plt.savefig("results_file/feat_plots_262_h.png", format="png", dpi = 200)

###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
### prop = 343


# Read data
name = "results_file/n_2000/res_343"
f=open( name, "rb")
b=pickle.load(f)
f.close()

X = b[2]
y = b[3]
dataset = SimulationDataset(X, y, scale_data=True)
X2 = dataset.X.numpy().astype(np.float32)
y2 = dataset.y.numpy().astype(np.float32)


# Fit model
final_model_fitted = final_model(b[1][4], X2, y2, b[7])

model_new = model_regression_copy(X2.shape[1])
model_new.load_state_dict(final_model_fitted.module_.state_dict())
x1 = np.linspace(-1.7, 1.7, 100)


# Define true functions
f0 = -1.8*x1
f1 = -0.5*x1
f2 = x1**3
f3 = -1 * x1
f4 = 0 * x1
f5 = -2*x1
f6 = -np.sin(2*x1)
f7 = np.exp(x1)
f8 = -0 * x1
f9 = 0 * x1

f= [f0 - f0.mean(), f1 - f1.mean(), f2 - f2.mean(), f3 - f3.mean(), f4 - f4.mean(), f5 - f5.mean(), f6 - f6.mean(), f7 - f7.mean(), f8 - f8.mean(), f9 - f9.mean()]


# Plot the results
font = {'family' : 'normal',
        'size'   : 27}

matplotlib.rc('font', **font)

colors = cm.rainbow(np.linspace(0, 1, 50))
plt.rcParams["figure.figsize"] = (50,19) 
final, non_linear1, linear1  = model_new(dataset.X)
items = list(model_new.state_dict().items())
for i in range(X2.shape[1]):
    plt.subplot(2, 5, i+1)
    axis_x = X2[:,i]
    axis_y = linear1[:,i].detach().numpy() + non_linear1[:,i].detach().numpy()
    axis_y = axis_y - axis_y.mean()
    plt.scatter(axis_x, axis_y, color = colors[5*i], label = "Prediction")
    axis_x = x1
    axis_y = f[i]
    y_lim = plt.ylim()
    x_lim = plt.xlim()
    plt.plot(x1, f[i], c = 'r', label = "Truth", linestyle='dashed',  linewidth=3.5)
    plt.title("Feature " + str(i+1))
    plt.legend()
    
# Save figure
plt.savefig("results_file/feat_plots_343_h.png", format="png", dpi = 200)

###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
### prop = 226


# Read data
name = "results_file/n_2000/res_226"
f=open( name, "rb")
b=pickle.load(f)
f.close()

X = b[2]
y = b[3]
dataset = SimulationDataset(X, y, scale_data=True)
X2 = dataset.X.numpy().astype(np.float32)
y2 = dataset.y.numpy().astype(np.float32)


# Fit model
final_model_fitted = final_model(b[1][2], X2, y2, b[7])

model_new = model_regression_copy(X2.shape[1])
model_new.load_state_dict(final_model_fitted.module_.state_dict())
x1 = np.linspace(-1.7, 1.7, 100)

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
f0 = - x0**2
f1 = -np.sin(2*x1)
f2 = 1.8*x2
f3 = np.log(abs(x3) + 0.01)
f4 = 1.2*x4
f5 = 0*x5
f6 = 0*x6
f7 = -np.sin(2*x7)
f8 = np.cos(2*x8)
f9 = -np.log(abs(x9) + 0.01)

f= [f0 - f0.mean(), f1 - f1.mean(), f2 - f2.mean(), f3 - f3.mean(), f4 - f4.mean(), f5 - f5.mean(), f6 - f6.mean(), f7 - f7.mean(), f8 - f8.mean(), f9 - f9.mean()]


# Plot the results
font = {'family' : 'normal',
        'size'   : 27}

matplotlib.rc('font', **font)

colors = cm.rainbow(np.linspace(0, 1, 50))
plt.rcParams["figure.figsize"] = (50,19) 
final, non_linear1, linear1  = model_new(dataset.X)
items = list(model_new.state_dict().items())
for i in range(X2.shape[1]):
    plt.subplot(2, 5, i+1)
    axis_x = X2[:,i]
    axis_y = linear1[:,i].detach().numpy() + non_linear1[:,i].detach().numpy()
    axis_y = axis_y - axis_y.mean()
    plt.scatter(axis_x, axis_y, color = colors[5*i], label = "Prediction")
    axis_x = x1
    axis_y = f[i]
    y_lim = plt.ylim()
    x_lim = plt.xlim()
    plt.plot(np.sort(X2[:,i]), f[i], c = 'r', label = "Truth", linestyle='dashed',  linewidth=3.5)
    plt.title("Feature " + str(i+1))
    plt.legend()
    
# Save figure
plt.savefig("results_file/feat_plots_226_h.png", format="png", dpi = 200)

    
    
    