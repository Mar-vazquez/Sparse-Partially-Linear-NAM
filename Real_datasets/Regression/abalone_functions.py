#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 13:54:46 2023

@author: marvazquezrabunal
"""

"""
FUNCTIONS ABALONE


Description:

Code to apply our model and the competitive methods to the abalone regression
dataset. Also to plot the shape functions for all the features. 

"""


###----------------------------------------------------------------------------

### Call libraries
import pandas as pd
import sys
import numpy as np
from sklearn.model_selection import KFold
import random
import torch
import pickle
from model_regression import model_regression_copy
import matplotlib.cm as cm
import matplotlib
import matplotlib.pyplot as plt
import copy

sys.path.append('/Users/marvazquezrabunal/Library/Mobile Documents/com~apple~CloudDocs/ETH/Spring 2023/Thesis/Real_datasets/Regression')
from functions_data_reg import SimulationDataset, results_data_reg, results_sparse_NAM_reg, results_data_NAM_reg, final_model

###----------------------------------------------------------------------------
### Read data

data_abalone =  pd.read_csv("Real_datasets/Regression/abalone/abalone.data", sep = ",", header = None).to_numpy()
y = data_abalone[:, -1].astype('float64')
X = data_abalone[:, 1:8].astype('float64')

###----------------------------------------------------------------------------
### Apply transformations (center and scale X and center y) to the data

dataset = SimulationDataset(X, y)
X2 = dataset.X.numpy().astype(np.float32)
y2 = dataset.y.numpy().astype(np.float32)

###----------------------------------------------------------------------------
### Split in k = 5 folds

# Randomly generate index
np.random.seed(230)    
random.seed(230)
torch.manual_seed(230)
nrows = X2.shape[0]
data_index = np.random.choice(nrows, nrows, replace = False)

# Split data into k folds
k_folds = KFold(n_splits = 5).split(data_index)
cv_index = [(data_index[train], data_index[validate]) for train, validate in list(k_folds)]

###----------------------------------------------------------------------------
### Apply our model to each fold and obtain structure found and MSE
res = []
i = 0
for zip_index in cv_index:
    train_index, test_index = zip_index
    X_train, X_test = (X2[train_index, :], X2[test_index, :])
    y_train, y_test = (y2[train_index], y2[test_index])
    
    # Save data for each fold
    data_train = np.hstack((X_train, y_train.reshape(y_train.shape[0],1)))
    data_test = np.hstack((X_test, y_test.reshape(y_test.shape[0],1)))
    pd.DataFrame(data_train).to_csv('Real_datasets/Regression/abalone/abalone_train_' + str(i) +'.csv')
    pd.DataFrame(data_test).to_csv('Real_datasets/Regression/abalone/abalone_test_' + str(i) +'.csv') 
    
    # Fit model for each fold
    results = results_data_reg(X_train, y_train, X_test, y_test, 0)
    res.append(results)
    i = i + 1

# Save results
f=open( "Real_datasets/Regression/abalone/results_abalone", "wb")
pickle.dump(res,f)
f.close()  

###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
### Obtain results for NAM

np.random.seed(134)    
random.seed(134)
torch.manual_seed(134)

res_NAM = []
for zip_index in cv_index:
    train_index, test_index = zip_index
    X_train, X_test = (X2[train_index, :], X2[test_index, :])
    y_train, y_test = (y2[train_index], y2[test_index])
    # Fit NAM for each fold
    results_NAM = results_data_NAM_reg(X_train, y_train, X_test, y_test)
    res_NAM.append(results_NAM)

# Save results
f=open( "Real_datasets/Regression/abalone/res_NAM_abalone", "wb")
pickle.dump(res_NAM,f)
f.close() 
    
###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
### Obtain results for sparse NAM

np.random.seed(250)    
random.seed(250)
torch.manual_seed(250)

res_sparse_NAM = []
i = 0
for zip_index in cv_index:
    train_index, test_index = zip_index
    X_train, X_test = (X2[train_index, :], X2[test_index, :])
    y_train, y_test = (y2[train_index], y2[test_index])
    # Fit sparse NAM for each fold
    results_sparse_NAM = results_sparse_NAM_reg(X_train, y_train, X_test, y_test, i)
    res_sparse_NAM.append(results_sparse_NAM)
    i = i + 1

# Save results
f=open( "Real_datasets/Regression/abalone/results_sparse_NAM_abalone", "wb")
pickle.dump(res_sparse_NAM,f)
f.close()  
###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
###----------------------------------------------------------------------------

### Summary of the results

### Obtain mean and sd of MSE and of the number of features with each structure
### for our model
mse_model = []
sparse = []
linear = []
non_lin = []
for i in range(5):
    mse_model.append(res[i][2])
    sparse.append(len(res[i][1][0]))
    linear.append(len(res[i][1][1]))
    non_lin.append(len(res[i][1][2]))
    
np.mean(mse_model)
np.std(mse_model)

np.mean(sparse)
np.std(sparse)

np.mean(linear)
np.std(linear)

np.mean(non_lin)
np.std(non_lin)

###----------------------------------------------------------------------------

### Obtain mean and sd of MSE for NAM

np.mean(res_NAM)
np.std(res_NAM)

###----------------------------------------------------------------------------

### Obtain mean and sd of MSE and of the number of features with each structure
### for sparse NAM
mse_model = []
sparse = []
linear = []
non_lin = []
for i in range(5):
    mse_model.append(res_sparse_NAM[i][2])
    sparse.append(len(res_sparse_NAM[i][1][0]))
    linear.append(len(res_sparse_NAM[i][1][1]))
    non_lin.append(len(res_sparse_NAM[i][1][2]))
    
np.mean(mse_model)
np.std(mse_model)

np.mean(sparse)
np.std(sparse)

np.mean(linear)
np.std(linear)

np.mean(non_lin)
np.std(non_lin)


###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
### Plot shape functions features

# Load the data
f=open( "Real_datasets/Regression/Abalone/results_abalone", "rb")
res = pickle.load(f)
f.close()  

# Fit the model using the optimal hyperparameters found
final_res = final_model(X2,y2, res[3][0], 18)
model_new = model_regression_copy(X2.shape[1])
model_new.load_state_dict(final_res.module_.state_dict())

# Set some characzeristics of the plot 
font = {'family' : 'normal',
        'size'   : 40}
matplotlib.rc('font', **font)
colors = cm.rainbow(np.linspace(0, 1, 20))
plt.rcParams["figure.figsize"] = (60,25) 

# Read data and transform it 
data_abalone =  pd.read_csv("Real_datasets/Regression/abalone/abalone.data", sep = ",", header = None).to_numpy()
y = data_abalone[:, -1].astype('float64')
X = data_abalone[:, 1:8].astype('float64')
labels_feat = ["length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight"]
x0 = np.linspace(0.075, 0.815, 4177)
new_x0 = (x0-X[:,0].mean())/X[:,0].std()
x4 = np.linspace(0.001, 1.49, 4177)
new_x4 = (x4-X[:,4].mean())/X[:,4].std()
x6 = np.linspace(0.0015, 1.005, 4177)
new_x6 = (x6-X[:,6].mean())/X[:,6].std()

new_datasetX = copy.deepcopy(dataset.X)
new_datasetX[:,0]= torch.from_numpy(new_x0)
new_datasetX[:,4]= torch.from_numpy(new_x4)
new_datasetX[:,6]= torch.from_numpy(new_x6)

new_X = copy.deepcopy(X)
new_X[:,0] = x0
new_X[:,4] = x4
new_X[:,6] = x6

# Plot shape functions
final, non_linear1, linear1  = model_new(new_datasetX)
items = list(model_new.state_dict().items())
for i in range(X2.shape[1]):
    plt.subplot(2, 4, i+1)
    axis_x = new_X[:,i]
    axis_y = linear1[:,i].detach().numpy() + non_linear1[:,i].detach().numpy()
    plt.plot(axis_x, axis_y, color = colors[3*i], label = "Prediction", linewidth = 5)
    y_lim = plt.ylim()
    x_lim = plt.xlim()
    plt.title(str(labels_feat[i]))
    plt.legend()
plt.suptitle("Shape functions Abalone", fontweight='bold', fontsize="50")

# Save figure
plt.savefig("Real_datasets/Regression/abalone/shape_abalone.png", format="png", dpi = 200)
