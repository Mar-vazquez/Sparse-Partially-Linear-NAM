#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 11:59:07 2023

@author: marvazquezrabunal
"""


"""
FUNCTIONS WINE


Description:

Code to apply our model and the competitive methods to the wine regression
dataset.

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

sys.path.append('/Users/marvazquezrabunal/Library/Mobile Documents/com~apple~CloudDocs/ETH/Spring 2023/Thesis/Real_datasets/Regression')
from functions_data_reg import SimulationDataset, results_data_reg, results_sparse_NAM_reg, results_data_NAM_reg

###----------------------------------------------------------------------------
### Read data

data_wine =  pd.read_csv("Real_datasets/Regression/wine/winequality-white.csv", sep = ";").to_numpy()
y = data_wine[:, -1]
X = data_wine[:, 0:11]

###----------------------------------------------------------------------------
### Apply transformations (center and scale X and center y) to the data

dataset = SimulationDataset(X, y)
X2 = dataset.X.numpy().astype(np.float32)
y2 = dataset.y.numpy().astype(np.float32)

###----------------------------------------------------------------------------
### Split in k = 5 folds

# Randomly generate indices
np.random.seed(0)    
random.seed(0)
torch.manual_seed(0)
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
    pd.DataFrame(data_train).to_csv('Real_datasets/Regression/wine/wine_train_' + str(i) +'.csv')
    pd.DataFrame(data_test).to_csv('Real_datasets/Regression/wine/wine_test_' + str(i) +'.csv') 
    
    # Fit model for each fold
    results = results_data_reg(X_train, y_train, X_test, y_test, 0)
    res.append(results)
    i = i + 1

# Save results
f=open( "Real_datasets/Regression/wine/results_wine", "wb")
pickle.dump(res,f)
f.close()  

###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
### Obtain results for NAM

np.random.seed(10)    
random.seed(10)
torch.manual_seed(10)

res_NAM = []
for zip_index in cv_index:
    train_index, test_index = zip_index
    X_train, X_test = (X2[train_index, :], X2[test_index, :])
    y_train, y_test = (y2[train_index], y2[test_index])
    
    # Fit NAM for each fold
    results_NAM = results_data_NAM_reg(X_train, y_train, X_test, y_test)
    res_NAM.append(results_NAM)

# Save results
f=open( "Real_datasets/Regression/wine/res_NAM_wine", "wb")
pickle.dump(res_NAM,f)
f.close() 
    
###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
### Obtain results for sparse NAM

np.random.seed(20)    
random.seed(20)
torch.manual_seed(20)

res_sparse_NAM = []
i = 0
for zip_index in cv_index:
    train_index, test_index = zip_index
    X_train, X_test = (X2[train_index, :], X2[test_index, :])
    y_train, y_test = (y2[train_index], y2[test_index])
    
    # Fit sparse NAM model for each fold
    results_sparse_NAM = results_sparse_NAM_reg(X_train, y_train, X_test, y_test, i)
    res_sparse_NAM.append(results_sparse_NAM)
    i = i + 1

# Save results
f=open( "Real_datasets/Regression/wine/results_sparse_NAM_wine", "wb")
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




