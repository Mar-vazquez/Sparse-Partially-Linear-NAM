#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:11:44 2023

@author: marvazquezrabunal
"""


"""
FUNCTIONS METABRIC


Description:

Code to apply our model and the competitive methods to the METABRIC survival
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
import copy

sys.path.append('/Users/marvazquezrabunal/Library/Mobile Documents/com~apple~CloudDocs/ETH/Spring 2023/Thesis/Real_datasets/Survival')
from functions_data_surv import SimulationDataset, results_data_surv, results_data_NAM_surv, results_sparse_NAM_surv



###----------------------------------------------------------------------------
### Read data

data_met =  pd.read_csv("Real_datasets/Survival/METABRIC/METABRIC_adapted.csv", sep = ",").to_numpy()
y = copy.deepcopy(data_met[:, 9:11])
X = copy.deepcopy(data_met[:, 0:10])

###----------------------------------------------------------------------------
### Apply transformations (center and scale X and center y) to the data

dataset = SimulationDataset(X, y, binary = [4,5,6,7])
X2 = dataset.X.numpy().astype(np.float32)
y2 = dataset.y.numpy().astype(np.float32)

###----------------------------------------------------------------------------
### Split in k = 5 folds

# Randomly generate index
np.random.seed(30)    
random.seed(30)
torch.manual_seed(30)
nrows = X2.shape[0]
data_index = np.random.choice(nrows, nrows, replace = False)

# Split data into k folds
k_folds = KFold(n_splits = 5).split(data_index)
cv_index = [(data_index[train], data_index[validate]) for train, validate in list(k_folds)]

###----------------------------------------------------------------------------
### Apply our model to each fold and obtain structure found, C-index and IBS
res = []
i = 0
for zip_index in cv_index:
    train_index, test_index = zip_index
    X_train, X_test = (X2[train_index, :], X2[test_index, :])
    y_train, y_test = (y2[train_index, :], y2[test_index, :])
    
    # Save data for each fold
    data_train = np.hstack((X_train, y_train.reshape(y_train.shape[0], 2)))
    data_test = np.hstack((X_test, y_test.reshape(y_test.shape[0], 2)))
    pd.DataFrame(data_train).to_csv('Real_datasets/Survival/METABRIC/met_train_' + str(i) +'.csv')
    pd.DataFrame(data_test).to_csv('Real_datasets/Survival/METABRIC/met_test_' + str(i) +'.csv') 
    
    # Fit model for each fold
    results = results_data_surv(X_train, y_train, X_test, y_test, i)
    res.append(results)
    i = i + 1

# Save results
f=open( "Real_datasets/Survival/METABRIC/results_meta", "wb")
pickle.dump(res,f)
f.close()  

###----------------------------------------------------------------------------
### Mean and sd of the results of the model
cindex_model = []
ibs_model = []
for i in range(5):
    cindex_model.append(res[i][3])
    ibs_model.append(res[i][4])
    
np.mean(cindex_model)
np.std(cindex_model)

np.mean(ibs_model)
np.std(ibs_model)


###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
### Obtain results for NAM

np.random.seed(40)    
random.seed(40)
torch.manual_seed(40)

res_NAM = []
for zip_index in cv_index:
    train_index, test_index = zip_index
    X_train, X_test = (X2[train_index, :], X2[test_index, :])
    y_train, y_test = (y2[train_index], y2[test_index])
    # Fit NAM for each fold
    results_NAM = results_data_NAM_surv(X_train, y_train, X_test, y_test)
    res_NAM.append(results_NAM)

# Save results
f=open( "Real_datasets/Survival/METABRIC/res_NAM_meta", "wb")
pickle.dump(res_NAM,f)
f.close()  

###----------------------------------------------------------------------------
### Mean and sd of the results of the NAM model
cindex_model = []
ibs_model = []
for i in range(5):
    cindex_model.append(res_NAM[i][1])
    ibs_model.append(res_NAM[i][2])
    
np.mean(cindex_model)
np.std(cindex_model)

np.mean(ibs_model)
np.std(ibs_model)


###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
### Obtain results for sparse NAM

np.random.seed(80)    
random.seed(80)
torch.manual_seed(80)

res_sparse_NAM = []
i = 0
for zip_index in cv_index:
    train_index, test_index = zip_index
    X_train, X_test = (X2[train_index, :], X2[test_index, :])
    y_train, y_test = (y2[train_index], y2[test_index])
    # Fit sparse NAM for each fold
    results_sparse_NAM = results_sparse_NAM_surv(X_train, y_train, X_test, y_test, i)
    res_sparse_NAM.append(results_sparse_NAM)
    i = i + 1

# Save results
f=open( "Real_datasets/Survival/METABRIC/res_sparse_NAM_meta", "wb")
pickle.dump(res_sparse_NAM,f)
f.close()

###----------------------------------------------------------------------------
### Mean and sd of the results of the sparse NAM model
cindex_model = []
ibs_model = []
for i in range(5):
    cindex_model.append(res_sparse_NAM[i][3])
    ibs_model.append(res_sparse_NAM[i][4])
    
np.mean(cindex_model)
np.std(cindex_model)

np.mean(ibs_model)
np.std(ibs_model)

