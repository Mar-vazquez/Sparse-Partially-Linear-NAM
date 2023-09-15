#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:21:19 2023

@author: marvazquezrabunal
"""


"""
FUNCTIONS ROT. & GBSG


Description:

Code to apply our model and the competitive methods to the Rot. & GBSG survival
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
from model_survival import model_survival_copy
import matplotlib.cm as cm
import matplotlib
import matplotlib.pyplot as plt

sys.path.append('/Users/marvazquezrabunal/Library/Mobile Documents/com~apple~CloudDocs/ETH/Spring 2023/Thesis/Real_datasets/Survival')
from functions_data_surv import SimulationDataset, results_data_surv, results_data_NAM_surv, results_sparse_NAM_surv, final_model


###----------------------------------------------------------------------------
### Read data

data_rgbsg =  pd.read_csv("Real_datasets/Survival/RGBSG/RGBSG_adapted.csv", sep = ",").to_numpy()
y = copy.deepcopy(data_rgbsg[:, 7:9])
X = np.delete(copy.deepcopy(data_rgbsg[:, 0:8]), 1, 1)

###----------------------------------------------------------------------------
### Apply transformations (center and scale X and center y) to the data

dataset = SimulationDataset(X, y, binary = [0, 1])
X2 = dataset.X.numpy().astype(np.float32)
y2 = dataset.y.numpy().astype(np.float32)

###----------------------------------------------------------------------------
### Split in k = 5 folds

# Randomly generate index
np.random.seed(22)    
random.seed(22)
torch.manual_seed(22)
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
    pd.DataFrame(data_train).to_csv('Real_datasets/Survival/RGBSG/rgbsg_train_' + str(i) +'.csv')
    pd.DataFrame(data_test).to_csv('Real_datasets/Survival/RGBSG/rgbsg_test_' + str(i) +'.csv') 
    
    # Fit model for each fold
    results = results_data_surv(X_train, y_train, X_test, y_test, 2*i)
    res.append(results)
    i = i + 1

# Save results
f=open( "Real_datasets/Survival/RGBSG/results_rgbsg", "wb")
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

np.random.seed(423)    
random.seed(423)
torch.manual_seed(423)

res_NAM = []
for zip_index in cv_index:
    train_index, test_index = zip_index
    X_train, X_test = (X2[train_index, :], X2[test_index, :])
    y_train, y_test = (y2[train_index], y2[test_index])
    # Fit NAM for each fold
    results_NAM = results_data_NAM_surv(X_train, y_train, X_test, y_test)
    res_NAM.append(results_NAM)

# Save results
f=open( "Real_datasets/Survival/RGBSG/res_NAM_rgbsg", "wb")
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

np.random.seed(98)    
random.seed(98)
torch.manual_seed(98)

res_sparse_NAM = []
i = 0
for zip_index in cv_index:
    train_index, test_index = zip_index
    X_train, X_test = (X2[train_index, :], X2[test_index, :])
    y_train, y_test = (y2[train_index], y2[test_index])
    # Fit sparse NAM for each fold
    results_sparse_NAM = results_sparse_NAM_surv(X_train, y_train, X_test, y_test, 2*i)
    res_sparse_NAM.append(results_sparse_NAM)
    i = i + 1

# Save results
f=open( "Real_datasets/Survival/RGBSG/res_sparse_NAM_rgbsg", "wb")
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


###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
### Plot shape functions features

# Load the results
f=open( "Real_datasets/Survival/RGBSG/results_rgbsg", "rb")
res = pickle.load(f)
f.close()  

# Fit the model using the optimal hyperparameters found
final_res = final_model(X2,y2, res[4][1], 212)
model_new = model_survival_copy(X2.shape[1]-1)
model_new.load_state_dict(final_res.module_.state_dict())

# Set some characzeristics of the plot 
font = {'family' : 'normal',
        'size'   : 40}
matplotlib.rc('font', **font)
colors = cm.rainbow(np.linspace(0, 1,20))
plt.rcParams["figure.figsize"] = (60,25) 

# Read data and transform it 
data_rgbsg =  pd.read_csv("Real_datasets/Survival/RGBSG/RGBSG_adapted.csv", sep = ",").to_numpy()
y = copy.deepcopy(data_rgbsg[:, 7:9])
X = np.delete(copy.deepcopy(data_rgbsg[:, 0:8]), 1, 1)
labels_feat = ["horm_treatment", "menopause", "age", "n_positive_nodes", "progesterone", "estrogene"]
x2 = np.linspace(21, 90, 2232)
new_x2 = (x2-X[:,2].mean())/X[:,2].std()
x3 = np.linspace(1, 51, 2232)
new_x3 = (x3-X[:,3].mean())/X[:,3].std()

new_datasetX = copy.deepcopy(X2)
new_datasetX[:,2]= new_x2
new_datasetX[:,3]= new_x3

new_X = copy.deepcopy(X)
new_X[:,2] = x2
new_X[:,3] = x3

# Plot shape functions
final, non_linear1, linear1, time1  = model_new(new_datasetX)
items = list(model_new.state_dict().items())
for i in range(X2.shape[1]-1):
    plt.subplot(2, 3, i+1)
    if i in [0,1]:
        axis_x = ["0","1"]
        axis_y = linear1[:,i].detach().numpy() + non_linear1[:,i].detach().numpy() + time1[:,i].detach().numpy()
        plt.scatter(axis_x, axis_y[0:2], color = colors[3*i], label = "Prediction", s = 180)
        y_lim = plt.ylim()
        x_lim = plt.xlim()
        plt.title(str(labels_feat[i]))
        plt.legend()
    else:
        axis_x = new_X[:,i]
        axis_y = linear1[:,i].detach().numpy() + non_linear1[:,i].detach().numpy() + time1[:,i].detach().numpy()
        plt.plot(axis_x, axis_y, color = colors[3*i], label = "Prediction", linewidth = 5)
        y_lim = plt.ylim()
        x_lim = plt.xlim()
        plt.title(str(labels_feat[i]))
        plt.legend()
plt.suptitle("Shape functions Rot. & GBSG", fontweight='bold', fontsize="50")

# Save figure
plt.savefig("Real_datasets/Survival/RGBSG/shape_RGBSG.png", format="png", dpi = 200)

