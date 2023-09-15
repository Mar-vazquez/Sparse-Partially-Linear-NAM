#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 17:21:13 2023

@author: marvazquezrabunal
"""


"""
FUNCTIONS TITANIC


Description:

Code to apply our model and the competitive methods to the Titanic classification
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
import copy
from model_classification import model_classification_copy
import matplotlib.cm as cm
import matplotlib
import matplotlib.pyplot as plt


sys.path.append('/Users/marvazquezrabunal/Library/Mobile Documents/com~apple~CloudDocs/ETH/Spring 2023/Thesis/Real_datasets/Classification')
from functions_data_classification import SimulationDataset, results_data_classif, results_data_NAM_classif, results_sparse_NAM_classif, final_model



###----------------------------------------------------------------------------
### Read data

data_class = pd.read_csv("Real_datasets/Classification/Titanic/titanic_adapted.csv", sep = ",").to_numpy()
y = copy.deepcopy(data_class[:, 1])
X = copy.deepcopy(data_class[:, 2:7])



###----------------------------------------------------------------------------
### Apply transformations (center and scale X and center y) to the data

dataset = SimulationDataset(X, y, binary = [0])
X2 = dataset.X.numpy().astype(np.float32)
y2 = dataset.y.numpy().astype(np.float32)

###----------------------------------------------------------------------------
### Split in k = 5 folds

# Randomly generate index
np.random.seed(1232)    
random.seed(1232)
torch.manual_seed(1232)
nrows = X2.shape[0]
data_index = np.random.choice(nrows, nrows, replace = False)

# Split data into k folds
k_folds = KFold(n_splits = 5).split(data_index)
cv_index = [(data_index[train], data_index[validate]) for train, validate in list(k_folds)]

###----------------------------------------------------------------------------
### Apply our model to each fold and obtain structure found, F1 score and AUC

res = []
i = 0
for zip_index in cv_index:
    train_index, test_index = zip_index
    X_train, X_test = (X2[train_index, :], X2[test_index, :])
    y_train, y_test = (y2[train_index], y2[test_index])
    
    # Save data for each fold
    data_train = np.hstack((X_train, y_train.reshape(y_train.shape[0], 1)))
    data_test = np.hstack((X_test, y_test.reshape(y_test.shape[0], 1)))
    pd.DataFrame(data_train).to_csv('Real_datasets/Classification/Titanic/titanic_train_' + str(i) +'.csv')
    pd.DataFrame(data_test).to_csv('Real_datasets/Classification/Titanic/titanic_test_' + str(i) +'.csv') 
    
    # Fit model for each fold
    results = results_data_classif(X_train, y_train, X_test, y_test, 3*i)
    res.append(results)
    i = i + 1

# Save results
f=open( "Real_datasets/Classification/Titanic/results_titanic", "wb")
pickle.dump(res,f)
f.close()  

###----------------------------------------------------------------------------
### Mean and sd of the results of the model (F1 score, AUC and structure)
f1_model = []
for i in range(5):
    f1_model.append(res[i][3])
    
np.mean(f1_model)
np.std(f1_model)

auc_model = []
for i in range(5):
    auc_model.append(res[i][4])
    
np.mean(auc_model)
np.std(auc_model)

sparse = []
linear = []
non_lin = []
for i in range(5):
    sparse.append(len(res[i][2][0]))
    linear.append(len(res[i][2][1]))
    non_lin.append(len(res[i][2][2]))
    

np.mean(sparse)
np.std(sparse)


np.mean(linear)
np.std(linear)


np.mean(non_lin)
np.std(non_lin)



###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
### Obtain results for NAM

np.random.seed(1640)    
random.seed(1640)
torch.manual_seed(1640)

res_NAM = []
for zip_index in cv_index:
    train_index, test_index = zip_index
    X_train, X_test = (X2[train_index, :], X2[test_index, :])
    y_train, y_test = (y2[train_index], y2[test_index])
    # Fit NAM for each fold
    results_NAM = results_data_NAM_classif(X_train, y_train, X_test, y_test)
    res_NAM.append(results_NAM)

# Save results
f=open( "Real_datasets/Classification/Titanic/res_NAM_titanic", "wb")
pickle.dump(res_NAM,f)
f.close()  

###----------------------------------------------------------------------------
### Mean and sd of the results of the NAM (F1 score, AUC and structure)
f1_model = []
for i in range(5):
    f1_model.append(res_NAM[i][1])
    
np.mean(f1_model)
np.std(f1_model)  

auc_model = []
for i in range(5):
    auc_model.append(res_NAM[i][2])
    
np.mean(auc_model)
np.std(auc_model)  


###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
### Obtain results for sparse NAM

np.random.seed(830)    
random.seed(830)
torch.manual_seed(830)

res_sparse_NAM = []
i = 0
for zip_index in cv_index:
    train_index, test_index = zip_index
    X_train, X_test = (X2[train_index, :], X2[test_index, :])
    y_train, y_test = (y2[train_index], y2[test_index])
    
    # Fit sparse NAM for each fold
    results_sparse_NAM = results_sparse_NAM_classif(X_train, y_train, X_test, y_test, i)
    res_sparse_NAM.append(results_sparse_NAM)
    i = i + 1

# Save results
f=open( "Real_datasets/Classification/Titanic/res_sparse_NAM_titanic", "wb")
pickle.dump(res_sparse_NAM,f)
f.close()


###----------------------------------------------------------------------------
### Mean and sd of the results of the sparse NAM (F1 score, AUC and structure)
f1_model = []
for i in range(5):
    f1_model.append(res_sparse_NAM[i][3])
    
np.mean(f1_model)
np.std(f1_model)

auc_model = []
for i in range(5):
    auc_model.append(res_sparse_NAM[i][4])
    
np.mean(auc_model)
np.std(auc_model)


sparse = []
linear = []
non_lin = []
for i in range(5):
    sparse.append(len(res_sparse_NAM[i][2][0]))
    linear.append(len(res_sparse_NAM[i][2][1]))
    non_lin.append(len(res_sparse_NAM[i][2][2]))
    

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

# Load the results
f=open( "Real_datasets/Classification/Titanic/results_titanic", "rb")
res = pickle.load(f)
f.close()  

# Fit model with one set of optimal hyperparameters
final_res = final_model(X2,y2, res[2][1], 1340)
model_new = model_classification_copy(X2.shape[1])
model_new.load_state_dict(final_res.module_.state_dict())

# Set parameters plot
font = {'family' : 'normal',
        'size'   : 40}
matplotlib.rc('font', **font)
colors = cm.rainbow(np.linspace(0, 1, 20))
plt.rcParams["figure.figsize"] = (60,15) 

# Transform data
data_class = pd.read_csv("Real_datasets/Classification/Titanic/titanic_adapted.csv", sep = ",").to_numpy()
y = copy.deepcopy(data_class[:, 1])
X = copy.deepcopy(data_class[:, 2:7])
labels_feat = ["Sex", "Age", "Sibsp", "Parch", "Fare"]
x4 = np.linspace(0, 512, 891)
new_x4 = (x4-X[:,4].mean())/X[:,4].std()

new_datasetX = copy.deepcopy(dataset.X)
new_datasetX[:,4]= torch.from_numpy(new_x4)

new_X = copy.deepcopy(X)
new_X[:,4] = x4

# Plot shape functions
final, non_linear1, linear1  = model_new(new_datasetX)
items = list(model_new.state_dict().items())
for i in range(X2.shape[1]):
    plt.subplot(1, 5, i+1)
    if i == 0:
        axis_x = ["Female", "Male"]
        axis_y = linear1[:,i].detach().numpy() + non_linear1[:,i].detach().numpy()
        plt.scatter(axis_x, axis_y[3:5], color = colors[4*i], label = "Prediction", s=120)
        plt.title(str(labels_feat[i]))
        plt.legend()
    else:
        axis_x = new_X[:,i]
        axis_y = linear1[:,i].detach().numpy() + non_linear1[:,i].detach().numpy()
        plt.plot(axis_x, axis_y, color = colors[4*i], label = "Prediction", linewidth = 5)
        y_lim = plt.ylim()
        x_lim = plt.xlim()
        plt.title(str(labels_feat[i]))
        plt.legend()
plt.suptitle("Shape functions Titanic", fontweight='bold', fontsize="50")

# Save figure
plt.savefig("Real_datasets/Classification/Titanic/shape_titanic.png", format="png", dpi = 200)


###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
### Plot shape functions features for all the folds

# Load the results
f=open( "Real_datasets/Classification/Titanic/results_titanic", "rb")
res = pickle.load(f)
f.close()  

# Fit all the models with the all the optimal hyperparameters
final_res0 = final_model(X2,y2, res[0][1], 1340)
model_new0 = model_classification_copy(X2.shape[1])
model_new0.load_state_dict(final_res0.module_.state_dict())

final_res1 = final_model(X2,y2, res[1][1], 1340)
model_new1 = model_classification_copy(X2.shape[1])
model_new1.load_state_dict(final_res1.module_.state_dict())

final_res2 = final_model(X2,y2, res[2][1], 1340)
model_new2 = model_classification_copy(X2.shape[1])
model_new2.load_state_dict(final_res2.module_.state_dict())

final_res3 = final_model(X2,y2, res[3][1], 1340)
model_new3 = model_classification_copy(X2.shape[1])
model_new3.load_state_dict(final_res3.module_.state_dict())

final_res4 = final_model(X2,y2, res[4][1], 1340)
model_new4 = model_classification_copy(X2.shape[1])
model_new4.load_state_dict(final_res4.module_.state_dict())

# Set some settings of the plots
font = {'family' : 'normal',
        'size'   : 40}
matplotlib.rc('font', **font)
colors = cm.rainbow(np.linspace(0, 1, 20))
plt.rcParams["figure.figsize"] = (60,15) 

# Transform data
data_class = pd.read_csv("Real_datasets/Classification/Titanic/titanic_adapted.csv", sep = ",").to_numpy()
y = copy.deepcopy(data_class[:, 1])
X = copy.deepcopy(data_class[:, 2:7])
labels_feat = ["Sex", "Age", "Sibsp", "Parch", "Fare"]
x2 = np.linspace(0, 8, 891)
new_x2 = (x2-X[:,2].mean())/X[:,2].std()
x4 = np.linspace(0, 512, 891)
new_x4 = (x4-X[:,4].mean())/X[:,4].std()


new_datasetX = copy.deepcopy(dataset.X)
new_datasetX[:,4]= torch.from_numpy(new_x4)
new_datasetX[:,2]= torch.from_numpy(new_x2)

new_X = copy.deepcopy(X)
new_X[:,4] = x4
new_X[:,2] = x2

# Plot all the shape functions
final0, non_linear0, linear0  = model_new0(new_datasetX)
final1, non_linear1, linear1  = model_new1(new_datasetX)
final2, non_linear2, linear2  = model_new2(new_datasetX)
final3, non_linear3, linear3  = model_new3(new_datasetX)
final4, non_linear4, linear4  = model_new4(new_datasetX)
items = list(model_new.state_dict().items())
for i in range(X2.shape[1]):
    plt.subplot(1, 5, i+1)
    if i == 0:
        axis_x = ["Female", "Male"]
        axis_y0 = linear0[:,i].detach().numpy() + non_linear0[:,i].detach().numpy()
        axis_y1 = linear1[:,i].detach().numpy() + non_linear1[:,i].detach().numpy()
        axis_y2 = linear2[:,i].detach().numpy() + non_linear2[:,i].detach().numpy()
        axis_y3 = linear3[:,i].detach().numpy() + non_linear3[:,i].detach().numpy()
        axis_y4 = linear4[:,i].detach().numpy() + non_linear4[:,i].detach().numpy()
        plt.scatter(axis_x, axis_y0[3:5], color = colors[0], label = "Prediction 1", s=150, rasterized = True)
        plt.scatter(axis_x, axis_y1[3:5], color = colors[4], label = "Prediction 2", s=120, rasterized = True)
        plt.scatter(axis_x, axis_y2[3:5], color = colors[8], label = "Prediction 3", s=120, rasterized = True)
        plt.scatter(axis_x, axis_y3[3:5], color = colors[12], label = "Prediction 4", s=120, rasterized = True)
        plt.scatter(axis_x, axis_y4[3:5], color = colors[16], label = "Prediction 5", s=120, rasterized = True)
        plt.title(str(labels_feat[i]))
        plt.legend()
    else:
        axis_x = new_X[:,i]
        axis_y0 = linear0[:,i].detach().numpy() + non_linear0[:,i].detach().numpy()
        axis_y1 = linear1[:,i].detach().numpy() + non_linear1[:,i].detach().numpy()
        axis_y2 = linear2[:,i].detach().numpy() + non_linear2[:,i].detach().numpy()
        axis_y3 = linear3[:,i].detach().numpy() + non_linear3[:,i].detach().numpy()
        axis_y4 = linear4[:,i].detach().numpy() + non_linear4[:,i].detach().numpy()
        plt.plot(axis_x, axis_y0, color = colors[0], label = "Prediction 1", linewidth = 8, rasterized = True)
        plt.plot(axis_x, axis_y1, color = colors[4], label = "Prediction 2", linewidth = 7, rasterized = True)
        plt.plot(axis_x, axis_y2, color = colors[8], label = "Prediction 3", linewidth = 6, rasterized = True)
        plt.plot(axis_x, axis_y3, color = colors[12], label = "Prediction 4", linewidth = 5,rasterized = True)
        plt.plot(axis_x, axis_y4, color = colors[16], label = "Prediction 5", linewidth = 4,rasterized = True)
        y_lim = plt.ylim()
        x_lim = plt.xlim()
        plt.title(str(labels_feat[i]))
        plt.legend()
plt.suptitle("Shape functions Titanic 5 folds", fontweight='bold', fontsize="50")

# Save the figure
plt.savefig("Real_datasets/Classification/Titanic/shape_titanic2.png", format="png", dpi = 200)


