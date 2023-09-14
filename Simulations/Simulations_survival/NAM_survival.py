#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 09:46:01 2023

@author: marvazquezrabunal
"""


"""
NAM SURVIVAL


Description:

Code to fit a NAM to survival data.

"""

###----------------------------------------------------------------------------

### Call libraries

from skorch_survival import NeuralNet_survival
from model_survival import NAM_survival, CustomLosstime
from sklearn.model_selection import train_test_split
from skorch.dataset import Dataset
import skorch
from skorch.helper import predefined_split
import torch
import pandas as pd
from functions_survival import metrics
import numpy as np
import random

###----------------------------------------------------------------------------
# Function to fit a NAM on survival data

def NAM_fit_surv(X, y):
    """Fit a NAM model to survival data.

    Parameters
    ----------
    X: explanatory data.
    y: response.
        
    Returns
    -------
    Fitted NAM.
    
    """
    module_ = NAM_survival(X.shape[1]-1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    valid_ds = Dataset(X_test, y_test)
    es = skorch.callbacks.EarlyStopping()
    net = NeuralNet_survival(
        module_,
        max_epochs = 100,
        criterion = CustomLosstime(),
        lr = 0.001,
        lambda1 = 0.0,
        train_split = predefined_split(valid_ds),
        optimizer = torch.optim.Adam,
        callbacks = [es],
        iterator_train__shuffle = True)
    net.fit(X_train, y_train)
    return net


###----------------------------------------------------------------------------
###----------------------------------------------------------------------------

### Fit a NAM on the simulated datasets for n = 1000

seq_names = ["1900", "3700", "4600", "5500", "6400", "2080", "3070", "4060", "5050", "7030", "1810", "2620", "3430", "1360", "5230", "1009", "2008", "3007", "5005", "7003", "1711", "1261", "2332", "2521", "2224"]
cindex_NAM = []
ibs_NAM = []
seed_list = range(1, 1000)
for i in range(len(seq_names)):
    name_train = "results_file/n_1000/train_" + seq_names[i] + '.csv'
    name_test = "results_file/n_1000/test_" + seq_names[i] + '.csv'
    data_train = pd.read_csv(name_train, usecols=range(1,14)).to_numpy()
    data_test = pd.read_csv(name_test, usecols=range(1,14)).to_numpy()
    
    X_train = data_train[:, 0:11]
    y_train = data_train[:, 11:13]
    X_test = data_test[:, 0:11]
    y_test = data_test[:, 11:13]
    
    seed2 = seed_list[5*i :(5*(i + 1))]
    cindex_2 = []
    ibs_2 = []
    for j in seed2:
        torch.manual_seed(j)
        np.random.seed(j)    
        random.seed(j)
        NAM_surv = NAM_fit_surv(X_train, y_train)
        cindex, ibs = metrics(j, NAM_surv, X_train, y_train, X_test, y_test)
        cindex_2.append(cindex)
        ibs_2.append(ibs)
    cindex_NAM.append(cindex_2)
    ibs_NAM.append(ibs_2)
    
cindex_NAM = np.array(cindex_NAM)
ibs_NAM = np.array(ibs_NAM)
    
pd.DataFrame(cindex_NAM).to_csv('results_file/n_1000/cindex_NAM_1000.csv') 
pd.DataFrame(ibs_NAM).to_csv('results_file/n_1000/ibs_NAM_1000.csv') 


###----------------------------------------------------------------------------
### Fit a NAM on the simulated datasets for n = 2000

seq_names = ["1900", "3700", "4600", "5500", "6400", "2080", "3070", "4060", "5050", "7030", "1810", "2620", "3430", "1360", "5230", "1009", "2008", "3007", "5005", "7003", "1711", "1261", "2332", "2521", "2224"]
cindex_NAM = []
ibs_NAM = []
seed_list = range(1, 1000)
for i in range(len(seq_names)):
    name_train = "results_file/n_2000/train_" + seq_names[i] + '.csv'
    name_test = "results_file/n_2000/test_" + seq_names[i] + '.csv'
    data_train = pd.read_csv(name_train, usecols=range(1,14)).to_numpy()
    data_test = pd.read_csv(name_test, usecols=range(1,14)).to_numpy()
    
    X_train = data_train[:, 0:11]
    y_train = data_train[:, 11:13]
    X_test = data_test[:, 0:11]
    y_test = data_test[:, 11:13]
    
    seed2 = seed_list[5*i :(5*(i + 1))]
    cindex_2 = []
    ibs_2 = []
    for j in seed2:
        torch.manual_seed(j)
        np.random.seed(j)    
        random.seed(j)
        NAM_surv = NAM_fit_surv(X_train, y_train)
        cindex, ibs = metrics(j, NAM_surv, X_train, y_train, X_test, y_test)
        cindex_2.append(cindex)
        ibs_2.append(ibs)
    cindex_NAM.append(cindex_2)
    ibs_NAM.append(ibs_2)
    
cindex_NAM = np.array(cindex_NAM)
ibs_NAM = np.array(ibs_NAM)
    
pd.DataFrame(cindex_NAM).to_csv('results_file/n_2000/cindex_NAM_2000.csv') 
pd.DataFrame(ibs_NAM).to_csv('results_file/n_2000/ibs_NAM_2000.csv') 
    
