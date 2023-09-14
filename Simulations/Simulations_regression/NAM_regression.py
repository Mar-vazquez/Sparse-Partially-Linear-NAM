#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 12:20:18 2023

@author: marvazquezrabunal
"""


"""
NAM REGRESSION


Description:

Code to fit a NAM to regression data.

"""

###----------------------------------------------------------------------------

### Call libraries
from skorch_regression import NeuralNet_regression
from model_regression import NAM_regression
from sklearn.model_selection import train_test_split
from skorch.dataset import Dataset
import skorch
from skorch.helper import predefined_split
import torch
import pandas as pd
import numpy as np
import random
from sklearn.metrics import mean_squared_error


###----------------------------------------------------------------------------
### Function to fit the NAM

def NAM_fit_reg(X, y):
    """Fit a NAM model to regression data.

    Parameters
    ----------
    X: explanatory data.
    y: response.
        
    Returns
    -------
    Fitted NAM.
    
    """
    module_ = NAM_regression(X.shape[1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    valid_ds = Dataset(X_test, y_test)
    es = skorch.callbacks.EarlyStopping()
    net = NeuralNet_regression(
        module_,
        max_epochs = 100,
        criterion = torch.nn.MSELoss(),
        lr = 0.01,
        lambda1 = 0.0,
        train_split = predefined_split(valid_ds),
        optimizer = torch.optim.Adam,
        callbacks = [es],
        iterator_train__shuffle = True)
    net.fit(X_train, y_train)
    return net

###----------------------------------------------------------------------------
### Fit the model to the simulated data for the different sample sizes.

# Change the seq_names depending on the sample size
#seq_names = ["190", "370", "460", "550", "730", "208", "307", "505", "604", "703", "172", "352", "343", "235", "424"]
#seq_names = ["280", "370", "460", "550", "640", "208", "307", "406", "505", "703", "181", "262", "343", "136", "523"]
seq_names = ["190", "280", "460", "550", "730", "109", "208", "406", "505", "604", "172", "262", "343", "226", "451"]


# Change the n_ depending on the sample size. The example is for n_2000
mse_NAM = []
seed_list = range(1, 1000)
for i in range(len(seq_names)):
    # Read data
    name_train = "results_file/n_2000/train_" + seq_names[i] + '.csv'
    name_test = "results_file/n_2000/test_" + seq_names[i] + '.csv'
    data_train = pd.read_csv(name_train, usecols=range(1,12)).to_numpy()
    data_test = pd.read_csv(name_test, usecols=range(1,12)).to_numpy()
    
    X_train = data_train[:, 0:10]
    y_train = data_train[:, 10:11]
    X_test = data_test[:, 0:10]
    y_test = data_test[:, 10:11]
    
    result = np.vstack((X_train, X_test))
    
    # Fit model and obtain MSE
    seed2 = seed_list[5*i :(5*(i + 1))]
    mse_2 = []
    for j in seed2:
        torch.manual_seed(j)
        np.random.seed(j)    
        random.seed(j)
        NAM_reg = NAM_fit_reg(X_train.astype(np.float32), y_train.astype(np.float32).reshape((y_train.shape[0])))
        pred = NAM_reg.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        mse_2.append(mse)
    mse_NAM.append(mse_2)
    
mse_NAM = np.array(mse_NAM)

# Save results
pd.DataFrame(mse_NAM).to_csv('results_file/n_2000/mse_NAM_2000.csv') 




