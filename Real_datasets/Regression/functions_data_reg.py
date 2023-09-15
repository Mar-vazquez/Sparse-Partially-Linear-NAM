#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 11:42:17 2023

@author: marvazquezrabunal
"""


"""
FUNCTIONS REAL DATA REGRESSION


Description:

Code with the functions to apply our model to real-world datasets, doing CV,
obtaining the optimal hyperparameters and the evaluation metrics. It also contains
the functions to fit the NAM and the sparse NAM model.

"""


###----------------------------------------------------------------------------

### Call libraries
import sys
sys.path.append('/Users/marvazquezrabunal/Library/Mobile Documents/com~apple~CloudDocs/ETH/Spring 2023/Thesis/Simulations/Regression')
import numpy as np
import torch
import random
from cross_validation_regression import CrossValidation_regression, final_model_regression
from joblib import Parallel, delayed
from model_regression import model_regression
from sklearn.preprocessing import StandardScaler
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from skorch_regression import NeuralNet_regression
from model_regression import NAM_regression
from sklearn.model_selection import train_test_split
from skorch.dataset import Dataset
import skorch
from skorch.helper import predefined_split


###----------------------------------------------------------------------------
### Given the full data obtain the transformed data
class SimulationDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, scale_data=True):
      if not torch.is_tensor(X) and not torch.is_tensor(y):
        if scale_data:
            X = StandardScaler().fit_transform(X)
            y = y - y.mean()
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


###----------------------------------------------------------------------------
### Given the train and test data let's do cross validation 

def fit_model_reg(X_train, y_train, seed):
    """Given the training data, do CV on lr, alpha and lambda.

    Parameters
    ----------
    X_train: train explanatory variables.
    y_train: train response variable.
    seed: random seed.
    
    Returns
    -------
    Result of the CV.

    """
    lr_array = np.array([0.001, 0.005, 0.01])
    alpha_array = np.array([0.01, 0.05, 0.1])

    alpha_list= list(np.tile(alpha_array, len(lr_array)))
    lr_list = list(np.repeat(lr_array, len(alpha_array)))

    res_zip = list(zip(lr_list, alpha_list))
    
    example = CrossValidation_regression(model_regression, max_epochs = 50, criterion = torch.nn.MSELoss(),
                                        optimizer = torch.optim.Adam, seed = seed)

    jobs = (
            delayed(example.cross_validation)(
                X_train, y_train, nfolds= 3, nlambdas = 30,
                alpha= alpha,
                lr = lr
            )
            for lr, alpha in res_zip
        )

    start = time.time()
    sol = Parallel(n_jobs=-1)(jobs)
    end = time.time()
    print((end - start)/60)
    return sol


###----------------------------------------------------------------------------
### Let's find the optimal hyperparameters from the cross validation

def optimal_param(sol):
    """Obtain optimal hyperparameters from the CV results and plot the CV error
    curve.

    Parameters
    ----------
    sol: list of results obtained after fitting the models doing CV.
        
    Returns
    -------
    List with the optimal learning rate, alpha and lambda
    
    """
    # Obtain the optimal hyperparameters
    lr_array = np.array([0.001, 0.005, 0.01])
    alpha_array = np.array([0.01, 0.05, 0.1])

    alpha_list= list(np.tile(alpha_array, len(lr_array)))
    lr_list = list(np.repeat(lr_array, len(alpha_array)))

    res_zip = list(zip(lr_list, alpha_list))
    
    lr_list = np.empty(0)
    err_list = np.empty(0)
    zero_list = np.empty(0)
    lamb_list = np.empty(0)
    alpha_list = np.empty(0)
    err_std_list = np.empty(0)
    for i in range(len(res_zip)):
        lr_i = res_zip[i][0]
        alpha_i = res_zip[i][1]
        err_i, lamb_i, zero_i = sol[i]
        lambda_error = err_i.mean(axis = 1)
        lambda_error_std = err_i.std(axis = 1)
        lambda_zero = zero_i.mean(axis = 1)
        alpha_vect = np.repeat(alpha_i, len(lambda_zero))
        lr_vect = np.repeat(lr_i,len(lambda_zero))
        err_list= np.concatenate((err_list, lambda_error))
        err_std_list= np.concatenate((err_std_list, lambda_error_std))
        zero_list= np.concatenate((zero_list, lambda_zero))
        lamb_list = np.concatenate((lamb_list, lamb_i))
        alpha_list = np.concatenate((alpha_list, alpha_vect))
        lr_list = np.concatenate((lr_list, lr_vect))

    index_min = np.argmin(err_list)
    limit = lambda_error.min() + err_std_list[index_min]/np.sqrt(3)
    list_index = [j for j,x in enumerate(err_list) if x <= limit]
    opt_ind = list_index[np.argmax(zero_list[list_index])]
    lambda_opt = lamb_list[opt_ind]
    alpha_opt = alpha_list[opt_ind]
    lr_opt = lr_list[opt_ind]

    # Plot the CV error curve
    l = 0
    for i in range(len(res_zip)):
        lr_i = res_zip[i][0]
        alpha_i = res_zip[i][1]
        ind_i = [k for k,x in enumerate(lr_list) if x == lr_i]
        ind_j = [k for k,x in enumerate(alpha_list) if x == alpha_i]
        lambda_ij = [x for k,x in enumerate(lamb_list) if k in ind_i and k in ind_j]
        error_ij = [x for k,x in enumerate(err_list) if k in ind_i and k in ind_j] 
        std_ij = [x for k,x in enumerate(err_std_list) if k in ind_i and k in ind_j] 
        plt.subplot(len(lr_array), len(alpha_array), l+1)
        plt.errorbar(np.log(lambda_ij), error_ij,  yerr = std_ij, marker = 'o', linestyle = " ",  linewidth = 2, capsize = 13)
        plt.xlabel("log(lambda)")
        plt.ylabel("CV error")
        plt.title("lr = " + str(lr_i) + " and alpha = " + str(alpha_i))
        l = l + 1
        
    return [lr_opt, alpha_opt, lambda_opt]


###----------------------------------------------------------------------------

### Fit the final model given the optimal hyperparameters

def final_model(X, y, opt_param, seed):
    """Given the optimal hyperparameters, fit the final model.
    
    Parameters
    ----------
    X: explanatory variables.
    y: response.
    opt_param: optimal hyperparameters.
    seed: random seed.
        
    Returns
    -------
    List with the optimal learning rate, alpha and lambda
    
    """
    np.random.seed(seed)    
    random.seed(seed)
    torch.manual_seed(seed)
    lr_opt = opt_param[0]
    alpha_opt = opt_param[1]
    lambda_opt = opt_param[2]
    val = final_model_regression(model_regression, max_epochs = 50, criterion = torch.nn.MSELoss, alpha = alpha_opt, lr = lr_opt, optimizer = torch.optim.Adam)
    res = val.fit(X, y, lambda_opt)
    return res


###----------------------------------------------------------------------------

### Obtain the structure found

def structure_found(nfeat, res):
    """Obtain which feature belongs to each structure category (sparse/linear/
    non-linear)

    Parameters
    ----------
    nfeat: number of explanatory features.
    res: fitted model.
        
    Returns
    -------
    List with the indices of the features that belong to the sparse/linear/non-linear
    structure category.
    
    """
    non_lin = []
    lin = []
    sparse = []
    for i in range(nfeat):
        si = 0
        for param in res.module_.non_linear[i].parameters():
            si += param.sum()
        if si == 0:
            lin.append(i)
        else:
            non_lin.append(i)
        for param in res.module_.linear[i].parameters():
            if param == 0:
                sparse.append(i)
            
    for j in sparse:
        lin.remove(j)
    return [sparse, lin, non_lin]


###----------------------------------------------------------------------------

### Function to fit the model and obtain the relevant information

def results_data_reg(X_train, y_train, X_test, y_test, seed):
    """Do CV, obtain optimal hyperparameters, fit final model, obtain structure
    of the features and MSE.

    Parameters
    ----------
    X_train: train explanatory features.
    y_train: train response variable.
    X_test: test explanatory features.
    y_test: test response variable.
    seed: random seed.
        
    Returns
    -------
    List with the optimal hyperparameters, structure obtained and MSE.
    
    """
    model = fit_model_reg(X_train, y_train, seed)
    opt_param = optimal_param(model)
    final_model_res = final_model(X_train, y_train, opt_param, seed)
    structure = structure_found(X_train.shape[1], final_model_res)
    pred = final_model_res.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    return [opt_param, structure, mse]

###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
"""
FUNCTIONS TO FIT THE NAM
"""

### Fit the NAM model

def NAM_fit_reg(X, y):
    """Fit NAM on the data.

    Parameters
    ----------
    X: explanatory features.
    y: response variable.

        
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

### Function to obtain all the info of the NAM 

def results_data_NAM_reg(X_train, y_train, X_test, y_test):
    """Fit NAM and obtain MSE.

    Parameters
    ----------
    X_train: train explanatory features.
    y_train: train response variable.
    X_test: test explanatory features.
    y_test: test response variable.
        
    Returns
    -------
    MSE.
    
    """
    model = NAM_fit_reg(X_train, y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    return mse


###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
"""
FUNCTIONS TO FIT THE sparse NAM
"""


### Fit sparse NAM model

def sparse_NAM_reg(X_train, y_train, seed):
    """For the sparses NAM model do CV of lr and lambda.

    Parameters
    ----------
    X_train: train explanatory features.
    y_train: train response variable.
    seed: random seed.
        
    Returns
    -------
    Result of the CV.
    
    """
    
    lr_array = np.array([0.001, 0.005, 0.01])
    alpha_array = np.array([1])
    

    alpha_list= list(np.tile(alpha_array, len(lr_array)))
    lr_list = list(np.repeat(lr_array, len(alpha_array)))

    res_zip = list(zip(lr_list, alpha_list))
    

    example = CrossValidation_regression(model_regression, max_epochs = 50, criterion = torch.nn.MSELoss(),
                                        optimizer = torch.optim.Adam, seed = seed)

    jobs = (
            delayed(example.cross_validation)(
                X_train, y_train, nfolds= 3, nlambdas = 30,
                alpha= alpha,
                lr = lr
            )
            for lr, alpha in res_zip
        )

    start = time.time()
    sol = Parallel(n_jobs=-1)(jobs)
    end = time.time()
    print((end - start)/60)
    
    return sol

###----------------------------------------------------------------------------

### Find the optimal hyperparameters from the cross validation for sparse NAM

def optimal_param_sparse(sol):
    """Obtain optimal hyperparameters from the CV results and plot the CV error
    curve.

    Parameters
    ----------
    sol: list of results obtained after fitting the models doing CV.
        
    Returns
    -------
    List with the optimal learning rate, alpha and lambda
    
    """
    # Obtain the optimal hyperparameters
    lr_array = np.array([0.001, 0.005, 0.01])
    alpha_array = np.array([1])

    alpha_list= list(np.tile(alpha_array, len(lr_array)))
    lr_list = list(np.repeat(lr_array, len(alpha_array)))

    res_zip = list(zip(lr_list, alpha_list))
    
    lr_list = np.empty(0)
    err_list = np.empty(0)
    zero_list = np.empty(0)
    lamb_list = np.empty(0)
    alpha_list = np.empty(0)
    err_std_list = np.empty(0)
    for i in range(len(res_zip)):
        lr_i = res_zip[i][0]
        alpha_i = res_zip[i][1]
        err_i, lamb_i, zero_i = sol[i]
        lambda_error = err_i.mean(axis = 1)
        lambda_error_std = err_i.std(axis = 1)
        lambda_zero = zero_i.mean(axis = 1)
        alpha_vect = np.repeat(alpha_i, len(lambda_zero))
        lr_vect = np.repeat(lr_i,len(lambda_zero))
        err_list= np.concatenate((err_list, lambda_error))
        err_std_list= np.concatenate((err_std_list, lambda_error_std))
        zero_list= np.concatenate((zero_list, lambda_zero))
        lamb_list = np.concatenate((lamb_list, lamb_i))
        alpha_list = np.concatenate((alpha_list, alpha_vect))
        lr_list = np.concatenate((lr_list, lr_vect))

    index_min = np.argmin(err_list)
    limit = lambda_error.min() + err_std_list[index_min]/np.sqrt(3)
    list_index = [j for j,x in enumerate(err_list) if x <= limit]
    opt_ind = list_index[np.argmax(zero_list[list_index])]
    lambda_opt = lamb_list[opt_ind]
    alpha_opt = alpha_list[opt_ind]
    lr_opt = lr_list[opt_ind]

    # Plot the CV error curve
    l = 0
    for i in range(len(res_zip)):
        lr_i = res_zip[i][0]
        alpha_i = res_zip[i][1]
        ind_i = [k for k,x in enumerate(lr_list) if x == lr_i]
        ind_j = [k for k,x in enumerate(alpha_list) if x == alpha_i]
        lambda_ij = [x for k,x in enumerate(lamb_list) if k in ind_i and k in ind_j]
        error_ij = [x for k,x in enumerate(err_list) if k in ind_i and k in ind_j] 
        std_ij = [x for k,x in enumerate(err_std_list) if k in ind_i and k in ind_j] 
        plt.subplot(len(lr_array), len(alpha_array), l+1)
        plt.errorbar(np.log(lambda_ij), error_ij,  yerr = std_ij, marker = 'o', linestyle = " ",  linewidth = 2, capsize = 13)
        plt.xlabel("log(lambda)")
        plt.ylabel("CV error")
        plt.title("lr = " + str(lr_i) + " and alpha = " + str(alpha_i))
        l = l + 1
        
    return [lr_opt, alpha_opt, lambda_opt]

###----------------------------------------------------------------------------

### Function to fit the sparse NAM model and obtain the relevant information

def results_sparse_NAM_reg(X_train, y_train, X_test, y_test, seed):
    """Do CV, obtain optimal hyperparameters, fit final model of sparse NAM,
    obtain structure of the features and MSE.

    Parameters
    ----------
    X_train: train explanatory features.
    y_train: train response variable.
    X_test: test explanatory features.
    y_test: test response variable.
    seed: random seed.
        
    Returns
    -------
    List with the optimal hyperparameters, structure obtained and MSE.
    
    """
    model = sparse_NAM_reg(X_train, y_train, seed)
    opt_param_sparse = optimal_param_sparse(model)
    final_model_sparse = final_model(X_train, y_train, opt_param_sparse, seed)
    structure = structure_found(X_train.shape[1], final_model_sparse)
    pred = final_model_sparse.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    return [opt_param_sparse, structure, mse]




