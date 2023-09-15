#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 10:35:54 2023

@author: marvazquezrabunal
"""


"""
FUNCTIONS REAL DATA SURVIVAL


Description:

Code with the functions to apply our model to real-world datasets, doing CV,
obtaining the optimal hyperparameters and the evaluation metrics. It also contains
the functions to fit the NAM and the sparse NAM model.

"""


###----------------------------------------------------------------------------

### Call libraries
import sys
sys.path.append('/Users/marvazquezrabunal/Library/Mobile Documents/com~apple~CloudDocs/ETH/Spring 2023/Thesis/Simulations/Survival')
import numpy as np
import torch
import random
from cross_validation_survival import CrossValidation_survival, final_model_survival
from joblib import Parallel, delayed
from model_survival import model_survival, CustomLosstime, sparse_NAM_survival
import time
import matplotlib.pyplot as plt
from pycox.evaluation import EvalSurv
import pandas as pd
from sksurv.metrics import integrated_brier_score
from cross_validation_notime_survival import CrossValidation_notime_survival, final_model_notime_survival
from skorch_survival import NeuralNet_survival
from model_survival import NAM_survival
from sklearn.model_selection import train_test_split
from skorch.dataset import Dataset
import skorch
from skorch.helper import predefined_split


###----------------------------------------------------------------------------
### Given the full data obtain the transformed data
class SimulationDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, binary):
      if not torch.is_tensor(X) and not torch.is_tensor(y):
        for i in range(X.shape[1]):
            if i not in binary:
                X[:, i] = (X[:,i] - X[:,i].mean())/X[:,i].std()
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
    
###----------------------------------------------------------------------------
### Given the train and test data let's do cross validation 

def fit_model_surv(X_train, y_train, seed):
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
    alpha_array = np.array([0.01, 0.1])
   

    alpha_list= list(np.tile(alpha_array, len(lr_array)))
    lr_list = list(np.repeat(lr_array, len(alpha_array)))

    res_zip = list(zip(lr_list, alpha_list))
    
    example = CrossValidation_survival(model_survival, max_epochs = 50, criterion = CustomLosstime(),
                                        optimizer = torch.optim.Adam, gamma = 0.5, seed = seed)

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
### Find the optimal hyperparameters from the cross validation
    
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
    alpha_array = np.array([0.01,  0.1])


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
    limit = err_list.min() + err_std_list[index_min]/np.sqrt(3)
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
    Fitted model.
    
    """
    np.random.seed(seed)    
    random.seed(seed)
    torch.manual_seed(seed)
    lr_opt = opt_param[0]
    alpha_opt = opt_param[1]
    lambda_opt = opt_param[2]
    val = final_model_survival(model_survival, max_epochs = 50, criterion = CustomLosstime(), alpha = alpha_opt, lr = lr_opt, optimizer = torch.optim.Adam, gamma = 0.5)
    res = val.fit(X, y, lambda_opt)
    return res

###----------------------------------------------------------------------------

### Obtain the structure found

def structure_found(nfeat, res):
    """Obtain which feature belongs to each structure category (sparse/linear/
    non-linear/time)

    Parameters
    ----------
    nfeat: number of explanatory features.
    res: fitted model.
        
    Returns
    -------
    List with the indices of the features that belong to the sparse/linear/
    non-linear/time structure category.
    
    """
    time = []
    non_lin = []
    lin = []
    sparse = []
    for i in range(nfeat):
        sj = 0
        for param in res.module_.non_linear_time[i].parameters():
            sj += param.sum()
        if sj != 0:
            time.append(i)
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
    
    for j in time:
        non_lin.remove(j)
    return [sparse, lin, non_lin, time]

###----------------------------------------------------------------------------

### Function to obtain the survival probabilities for test individuals at test times.


def survival_curve2(model, x, y, x_try):
    """Function to obtain survival probabilities for test subjects at test times.

    Parameters
    ----------
    model: fitted model.
    x: training explanatory features to estimate H0.
    y: training response to estimate H0.
    x_try: test explanatory features.
        
    Returns
    -------
    Predicted survival probabilities for the test individuals at all testing
    times and at the ones that are smaller than the 70% percentile.
    
    """
    # Obtain Delta H0 using the training data
    x = torch.from_numpy(x)
    time = x[:, -1]
    idx = time.sort(descending = True)[1]
    events = y[idx, 1]
    x_sorted = x[idx, :]
    time_sorted = time[idx]
    events1 = [j for j, x in enumerate(events) if x == 1]
    time_noncens = time_sorted[events1]
    delta = []
    for i in range(len(time_noncens)):
        r_i = [j for j,x in enumerate(time_sorted) if x >= time_noncens[i]]
        new_x2 = x_sorted[r_i, :]
        new_x2[:, -1] = time_noncens[i]
        pred_j = np.exp(model.predict(new_x2))
        delta_i = 1/(sum(pred_j))
        delta.append(delta_i)
        
    ### Obtain H for the test data
    t_try = x_try[:,-1]
    time_interest_ind = ([j for j,x in enumerate(t_try) if x <= np.percentile(t_try, 70)])
    time_interest = ([x for j,x in enumerate(t_try) if x <= np.percentile(t_try, 70)])
    res = np.empty((len(t_try),0))
    for i in range(len(t_try)):
        surv_curve = []
        for ind_star in range(len(t_try)):
            ind_time = [j for j,x in enumerate(time_noncens) if x <= t_try[ind_star]]
            if len(ind_time) == 0:
                h = 0.00001
            else:
                x_star_rep = np.tile(x_try[i,:], (len(ind_time), 1))
                x_star_rep[:,-1] = time_noncens[ind_time]
                pred = np.exp(model.predict(x_star_rep))
                delta_i = [delta[i] for i in ind_time]
                h = sum(pred*delta_i)
            # Obtain survival probability
            surv_curve.append(np.exp(-h))
        curve2 = np.array(surv_curve).reshape(len(t_try), 1)
        res = np.hstack((res, curve2))
    res2 = res[time_interest_ind,:]
    res2 = res2[np.argsort(time_interest),:]
    return res, res2

###----------------------------------------------------------------------------

### Function to obtain the C-index and the IBS

def metrics(model, X_train, y_train, X_test, y_test):
    """Function that obtains the C-index and the IBS from a fitted model.

    Parameters
    ----------
    model: fitted model.
    X_train: training explanatory features.
    y_train: training response.
    X_test: test explanatory features.
    y_test: test response.
        
    Returns
    -------
    List with the C-index, the IBS and the predicted survival probabilities.
    
    """
    # Sample 100 observations from train set
    ind_sort = np.argsort(X_train[:,-1])
    ind_train = ind_sort[0::int(len(ind_sort)/100)]
    
    # Observations from test set (we use the ones with time larger than
    # the minimum training one and smaller than the maximum one because of
    # an error we were obtaining with the C-index or the IBS)
    ind_test2 = [j for j, x in enumerate(X_test[:,-1]) if x >= min(X_train[ind_train,-1]) and x < max(X_train[:,-1])]
    x_test2 = X_test[ind_test2,:]
    y_test2 = y_test[ind_test2,:]
    
    # Test survival times under the 70% percentile
    time_interest2 = sorted([x for x in y_test2[:,0] if x <= np.percentile(y_test2[:,0], 70)])
    
    # Survival probabilities for the test individuals at test times
    surv_curv3, survcurv4 = survival_curve2(model, X_train[ind_train,:], y_train, x_test2)
    
    # Obtain C-index
    eval_surv = EvalSurv(pd.DataFrame(surv_curv3), y_test2[:,0], y_test2[:,1], censor_surv='km')
    cindex = eval_surv.concordance_td('adj_antolini')
    
    # Obtain IBS
    add_list = []
    for idx in range(y_train.shape[0]):
        add_list.append((y_train[idx,1], y_train[idx,0]))    
    y_array_train = np.array(add_list, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
    add_list_test = []
    for idx in range(y_test2.shape[0]):
        add_list_test.append((y_test2[idx,1], y_test2[idx,0])) 
    y_array_test = np.array(add_list_test, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
    uniq_ind = np.unique(time_interest2, return_index=True)[1].astype(int)
    IBS = integrated_brier_score(y_array_train, y_array_test, survcurv4.transpose()[:, uniq_ind.astype(int)],  np.array(time_interest2)[uniq_ind.astype(int)])
    return [cindex, IBS, surv_curv3]
    

###----------------------------------------------------------------------------

### Function to fit the model and obtain the relevant information

def results_data_surv(X_train, y_train, X_test, y_test, seed):
    """Do CV, obtain optimal hyperparameters, fit final model, obtain structure
    of the features, C-index and IBS.

    Parameters
    ----------
    X_train: train explanatory features.
    y_train: train response variable.
    X_test: test explanatory features.
    y_test: test response variable.
    seed: random seed.
        
    Returns
    -------
    List with the optimal hyperparameters, structure obtained, C-index and IBS.
    
    """
    model = fit_model_surv(X_train, y_train, seed)
    opt_param = optimal_param(model)
    final_model_res = final_model(X_train, y_train, opt_param, seed)
    structure = structure_found(X_train.shape[1] - 1, final_model_res)
    cindex, ibs, pred = metrics(final_model_res, X_train, y_train, X_test, y_test)
    return [pred, opt_param, structure, cindex, ibs]

###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
"""
FUNCTIONS TO FIT THE NAM
"""

### Fit the NAM model

def NAM_fit_surv(X, y):
    """Fit NAM on the data.

    Parameters
    ----------
    X: explanatory features.
    y: response variable.

        
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

### Function to obtain all the info of the NAM 

def results_data_NAM_surv(X_train, y_train, X_test, y_test):
    """Fit NAM and obtain C-index and IBS.

    Parameters
    ----------
    X_train: train explanatory features.
    y_train: train response variable.
    X_test: test explanatory features.
    y_test: test response variable.
        
    Returns
    -------
    Predicted survival probabilities, C-index and IBS.
    
    """
    model = NAM_fit_surv(X_train, y_train)
    cindex, ibs, pred = metrics(model, X_train, y_train, X_test, y_test)
    return [pred, cindex, ibs]



###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
"""
FUNCTIONS TO FIT THE SPARSE NAM
"""


### Fit sparse NAM model


def sparse_NAM_surv(X_train, y_train, seed):
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
    

    example = CrossValidation_notime_survival(sparse_NAM_survival, max_epochs = 50, criterion = CustomLosstime(),
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
    limit = err_list.min() + err_std_list[index_min]/np.sqrt(3)
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

### Fit the sparse NAM given the optimal hyperparameters

def final_model_sparse(X, y, opt_param, seed):
    """Given the optimal hyperparameters, fit the final sparse NAM model.
    
    Parameters
    ----------
    X: explanatory variables.
    y: response.
    opt_param: optimal hyperparameters.
    seed: random seed.
        
    Returns
    -------
    Fitted model.
    
    """
    np.random.seed(seed)    
    random.seed(seed)
    torch.manual_seed(seed)
    lr_opt = opt_param[0]
    alpha_opt = opt_param[1]
    lambda_opt = opt_param[2]
    val = final_model_notime_survival(sparse_NAM_survival, max_epochs = 50, criterion = CustomLosstime(), alpha = alpha_opt, lr = lr_opt, optimizer = torch.optim.Adam)
    res = val.fit(X, y, lambda_opt)
    return res

###----------------------------------------------------------------------------

### Function to obtain the structure found by the sparse NAM

def structure_sparse(nfeat, res):
    """Obtain which feature belongs to each structure category (sparse/linear/
    non-linear)

    Parameters
    ----------
    nfeat: number of explanatory features.
    res: fitted model.
        
    Returns
    -------
    List with the indices of the features that belong to the sparse/linear/
    non-linear structure category.
    
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

### Function to fit the sparse NAM model and obtain the relevant information

def results_sparse_NAM_surv(X_train, y_train, X_test, y_test, seed):
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
    List with the predicted survival probabilities, optimal hyperparameters,
    structure obtained, C-index and IBS.
    
    """
    model = sparse_NAM_surv(X_train, y_train, seed)
    opt_param_sparse = optimal_param_sparse(model)
    final_model_spars = final_model_sparse(X_train, y_train, opt_param_sparse, seed)
    structure = structure_sparse((X_train.shape[1]-1), final_model_spars)
    cindex, ibs, pred = metrics(final_model_spars, X_train, y_train, X_test, y_test)
    return [pred, opt_param_sparse, structure, cindex, ibs]


