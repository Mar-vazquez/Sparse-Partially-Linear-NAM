#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 11:12:31 2023

@author: marvazquezrabunal
"""


"""
FUNCTIONS SIMULATION SURVIVAL


Description:

Functions to do CV, find the optimal hyperparameters, fit the final model and
obtain the evaluation metrics in the simulated datasets.

"""

###----------------------------------------------------------------------------

### Call libraries

import numpy as np
import torch
import random
from cross_validation_survival import CrossValidation_survival, final_model_survival
from simulationdata_survival import simulation_survival
from joblib import Parallel, delayed
from model_survival import model_survival, CustomLosstime
from sklearn.preprocessing import StandardScaler
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pycox.evaluation import EvalSurv
import pandas as pd
from sksurv.metrics import integrated_brier_score

###----------------------------------------------------------------------------

### Class to create a torch dataset and scale and center the X data
class SimulationDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, scale_data=True):
      if not torch.is_tensor(X) and not torch.is_tensor(y):
        if scale_data:
            X = StandardScaler().fit_transform(X)
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

###----------------------------------------------------------------------------

### Function to obtain the optimal hyperparameters from the CV results

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

### Function to fit the final model given the optimal hyperparameters

def final_model(seed, X, y, opt_param):
    """Fit the final model using the optimal hyperparameters.

    Parameters
    ----------
    seed: random seed.
    X: data with the explanatory variables.
    y: response.
    opt_param: list with the optimal hyperparameters from the CV.
        
    Returns
    -------
    Final fitted model.
    
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

### Function to obtain the structures found and the proportion of wrong ones

def wrong_classif_survival(nfeat, res, rest):
    """Obtain the type of structure of each feature and the proportion of 
    wrongly found ones.
    
    Parameters
    ----------
    nfeat: number of explanatory features in the data.
    res: final fitted model.
    rest: list with the information from the simulated data. 
        
    Returns
    -------
    Proportion of wrongly found structure and list with the features
    that were found in each category. 
    
    """
    
    # Find the features belonging to each structure category
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

    elem = rest[0]
    ind = rest[1]

    real_sparse = ind[0:elem[0]]
    real_lin = ind[elem[0]:elem[0] + elem[1]]
    real_non_lin = ind[elem[0] + elem[1]: elem[0] + elem[1] + elem[2]]
    real_time = ind[sum(elem[0:3]):]

    # Find the proportion of wrongly found structure
    wrong_classif = 0
    for i in sparse:
        if i not in real_sparse:
            wrong_classif += 1
    for i in lin:
        if i not in real_lin:
            wrong_classif += 1
    for i in non_lin:
        if i not in real_non_lin:
            wrong_classif += 1
    for i in time:
        if i not in real_time:
            wrong_classif += 1

    wrong_classif /= nfeat
    
    return wrong_classif, [sparse, lin, non_lin, time]


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
        
    # Obtain H for the test data
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

def metrics(seed, model, x_train, y_train, x_test, y_test):
    """Function that obtains the C-index and the IBS from a fitted model.

    Parameters
    ----------
    seed: initial random seed.
    model: fitted model.
    x_train: training explanatory features.
    y_train: training response.
    x_test: test explanatory features.
    y_test: test response.
        
    Returns
    -------
    List with the C-index and the IBS.
    
    """
    random.seed(seed)
    
    # Sample 100 observations from train set
    ind_sort = np.argsort(x_train[:,-1])
    ind_train = ind_sort[0::int(len(ind_sort)/100)]
    
    # Observations from test set (we use the ones with time larger than
    # the minimum training one because of an error we werer obtaining with the
    # C-index or the IBS)
    ind_test2 = [j for j, x in enumerate(x_test[:,-1]) if x > min(x_train[ind_train,-1])]
    x_test2 = x_test[ind_test2,:]
    y_test2 = y_test[ind_test2,:]
    
    # Test survival times under the 70% percentile
    time_interest2 = sorted([x for x in y_test2[:,0] if x <= np.percentile(y_test2[:,0], 70)])
    
    # Survival probabilities for the test individuals at test times
    surv_curv3, survcurv4 = survival_curve2(model, x_train[ind_train,:], y_train, x_test2)
    
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
    IBS = integrated_brier_score(y_array_train, y_array_test, survcurv4.transpose(), time_interest2)
    
    return [cindex, IBS]
    
###----------------------------------------------------------------------------

###  Function to fit the model and obtain the results

def simulation_CV_surv(nfeat, prop, size, seed, seed2):
    """Function that obtains the simulated data, applies CV, fits the final 
    model and obtains the evaluation metrics.

    Parameters
    ----------
    nfeat: number of explanatory features in the data.
    prop: proportion of sparse/linear/non-linear/time features.
    size: sample size of the data.
    seed: initial random seed.
    seed2: list of random seeds with which we fit the final model.
        
    Returns
    -------
    List with the information about the results of the simulation experiment.
    We return the seeds, the simulated data, the info about this simulated data,
    the lr and alphas used in CV, the result of CV, the optimal hyperparameters,
    the C-index, the IBS, the wrong proportion found and the structure of each
    feature for the final models.
    
    """
    
    np.random.seed(seed)    
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Simulate the data and transform it
    X, y, rest = simulation_survival(nfeat, prop, size)
    dataset = SimulationDataset(X, y, scale_data=True)
    X2 = dataset.X.numpy().astype(np.float32)
    y2 = dataset.y.numpy().astype(np.float32)
    
    # Do the CV
    lr_array = np.array([0.001, 0.005, 0.01])
    alpha_array = np.array([0.01, 0.1])


    alpha_list= list(np.tile(alpha_array, len(lr_array)))
    lr_list = list(np.repeat(lr_array, len(alpha_array)))

    res_zip = list(zip(lr_list, alpha_list))
    
    X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size = 0.1, random_state = seed)

    example = CrossValidation_survival(model_survival, max_epochs = 50, criterion = CustomLosstime(),
                                        optimizer = torch.optim.Adam, gamma = 0.5)

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
    
    # Obtain the optimal hyperparameters
    opt_param = optimal_param(sol)
    
    # Fit the final models and obtain the evaluation metrics
    wrong_list =[]
    classif_list = []
    cindex_list = []
    ibs_list = []
    
    for j in seed2:
        final_model_res = final_model(j, X_train, y_train, opt_param)
        wrong_classif_surv, classif = wrong_classif_survival(nfeat, final_model_res, rest)
        cindex, ibs = metrics(j, final_model_res, X_train, y_train, X_test, y_test)
        wrong_list.append(wrong_classif_surv)
        classif_list.append(classif)
        cindex_list.append(cindex)
        ibs_list.append(ibs)
    
    info = [seed, seed2, X2, y2, rest, res_zip, sol, opt_param, cindex_list, ibs_list, wrong_list, classif_list]

    return info


