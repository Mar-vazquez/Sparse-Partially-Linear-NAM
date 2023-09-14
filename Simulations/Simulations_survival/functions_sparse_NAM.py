#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 17:26:15 2023

@author: marvazquezrabunal
"""


"""
FUNCTIONS SPARSE NAM SURVIVAL


Description:

Code with auxiliar functions to fit the sparse NAM model for survival data and
obtain the evaluation metrics.

"""

###----------------------------------------------------------------------------

### Call libraries
import numpy as np
import matplotlib.pyplot as plt 
import torch
import random
from cross_validation_notime_survival import final_model_notime_survival
from model_survival import sparse_NAM_survival, CustomLosstime

###----------------------------------------------------------------------------

# Fit final sparse NAM model given the optimal hyperparameters

def final_model_sparse(seed, X, y, opt_param):
    """Fit the final sparse NAM model using the optimal hyperparameters.

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
    val = final_model_notime_survival(sparse_NAM_survival, max_epochs = 50, criterion = CustomLosstime(), alpha = alpha_opt, lr = lr_opt, optimizer = torch.optim.Adam)
    res = val.fit(X, y, lambda_opt)
    return res

###----------------------------------------------------------------------------

### Function to obtain the optimal hyperparameters from the CV results

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

### Function to obtain the structures found and the proportion of wrong ones

def wrong_classif_survival_sparse(nfeat, res, rest):
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
    

    elem = rest[0]
    ind = rest[1]

    real_sparse = ind[0:elem[0]]
    real_lin = ind[elem[0]:elem[0] + elem[1]]
    real_non_lin = ind[elem[0] + elem[1]: elem[0] + elem[1] + elem[2]]

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

    wrong_classif /= nfeat
    
    return wrong_classif, [sparse, lin, non_lin]

