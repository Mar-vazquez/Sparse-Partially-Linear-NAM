#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 16:06:39 2023

@author: marvazquezrabunal
"""

from model_regression import model_regression
import torch
import pandas as pd
import numpy as np
import random
from sklearn.metrics import mean_squared_error
from cross_validation_regression import CrossValidation_regression
from joblib import Parallel, delayed
import time
from functions_regression import wrong_classif_regression, final_model
import pickle
import matplotlib.pyplot as plt

### SPARSE NAM REGRESSION

## Code to fit the sparse NAM

###----------------------------------------------------------------------------
### Function to obtain the optimal hyperparameters

def optimal_param(sol):
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
    limit = lambda_error.min() + err_std_list[index_min]/np.sqrt(5)
    list_index = [j for j,x in enumerate(err_list) if x <= limit]
    opt_ind = list_index[np.argmax(zero_list[list_index])]
    lambda_opt = lamb_list[opt_ind]
    alpha_opt = alpha_list[opt_ind]
    lr_opt = lr_list[opt_ind]

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
### Function to fit the sparse NAM 

def sparse_NAM_fit_regression(X_train, y_train, X_test, y_test, seed, rest, seed2):
    np.random.seed(seed)    
    random.seed(seed)
    torch.manual_seed(seed)
    
    lr_array = np.array([0.001, 0.005, 0.01])
    alpha_array = np.array([1])
    
    # lr_array = np.array([0.005, 0.01])
    # alpha_array = np.array([0.05, 0.1])

    alpha_list= list(np.tile(alpha_array, len(lr_array)))
    lr_list = list(np.repeat(lr_array, len(alpha_array)))

    res_zip = list(zip(lr_list, alpha_list))
    

    example = CrossValidation_regression(model_regression, max_epochs = 50, criterion = torch.nn.MSELoss(),
                                        optimizer = torch.optim.Adam, seed = seed)

    jobs = (
            delayed(example.cross_validation)(
                X_train, y_train, nfolds= 5, nlambdas = 30,
                alpha= alpha,
                lr = lr
            )
            for lr, alpha in res_zip
        )

    start = time.time()
    sol = Parallel(n_jobs=-1)(jobs)
    end = time.time()
    print((end - start)/60)
    
    
    opt_param = optimal_param(sol)
    wrong_list =[]
    mse_list = []
    
    for j in seed2:
        final_model_res = final_model(j, X_train, y_train, opt_param)
        wrong_classif_reg = wrong_classif_regression(rest[-3], final_model_res, rest)
        pred = final_model_res.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        mse_list.append(mse)
        wrong_list.append(wrong_classif_reg)

    
    info = [seed, seed2, rest, res_zip, sol, opt_param, mse_list, wrong_list]

    return info

###----------------------------------------------------------------------------
### Fit the sparse NAM on the simulated datasets 

#seq_names = ["208", "307", "505", "604", "703"]
#seq_names = ["208", "307", "406", "505", "703"]
seq_names = ["109", "208", "406", "505", "604"]

mse_sparse_NAM = []
wrong_sparse_NAM = []
seed_list = range(1, 1000)
for i in range(len(seq_names)):
    # Read the data
    name_train = "results_file/n_2000/train_" + seq_names[i] + '.csv'
    name_test = "results_file/n_2000/test_" + seq_names[i] + '.csv'
    data_train = pd.read_csv(name_train, usecols=range(1,12)).to_numpy()
    data_test = pd.read_csv(name_test, usecols=range(1,12)).to_numpy()
    
    X_train = data_train[:, 0:10]
    y_train = data_train[:, 10:11]
    X_test = data_test[:, 0:10]
    y_test = data_test[:, 10:11]
    
    # Read the info
    name_info = "results_file/n_2000/res_" + seq_names[i]
    f=open(name_info, "rb")
    b=pickle.load(f)
    f.close()
    
    rest = b[4]
    
    seed2 = seed_list[5*i :(5*(i + 1))]
    
    # Fit the sparse NAM model and obtain structure and MSE
    sparse_NAM_reg = sparse_NAM_fit_regression(X_train.astype(np.float32), y_train.astype(np.float32).reshape((y_train.shape[0])), X_test.astype(np.float32), y_test.astype(np.float32), i, rest, seed2)
    mse_sparse_NAM.append(sparse_NAM_reg[6])
    wrong_sparse_NAM.append(sparse_NAM_reg[7])
    
# Save results of the model
sparse_results_2000 = [mse_sparse_NAM, wrong_sparse_NAM]
f=open( "results_file/n_2000/sparse_NAM_res_2000", "wb")
pickle.dump(sparse_results_2000,f)
f.close()    
    