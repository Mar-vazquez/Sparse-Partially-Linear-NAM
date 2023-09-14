#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 15:30:02 2023

@author: marvazquezrabunal
"""


"""
SPARSE NAM SURVIVAL


Description:

Code to fit the sparse NAM to survival data.

"""

###----------------------------------------------------------------------------

### Call libraries
import numpy as np
import torch
import random
from cross_validation_notime_survival import CrossValidation_notime_survival
from joblib import Parallel, delayed
from model_survival import CustomLosstime, sparse_NAM_survival
import time
import pandas as pd
from functions_survival import  metrics
import pickle
from functions_sparse_NAM import optimal_param_sparse, final_model_sparse, wrong_classif_survival_sparse

###----------------------------------------------------------------------------

### Function to fit the sparse NAM model on survival data


def sparse_NAM_fit_surv(X_train, y_train, X_test, y_test, seed, rest, seed2):
    """Does CV, fits the final sparse NAM model and obtains the evaluation metrics.

    Parameters
    ----------
    X_train: training explanatory features.
    y_train: training response.
    X_test: test explanatory features.
    y_test: test response.
    seed: initial random seed.
    rest: information about the simulated data.
    seed2: list of random seeds with which we fit the final models.
        
    Returns
    -------
    List with the seeds, information about the simulated data, lr and alphas
    used for the CV, result of the CV, optimal hyperparameters, C-index,
    IBS, proportion of wrong structure found and predicted structure of each
    feature.
    
    """
    np.random.seed(seed)    
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Do CV
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
    
    # Obtain optimal hyperparameters
    opt_param = optimal_param_sparse(sol)
    
    # Fit final models and obtain evaluation metrics
    wrong_list =[]
    classif_list = []
    cindex_list = []
    ibs_list = []
    
    for j in seed2:
        final_model_res = final_model_sparse(j, X_train, y_train, opt_param)
        wrong_classif_surv, classif = wrong_classif_survival_sparse(rest[-3], final_model_res, rest)
        cindex, ibs = metrics(j, final_model_res, X_train, y_train, X_test, y_test)
        wrong_list.append(wrong_classif_surv)
        classif_list.append(classif)
        cindex_list.append(cindex)
        ibs_list.append(ibs)
    
    info = [seed, seed2, rest, res_zip, sol, opt_param, cindex_list, ibs_list, wrong_list, classif_list]

    return info

###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
### Fit the sparse NAM on the simulated datasets for n = 1000

seq_names = ["2080", "3070", "4060", "5050", "7030"]
cindex_sparse_NAM = []
ibs_sparse_NAM = []
wrong_sparse_NAM = []
classif_NAM = []
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
    
    name_info = "results_file/n_1000/res_" + seq_names[i]
    f=open( name_info, "rb")
    b=pickle.load(f)
    f.close()
    
    rest = b[4]
    
    seed2 = seed_list[5*i :(5*(i + 1))]
    
    # Fit sparse NAM model and obtain evaluation metrics
    sparse_NAM_surv = sparse_NAM_fit_surv(X_train, y_train, X_test, y_test, i, rest, seed2)
    cindex_sparse_NAM.append(sparse_NAM_surv[6])
    ibs_sparse_NAM.append(sparse_NAM_surv[7])
    wrong_sparse_NAM.append(sparse_NAM_surv[8])
    classif_NAM.append(sparse_NAM_surv[-1])

sparse_results_1000 = [wrong_sparse_NAM, cindex_sparse_NAM, ibs_sparse_NAM, classif_NAM]
f=open( "results_file/n_1000/sparse_NAM_res_1000", "wb")
pickle.dump(sparse_results_1000,f)
f.close()    


###----------------------------------------------------------------------------
### Fit the sparse NAM on the simulated datasets for n = 2000

seq_names = ["2080", "3070", "4060", "5050", "7030"]
cindex_sparse_NAM = []
ibs_sparse_NAM = []
wrong_sparse_NAM = []
classif_NAM = []
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
    
    name_info = "results_file/n_2000/res_" + seq_names[i]
    f=open( name_info, "rb")
    b=pickle.load(f)
    f.close()
    
    rest = b[4]
    
    seed2 = seed_list[5*i :(5*(i + 1))]
    
    # Fit sparse NAM model and obtain evaluation metrics
    sparse_NAM_surv = sparse_NAM_fit_surv(X_train, y_train, X_test, y_test, i, rest, seed2)
    cindex_sparse_NAM.append(sparse_NAM_surv[6])
    ibs_sparse_NAM.append(sparse_NAM_surv[7])
    wrong_sparse_NAM.append(sparse_NAM_surv[8])
    classif_NAM.append(sparse_NAM_surv[-1])

sparse_results_2000 = [wrong_sparse_NAM, cindex_sparse_NAM, ibs_sparse_NAM, classif_NAM]
f=open( "results_file/n_2000/sparse_NAM_res_2000", "wb")
pickle.dump(sparse_results_2000,f)
f.close()    
