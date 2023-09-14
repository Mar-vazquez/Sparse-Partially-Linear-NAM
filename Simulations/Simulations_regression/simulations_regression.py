#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 16:33:39 2023

@author: marvazquezrabunal
"""


"""
SIMULATIONS REGRESSION


Description:
    
Code to obtain the results of the regression simulations for all 
the sample sizes (considering different proportions of sparse/linear/non_linear).

"""


###----------------------------------------------------------------------------

### Call libraries

import numpy as np
from functions_regression import simulation_CV
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split


###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
### Simulation Regression n = 700

seq = [190, 370, 460, 550, 730, 208, 307, 505, 604, 703, 172, 352, 343, 235, 424]
acc_list = []
mse_list = []
ind_list = []
elem_list = []

for i in range(len(seq)):
    # Fit model
    prop = seq[i]
    sparse_prop = (prop // 10**2 % 10)/10
    lin_prop = (prop // 10**1 % 10)/10
    non_lin_prop = (prop // 10**0 % 10)/10
    info = simulation_CV(10, [sparse_prop, lin_prop, non_lin_prop], 700, i, [5*(i+1), 5*(i+1) + 1, 5*(i+1) + 2, 5*(i+1) + 3, 5*(i+1) + 4])
    
    # Save results model
    name = 'results_file/n_700/res_' + str(prop)
    f=open(name, "wb")
    pickle.dump(info,f)
    f.close()
    
    # Update in which iteration we are
    f = open("results_file/n_700/update.txt",'w')
    f.write(str(i))
    f.close()
    
    # Save the training and test data
    X_sim = info[2]
    y_sim = info[3].reshape(700, 1)
    X_train, X_test, y_train, y_test = train_test_split(X_sim, y_sim, test_size = 0.1, random_state = i)

    data_train = np.hstack((X_train, y_train))
    data_test = np.hstack((X_test, y_test))
    name_train = 'results_file/n_700/train_' + str(prop) + '.csv'
    name_test = 'results_file/n_700/test_' + str(prop) + '.csv'
    pd.DataFrame(data_train).to_csv(name_train) 
    pd.DataFrame(data_test).to_csv(name_test) 
    
    # Lists of accuracy, MSE, indices and elements
    acc_list.append(info[9])
    mse_list.append(info[8])
    
    ind_list.append(info[4][1])
    elem_list.append(info[4][0])

# Save the results of structure and MSE for n = 700
results_700 = [acc_list, mse_list]
f=open( "results_file/n_700/results_700", "wb")
pickle.dump(results_700,f)
f.close()

# Save the list of indices and elements for n = 700
pd.DataFrame(ind_list).to_csv('results_file/n_700/ind_list.csv') 
pd.DataFrame(elem_list).to_csv('results_file/n_700/elem_list.csv') 


###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
### Simulation Regression n = 1000

seq = [280, 370, 460, 550, 640, 208, 307, 406, 505, 703, 181, 262, 343, 136, 523]
acc_list = []
mse_list = []
ind_list = []
elem_list = []
for i in range(len(seq)):
    # Fit model
    prop = seq[i]
    sparse_prop = (prop // 10**2 % 10)/10
    lin_prop = (prop // 10**1 % 10)/10
    non_lin_prop = (prop // 10**0 % 10)/10
    seed = (i*100 + 103*(i == 0)) * (i < 10) + (i*10 + 5) * (i >= 10)
    info = simulation_CV(10, [sparse_prop, lin_prop, non_lin_prop], 1000, seed, [5*(i+1) + 400, 5*(i+1) + 1 + 400, 5*(i+1) + 2 + 400, 5*(i+1) + 3 + 400, 5*(i+1) + 4 + 400])
    
    # Save results model
    name = 'results_file/n_1000/res_' + str(prop)
    f=open(name, "wb")
    pickle.dump(info,f)
    f.close()
    
    # Update in which iteration we are
    f = open("results_file/n_1000/update.txt",'w')
    f.write(str(i))
    f.close()
    
    # Save the training and test data
    X_sim = info[2]
    y_sim = info[3].reshape(1000,1)
    X_train, X_test, y_train, y_test = train_test_split(X_sim, y_sim, test_size = 0.1, random_state = seed)

    data_train = np.hstack((X_train, y_train))
    data_test = np.hstack((X_test, y_test))
    name_train = 'results_file/n_1000/train_' + str(prop) + '.csv'
    name_test = 'results_file/n_1000/test_' + str(prop) + '.csv'
    pd.DataFrame(data_train).to_csv(name_train) 
    pd.DataFrame(data_test).to_csv(name_test) 
    
    # Lists of accuracy, MSE, indices and elements
    acc_list.append(info[9])
    mse_list.append(info[8])
    
    ind_list.append(info[4][1])
    elem_list.append(info[4][0])

# Save the results of structure and MSE for n = 1000
results_1000 = [acc_list, mse_list]
f=open( "results_file/n_1000/results_1000", "wb")
pickle.dump(results_1000,f)
f.close()

# Save the list of indices and elements for n = 1000
pd.DataFrame(ind_list).to_csv('results_file/n_1000/ind_list.csv') 
pd.DataFrame(elem_list).to_csv('results_file/n_1000/elem_list.csv') 


###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
### Simulation Regression n = 2000

seq = [190, 280, 460, 550, 730, 109, 208, 406, 505, 604, 172, 262, 343, 226, 451]
acc_list = []
mse_list = []
ind_list = []
elem_list = []
for i in range(len(seq)):
    # Fit model
    prop = seq[i]
    sparse_prop = (prop // 10**2 % 10)/10
    lin_prop = (prop // 10**1 % 10)/10
    non_lin_prop = (prop // 10**0 % 10)/10
    seed = (i*1000 + 5 + 1030*(i == 0)) * (i < 10) + (i*100 + 55) * (i >= 10)
    info = simulation_CV(10, [sparse_prop, lin_prop, non_lin_prop], 2000, seed, [5*(i+1) + 900, 5*(i+1) + 1 + 900, 5*(i+1) + 2 + 900, 5*(i+1) + 3 + 900, 5*(i+1) + 4 + 900])
    
    # Save results model
    name = 'results_file/n_2000/res_' + str(prop)
    f=open(name, "wb")
    pickle.dump(info,f)
    f.close()
    
    # Update in which iteration we are
    f = open("results_file/n_2000/update.txt",'w')
    f.write(str(i))
    f.close()
    
    # Save the training and test data
    X_sim = info[2]
    y_sim = info[3].reshape(2000,1)
    X_train, X_test, y_train, y_test = train_test_split(X_sim, y_sim, test_size = 0.1, random_state = seed)

    data_train = np.hstack((X_train, y_train))
    data_test = np.hstack((X_test, y_test))
    name_train = 'results_file/n_2000/train_' + str(prop) + '.csv'
    name_test = 'results_file/n_2000/test_' + str(prop) + '.csv'
    pd.DataFrame(data_train).to_csv(name_train) 
    pd.DataFrame(data_test).to_csv(name_test) 
    
    # Lists of accuracy, MSE, indices and elements
    acc_list.append(info[9])
    mse_list.append(info[8])
    
    ind_list.append(info[4][1])
    elem_list.append(info[4][0])

# Save the results of structure and MSE for n = 2000
results_2000 = [acc_list, mse_list]
f=open( "results_file/n_2000/results_2000", "wb")
pickle.dump(results_2000,f)
f.close()

# Save the list of indices and elements for n = 2000
pd.DataFrame(ind_list).to_csv('results_file/n_2000/ind_list.csv') 
pd.DataFrame(elem_list).to_csv('results_file/n_2000/elem_list.csv') 


