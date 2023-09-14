#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 15:22:53 2023

@author: marvazquezrabunal
"""


"""
SIMULATIONS SURVIVAL


Description:
    
Code to obtain the results of the survival simulations for all the sample sizes
(considering different proportions of sparse/linear/non_linear/time).

"""


###----------------------------------------------------------------------------
### Call libraries

from functions_survival import simulation_CV_surv
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
### Simulation Survival n = 1000

seq = [1900, 3700, 4600, 5500, 6400, 2080, 3070, 4060, 5050, 7030, 1009, 2008, 3007, 5005, 7003, 1810, 2620, 3430, 1360, 5230, 1711, 1261, 2332, 2521, 2224]
acc_list = []
cindex_list = []
ibs_list = []
ind_list = []
elem_list = []

seq_ind = [0,1,2,3,4,5,6,7,8,9,15,16,17,18,19,10,11,12,13,14,20,21,22,23,24]
seq_ind = [20,21,22,23,24]
for i in seq_ind:
    # Fit model
    prop = seq[i]
    sparse_prop = (prop // 10**3 % 10)/10
    lin_prop = (prop // 10**2 % 10)/10
    non_lin_prop = (prop // 10**1 % 10)/10
    time_prop = (prop // 10**0 % 10)/10
    info = simulation_CV_surv(10, [sparse_prop, lin_prop, non_lin_prop, time_prop], 1000, 6*i + 1, [6*i + 2, 6*i + 3, 6*i + 4, 6*i + 5, 6*i + 6])
    
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
    y_sim = info[3].reshape(1000, 2)
    X_train, X_test, y_train, y_test = train_test_split(X_sim, y_sim, test_size = 0.1, random_state = 6*i + 1)

    data_train = np.hstack((X_train, y_train))
    data_test = np.hstack((X_test, y_test))
    name_train = 'results_file/n_1000/train_' + str(prop) + '.csv'
    name_test = 'results_file/n_1000/test_' + str(prop) + '.csv'
    pd.DataFrame(data_train).to_csv(name_train) 
    pd.DataFrame(data_test).to_csv(name_test) 
    
    # Lists of accuracy, C-index, IBS, indices and elements
    acc_list.append(info[10])
    cindex_list.append(info[8])
    ibs_list.append(info[9])
    
    ind_list.append(info[4][1])
    elem_list.append(info[4][0])

# Save the results of structure, C-index and IBS for n = 1000
results_1000 = [acc_list, cindex_list, ibs_list]
f=open( "results_file/n_1000/results_1000", "wb")
pickle.dump(results_1000,f)
f.close()

# Save the list of indices and elements for n = 1000
pd.DataFrame(ind_list).to_csv('results_file/n_1000/ind_list.csv') 
pd.DataFrame(elem_list).to_csv('results_file/n_1000/elem_list.csv') 



###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
### Simulation Survival n = 2000

seq = [1900, 3700, 4600, 5500, 6400, 2080, 3070, 4060, 5050, 7030, 1009, 2008, 3007, 5005, 7003, 1810, 2620, 3430, 1360, 5230, 1711, 1261, 2332, 2521, 2224]
acc_list = []
cindex_list = []
ibs_list = []
ind_list = []
elem_list = []

seq_ind = [0,1,2,3,4,5,6,7,8,9,15,16,17,18,19,10,11,12,13,14,20,21,22,23,24]
seq_ind = [20,21,22,23,24]

for i in seq_ind:
    # Fit model
    prop = seq[i]
    sparse_prop = (prop // 10**3 % 10)/10
    lin_prop = (prop // 10**2 % 10)/10
    non_lin_prop = (prop // 10**1 % 10)/10
    time_prop = (prop // 10**0 % 10)/10
    info = simulation_CV_surv(10, [sparse_prop, lin_prop, non_lin_prop, time_prop], 2000, 6*i + 1 + 1000, [6*i + 2 + 1000, 6*i + 3 + 1000, 6*i + 4 + 1000, 6*i + 5 + 1000, 6*i + 6 + 1000])
    
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
    y_sim = info[3].reshape(2000, 2)
    X_train, X_test, y_train, y_test = train_test_split(X_sim, y_sim, test_size = 0.1, random_state = 6*i + 1 + 1000)

    data_train = np.hstack((X_train, y_train))
    data_test = np.hstack((X_test, y_test))
    name_train = 'results_file/n_2000/train_' + str(prop) + '.csv'
    name_test = 'results_file/n_2000/test_' + str(prop) + '.csv'
    pd.DataFrame(data_train).to_csv(name_train) 
    pd.DataFrame(data_test).to_csv(name_test) 
    
    # Lists of accuracy, C-index, IBS, indices and elements
    acc_list.append(info[10])
    cindex_list.append(info[8])
    ibs_list.append(info[9])
    
    ind_list.append(info[4][1])
    elem_list.append(info[4][0])

# Save the results of structure, C-index and IBS for n = 2000
results_2000 = [acc_list, cindex_list, ibs_list]
f=open( "results_file/n_2000/results_2000", "wb")
pickle.dump(results_2000,f)
f.close()

# Save the list of indices and elements for n = 2000
pd.DataFrame(ind_list).to_csv('results_file/n_2000/ind_list.csv') 
pd.DataFrame(elem_list).to_csv('results_file/n_2000/elem_list.csv') 

