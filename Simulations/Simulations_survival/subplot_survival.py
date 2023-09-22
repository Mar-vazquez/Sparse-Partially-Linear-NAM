#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 11:10:26 2023

@author: marvazquezrabunal
"""


"""
PLOT RESULTS SURVIVAL SIMULATION


Description:

Function to plot the results of the survival simulations, in terms of 
structure, C-index and IBS.

"""

###----------------------------------------------------------------------------

### Call libraries
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

###----------------------------------------------------------------------------

# Fix some parameters of the plots
plt.rcParams["figure.figsize"] = (20,25)
font = {'family' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)

CB_color = '#377eb8', '#ff7f00', '#4daf4a','#f781bf', '#a65628', '#984ea3','#999999', '#e41a1c', '#dede00'


###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
### n = 1000

### Plotting the wrong structure proportion n = 1000

# Read data and obtain mean and std of structure
f=open( "results_file/n_1000/results_1000", "rb")
b=pickle.load(f)
f.close()

wrong = np.array(b[0])
wrong_mean = wrong.mean(axis = 1)
wrong_std = wrong.std(axis = 1)

cind = np.array(b[1])
cind_mean = cind.mean(axis = 1)
cind_std = cind.std(axis = 1)

ibs = np.array(b[2])
ibs_mean = ibs.mean(axis = 1)
ibs_std = ibs.std(axis = 1)

scenario_factor = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

f=open( "results_file/n_1000/sparse_NAM_res_1000", "rb")
b=pickle.load(f)
f.close()
sparse_wrong = np.array(b[0])
sparse_wrong_mean = sparse_wrong.mean(axis = 1)
sparse_wrong_std = sparse_wrong.std(axis = 1)

sparse_cindex = np.array(b[1])
sparse_cindex_mean = sparse_cindex.mean(axis = 1)
sparse_cindex_std = sparse_cindex.std(axis = 1)

sparse_ibs = np.array(b[2])
sparse_ibs_mean = sparse_ibs.mean(axis = 1)
sparse_ibs_std = sparse_ibs.std(axis = 1)

rel_wrong = pd.read_csv('results_file/n_1000/wrong_rel_1000.csv', usecols=range(1,6)).to_numpy()
lasso_wrong = pd.read_csv('results_file/n_1000/wrong_lasso_1000.csv', usecols=range(1,6)).to_numpy()

rel_wrong_mean = rel_wrong.mean(axis = 1)
lasso_wrong_mean = lasso_wrong.mean(axis = 1)

rel_wrong_std = rel_wrong.std(axis = 1)
lasso_wrong_std = lasso_wrong.std(axis = 1)


# Plot
plt.subplot(321)

plt.scatter(scenario_factor, wrong_mean, marker='o', s=70, c = CB_color[7], label = "Our model")
plt.errorbar(scenario_factor, wrong_mean, wrong_std, fmt = 'none', ecolor = CB_color[7], capsize = 3)

plt.scatter(scenario_factor[0:15], rel_wrong_mean, marker='X', s=70, c = CB_color[0], label = "Relgam")
plt.errorbar(scenario_factor[0:15], rel_wrong_mean, rel_wrong_std, fmt = 'none', ecolor = CB_color[0], capsize = 3)

plt.scatter(scenario_factor[0:5], lasso_wrong_mean, marker='*', s=70, c = CB_color[3], label = "Lasso")
plt.errorbar(scenario_factor[0:5], lasso_wrong_mean, lasso_wrong_std, fmt = 'none', ecolor = CB_color[3], capsize = 3)

plt.scatter(scenario_factor[5:10], sparse_wrong_mean, marker='d', s=70, c = CB_color[8], label = "Sparse NAM")
plt.errorbar(scenario_factor[5:10], sparse_wrong_mean, sparse_wrong_std, fmt = 'none', ecolor = CB_color[8], capsize = 3)

# Add labels and title
plt.ylabel('Wrong structure proportion', fontweight='bold')
plt.title('n = 1000', fontweight='bold')

# Add vertical lines
plt.axvline(x=5.5, color='gray', linestyle='dashed')
plt.axvline(x=10.5, color='gray', linestyle='dashed')
plt.axvline(x=15.5, color='gray', linestyle='dashed')
plt.axvline(x=20.5, color='gray', linestyle='dashed')

# Remove x-axis labels and set only ticks at scenario boundaries
scenario_ticks = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
scenario_labels = ['','','Linear','','','','', 'Non-Linear','','','','' ,'Mix1','','','','', 'Time','','','','' ,'Mix2','','']
plt.xticks(scenario_ticks, scenario_labels, fontsize=20)


# Add a legend
plt.legend(fontsize="19")

plt.ylim(-0.05, 1.05)



###-------------------------------------

### Plotting C-index n = 1000

# Read data and obtain mean and std of C-index
rel_cind = pd.read_csv('results_file/n_1000/cindex_rel_1000.csv', usecols=range(1,6)).to_numpy()
lasso_cind = pd.read_csv('results_file/n_1000/cindex_lasso_1000.csv', usecols=range(1,6)).to_numpy()
nam_cind = pd.read_csv('results_file/n_1000/cindex_NAM_1000.csv', usecols=range(1,6)).to_numpy()

rel_cind_mean = rel_cind.mean(axis = 1)
lasso_cind_mean = lasso_cind.mean(axis = 1)
nam_cind_mean = nam_cind.mean(axis = 1)

rel_cind_std = rel_cind.std(axis = 1)
lasso_cind_std = lasso_cind.std(axis = 1)
nam_cind_std = nam_cind.std(axis = 1)


# Plot
plt.subplot(323)

plt.scatter(scenario_factor, cind_mean, marker='o', s=70, c = CB_color[7], label = "Our model")
plt.errorbar(scenario_factor, cind_mean, cind_std, fmt = 'none', ecolor = CB_color[7], capsize = 3)

plt.scatter(scenario_factor[0:15], rel_cind_mean, marker='X', s=70, c = CB_color[0], label = "Relgam")
plt.errorbar(scenario_factor[0:15], rel_cind_mean, rel_cind_std, fmt = 'none', ecolor = CB_color[0], capsize = 3)

plt.scatter(scenario_factor[0:5], lasso_cind_mean, marker='*', s=70, c = CB_color[3], label = "Lasso")
plt.errorbar(scenario_factor[0:5], lasso_cind_mean, lasso_cind_std, fmt = 'none', ecolor = CB_color[3], capsize = 3)

plt.scatter(scenario_factor[5:10], sparse_cindex_mean, marker='d', s=70, c = CB_color[8], label = "Sparse NAM")
plt.errorbar(scenario_factor[5:10], sparse_cindex_mean, sparse_cindex_std, fmt = 'none', ecolor = CB_color[8], capsize = 3)


plt.scatter(scenario_factor, nam_cind_mean, marker='p', s=70, c = CB_color[2], label = "NAM")
plt.errorbar(scenario_factor, nam_cind_mean, nam_cind_std, fmt = 'none', ecolor = CB_color[2], capsize = 3)

# Add labels and title
plt.ylabel('C-index', fontweight='bold')
plt.title('n = 1000', fontweight='bold')

# Add vertical lines at x = 5.5 and x = 10.5
plt.axvline(x=5.5, color='gray', linestyle='dashed')
plt.axvline(x=10.5, color='gray', linestyle='dashed')
plt.axvline(x=15.5, color='gray', linestyle='dashed')
plt.axvline(x=20.5, color='gray', linestyle='dashed')

# Remove x-axis labels and set only ticks at scenario boundaries
scenario_ticks = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
scenario_labels = ['','','Linear','','','','', 'Non-Linear','','','','' ,'Mix1','','','','', 'Time','','','','' ,'Mix2','','']
plt.xticks(scenario_ticks, scenario_labels, fontsize=20)

# Add a legend
plt.legend(fontsize="19")

plt.ylim(-0.05, 1)


###-------------------------------------

### Plotting IBS n = 1000

# Read data and obtain mean and std of IBS
rel_ibs = pd.read_csv('results_file/n_1000/ibs_rel_1000.csv', usecols=range(1,6)).to_numpy()
lasso_ibs = pd.read_csv('results_file/n_1000/ibs_lasso_1000.csv', usecols=range(1,6)).to_numpy()
nam_ibs = pd.read_csv('results_file/n_1000/ibs_NAM_1000.csv', usecols=range(1,6)).to_numpy()


rel_ibs_mean = rel_ibs.mean(axis = 1)
lasso_ibs_mean = lasso_ibs.mean(axis = 1)
nam_ibs_mean = nam_ibs.mean(axis = 1)

rel_ibs_std = rel_ibs.std(axis = 1)
lasso_ibs_std = lasso_ibs.std(axis = 1)
nam_ibs_std = nam_ibs.std(axis = 1)

# Plot
plt.subplot(325)

plt.scatter(scenario_factor, ibs_mean, marker='o', s=70, c = CB_color[7], label = "Our model")
plt.errorbar(scenario_factor, ibs_mean, ibs_std, fmt = 'none', ecolor = CB_color[7], capsize = 3)

plt.scatter(scenario_factor[0:15], rel_ibs_mean, marker='X', s=70, c = CB_color[0], label = "Relgam")
plt.errorbar(scenario_factor[0:15], rel_ibs_mean, rel_ibs_std, fmt = 'none', ecolor = CB_color[0], capsize = 3)

plt.scatter(scenario_factor[0:5], lasso_ibs_mean, marker='*', s=70, c = CB_color[3], label = "Lasso")
plt.errorbar(scenario_factor[0:5], lasso_ibs_mean, lasso_ibs_std, fmt = 'none', ecolor = CB_color[3], capsize = 3)

plt.scatter(scenario_factor[5:10], sparse_ibs_mean, marker='d', s=70, c = CB_color[8], label = "Sparse NAM")
plt.errorbar(scenario_factor[5:10], sparse_ibs_mean, sparse_ibs_std, fmt = 'none', ecolor = CB_color[8], capsize = 3)

plt.scatter(scenario_factor, nam_ibs_mean, marker='p', s=70, c = CB_color[2], label = "NAM")
plt.errorbar(scenario_factor, nam_ibs_mean, nam_ibs_std, fmt = 'none', ecolor = CB_color[2], capsize = 3)

# Add labels and title
plt.ylabel('IBS', fontweight='bold')
plt.title('n = 1000', fontweight='bold')

# Add vertical lines at x = 5.5 and x = 10.5
plt.axvline(x=5.5, color='gray', linestyle='dashed')
plt.axvline(x=10.5, color='gray', linestyle='dashed')
plt.axvline(x=15.5, color='gray', linestyle='dashed')
plt.axvline(x=20.5, color='gray', linestyle='dashed')

# Remove x-axis labels and set only ticks at scenario boundaries
scenario_ticks = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
scenario_labels = ['','','Linear','','','','', 'Non-Linear','','','','' ,'Mix1','','','','', 'Time','','','','' ,'Mix2','','']
plt.xticks(scenario_ticks, scenario_labels, fontsize=20)

# Add a legend
plt.legend(fontsize="19")

plt.ylim(-0.05, 1)


###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
### n = 2000

### Plotting the wrong structure proportion n = 2000

# Read data and obtain mean and std of structure
f=open( "results_file/n_2000/results_2000", "rb")
b=pickle.load(f)
f.close()

wrong = np.array(b[0])
wrong_mean = wrong.mean(axis = 1)
wrong_std = wrong.std(axis = 1)

cind = np.array(b[1])
cind_mean = cind.mean(axis = 1)
cind_std = cind.std(axis = 1)

ibs = np.array(b[2])
ibs_mean = ibs.mean(axis = 1)
ibs_std = ibs.std(axis = 1)

scenario_factor = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

f=open( "results_file/n_2000/sparse_NAM_res_2000", "rb")
b=pickle.load(f)
f.close()
sparse_wrong = np.array(b[0])
sparse_wrong_mean = sparse_wrong.mean(axis = 1)
sparse_wrong_std = sparse_wrong.std(axis = 1)

sparse_cindex = np.array(b[1])
sparse_cindex_mean = sparse_cindex.mean(axis = 1)
sparse_cindex_std = sparse_cindex.std(axis = 1)

sparse_ibs = np.array(b[2])
sparse_ibs_mean = sparse_ibs.mean(axis = 1)
sparse_ibs_std = sparse_ibs.std(axis = 1)

rel_wrong = pd.read_csv('results_file/n_2000/wrong_rel_2000.csv', usecols=range(1,6)).to_numpy()
lasso_wrong = pd.read_csv('results_file/n_2000/wrong_lasso_2000.csv', usecols=range(1,6)).to_numpy()

rel_wrong_mean = rel_wrong.mean(axis = 1)
lasso_wrong_mean = lasso_wrong.mean(axis = 1)

rel_wrong_std = rel_wrong.std(axis = 1)
lasso_wrong_std = lasso_wrong.std(axis = 1)

# Plot
plt.subplot(322)

plt.scatter(scenario_factor, wrong_mean, marker='o', s=70, c = CB_color[7], label = "Our model")
plt.errorbar(scenario_factor, wrong_mean, wrong_std, fmt = 'none', ecolor = CB_color[7], capsize = 3)

plt.scatter(scenario_factor[0:15], rel_wrong_mean, marker='X', s=70, c = CB_color[0], label = "Relgam")
plt.errorbar(scenario_factor[0:15], rel_wrong_mean, rel_wrong_std, fmt = 'none', ecolor = CB_color[0], capsize = 3)

plt.scatter(scenario_factor[0:5], lasso_wrong_mean, marker='*', s=70, c = CB_color[3], label = "Lasso")
plt.errorbar(scenario_factor[0:5], lasso_wrong_mean, lasso_wrong_std, fmt = 'none', ecolor = CB_color[3], capsize = 3)

plt.scatter(scenario_factor[5:10], sparse_wrong_mean, marker='d', s=70, c = CB_color[8], label = "Sparse NAM")
plt.errorbar(scenario_factor[5:10], sparse_wrong_mean, sparse_wrong_std, fmt = 'none', ecolor = CB_color[8], capsize = 3)


# Add labels and title
plt.ylabel('Wrong structure proportion', fontweight='bold')
plt.title('n = 2000', fontweight='bold')

# Add vertical lines
plt.axvline(x=5.5, color='gray', linestyle='dashed')
plt.axvline(x=10.5, color='gray', linestyle='dashed')
plt.axvline(x=15.5, color='gray', linestyle='dashed')
plt.axvline(x=20.5, color='gray', linestyle='dashed')

# Remove x-axis labels and set only ticks at scenario boundaries
scenario_ticks = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
scenario_labels = ['','','Linear','','','','', 'Non-Linear','','','','' ,'Mix1','','','','', 'Time','','','','' ,'Mix2','','']
plt.xticks(scenario_ticks, scenario_labels, fontsize=20)

# Add a legend
plt.legend(fontsize="19")

plt.ylim(-0.05, 1.05)



###-------------------------------------

### Plotting C-index n = 2000

# Read data and obtain mean and std of C-index
rel_cind = pd.read_csv('results_file/n_2000/cindex_rel_2000.csv', usecols=range(1,6)).to_numpy()
lasso_cind = pd.read_csv('results_file/n_2000/cindex_lasso_2000.csv', usecols=range(1,6)).to_numpy()
nam_cind = pd.read_csv('results_file/n_2000/cindex_NAM_2000.csv', usecols=range(1,6)).to_numpy()

rel_cind_mean = rel_cind.mean(axis = 1)
lasso_cind_mean = lasso_cind.mean(axis = 1)
nam_cind_mean = nam_cind.mean(axis = 1)

rel_cind_std = rel_cind.std(axis = 1)
lasso_cind_std = lasso_cind.std(axis = 1)
nam_cind_std = nam_cind.std(axis = 1)

# Plot
plt.subplot(324)

plt.scatter(scenario_factor, cind_mean, marker='o', s=70, c = CB_color[7], label = "Our model")
plt.errorbar(scenario_factor, cind_mean, cind_std, fmt = 'none', ecolor = CB_color[7], capsize = 3)

plt.scatter(scenario_factor[0:15], rel_cind_mean, marker='X', s=70, c = CB_color[0], label = "Relgam")
plt.errorbar(scenario_factor[0:15], rel_cind_mean, rel_cind_std, fmt = 'none', ecolor = CB_color[0], capsize = 3)

plt.scatter(scenario_factor[0:5], lasso_cind_mean, marker='*', s=70, c = CB_color[3], label = "Lasso")
plt.errorbar(scenario_factor[0:5], lasso_cind_mean, lasso_cind_std, fmt = 'none', ecolor = CB_color[3], capsize = 3)

plt.scatter(scenario_factor[5:10], sparse_cindex_mean, marker='d', s=70, c = CB_color[8], label = "Sparse NAM")
plt.errorbar(scenario_factor[5:10], sparse_cindex_mean, sparse_cindex_std, fmt = 'none', ecolor = CB_color[8], capsize = 3)


plt.scatter(scenario_factor, nam_cind_mean, marker='p', s=70, c = CB_color[2], label = "NAM")
plt.errorbar(scenario_factor, nam_cind_mean, nam_cind_std, fmt = 'none', ecolor = CB_color[2], capsize = 3)

# Add labels and title
plt.ylabel('C-index', fontweight='bold')
plt.title('n = 2000', fontweight='bold')

# Add vertical lines 
plt.axvline(x=5.5, color='gray', linestyle='dashed')
plt.axvline(x=10.5, color='gray', linestyle='dashed')
plt.axvline(x=15.5, color='gray', linestyle='dashed')
plt.axvline(x=20.5, color='gray', linestyle='dashed')

# Remove x-axis labels and set only ticks at scenario boundaries
scenario_ticks = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
scenario_labels = ['','','Linear','','','','', 'Non-Linear','','','','' ,'Mix1','','','','', 'Time','','','','' ,'Mix2','','']
plt.xticks(scenario_ticks, scenario_labels, fontsize=20)

# Add a legend
plt.legend(fontsize="19")

plt.ylim(-0.05, 1)



###-------------------------------------

### Plotting IBS n = 2000

# Read data and obtain mean and std of IBS
rel_ibs = pd.read_csv('results_file/n_2000/ibs_rel_2000.csv', usecols=range(1,6)).to_numpy()
lasso_ibs = pd.read_csv('results_file/n_2000/ibs_lasso_2000.csv', usecols=range(1,6)).to_numpy()
nam_ibs = pd.read_csv('results_file/n_2000/ibs_NAM_2000.csv', usecols=range(1,6)).to_numpy()

rel_ibs_mean = rel_ibs.mean(axis = 1)
lasso_ibs_mean = lasso_ibs.mean(axis = 1)
nam_ibs_mean = nam_ibs.mean(axis = 1)

rel_ibs_std = rel_ibs.std(axis = 1)
lasso_ibs_std = lasso_ibs.std(axis = 1)
nam_ibs_std = nam_ibs.std(axis = 1)

# Plot
plt.subplot(326)

plt.scatter(scenario_factor, ibs_mean, marker='o', s=70, c = CB_color[7], label = "Our model")
plt.errorbar(scenario_factor, ibs_mean, ibs_std, fmt = 'none', ecolor = CB_color[7], capsize = 3)

plt.scatter(scenario_factor[0:15], rel_ibs_mean, marker='X', s=70, c = CB_color[0], label = "Relgam")
plt.errorbar(scenario_factor[0:15], rel_ibs_mean, rel_ibs_std, fmt = 'none', ecolor = CB_color[0], capsize = 3)

plt.scatter(scenario_factor[0:5], lasso_ibs_mean, marker='*', s=70, c = CB_color[3], label = "Lasso")
plt.errorbar(scenario_factor[0:5], lasso_ibs_mean, lasso_ibs_std, fmt = 'none', ecolor = CB_color[3], capsize = 3)

plt.scatter(scenario_factor[5:10], sparse_ibs_mean, marker='d', s=70, c = CB_color[8], label = "Sparse NAM")
plt.errorbar(scenario_factor[5:10], sparse_ibs_mean, sparse_ibs_std, fmt = 'none', ecolor = CB_color[8], capsize = 3)

plt.scatter(scenario_factor, nam_ibs_mean, marker='p', s=70, c = CB_color[2], label = "NAM")
plt.errorbar(scenario_factor, nam_ibs_mean, nam_ibs_std, fmt = 'none', ecolor = CB_color[2], capsize = 3)

# Add labels and title
plt.ylabel('IBS', fontweight='bold')
plt.title('n = 2000', fontweight='bold')

# Add vertical lines 
plt.axvline(x=5.5, color='gray', linestyle='dashed')
plt.axvline(x=10.5, color='gray', linestyle='dashed')
plt.axvline(x=15.5, color='gray', linestyle='dashed')
plt.axvline(x=20.5, color='gray', linestyle='dashed')

# Remove x-axis labels and set only ticks at scenario boundaries
scenario_ticks = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
scenario_labels = ['','','Linear','','','','', 'Non-Linear','','','','' ,'Mix1','','','','', 'Time','','','','' ,'Mix2','','']
plt.xticks(scenario_ticks, scenario_labels, fontsize=20)


# Add a legend
plt.legend(fontsize="19")

plt.ylim(-0.05, 1)

# Show the plot
plt.suptitle("Simulations Survival", fontweight='bold', fontsize="34")
plt.savefig("results_file/subplots_surv.png", format="png", dpi = 150)

plt.show()
