#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:50:37 2023

@author: marvazquezrabunal
"""


"""
PLOT RESULTS REGRESSION SIMULATION


Description:

Function to plot the results of the regression simulations, both in terms of 
structure and in terms of MSE.

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
        'size'   : 18}

matplotlib.rc('font', **font)

###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
### n = 700

### Plotting the wrong structure proportion n = 700

# Read data and obtain mean and std of structure
f=open( "results_file/n_700/results_700", "rb")
b=pickle.load(f)
f.close()

wrong = np.array(b[0])
wrong_mean = wrong.mean(axis = 1)
wrong_std = wrong.std(axis = 1)

mse = np.array(b[1])
mse_mean = mse.mean(axis = 1)
mse_std = mse.std(axis = 1)

scenario_factor = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

f=open( "results_file/n_700/sparse_NAM_res_700", "rb")
b=pickle.load(f)
f.close()
sparse_wrong = np.array(b[1])
sparse_wrong_mean = sparse_wrong.mean(axis = 1)
sparse_wrong_std = sparse_wrong.std(axis = 1)

sparse_mse = np.array(b[0])
sparse_mse_mean = sparse_mse.mean(axis = 1)
sparse_mse_std = sparse_mse.std(axis = 1)

gamsel_wrong = pd.read_csv('results_file/n_700/wrong_gamsel_700.csv', usecols=range(1,6)).to_numpy()
rel_wrong = pd.read_csv('results_file/n_700/wrong_rel_700.csv', usecols=range(1,6)).to_numpy()
lasso_wrong = pd.read_csv('results_file/n_700/wrong_lasso_700.csv', usecols=range(1,6)).to_numpy()

gamsel_wrong_mean = gamsel_wrong.mean(axis = 1)
rel_wrong_mean = rel_wrong.mean(axis = 1)
lasso_wrong_mean = lasso_wrong.mean(axis = 1)

gamsel_wrong_std = gamsel_wrong.std(axis = 1)
rel_wrong_std = rel_wrong.std(axis = 1)
lasso_wrong_std = lasso_wrong.std(axis = 1)

# Plot
plt.subplot(321)

plt.scatter(scenario_factor, wrong_mean, marker='o', s=50, c = 'deeppink', label = "Our model")
plt.errorbar(scenario_factor, wrong_mean, wrong_std, fmt = 'none', ecolor = 'deeppink', capsize = 3)

plt.scatter(scenario_factor, gamsel_wrong_mean, marker='^', s=50, c = 'turquoise', label = "Gamsel")
plt.errorbar(scenario_factor, gamsel_wrong_mean, gamsel_wrong_std, fmt = 'none', ecolor = 'turquoise', capsize = 3)

plt.scatter(scenario_factor, rel_wrong_mean, marker='X', s=35, c = 'darkviolet', label = "Relgam")
plt.errorbar(scenario_factor, rel_wrong_mean, rel_wrong_std, fmt = 'none', ecolor = 'darkviolet', capsize = 3)

plt.scatter(scenario_factor[0:5], lasso_wrong_mean, marker='*', s=50, c = 'orange', label = "Lasso")
plt.errorbar(scenario_factor[0:5], lasso_wrong_mean, lasso_wrong_std, fmt = 'none', ecolor = 'orange', capsize = 3)

plt.scatter(scenario_factor[5:10], sparse_wrong_mean, marker='d', s=50, c = 'greenyellow', label = "Sparse NAM")
plt.errorbar(scenario_factor[5:10], sparse_wrong_mean, sparse_wrong_std, fmt = 'none', ecolor = 'greenyellow', capsize = 3)


# Add labels and title
plt.ylabel('Wrong structure proportion', fontweight='bold')
plt.title('n = 700', fontweight='bold')

# Add vertical lines at x = 5.5 and x = 10.5
plt.axvline(x=5.5, color='gray', linestyle='dashed')
plt.axvline(x=10.5, color='gray', linestyle='dashed')

# Remove x-axis labels and set only ticks at scenario boundaries
scenario_ticks = [1,2,3,4,5,6, 7, 8,9,10,11,12, 13,14,15]
scenario_labels = ['','','Linear','','','','', 'Non-Linear','','','','' ,'Mix','','']
plt.xticks(scenario_ticks, scenario_labels)


# Add a legend
plt.legend(fontsize="16")

plt.ylim(-0.05, 1)


###-------------------------------------

### Plotting MSE n = 700

# Read data and obtain mean and std of MSE
gamsel_mse = pd.read_csv('results_file/n_700/mse_gamsel_700.csv', usecols=range(1,6)).to_numpy()
rel_mse = pd.read_csv('results_file/n_700/mse_rel_700.csv', usecols=range(1,6)).to_numpy()
lasso_mse = pd.read_csv('results_file/n_700/mse_lasso_700.csv', usecols=range(1,6)).to_numpy()
NAM_mse = pd.read_csv('results_file/n_700/mse_NAM_700.csv', usecols=range(1,6)).to_numpy()

gamsel_mse_mean = gamsel_mse.mean(axis = 1)
rel_mse_mean = rel_mse.mean(axis = 1)
lasso_mse_mean = lasso_mse.mean(axis = 1)
NAM_mse_mean = NAM_mse.mean(axis = 1)

gamsel_mse_std = gamsel_mse.std(axis = 1)
rel_mse_std = rel_mse.std(axis = 1)
lasso_mse_std = lasso_mse.std(axis = 1)
NAM_mse_std = NAM_mse.std(axis = 1)

# Plot
plt.subplot(322)
plt.scatter(scenario_factor, mse_mean, marker='o', s=50, c = 'deeppink', label = "Our model")
plt.errorbar(scenario_factor, mse_mean, mse_std, fmt = 'none', ecolor = 'deeppink', capsize = 3)

plt.scatter(scenario_factor, gamsel_mse_mean, marker='^', s=50, c = 'turquoise', label = "Gamsel")
plt.errorbar(scenario_factor, gamsel_mse_mean, gamsel_mse_std, fmt = 'none', ecolor = 'turquoise', capsize = 3)

plt.scatter(scenario_factor, rel_mse_mean, marker='X', s=35, c = 'darkviolet', label = "Relgam")
plt.errorbar(scenario_factor, rel_mse_mean, rel_mse_std, fmt = 'none', ecolor = 'darkviolet', capsize = 3)

plt.scatter(scenario_factor[0:5], lasso_mse_mean, marker='*', s=50, c = 'orange', label = "Lasso")
plt.errorbar(scenario_factor[0:5], lasso_mse_mean, lasso_mse_std, fmt = 'none', ecolor = 'orange', capsize = 3)

plt.scatter(scenario_factor[5:10], sparse_mse_mean, marker='d', s=50, c = 'greenyellow', label = "Sparse NAM")
plt.errorbar(scenario_factor[5:10], sparse_mse_mean, sparse_mse_std, fmt = 'none', ecolor = 'greenyellow', capsize = 3)

plt.scatter(scenario_factor, NAM_mse_mean, marker='2', s=35, c = 'green', label = "NAM")
plt.errorbar(scenario_factor, NAM_mse_mean, NAM_mse_std, fmt = 'none', ecolor = 'green', capsize = 3)

# Add labels and title
plt.ylabel('MSE', fontweight='bold')
plt.title('n = 700', fontweight='bold')

# Add vertical lines at x = 5.5 and x = 10.5
plt.axvline(x=5.5, color='gray', linestyle='dashed')
plt.axvline(x=10.5, color='gray', linestyle='dashed')

# Remove x-axis labels and set only ticks at scenario boundaries
scenario_ticks = [1,2,3,4,5,6, 7, 8,9,10,11,12, 13,14,15]
scenario_labels = ['','','Linear','','','','', 'Non-Linear','','','','' ,'Mix','','']
plt.xticks(scenario_ticks, scenario_labels)


# Add a legend
plt.legend(fontsize="16")



###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
### n = 1000


### Plotting the wrong structure proportion n = 1000

# Read data and obtain mean and std of strucure
f=open( "results_file/n_1000/results_1000", "rb")
b=pickle.load(f)
f.close()

wrong = np.array(b[0])
wrong_mean = wrong.mean(axis = 1)
wrong_std = wrong.std(axis = 1)

mse = np.array(b[1])
mse_mean = mse.mean(axis = 1)
mse_std = mse.std(axis = 1)

scenario_factor = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

f=open( "results_file/n_1000/sparse_NAM_res_1000", "rb")
b=pickle.load(f)
f.close()
sparse_wrong = np.array(b[1])
sparse_wrong_mean = sparse_wrong.mean(axis = 1)
sparse_wrong_std = sparse_wrong.std(axis = 1)

sparse_mse = np.array(b[0])
sparse_mse_mean = sparse_mse.mean(axis = 1)
sparse_mse_std = sparse_mse.std(axis = 1)

gamsel_wrong = pd.read_csv('results_file/n_1000/wrong_gamsel_1000.csv', usecols=range(1,6)).to_numpy()
rel_wrong = pd.read_csv('results_file/n_1000/wrong_rel_1000.csv', usecols=range(1,6)).to_numpy()
lasso_wrong = pd.read_csv('results_file/n_1000/wrong_lasso_1000.csv', usecols=range(1,6)).to_numpy()

gamsel_wrong_mean = gamsel_wrong.mean(axis = 1)
rel_wrong_mean = rel_wrong.mean(axis = 1)
lasso_wrong_mean = lasso_wrong.mean(axis = 1)

gamsel_wrong_std = gamsel_wrong.std(axis = 1)
rel_wrong_std = rel_wrong.std(axis = 1)
lasso_wrong_std = lasso_wrong.std(axis = 1)


# Plot
plt.subplot(323)

plt.scatter(scenario_factor, wrong_mean, marker='o', s=50, c = 'deeppink', label = "Our model")
plt.errorbar(scenario_factor, wrong_mean, wrong_std, fmt = 'none', ecolor = 'deeppink', capsize = 3)

plt.scatter(scenario_factor, gamsel_wrong_mean, marker='^', s=50, c = 'turquoise', label = "Gamsel")
plt.errorbar(scenario_factor, gamsel_wrong_mean, gamsel_wrong_std, fmt = 'none', ecolor = 'turquoise', capsize = 3)

plt.scatter(scenario_factor, rel_wrong_mean, marker='X', s=35, c = 'darkviolet', label = "Relgam")
plt.errorbar(scenario_factor, rel_wrong_mean, rel_wrong_std, fmt = 'none', ecolor = 'darkviolet', capsize = 3)

plt.scatter(scenario_factor[0:5], lasso_wrong_mean, marker='*', s=50, c = 'orange', label = "Lasso")
plt.errorbar(scenario_factor[0:5], lasso_wrong_mean, lasso_wrong_std, fmt = 'none', ecolor = 'orange', capsize = 3)

plt.scatter(scenario_factor[5:10], sparse_wrong_mean, marker='d', s=50, c = 'greenyellow', label = "Sparse NAM")
plt.errorbar(scenario_factor[5:10], sparse_wrong_mean, sparse_wrong_std, fmt = 'none', ecolor = 'greenyellow', capsize = 3)


# Add labels and title
plt.ylabel('Wrong structure proportion', fontweight='bold')
plt.title('n = 1000', fontweight='bold')

# Add vertical lines at x = 5.5 and x = 10.5
plt.axvline(x=5.5, color='gray', linestyle='dashed')
plt.axvline(x=10.5, color='gray', linestyle='dashed')

# Remove x-axis labels and set only ticks at scenario boundaries
#plt.xticks(np.arange(len(set(scenario_factor)) + 1), [])
scenario_ticks = [1,2,3,4,5,6, 7, 8,9,10,11,12, 13,14,15]
scenario_labels = ['','','Linear','','','','', 'Non-Linear','','','','' ,'Mix','','']
plt.xticks(scenario_ticks, scenario_labels)


# Add a legend
plt.legend(fontsize="16")

plt.ylim(-0.05, 1)


###-------------------------------------

### Plotting MSE n = 1000

# Read data and obtain mean and std of MSE
gamsel_mse = pd.read_csv('results_file/n_1000/mse_gamsel_1000.csv', usecols=range(1,6)).to_numpy()
rel_mse = pd.read_csv('results_file/n_1000/mse_rel_1000.csv', usecols=range(1,6)).to_numpy()
lasso_mse = pd.read_csv('results_file/n_1000/mse_lasso_1000.csv', usecols=range(1,6)).to_numpy()
NAM_mse = pd.read_csv('results_file/n_1000/mse_NAM_1000.csv', usecols=range(1,6)).to_numpy()

gamsel_mse_mean = gamsel_mse.mean(axis = 1)
rel_mse_mean = rel_mse.mean(axis = 1)
lasso_mse_mean = lasso_mse.mean(axis = 1)
NAM_mse_mean = NAM_mse.mean(axis = 1)


gamsel_mse_std = gamsel_mse.std(axis = 1)
rel_mse_std = rel_mse.std(axis = 1)
lasso_mse_std = lasso_mse.std(axis = 1)
NAM_mse_std = NAM_mse.std(axis = 1)

# Plot
plt.subplot(324)

plt.scatter(scenario_factor, mse_mean, marker='o', s=50, c = 'deeppink', label = "Our model")
plt.errorbar(scenario_factor, mse_mean, mse_std, fmt = 'none', ecolor = 'deeppink', capsize = 3)

plt.scatter(scenario_factor, gamsel_mse_mean, marker='^', s=50, c = 'turquoise', label = "Gamsel")
plt.errorbar(scenario_factor, gamsel_mse_mean, gamsel_mse_std, fmt = 'none', ecolor = 'turquoise', capsize = 3)

plt.scatter(scenario_factor, rel_mse_mean, marker='X', s=35, c = 'darkviolet', label = "Relgam")
plt.errorbar(scenario_factor, rel_mse_mean, rel_mse_std, fmt = 'none', ecolor = 'darkviolet', capsize = 3)

plt.scatter(scenario_factor[0:5], lasso_mse_mean, marker='*', s=50, c = 'orange', label = "Lasso")
plt.errorbar(scenario_factor[0:5], lasso_mse_mean, lasso_mse_std, fmt = 'none', ecolor = 'orange', capsize = 3)

plt.scatter(scenario_factor[5:10], sparse_mse_mean, marker='d', s=50, c = 'greenyellow', label = "Sparse NAM")
plt.errorbar(scenario_factor[5:10], sparse_mse_mean, sparse_mse_std, fmt = 'none', ecolor = 'greenyellow', capsize = 3)

plt.scatter(scenario_factor, NAM_mse_mean, marker='2', s=35, c = 'green', label = "NAM")
plt.errorbar(scenario_factor, NAM_mse_mean, NAM_mse_std, fmt = 'none', ecolor = 'green', capsize = 3)


# Add labels and title
plt.ylabel('MSE', fontweight='bold')
plt.title('n = 1000', fontweight='bold')

# Add vertical lines at x = 5.5 and x = 10.5
plt.axvline(x=5.5, color='gray', linestyle='dashed')
plt.axvline(x=10.5, color='gray', linestyle='dashed')

# Remove x-axis labels and set only ticks at scenario boundaries
scenario_ticks = [1,2,3,4,5,6, 7, 8,9,10,11,12, 13,14,15]
scenario_labels = ['','','Linear','','','','', 'Non-Linear','','','','' ,'Mix','','']
plt.xticks(scenario_ticks, scenario_labels)


# Add a legend
plt.legend(fontsize="16")


###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
### n = 2000

### Plotting the wrong structure proportion n = 2000

# Read data and obtain mean and std structure
f=open( "results_file/n_2000/results_2000", "rb")
b=pickle.load(f)
f.close()

wrong = np.array(b[0])
wrong_mean = wrong.mean(axis = 1)
wrong_std = wrong.std(axis = 1)

mse = np.array(b[1])
mse_mean = mse.mean(axis = 1)
mse_std = mse.std(axis = 1)

scenario_factor = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

f=open( "results_file/n_2000/sparse_NAM_res_2000", "rb")
b=pickle.load(f)
f.close()
sparse_wrong = np.array(b[1])
sparse_wrong_mean = sparse_wrong.mean(axis = 1)
sparse_wrong_std = sparse_wrong.std(axis = 1)

sparse_mse = np.array(b[0])
sparse_mse_mean = sparse_mse.mean(axis = 1)
sparse_mse_std = sparse_mse.std(axis = 1)

gamsel_wrong = pd.read_csv('results_file/n_2000/wrong_gamsel_2000.csv', usecols=range(1,6)).to_numpy()
rel_wrong = pd.read_csv('results_file/n_2000/wrong_rel_2000.csv', usecols=range(1,6)).to_numpy()
lasso_wrong = pd.read_csv('results_file/n_2000/wrong_lasso_2000.csv', usecols=range(1,6)).to_numpy()

gamsel_wrong_mean = gamsel_wrong.mean(axis = 1)
rel_wrong_mean = rel_wrong.mean(axis = 1)
lasso_wrong_mean = lasso_wrong.mean(axis = 1)

gamsel_wrong_std = gamsel_wrong.std(axis = 1)
rel_wrong_std = rel_wrong.std(axis = 1)
lasso_wrong_std = lasso_wrong.std(axis = 1)

# Plot
plt.subplot(325)

plt.scatter(scenario_factor, wrong_mean, marker='o', s=50, c = 'deeppink', label = "Our model")
plt.errorbar(scenario_factor, wrong_mean, wrong_std, fmt = 'none', ecolor = 'deeppink', capsize = 3)

plt.scatter(scenario_factor, gamsel_wrong_mean, marker='^', s=50, c = 'turquoise', label = "Gamsel")
plt.errorbar(scenario_factor, gamsel_wrong_mean, gamsel_wrong_std, fmt = 'none', ecolor = 'turquoise', capsize = 3)

plt.scatter(scenario_factor, rel_wrong_mean, marker='X', s=35, c = 'darkviolet', label = "Relgam")
plt.errorbar(scenario_factor, rel_wrong_mean, rel_wrong_std, fmt = 'none', ecolor = 'darkviolet', capsize = 3)

plt.scatter(scenario_factor[0:5], lasso_wrong_mean, marker='*', s=50, c = 'orange', label = "Lasso")
plt.errorbar(scenario_factor[0:5], lasso_wrong_mean, lasso_wrong_std, fmt = 'none', ecolor = 'orange', capsize = 3)

plt.scatter(scenario_factor[5:10], sparse_wrong_mean, marker='d', s=50, c = 'greenyellow', label = "Sparse NAM")
plt.errorbar(scenario_factor[5:10], sparse_wrong_mean, sparse_wrong_std, fmt = 'none', ecolor = 'greenyellow', capsize = 3)


# Add labels and title
plt.ylabel('Wrong structure proportion', fontweight='bold')
plt.title('n = 2000', fontweight='bold')

# Add vertical lines at x = 5.5 and x = 10.5
plt.axvline(x=5.5, color='gray', linestyle='dashed')
plt.axvline(x=10.5, color='gray', linestyle='dashed')

# Remove x-axis labels and set only ticks at scenario boundaries
scenario_ticks = [1,2,3,4,5,6, 7, 8,9,10,11,12, 13,14,15]
scenario_labels = ['','','Linear','','','','', 'Non-Linear','','','','' ,'Mix','','']
plt.xticks(scenario_ticks, scenario_labels)


# Add a legend
plt.legend(fontsize="16")

plt.ylim(-0.05, 1)


###-------------------------------------

### Plotting MSE n = 2000

# Read data and obtain mean and std of MSE
gamsel_mse = pd.read_csv('results_file/n_2000/mse_gamsel_2000.csv', usecols=range(1,6)).to_numpy()
rel_mse = pd.read_csv('results_file/n_2000/mse_rel_2000.csv', usecols=range(1,6)).to_numpy()
lasso_mse = pd.read_csv('results_file/n_2000/mse_lasso_2000.csv', usecols=range(1,6)).to_numpy()
NAM_mse = pd.read_csv('results_file/n_2000/mse_NAM_2000.csv', usecols=range(1,6)).to_numpy()

gamsel_mse_mean = gamsel_mse.mean(axis = 1)
rel_mse_mean = rel_mse.mean(axis = 1)
lasso_mse_mean = lasso_mse.mean(axis = 1)
NAM_mse_mean = NAM_mse.mean(axis = 1)


gamsel_mse_std = gamsel_mse.std(axis = 1)
rel_mse_std = rel_mse.std(axis = 1)
lasso_mse_std = lasso_mse.std(axis = 1)
NAM_mse_std = NAM_mse.std(axis = 1)

# Plot 
plt.subplot(326)

plt.scatter(scenario_factor, mse_mean, marker='o', s=50, c = 'deeppink', label = "Our model")
plt.errorbar(scenario_factor, mse_mean, mse_std, fmt = 'none', ecolor = 'deeppink', capsize = 3)

plt.scatter(scenario_factor, gamsel_mse_mean, marker='^', s=50, c = 'turquoise', label = "Gamsel")
plt.errorbar(scenario_factor, gamsel_mse_mean, gamsel_mse_std, fmt = 'none', ecolor = 'turquoise', capsize = 3)

plt.scatter(scenario_factor, rel_mse_mean, marker='X', s=35, c = 'darkviolet', label = "Relgam")
plt.errorbar(scenario_factor, rel_mse_mean, rel_mse_std, fmt = 'none', ecolor = 'darkviolet', capsize = 3)

plt.scatter(scenario_factor[0:5], lasso_mse_mean, marker='*', s=50, c = 'orange', label = "Lasso")
plt.errorbar(scenario_factor[0:5], lasso_mse_mean, lasso_mse_std, fmt = 'none', ecolor = 'orange', capsize = 3)

plt.scatter(scenario_factor[5:10], sparse_mse_mean, marker='d', s=50, c = 'greenyellow', label = "Sparse NAM")
plt.errorbar(scenario_factor[5:10], sparse_mse_mean, sparse_mse_std, fmt = 'none', ecolor = 'greenyellow', capsize = 3)

plt.scatter(scenario_factor, NAM_mse_mean, marker='2', s=35, c = 'green', label = "NAM")
plt.errorbar(scenario_factor, NAM_mse_mean, NAM_mse_std, fmt = 'none', ecolor = 'green', capsize = 3)


# Add labels and title
plt.ylabel('MSE', fontweight='bold')
plt.title('n = 2000', fontweight='bold')

# Add vertical lines at x = 5.5 and x = 10.5
plt.axvline(x=5.5, color='gray', linestyle='dashed')
plt.axvline(x=10.5, color='gray', linestyle='dashed')

# Remove x-axis labels and set only ticks at scenario boundaries
scenario_ticks = [1,2,3,4,5,6, 7, 8,9,10,11,12, 13,14,15]
scenario_labels = ['','','Linear','','','','', 'Non-Linear','','','','' ,'Mix','','']
plt.xticks(scenario_ticks, scenario_labels)


# Add a legend
plt.legend(fontsize="16")


###----------------------------------------------------------------------------
### Add main title and save plot

plt.suptitle("Simulations Regression", fontweight='bold', fontsize="34")
plt.savefig("results_file/subplots_reg.png", format="png", dpi = 150)
plt.show()














