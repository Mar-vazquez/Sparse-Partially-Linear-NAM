# Final_master_thesis

## Simulations

In this directory we are going to show all the code needed to replicate the simulations done with the regression and the survival data.

### Simulations regression

In this subdirectory we present all the functions to simulate the regression data, fit our model and the competing methods to the simulated data and plot the results.

  * simulations_regression: script that obtains, for all the sample sizes and the different scenarios, the simulated data and the results of our model.
  * functions_regression: script with the functions to fit the model and obtain the final results for the regression simulated data. We do the cross validation, we obtain the optimal hyperparameters, we fit the final models and we obtain the structure found and the MSE.
  * cross_validation_regression: script with the Cross validation class for regression data that does Cross validation for the learning rate, alpha and lambda.
  * simulationdata_regression: script that generates the data for the different scenarios of the regression simulations.
  * model_regression: script with our model for regression data and the NAM model for regression data.
  * skorch_regression: code for the subclass of skorch that allows fitting our model for regression data (adding the proximal operator).
  * sparse_NAM_regression: script with the function to fit the sparse NAM model, doing the CV and obtaining the optimal hyperparameters for the corresponding datasets from all sample sizes.
  * NAM_regression: script to fit the NAM model for all the scenarios from the different sample sizes.
  * simulations_reg_R: R script to fit the Lasso, Gamsel and Relgam models to the simulated regression datasets.
  * subplot_reg: script to plot the results of the regression simulation study. It will create for each sample size two plots (one for the wrong structure proportion and the other one for the MSE).
  * plot_feat_regression: script to plot the shape function of each one of the explanatory features in some simulated regression datasets.


### Simulations survival

In this subdirectory we present all the functions to simulate the survival data, fit our model and the competing methods to the simulated data and plot the results.

  * simulations_survival: script that obtains, for all the sample sizes and the different scenarios, the simulated data and the results of our model.
  * functions_survival: script with the functions to fit the model and obtain the final results for the survival simulated data. We do the cross validation, we obtain the optimal hyperparameters, we fit the final models and we obtain the structure found, the C-index and the IBS.
  * cross_validation_survival: script with the Cross validation class for survival data that does Cross validation for the learning rate, alpha and lambda (in the case in which the model has time dependence).
  * simulationdata_survival: script that generates the data for the different scenarios of the survival simulations.
  * model_survival: script with our model for survival data, the sparse NAM model and the NAM for survival data.
  * skorch_survival: code for the subclass of skorch that allows fitting our model for survival data (adding the proximal operator).
  * sparse_NAM_survival: script with the function to fit the sparse NAM model, doing the CV and obtaining the optimal hyperparameters for the corresponding datasets from all sample sizes.
  * NAM_survival: script to fit the NAM model for all the scenarios from the different sample sizes.
  * functions_sparse_NAM: script with the functions needed to fit the sparse NAM model to the survival data and obtain the evaluation metrics.
  * cross_validation_notime_surv: script with the Cross validation class for survival data that does Cross validation for the learning rate, alpha and lambda (in the case in which the model has no time dependence). It is needed to fit the sparse NAM.
  * skorch_notime_survival: code for the subclass of skorch that allows fitting our model for survival data without time dependence (adding the proximal operator). It is needed to fit the sparse NAM.
  * simulations_surv_R: R script to fit the Lasso and Relgam models to the simulated survival datasets.
  * subplot_surv: script to plot the results of the survival simulation study. It will create for each sample size three plots (one for the wrong structure proportion, one for the C-index and the other one for the IBS).
  * plot_feat_survival:  script to plot the shape function of each one of the explanatory features in some simulated survival datasets.


## Real_datasets

In this directory we are going to show all the code needed to replicate the results obtained with real-world datasets for regression, classification and survival.


### Regression

In this subdirectory we present all the functions to fit our models to real-world regression datasets (wine and abalone).

  * cross_validation_regression: script with the Cross validation class for regression data that does Cross validation for the learning rate, alpha and lambda (it is the same script shown in the simulation directory).
  * model_regression: script with our model for regression data and the NAM model for regression data (it is the same script shown in the simulation directory).
  * functions_data_reg: script with the functions to fit the models to the real-world regresssion data, obtain the MSE and the structure of each feature.
  * wine_functions: script with the code to fit the models to the Wine dataset and obtain the results.
  * abalone_functions: script with the code to fit the models to the Abalone dataset and obtain the results.
  * functions_R_data_reg: script with the code to fit the Lasso, Relgam and Gamsel models to the real-world regression datasets.




