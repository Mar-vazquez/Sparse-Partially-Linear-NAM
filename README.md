# Final_master_thesis

## Simulations

In this directory we are going to show the code needed to replicate the simulations done with the regression and the survival data.

### Simulations regression

In this subdirectory we present all the functions to simulate the regression data, fit our model and the competing methods to the simulated data and plot the results.

  * simulations_regression: script that obtains for all the sample sizes and the different scenarios the simulated data and the results of our model.
  * functions_regression: script with the functions to fit the model and obtain the final results for the regression simulated data. We do the cross validation, we obtain the optimal hyperparameters, we fit the final models and we obtain the structure found and the MSE.
  * cross_validation_regression: script with the Cross validation class for regression data that does Cross validation for the learning rate, alpha and lambda.
  * simulationdata_regression: script that generates the data for the different scenarios of the regression simulations.
  * model_regression: script with our model for regression data and the NAM model for regression data.
  * skorch_regression: code for the subclass of skorch that allows fitting our model (adding the proximal operator).
  * sparse_NAM_regression: script with the function to fit the sparse NAM model, doing the CV and obtaining the optimal hyperparameters for the corresponding datasets from all sample sizes.
  * NAM_regression: script to fit the NAM model for all the scenarios from the different sample sizes.
  * simulations_reg_R: R script to fit the lasso, gamsel and relgam models to the simulated regression datasets.
