#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 09:23:05 2023

@author: marvazquezrabunal
"""

"""
CROSS VALIDATION REGRESSION


Description:

Code to perform cross validation on regression datasets.

"""


###----------------------------------------------------------------------------

### Call libraries

import numpy as np
from sklearn.model_selection import KFold
import torch
from skorch.helper import predefined_split
from skorch.dataset import Dataset
import skorch
from sklearn.model_selection import train_test_split
import copy
import sys
import random

# Add the path to import the NeuralNet_regression class
sys.path.append("/Users/marvazquezrabunal/Library/Mobile Documents/com~apple~CloudDocs/ETH/Spring 2023/Thesis/Simulations/Regression")
from skorch_regression import NeuralNet_regression



###----------------------------------------------------------------------------

### Create the class to perform cross validation for regression

class CrossValidation_regression:
    def __init__(
            self,
            module,
            criterion,
            optimizer=torch.optim.SGD,
            max_epochs=10,
            batch_size=128,
            seed = 0
    ):
        self.module = module
        self.criterion = criterion
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.seed = seed

        
    def path(self, X_train, y_train, X_test, y_test, lambda_seq, alpha):
        """Obtain the path of the model for a sequence of lambdas.

        Parameters
        ----------
        X_train : train explanatory variables.
        y_train: train response variable.
        X_test : test explanatory variables.
        y_test: test response variable.
        lambda_seq: sequence of lambdas.
        alpha: penalization quantity.
        
        Returns
        -------
        List with the errors for the different lambdas and number of layers
        set to 0 for each value of lambda.

        """
        
        # Fit model without penalization
        module_ = self.module(X_train.shape[1])
        valid_ds = Dataset(X_test, y_test)
        net_new = NeuralNet_regression(
            module_,
            max_epochs = 40,
            criterion = self.criterion,
            lr = self.lr,
            lambda1 = 0.0,
            train_split = predefined_split(valid_ds),
            optimizer = self.optimizer,
            iterator_train__shuffle = True,
            alpha = alpha)
        net_new.fit(X_train, y_train)
        loss_list = []
        set_zero_list = []
        
        for lambda1 in lambda_seq:
            # For each lambda fit model with proximal operator
            net = copy.deepcopy(net_new)
            net = NeuralNet_regression(
                net.module_,
                max_epochs = self.max_epochs,
                criterion = self.criterion,
                lr = self.lr,
                lambda1 = lambda1,
                train_split = predefined_split(valid_ds),
                optimizer = self.optimizer,
                iterator_train__shuffle = True,
                alpha = alpha)
            net.fit(X_train, y_train)

            
            # Save linear features set to 0
            i = 0
            zeros_lin = []
            for param in net.module_.linear.parameters():
                if param.data == 0:
                    zeros_lin.append(i)
                i = i + 1
                
            # See how many layers were set to 0
            set_zero = 0
            for i in range(len(net.module_.linear)):
                non_linear_weights = np.empty(0)
                for param in net.module_.non_linear[i].parameters():
                    non_linear_weights = np.concatenate((non_linear_weights, param.data.reshape(-1).numpy()))
                linear_weights = np.empty(0)
                for param in net.module_.linear[i].parameters():
                    linear_weights = np.concatenate((linear_weights, param.data.reshape(-1).numpy()))
                if non_linear_weights.sum() == 0:
                    set_zero = set_zero + 1
                if linear_weights.sum() == 0:
                    set_zero = set_zero + 1
                    
            set_zero_list.append(set_zero)
            
            # If all the linearities were set to 0, stop
            if len(zeros_lin) == (X_train.shape[1] - 1):
                loss_list.append(net.history[:, 'valid_loss'][-1])
                continue
            
            # Retrain the model
            X_train2 = copy.deepcopy(X_train)
            X_train2[:, zeros_lin] = X_train[:, zeros_lin] * 0
        
            es = skorch.callbacks.EarlyStopping()
            net = NeuralNet_regression(
                net.module_,
                max_epochs = 100,
                criterion = self.criterion,
                lr = self.lr,
                lambda1 = 0.0,
                train_split=predefined_split(valid_ds),
                optimizer = self.optimizer,
                callbacks = [es],
                iterator_train__shuffle = True,
                alpha = alpha)
            net.fit(X_train2, y_train)
            
            non_lin_freez = []
            for name, param in net.module_.named_parameters():
                if [*name][0] == 'n':
                    non_lin_freez.append(name)
                    
            # Retrain just the linearities        
            cb = skorch.callbacks.Freezer(non_lin_freez)
            es = skorch.callbacks.EarlyStopping()
            net = NeuralNet_regression(
                net.module_,
                max_epochs = 100,
                criterion = self.criterion,
                lr = self.lr,
                lambda1 = 0.0,
                train_split = predefined_split(valid_ds),
                optimizer = self.optimizer,
                callbacks = [es, cb],
                iterator_train__shuffle = True,
                alpha = alpha)
            net.fit(X_train2, y_train)
            
            loss_list.append(net.history[:, 'valid_loss'][-1])
        return loss_list, set_zero_list
    
    
    def cross_validation_split(self, nrows, nfolds):
        """
        Split data based on kfold cross validation.

        Parameters
        ----------
        nrows: number of rows in the dataset.
        nfolds: number k of folds. 

        Returns
        -------
        List containing zips of (train, test) indices.

        """
        # Randomly generate index
        data_index = np.random.choice(nrows, nrows, replace = False)

        # Split data into k folds
        k_folds = KFold(n_splits = nfolds).split(data_index)

        # List containing zips of (train, test) indices
        response = [(data_index[train], data_index[validate]) for train, validate in list(k_folds)]
        return response
        
    def cross_validation(self, x, y, nfolds, nlambdas, alpha, lr):
        """
        Obtain the error for all train/test splits and for each value of lambda.

        Parameters
        ----------
        x: explanatory variables.
        y: response variable.
        nfolds: number k of folds.
        nlambdas: number of lambdas in the sequence.
        alpha: penalization quantity.
        lr: value of learning rate.

        Returns
        -------
        Error list, sequence of lambda and number of attributes set to 0.
        
        """
        np.random.seed(self.seed)    
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        # Obtain array of errors for each lambda and train/test split
        error_list = []
        zero_list = []
        cv_index = self.cross_validation_split(nrows = x.shape[0], nfolds = nfolds)
        self.lr = lr
        i = 1
        lambda_list = self.obtain_seq_lambda(x, y, nlambdas, alpha)
        for zip_index in cv_index:
            train_index, test_index = zip_index
            x_train, x_test = (x[train_index, :], x[test_index, :])
            y_train, y_test = (y[train_index], y[test_index])
            error, set_zero = self.path(x_train, y_train, x_test, y_test, lambda_list, alpha)
            error_list.append(error)
            zero_list.append(set_zero)
            i = i + 1
        return np.array(error_list).T, lambda_list, np.array(zero_list).T
    
    
    def dense_model(self, X, y, alpha):
        """Train dense model (without proximal operator).

        Parameters
        ----------
        X : explanatory variables.
        y: response variable.
        alpha: penalization quantity.
        
        Returns
        -------
        Fitted model.

        """
        
        # Fit model without PO
        model = self.module(X.shape[1])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        valid_ds = Dataset(X_test, y_test)
        es = skorch.callbacks.EarlyStopping()
        net = NeuralNet_regression(
            model,
            max_epochs= 40,
            criterion = self.criterion,
            lr = self.lr,
            lambda1 = 0.0,
            train_split = predefined_split(valid_ds),
            optimizer = self.optimizer,
            callbacks = [es],
            iterator_train__shuffle = True,
            alpha = alpha)
        net.fit(X_train, y_train)
        return(net)
    
    def gl_operator(self, u, lambda1):
        """Apply group lasso operator to a vector.

        Parameters
        ----------
        u : vector to which apply the group lasso operator.
        lambda1 : penalization parameter.
        
        Returns
        -------
        Vector with the group lasso operator applied.

        """
        norm2 = np.linalg.norm(u)
        if norm2 <= lambda1:
          u = np.zeros(len(u))
        else:
          u = (norm2-lambda1)*u/norm2
        return(u)
    
    def proximal_operator(self, non_linear_weights, linear_weights, lambda1, alpha):
        """Apply proximal operator to a vector.

        Parameters
        ----------
        x : vector to which apply the proximal operator.
        lambda1 : penalization parameter.
        alpha: penalization quantity.
        
        Returns
        -------
        Vector with the proximal operator applied.

        """
        l = np.sqrt(len(non_linear_weights) + len(linear_weights))
        v1 = self.gl_operator(non_linear_weights, l * lambda1 * (1 - alpha)).astype('float32')
        v1 = np.concatenate((v1, linear_weights))
        v2 = self.gl_operator(v1, l * lambda1 * alpha).astype('float32')
        return(v2)

    
    def obtain_lambda_start(self, X, y, alpha):
        """Obtain the lambda that starts the sequence of lambdas.

        Parameters
        ----------
        X : explanatory variables.
        y: response variable.
        alpha: penalization quantity.
        
        Returns
        -------
        Lambda in which start the sequence of lambdas.

        """
        # Fit dense model
        new_net = self.dense_model(X, y, alpha)
        
        #Find the lambda that sets one attribute to 0
        lambda_seq = np.linspace(0, 1300, 1000)
        k = 0
        for lambda1 in lambda_seq:
            for i in range(len(new_net.module_.linear)):
                non_linear_weights = np.empty(0)
                for param in new_net.module_.non_linear[i].parameters():
                    non_linear_weights = np.concatenate((non_linear_weights, param.data.reshape(-1).numpy()))
                linear_weights = np.empty(0)
                for param in new_net.module_.linear[i].parameters():
                    linear_weights = np.concatenate((linear_weights, param.data.reshape(-1).numpy()))
                prox_op = self.proximal_operator(non_linear_weights, linear_weights, self.lr * lambda1, alpha)
                if abs(prox_op[0: len(non_linear_weights)]).sum() == 0:
                    print("One weight set to 0")
                    return(lambda1/self.max_epochs/20)
            k = k + 1
            print(k)
        
    
    def obtain_seq_lambda(self, X, y, nlambdas, alpha):
        """Obtain the sequence of lambdas. It starts with lambda_start and
        finishes with the first lambda that puts all the weights to 0.

        Parameters
        ----------
        X : explanatory variables.
        y: response variable.
        nlambdas: number of lambdas in the sequence.
        alpha: penalization quantity.
        
        Returns
        -------
        Sequence of lambdas.

        """
        
        # Obtain initial lambda
        lambda_start = self.obtain_lambda_start(X, y, alpha)
        print(lambda_start)
        
        # Fit dense model
        new_net = self.dense_model(X, y, alpha)
        k = 0
        lambda1 = lambda_start
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        valid_ds = Dataset(X_test, y_test)
        
        # See what lambda sets everything to 0
        while k != "Stop":
            net = copy.deepcopy(new_net)
            net = NeuralNet_regression(
                net.module_,
                max_epochs= self.max_epochs,
                criterion= self.criterion,
                lr = self.lr,
                lambda1 = lambda1,
                train_split = predefined_split(valid_ds),
                optimizer = self.optimizer,
                iterator_train__shuffle=True,
                alpha = alpha)
            net.fit(X_train, y_train)
            
            sum_total = 0
            for name, param in list(net.module_.named_parameters()):
                sum_total = sum_total + abs(param).sum()
              
            if sum_total == 0:
                print("Everything set to 0:", lambda1)
                return(np.geomspace(lambda_start, lambda1, nlambdas))
            lambda1 = lambda1 * 1.2
    

###----------------------------------------------------------------------------

### Fit final model with optimal hyperparameters for regression model

class final_model_regression:
    def __init__(
            self,
            module,
            criterion,
            optimizer=torch.optim.SGD,
            lr=0.01,
            max_epochs=10,
            batch_size=128,
            alpha = 0.05
    ):
        self.module = module
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.alpha = alpha
        
    def fit(self, X, y, lambda_opt):
        """Fit the model with the optimal lambda and all the data.

        Parameters
        ----------
        X : explanatory variables.
        y: response variable.
        lambda_opt: optimal lambda.
        
        Returns
        -------
        Fitted model.

        """
        
        module_ = self.module(X.shape[1])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
        valid_ds = Dataset(X_test, y_test)
        
        # Fit model without PO
        es = skorch.callbacks.EarlyStopping()
        net = NeuralNet_regression(
            module_,
            max_epochs = 40,
            criterion = self.criterion,
            lr = self.lr,
            lambda1 = 0.0,
            train_split = predefined_split(valid_ds),
            optimizer = self.optimizer,
            callbacks = [es],
            iterator_train__shuffle = True,
            alpha = self.alpha)
        net.fit(X_train, y_train)
        
        # Fit model with PO
        net = NeuralNet_regression(
            net.module_,
            max_epochs = self.max_epochs,
            criterion = self.criterion,
            lr = self.lr,
            lambda1 = lambda_opt,
            train_split = predefined_split(valid_ds),
            optimizer = self.optimizer,
            iterator_train__shuffle = True,
            alpha = self.alpha)
        net.fit(X_train, y_train)
        
           
        req_grad_F = []
        for name, param in list(net.module_.named_parameters()):
            if abs(param).sum() == 0:
                req_grad_F.append(name)

           
        i = 0
        zeros_lin = []
        for param in net.module_.linear.parameters():
            if param.data == 0:
                zeros_lin.append(i)
            i = i + 1

        if len(zeros_lin) == (X_train.shape[1]):
           return net
       
        X_train[:, zeros_lin] = X_train[:, zeros_lin] * 0
        
        module2 = self.module(X.shape[1])
        
        for name, param in module2.named_parameters():
            if name in req_grad_F:
                param.data = 0*param.data
       
        # Refit model
        es = skorch.callbacks.EarlyStopping()
        net = NeuralNet_regression(
            module2,
            max_epochs = 100,
            criterion = self.criterion,
            lr = self.lr,
            lambda1 = 0.0,
            train_split = predefined_split(valid_ds),
            optimizer = self.optimizer,
            callbacks = [es],
            iterator_train__shuffle = True,
            alpha = self.alpha)
        net.fit(X_train, y_train)
        
        non_lin_freez = []
        for name, param in net.module_.named_parameters():
            if [*name][0] == 'n':
                non_lin_freez.append(name)
        
        # Refit model just with the linearities
        cb = skorch.callbacks.Freezer(non_lin_freez)
        es = skorch.callbacks.EarlyStopping()
        net = NeuralNet_regression(
            net.module_,
            max_epochs = 100,
            criterion = self.criterion,
            lr = self.lr,
            lambda1 = 0.0,
            train_split = predefined_split(valid_ds),
            optimizer = self.optimizer,
            callbacks = [es, cb],
            iterator_train__shuffle = True,
            alpha = self.alpha)
        net.fit(X_train, y_train)
        
        return net
        
