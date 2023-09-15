#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 10:10:59 2023

@author: marvazquezrabunal
"""


"""
MODEL SURVIVAL


Description:

Code with the architecture of the models used for survival

"""

###----------------------------------------------------------------------------

### Call libraries
import numpy as np
import torch
import random
from torch import nn
import torch.nn.functional as F


###----------------------------------------------------------------------------

### Linear, non-linear and time-dependent modules

class linear(nn.Module):
    def __init__(self):
        super(linear, self).__init__()
        self.fc1 = nn.Linear(1, 1, bias = False)
    def forward(self, x):
        x = x.to(torch.float32)
        out = self.fc1(x)
        return out
        
class non_linear(nn.Module):
  def __init__(self):
    super(non_linear, self).__init__()
    self.fc1 = nn.Linear(1, 100, bias = True)
    self.fc2 = nn.Linear(100, 50, bias = True)
    self.fc3 = nn.Linear(50, 1, bias = False)


  def forward(self, x):
    x = x.to(torch.float32)
    out = F.relu(self.fc1(x))
    out = F.relu(self.fc2(out))
    out = self.fc3(out)
    return out

class non_linear_time(nn.Module):
  def __init__(self):
    super(non_linear_time, self).__init__()
    self.fc1 = nn.Linear(2, 100, bias = True)
    self.fc2 = nn.Linear(100, 50, bias = True)
    self.fc3 = nn.Linear(50, 1, bias = False)

  def forward(self, x):
    x = x.to(torch.float32)
    out = F.relu(self.fc1(x))
    out = F.relu(self.fc2(out))
    out = self.fc3(out)
    return out

###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
### Create the model survival 

class model_survival(nn.Module):
  def __init__(self, nfeat):
    super(model_survival, self).__init__()
    self.nfeat = nfeat
    self.linear = torch.nn.ModuleList([
        linear()
        for i in range(nfeat)
    ])
    self.non_linear = torch.nn.ModuleList([
        non_linear()
        for i in range(nfeat)
    ])
    self.non_linear_time = torch.nn.ModuleList([
        non_linear_time()
        for i in range(nfeat)
    ])

  def forward(self, x):
    x = x.to(torch.float32)
    f_linear = torch.cat(self._feature_linear(x), dim = -1)
    f_non_linear = torch.cat(self._feature_non_linear(x), dim=-1)
    f_non_linear_time = torch.cat(self._feature_non_linear_time(x), dim=-1)
    output = f_linear.sum(axis = -1) + f_non_linear.sum(axis = -1) + f_non_linear_time.sum(axis = -1)
    return output

  def _feature_linear(self, x):
    return [self.linear[i](x[:, i].reshape(x.shape[0],1)) for i in range(self.nfeat)]
    
  def _feature_non_linear(self, x):
    return [self.non_linear[i](x[:, i].reshape(x.shape[0],1)) for i in range(self.nfeat)]
  
  def _feature_non_linear_time(self, x):
    return [self.non_linear_time[i](torch.stack((x[:, i], x[:, self.nfeat]), 1)) for i in range(self.nfeat)]

###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
### Create copy of the model (for plotting the behaviour of each feature separately)


class model_survival_copy(nn.Module):
  def __init__(self, nfeat):
    super(model_survival_copy, self).__init__()
    self.nfeat = nfeat
    self.linear = torch.nn.ModuleList([
        linear()
        for i in range(nfeat)
    ])
    self.non_linear = torch.nn.ModuleList([
        non_linear()
        for i in range(nfeat)
    ])
    self.non_linear_time = torch.nn.ModuleList([
        non_linear_time()
        for i in range(nfeat)
    ])

  def forward(self, x):
    x = torch.from_numpy(x)
    f_linear = torch.cat(self._feature_linear(x), dim = -1)
    f_non_linear = torch.cat(self._feature_non_linear(x), dim=-1)
    f_non_linear_time = torch.cat(self._feature_non_linear_time(x), dim=-1)
    output = f_linear.sum(axis = -1) + f_non_linear.sum(axis = -1) + f_non_linear_time.sum(axis = -1)
    return output, f_linear, f_non_linear, f_non_linear_time

  def _feature_linear(self, x):
    return [self.linear[i](x[:, i].reshape(x.shape[0],1)) for i in range(self.nfeat)]
    
  def _feature_non_linear(self, x):
    return [self.non_linear[i](x[:, i].reshape(x.shape[0],1)) for i in range(self.nfeat)]
  
  def _feature_non_linear_time(self, x):
    return [self.non_linear_time[i](torch.stack((x[:, i], x[:, self.nfeat]), 1)) for i in range(self.nfeat)]

###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
### Loss of the model 

def loss_coxtime(model, output, y, x, training):
    time = x[:, -1]
    idx = time.sort(descending = True)[1]
    events = y[idx, 1]
    x_sorted = x[idx, :]
    time_sorted = time[idx]
    output = output[idx]
    events1 = [j for j, x in enumerate(events) if x == 1]
    output_ev = output[events1]
    if training == True:
        l = [random.sample(range(j+1), 1)[0] for j in range(len(time_sorted))]
        new_x = x_sorted[l,:]
        new_x[:,-1] = time_sorted
        pred_j = model(new_x[events1,:])
        res = sum(torch.log(1 + torch.exp(pred_j - output_ev)))/sum(events)
    else:
        np.random.seed(0)
        random.seed(0)
        mid = 0
        for i in range(10):
            l = [random.sample(range(j+1), 1)[0] for j in range(len(time_sorted))]
            new_x = x_sorted[l,:]
            new_x[:,-1] = time_sorted
            pred_j = model(new_x[events1,:])
            mid = mid + torch.exp(pred_j - output_ev)
        res = sum(torch.log(1 + mid))/sum(events)
    return res

class CustomLosstime(nn.Module):
    def __init__(self):
        super(CustomLosstime, self).__init__()

    def forward(self, model, output, target, x, training):
        loss = loss_coxtime(model, output, target, x, training)
        return loss
    
###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
### Create the NAM model for survival data

class NAM_survival(nn.Module):
  def __init__(self, nfeat):
    super(NAM_survival, self).__init__()
    self.nfeat = nfeat
    self.non_linear = torch.nn.ModuleList([
        non_linear()
        for i in range(nfeat)
    ])

  def forward(self, x):
    x = x.to(torch.float32)
    f_non_linear = torch.cat(self._feature_non_linear(x), dim=-1)
    output = f_non_linear.sum(axis = -1)
    return output

  def _feature_non_linear(self, x):
    return [self.non_linear[i](x[:, i].reshape(x.shape[0],1)) for i in range(self.nfeat)]
  


###----------------------------------------------------------------------------
###----------------------------------------------------------------------------
### Create the sparse NAM model for survival data

class sparse_NAM_survival(nn.Module):
  def __init__(self, nfeat):
    super(sparse_NAM_survival, self).__init__()
    self.nfeat = nfeat
    self.linear = torch.nn.ModuleList([
        linear()
        for i in range(nfeat)
    ])
    self.non_linear = torch.nn.ModuleList([
        non_linear()
        for i in range(nfeat)
    ])


  def forward(self, x):
    x = x.to(torch.float32)
    f_linear = torch.cat(self._feature_linear(x), dim = -1)
    f_non_linear = torch.cat(self._feature_non_linear(x), dim=-1)
    output = f_linear.sum(axis = -1) + f_non_linear.sum(axis = -1) 
    return output

  def _feature_linear(self, x):
    return [self.linear[i](x[:, i].reshape(x.shape[0],1)) for i in range(self.nfeat)]
    
  def _feature_non_linear(self, x):
    return [self.non_linear[i](x[:, i].reshape(x.shape[0],1)) for i in range(self.nfeat)]
 



    