#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 12:11:31 2023

@author: marvazquezrabunal
"""


"""
MODEL CLASSIFICATION


Description:

Code with the architecture of the models used for classification

"""

###----------------------------------------------------------------------------

### Call libraries
from torch import nn
import torch.nn.functional as F
import torch


###----------------------------------------------------------------------------
### Linear and non linear modules

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

###----------------------------------------------------------------------------
### Create the classification model

class model_classification(nn.Module):
  def __init__(self, nfeat):
    super(model_classification, self).__init__()
    self.nfeat = nfeat
    self.linear = torch.nn.ModuleList([
        linear()
        for i in range(nfeat)
    ])
    self.non_linear = torch.nn.ModuleList([
        non_linear()
        for i in range(nfeat)
    ])
    self.sigmoid = nn.Sigmoid()
    self._bias = torch.nn.Parameter(data=torch.zeros(1))


  def forward(self, x):
    x = x.to(torch.float32)
    f_linear = torch.cat(self._feature_linear(x), dim = -1)
    f_non_linear = torch.cat(self._feature_non_linear(x), dim=-1)
    output = f_linear.sum(axis = - 1) + f_non_linear.sum(axis = -1) + self._bias
    output2 = self.sigmoid(output) 
    return output2

  def _feature_linear(self, x):
    return [self.linear[i](x[:, i].reshape(x.shape[0],1)) for i in range(self.nfeat)]
    
  def _feature_non_linear(self, x):
    return [self.non_linear[i](x[:, i].reshape(x.shape[0],1)) for i in range(self.nfeat)]

###----------------------------------------------------------------------------
###----------------------------------------------------------------------------

### Create copy of the model (for plotting the behaviour of each feature separately)

class model_classification_copy(nn.Module):
  def __init__(self, nfeat):
    super(model_classification_copy, self).__init__()
    self.nfeat = nfeat
    self.linear = torch.nn.ModuleList([
        linear()
        for i in range(nfeat)
    ])
    self.non_linear = torch.nn.ModuleList([
        non_linear()
        for i in range(nfeat)
    ])
    self.sigmoid = nn.Sigmoid()
    self._bias = torch.nn.Parameter(data=torch.zeros(1))

  def forward(self, x):
    x = x.to(torch.float32)
    f_linear = torch.cat(self._feature_linear(x), dim = -1)
    f_non_linear = torch.cat(self._feature_non_linear(x), dim=-1)
    output = f_linear.sum(axis = - 1) + f_non_linear.sum(axis = -1) + self._bias
    output2 = self.sigmoid(output) 
    return output2, f_non_linear, f_linear

  def _feature_linear(self, x):
    return [self.linear[i](x[:, i].reshape(x.shape[0],1)) for i in range(self.nfeat)]
    
  def _feature_non_linear(self, x):
    return [self.non_linear[i](x[:, i].reshape(x.shape[0],1)) for i in range(self.nfeat)]



###----------------------------------------------------------------------------
###----------------------------------------------------------------------------

### Create model to fit NAM for classification data

class NAM_classification(nn.Module):
  def __init__(self, nfeat):
    super(NAM_classification, self).__init__()
    self.nfeat = nfeat
    self.non_linear = torch.nn.ModuleList([
        non_linear()
        for i in range(nfeat)
    ])
    self.sigmoid = nn.Sigmoid()
    self._bias = torch.nn.Parameter(data=torch.zeros(1))

  def forward(self, x):
    x = x.to(torch.float32)
    f_non_linear = torch.cat(self._feature_non_linear(x), dim=-1)
    output = f_non_linear.sum(axis = -1) + self._bias
    output2 = self.sigmoid(output)
    return output2
    
  def _feature_non_linear(self, x):
    return [self.non_linear[i](x[:, i].reshape(x.shape[0],1)) for i in range(self.nfeat)]
