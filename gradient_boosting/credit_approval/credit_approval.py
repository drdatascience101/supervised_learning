#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 06:59:06 2019

@author: zxs
"""

# Import the required libraries
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

# Set the working directory
wd = '/Users/zxs/Documents/code/credit_approval/'
os.chdir(wd)

# Load the info
data = open('crx.data', 'rb')
data = pd.read_csv(data, header = None)

# Separate the variables
y = data[15]
x = data.iloc[:, 0:15]

# Define the column types
cat_cols = [0, 3, 4, 5, 6, 8, 9, 11, 12]
float_cols = [i for i in x.columns if i not in cat_cols]

# Initialize the label encoder
le = LabelEncoder()

# Convert the columns
for col in cat_cols:
    
    x[col] = le.fit_transform(x[col])

# Replace missing values
x = x.replace('?', np.NaN)
    
for col in float_cols:
    
    x[col] = x[col].astype('float')

# Convert the tArget variable
d = {'-': 0, '+': 1}
y = y.replace(d)

# Split the data for training / testing
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = .2, random_state = 100)

'''
    LGB Model
'''

# Define parameters
param = {'seed': 100,
         'feature_fraction_seed': 100,
         'bagging_seed': 100,
         'drop_seed': 100,
         'data_random_seed': 100,
         'objective': 'binary',
         'boosting_type': 'gbdt',
         'verbose': 1,
         'metric': 'auc',
         'is_unbalance': True,
         'boost_from_average': False,
         'learning_rate': .01,
         'num_leaves': 50,
         'max_depth': 5, # shallower trees reduce overfitting.
         'min_split_gain': 0, # minimal loss gain to perform a split.
         'min_child_samples': 21, # specifies the minimum samples per leaf node.
         'min_child_weight': 5, # minimal sum hessian in one leaf.
    
         'lambda_l1': 0.5, # L1 regularization.
         'lambda_l2': 0.5, # L2 regularization.
    
         # LightGBM can subsample the data for training (improves speed):
         'feature_fraction': 0.5, # randomly select a fraction of the features.
         'bagging_fraction': 0.5, # randomly bag or subsample training data.
         'bagging_freq': 0, # perform bagging every Kth iteration, disabled if 0.
        
         'subsample_for_bin': 200000, # sample size to determine histogram bins.
         'max_bin': 1000, # maximum number of bins to bucket feature values in.
    
         'nthread': 8}

# Initialize the data
x_train = lgb.Dataset(train_x.values, label = train_y.values)
x_test = lgb.Dataset(test_x.values, label = test_y.values) 

# Train the model
clf = lgb.train(param, x_train, valid_sets = [x_test])

preds = clf.predict(test_x.values, num_iteration = clf.best_iteration) 

auc = roc_auc_score(test_y, preds)  