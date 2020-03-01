# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 13:17:57 2020

@author: INE12363221
"""

import numpy as np
from sklearn import preprocessing
X = np.array([[ 1., -1.,  2.],
              [ 2.,  0.,  0.],
              [ 0.,  1., -1.]])
X_normalized_l1 = preprocessing.normalize(X, norm='l1')
print(X)
print(X_normalized_l1)
X_normalized_l2 = preprocessing.normalize(X, norm='l2')
print(X_normalized_l2)

#Standard scaler
X = np.array([[1., 2.,  3.],
              [ 4.,  5.,  6.],
              [ 7.,  8., 9.]])
scaler = preprocessing.StandardScaler()
#print(scaler.fit(X))
#scaler.fit( ) wont do any thing it just Compute the mean and std to be used for later scaling.
print(scaler.transform(X))
#scaler.transform : transforms data to mean =0 and standard deviation =1
#fit_transform:First fit then transforms it 
print(scaler.fit_transform(X))