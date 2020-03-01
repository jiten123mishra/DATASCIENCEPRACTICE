# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 23:06:43 2020

@author: INE12363221
"""
from sklearn.linear_model import LinearRegression 
regressor=LinearRegression()


x = dataset.iloc[:, [1,2,3]].values
y = dataset.iloc[:, 4].values
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'Advertising.csv')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
regressor.fit(x_train,y_train)

#making prediction 
y_pred=regressor.predict(x_test)

#visualizing homoscedacity 
#visualising training set with optimum line 
plt.scatter(y_pred,y_pred-y_test,color='red')        
plt.title('y pred vs diff')
plt.xlabel('y pred')
plt.ylabel('diff')
plt.show()

