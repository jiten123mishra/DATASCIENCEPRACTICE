# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 16:53:29 2019

@author: INE12363221
"""
#WHEN FEATURE SCALING IS REQUIRED ?
#Always use feature scaling we  need to calculate euclidian distance 
#like KNN or Kmeans, AHC 
#linear regression with gradient descent

#HOW to do 
#1. standard scaler
#2. min max scaler 
#3.Robust scaler 

#MinMaxScaler(feature_range = (0, 1)) will transform each value in the column proportionally within the range [0,1]. 
#Use this as the first scaler choice to transform a feature, as it will preserve the shape of the dataset (no distortion).

#StandardScaler() will transform each value in the column to range about the mean 0 and standard deviation 1, 
#ie, each value will be normalised by subtracting the mean and dividing by standard deviation. 
#Use StandardScaler if you know the data distribution is normal.

#If there are outliers, use RobustScaler(). Alternatively you could remove the outliers and use either of the above 2 scalers 
#(choice depends on whether data is normally distributed)

#WHY FEATURE SCALING 
#Every features has 2 parts magnitude and units
#suppose we have height as 180cm weight 1kg height 200 cm weight 2kg 
#As they are not in same scale result may not be actual representation 
#so we need to scale it down to same scale 
import pandas as pd
dataset = pd.read_csv('50_Startups.csv')
dataset1=pd.get_dummies(dataset)
df= dataset1
x = df.loc[:, df.columns != 'Profit'].values
y = df.iloc[:, df.columns == 'Profit'].values
#print(dataset1.info())
#print(x.info())
#print(type(y))
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
print("Before scalling ",x_train[0])
#165349.2	136897.8	471784.1	New York	192261.83


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
print("After scaling",x_train[0])

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)

#HOW TO  DO PREDICTION IN CASE  OF STANDARD SCALER
#1.SCALE INPUT 
#2.PREDICT GET OUTPUT
#3.INVERSE TRANSFORM THE OUTPUT 
input1=[[165349.2,136897.8,471784.1,0,1,0]]
input1_scaled=sc_X.transform(input1)
out=regressor.predict(input1_scaled)
print("output for 165349.2,136897.8,471784.1,0,1,0 input is ")
print(sc_y.inverse_transform(out))
