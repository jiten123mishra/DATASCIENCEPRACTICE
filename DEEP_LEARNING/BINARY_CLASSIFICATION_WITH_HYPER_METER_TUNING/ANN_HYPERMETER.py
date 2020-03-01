# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 11:18:56 2020

@author: INE12363221
"""
#BASED ON FEATURES DECIDE WHETHER LABEL IS 0 or 1 

#LIBRARIES
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

#PART:01 :DATA PROCESSING 
dataset=pd.read_csv('Churn_Modelling.csv')
#As we know Rownumber ,customer ID and surname not going to make any impact on prediction 
x=dataset.iloc[:,3:13]
y=dataset.iloc[:,13]

#Handle categorical features
x=pd.get_dummies(x,drop_first=True)


#splitting the dataset into the training set and test set 
x=x.values
y=y.values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.2,random_state=42 )

#Feature scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

#how to choose 
#number of hidden layer 
#number of nodes in that hidden layer 
#which  activation function 
#batchsize 
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Embedding, Flatten, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.activations import relu, sigmoid
#SOLUTION IS GRID  SEARCH 
def create_model(layers, activation):
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes,input_shape=(x_train.shape[1],)))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
            
    model.add(Dense(units = 1, kernel_initializer= 'glorot_uniform', activation = 'sigmoid')) # Note: no activation beyond this point
    
    model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
    return model
    
model = KerasClassifier(build_fn=create_model, verbose=1)

layers = [[20], [40, 20], [45, 30, 15]]
#First try with single hidden layer with 20 neurons #2nd try 2 layers first with 40 neuron second with 20 neuron 
activations = ['sigmoid', 'relu']
param_grid = dict(layers=layers, activation=activations, batch_size = [128, 256], epochs=[30])
grid = GridSearchCV(estimator=model, param_grid=param_grid,cv=5)

grid_result = grid.fit(x_train, y_train)

#print best parameters 
print("Best params obtained ",grid_result.best_params_)
print("Best score obtained ",grid_result.best_score_)

#CALCULATE ACCURACY 
pred_y=grid.predict(x_test)
y_pred =(pred_y>0.5)

from sklearn import metrics
 #ACCURACY
print("Accuracy score")
print(metrics.accuracy_score(y_test, y_pred))

