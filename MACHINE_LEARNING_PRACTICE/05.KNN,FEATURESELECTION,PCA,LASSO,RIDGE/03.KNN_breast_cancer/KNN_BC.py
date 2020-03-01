# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 10:42:58 2019

@author: INE12363221
"""

import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

cancer_data = np.genfromtxt(
 fname ='breast-cancer-wisconsin.data', delimiter= ',', dtype= float)
print(type(cancer_data))
cancer_data = np.delete(arr = cancer_data, obj= 0, axis = 1)

X = cancer_data[:,range(0,9)]
Y = cancer_data[:,9]
print(X[1][:])

imp = Imputer(missing_values="NaN", strategy='median', axis=0)
X = imp.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
 X, Y, test_size = 0.3, random_state = 100)
y_train = y_train.ravel()
y_test = y_test.ravel()

neigh = KNeighborsClassifier(n_neighbors = 5, weights='uniform', algorithm='auto')
neigh.fit(X_train, y_train) 
y_pred = neigh.predict(X_test)
print("Accuracy is ", accuracy_score(y_test,y_pred)*100,"% for K-Value:",3)

print("Prediction using sample data and knn  ",neigh.predict([[5.1,4.2,4.3,5.2,7.1,10.5,3.7,2.9,1.3]]))

import pickle
filename = 'finalized_model_adv.sav'
pickle.dump(neigh, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))
print("Prediction using sample data and loaded model file ",loaded_model.predict([[5.1,4.2,4.3,5.2,7.1,10.5,3.7,2.9,1.3]]))
