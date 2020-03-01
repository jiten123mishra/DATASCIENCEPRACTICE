import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def model_train(dataSet, filename):
    # Importing the dataset
    dataset = pd.read_csv(dataSet)
    X1 = dataset.iloc[:, [1,2,3]].values
    y = dataset.iloc[:, 4].values
    # Splitting the dataset into the Training set and Test set

    X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size = 0.2, random_state = 0)
    
    # Fitting Multiple Linear Regression to the Training set

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    
    #save the file as a model
    pickle.dump(regressor,open(filename, 'wb'))

