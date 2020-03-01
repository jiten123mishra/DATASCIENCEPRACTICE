import numpy as np
import pickle
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def model_train(dataSet,filename):
    # Importing the dataset
    cancer_data = np.genfromtxt(fname ='breast-cancer-wisconsin.data', delimiter= ',', dtype= float)
    cancer_data = np.delete(arr = cancer_data, obj= 0, axis = 1)

    X = cancer_data[:,range(0,9)]
    Y = cancer_data[:,9]

    imp = Imputer(missing_values="NaN", strategy='median', axis=0)
    X = imp.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    neigh = KNeighborsClassifier(n_neighbors = 5, weights='uniform', algorithm='auto')
    neigh.fit(X_train, y_train) 
    #save the file as a model
    pickle.dump(neigh,open(filename, 'wb'))

