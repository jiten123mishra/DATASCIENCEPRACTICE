import pandas as pd
from pandas import Series,DataFrame

import matplotlib.pyplot as plt
import seaborn as sns

# Gaussian Naive Bayes
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

# load the iris datasets
iris = datasets.load_iris()

# Grab features (X) and the Target (Y)
X = iris.data

Y = iris.target
model = GaussianNB()

from sklearn.model_selection import train_test_split
# Split the data into Trainging and Testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

# Fit the training model
model.fit(X_train,Y_train)

# Predicted outcomes
predicted = model.predict(X_test)

# Actual Expected Outvomes
expected = Y_test

train_score=model.score(X_train,Y_train)
test_score=model.score(X_test,Y_test)

print("using naivebayes")
print("training score =",train_score)
print("test score =",test_score)