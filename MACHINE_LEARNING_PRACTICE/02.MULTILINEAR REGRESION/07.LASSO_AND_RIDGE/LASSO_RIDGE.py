import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'Advertising.csv')
#Display head of data set 
print(dataset.head())

#DATA CLEANING 
# AS Unnamed column is just a  column is not contributing 
#column 1 TV 2. Radio 3. NewsPaper
x = dataset.iloc[:, [1,2,3]].values
y = dataset.iloc[:, 4].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
#linear regression 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
train_score=regressor.score(x_train,y_train)
test_score=regressor.score(x_test,y_test)
coefficients=regressor.coef_
print("using linear regression")
print("training score =",train_score)
print("test score =",test_score)
print("coefficients are =",coefficients)

#lasso regression 

from sklearn.linear_model import Lasso
regressor = Lasso(alpha=2)
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
train_score=regressor.score(x_train,y_train)
test_score=regressor.score(x_test,y_test)
coefficients=regressor.coef_
print("using lasso regression")
print("training score =",train_score)
print("test score =",test_score)
print("coefficients are =",coefficients)

#Ridge regression 

from sklearn.linear_model import Ridge
regressor = Ridge(alpha=2)
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
train_score=regressor.score(x_train,y_train)
test_score=regressor.score(x_test,y_test)
coefficients=regressor.coef_
print("using lasso regression")
print("training score =",train_score)
print("test score =",test_score)
print("coefficients are =",coefficients)