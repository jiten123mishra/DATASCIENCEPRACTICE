# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
dataset = pd.get_dummies(dataset,drop_first=True)
#Get columns 
#Get all rows and all columns except last one and store as numpy nd array as x
#Get all rows and 4th column i.e profit and store as numpy nd array as y 
x = dataset.iloc[:,[0,1,2,4,5]].values
y = dataset.iloc[:, [3]].values



## Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
#
## Predicting the Test set results
y_pred = regressor.predict(x_test)

#check accuracy of model
from sklearn.metrics import r2_score
print("R2 CSORE IS: ",r2_score(y_test,y_pred))

#REMOVE NON RELEVANT COLUMNS USING CORRELATIN 
import seaborn as sns 
plt.figure(figsize=(12,10))
cor = dataset.corr()
#sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
#plt.show()
cor_target = abs(cor["Profit"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.5]
#print(relevant_features)
#clearly R&D spend and Marketing spend are real contributor 
x = dataset.loc[:,['R&D Spend','Marketing Spend']].values
y = dataset.loc[:, ['Profit']].values

## Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
#
## Predicting the Test set results
y_pred = regressor.predict(x_test)

#check accuracy of model
from sklearn.metrics import r2_score
print("R2 CSORE IS: ",r2_score(y_test,y_pred))

from sklearn.externals import joblib
filename = 'finalized_model1.sav'
joblib.dump(regressor, filename)
loaded_model = joblib.load(filename)
result = loaded_model.score(x_test, y_test)
print("result from model joblibfile is ",result)