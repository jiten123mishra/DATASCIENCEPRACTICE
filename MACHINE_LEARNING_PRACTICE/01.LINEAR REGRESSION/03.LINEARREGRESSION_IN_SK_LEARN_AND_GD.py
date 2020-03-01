import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv(r"C:\ACAD_GLD\SELF\LR1\SALARY_DATA.csv")
x=df[['EXP']]
y=df[['SAL']]
######
#1ST WAY OF LINEAR REGRESSION 
from sklearn.model_selection import train_test_split
#1/3rd data is test data and 2/3 data is training data 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression(normalize=True)
#pass data to algorithm to adjust value of m and c 
#training phase 
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
#check accuracy of model
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))

#DOING IT OLD WAY TRAINING WITH WHOLE DATA 
#OLD WAY TAKES SERIES AS INPUT
def update_m_c(X,Y):
    m = 0
    c = 0

    L = 0.0001  # The learning Rate
    epochs = 1000 # The number of iterations to perform gradient descent

    n = float(len(X)) # Number of elements in X

        # Performing Gradient Descent 
    for i in range(epochs): 
        Y_pred = m*X + c  # The current predicted value of Y
        D_m = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt m
        D_c = (-2/n) * sum(Y - Y_pred)  # Derivative wrt c
        m = m - L * D_m  # Update m
        c = c - L * D_c  # Update c
    return m,c
#WAY-1
#x_train = df.iloc[:, 0]
#y_train = df.iloc[:, 1]
#WAY-2
x_train=x_train['EXP']
y_train=y_train['SAL']
m,b=update_m_c(x_train,y_train)
y_pred1 = b + m * x_test
print(r2_score(y_test,y_pred1))
