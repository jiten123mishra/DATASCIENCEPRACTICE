import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("SALARY_DATA.csv")
x=df[['EXP']]
y=df[['SAL']]
#SPLITTING DATA INTO TRAINING AND TEST
#1/3rd data is test data and 2/3 data is training data 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

#DOING IT OLD WAY TRAINING WITH WHOLE DATA 
#OLD WAY TAKES SERIES AS INPUT
def call_cost(m,c,X,Y):
    Y_true=Y
    Y_pred = m*X + c
    MSE = np.square(np.subtract(Y_true,Y_pred)).mean() 
    return MSE
def update_m_c(X,Y):
    m = 0
    c = 0

    L = 0.0001  # The learning Rate
    epochs = 1000 # The number of iterations to perform gradient descent

    n = float(len(X)) # Number of elements in X

        # Performing Gradient Descent 
    cost_history=np.zeros(epochs) 
    m_history=np.zeros(epochs) 
    c_history=np.zeros(epochs) 
    dm_history=np.zeros(epochs) 
    dc_history=np.zeros(epochs) 
    for i in range(epochs): 
        Y_pred = m*X + c  # The current predicted value of Y
        D_m = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt m
        D_c = (-2/n) * sum(Y - Y_pred)  # Derivative wrt c
        m = m - L * D_m  # Update m
        c = c - L * D_c  # Update c
        #MSE=1/n*sum(Y - Y_pred)**2
        MSE=call_cost(m,c,X,Y)
        cost_history[i]=MSE
        m_history[i]=m
        c_history[i]=c
        dm_history[i]=D_m
        dc_history[i]=D_c
        
    return m,c,cost_history,m_history,c_history,dm_history,dc_history

x_train=x_train['EXP']
y_train=y_train['SAL']
m,b,cost_history,m_history,c_history,dm_history,dc_history=update_m_c(x_train,y_train)
y_pred1 = b + m * x_test
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred1))
df = pd.DataFrame({'cost_history':cost_history, 'm_history':m_history ,'c_history':c_history,'dm_history':dm_history ,'dc_history':dc_history})

df['cost_history'].plot()
df['dm_history'].plot()
df['c_history'].plot()
from plotly.offline import iplot
import plotly as py
import plotly.tools as tls
import cufflinks as cf
py.offline.init_notebook_mode(connected=True
)
cf.go_offline()
#USE TOTAL CODE TO SEE DATA IN 3D
df[['m_history','c_history','cost_history']].iplot(kind='surface')
print(df.head())
